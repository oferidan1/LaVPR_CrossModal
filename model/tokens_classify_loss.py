import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

class TokensClassificationLoss_git(nn.Module):
    """
    Standalone Classification Supervision Module extracted directly from the 
    official SuperCLIP implementation (arXiv:2512.14480).
    
    Stripped of global contrastive CLIP loss and distributed DDP boilerplate 
    to provide pure token-grounding regularisation for local VPR runs.
    """
    def __init__(self, vision_dim=768, vocab_size=49408, idf_path=None, pad_id=49407):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        
        # ראש סיווג לניארי קל משקל מעל פיצ'רי הראייה הגולמיים (למשל ViT-B 768)
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        print(f'SuperCLIP Local Classification active - ignoring pad_id: {pad_id}')

    def loss(self, logits, targets):
        """
        Original SuperCLIP loss function syntax.
        Applies L1 normalization to the target distribution and evaluates 
        multinomial cross-entropy over log-softmax activations.
        """
        norm_item = F.normalize(targets, p=1, dim=1)
        loss = -(F.log_softmax(logits, dim=1) * norm_item).sum(dim=1).mean()
        return loss

    def reweight_targets(self, cap_fq, num_samples, targets):
        """
        Original SuperCLIP target reweighting function syntax.
        Computes dynamic online TF-IDF scaling on the local device.
        """
        # עדכון תדרי המילים הריצה מתוך ה-Batch הנוכחי
        cap_fq += targets.sum(dim=0, keepdim=True) / targets.shape[0]   
        num_samples += 1
            
        # חישוב משקולות ה-IDF המקוריות (מותאם למחשב בודד / world_size=1)
        batch_size = targets.shape[0]
        targets = targets * torch.log((num_samples + 1.0 / batch_size) / (cap_fq + 1.0 / batch_size)).to(dtype=targets.dtype)
        return targets

    def forward(self, cap_fq, num_samples, vision_embeddings, batch_text_ids):
        """
        Args:
            cap_fq (Tensor): Shared running buffer tracking token document frequencies [1, Vocab_Size]
            num_samples (Tensor): Shared running integer tracker tracking total processed batches [1]
            vision_embeddings (Tensor): Pooled raw visual features [Batch, vision_dim]
            batch_text_ids (Tensor): Input token IDs directly from your tokenizer [Batch, 77]
        """
        # 1. הפעלת ראש הסיווג מעל הפיצ'רים הויזואליים לקבלת לוג'יטים על פני כל המילון
        logits = self.classification_head(vision_embeddings) # [Batch, Vocab_Size]
        
        # 2. בניית מטריצת המטרות (Targets) בפורמט One-Hot / K-Hot
        B, C = logits.shape
        targets = torch.zeros(B, C, dtype=logits.dtype, device=logits.device)
        
        # פיזור ערכי 1.0 לתוך האינדקסים הלשוניים המקוריים של הצימוד
        targets.scatter_(dim=1, index=batch_text_ids, value=1.0) 
        
        # 🛡️ הגנה קריטית: איפוס ישיר של עמודת ה-Padding (49407) במטריצת ה-Targets
        # זה מונע מטוקני ה-Padding הריקים להשפיע על פונקציית הלמידה,
        # ומצד שני – שומר על אינדקס 0 נקי לחלוטין מרעשים!
        if self.pad_id < C:
            targets[:, self.pad_id] = 0.0
            
        # 3. שקלול המטרות באמצעות ה-Online TF-IDF המקורי
        targets = self.reweight_targets(cap_fq, num_samples, targets)
        
        # 4. חישוב פונקציית ההפסד המקורית (L1-Norm + Log-Softmax Cross Entropy)
        class_loss = self.loss(logits, targets)
        
        return class_loss
    

class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

class TokensClassificationLoss(nn.Module):
    """
    Strict SuperCLIP token-grounding loss optimized for Full-Weight Fine-Tuning.
    Protects the highly flexible ViT backbone from gradient flooding using internal scaling.
    """
    def __init__(self, vision_dim=768, vocab_size=49408, idf_path="dataset_token_idf.pt", pad_token_id=49407, grad_scale=0.05):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.grad_scale = grad_scale # שומר על משקולות ה-ViT מפני הצפה
        
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        
        try:
            idf_weights = torch.load(idf_path, weights_only=True)
            idf_weights = torch.clamp(idf_weights, min=0.0)
        except (FileNotFoundError, RuntimeError):
            print(f"Warning: {idf_path} not found. Defaulting to uniform weights.")
            idf_weights = torch.ones(vocab_size)
            
        self.register_buffer("idf_weights", idf_weights)

    def forward(self, vision_embeddings, batch_text_ids):
        # 🛡️ הגנה אקטיבית על ה-ViT באימון מלא: החלשת הגרדיאנטים הלשוניים ב-95%
        scaled_vision_features = GradientScaleFunction.apply(vision_embeddings, self.grad_scale)
        logits = self.classification_head(scaled_vision_features)
        
        B, C = logits.shape
        targets = torch.zeros(B, C, dtype=logits.dtype, device=logits.device)
        targets.scatter_(1, batch_text_ids, 1.0)
        
        # ניקוי קשיח של טוקני ה-Padding והמערכת
        if self.pad_token_id < C:
            targets[:, self.pad_token_id] = 0.0
        if (self.pad_token_id - 1) < C:
            targets[:, self.pad_token_id - 1] = 0.0
            
        # שקלול IDF סטטי ומיוצב
        weighted_targets = targets * self.idf_weights.unsqueeze(0)
        target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
        # חישוב ה-Loss המקורי (Multinomial Cross Entropy over Log-Softmax)
        log_probs = F.log_softmax(logits, dim=1)
        classification_loss = -torch.sum(target_distribution * log_probs, dim=1)
        
        return classification_loss.mean()
    


class HierarchicalTokensLoss(nn.Module):
    """
    Strict SuperCLIP Hierarchical token-grounding loss optimized for Full-Weight Fine-Tuning.
    Combines blended Hierarchical IDF (Image + Location) with dynamic Term Frequency (TF) scaling,
    while protecting the highly flexible ViT backbone from gradient flooding.
    """
    def __init__(self, vision_dim=768, vocab_size=49408, 
                 image_idf_path="datasets/gsv_cities_image_idf_clipb16.pt", 
                 location_idf_path="datasets/gsv_cities_location_idf_clipb16.pt", 
                 pad_token_id=49407, grad_scale=0.05,
                 alpha=0.6, target_initial_loss=4.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.grad_scale = grad_scale  # Protects ViT weights from gradient flooding
        self.alpha = alpha
        
        # 1. Linear Classification Head hooked onto raw visual dimensions
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        
        # 2. Securely load precomputed tensors with fallback safety
        try:
            img_idf = torch.load(image_idf_path, weights_only=True).clamp(min=0.0)
            loc_idf = torch.load(location_idf_path, weights_only=True).clamp(min=0.0)
        except (FileNotFoundError, RuntimeError):
            print(f"Warning: Hierarchical IDF files not found. Defaulting to uniform weights.")
            img_idf = torch.ones(vocab_size)
            loc_idf = torch.ones(vocab_size)
        
        # 3. Apply formula to precompute static global weights
        combined_idf = (self.alpha * img_idf) + ((1.0 - self.alpha) * loc_idf)
        self.register_buffer("global_idf", combined_idf)
        
        # 4. Dynamic scaling to balance auxiliary loss with ranking loss
        initial_mean_loss = -torch.log(torch.tensor(0.5)).item()
        self.loss_scale = target_initial_loss / initial_mean_loss

    def forward(self, vision_embeddings, batch_text_ids):
        """
        Args:
            vision_embeddings (Tensor): Unprojected pooled ViT features [Batch, vision_dim]
            batch_text_ids (Tensor): Target caption token IDs from tokenizer [Batch, Seq_Len]
        """
        batch_size = vision_embeddings.size(0)
        device = vision_embeddings.device
        
        # 🛡️ Active ViT Backbone Protection: Suppress linguistic gradients during full fine-tuning
        scaled_vision_features = GradientScaleFunction.apply(vision_embeddings, self.grad_scale)
        
        # =====================================================================
        # 1. VECTORIZED TERM FREQUENCIES (TF) CALCULATION
        # =====================================================================
        # Build an explicit mask for invalid/padding tokens based on your original rules
        invalid_mask = (batch_text_ids == 0) | \
                       (batch_text_ids == self.pad_token_id) | \
                       (batch_text_ids == (self.pad_token_id - 1))
        
        # Prepare updates: set step increments to 0 for padding tokens
        increments = torch.ones_like(batch_text_ids, dtype=torch.float32, device=device)
        increments[invalid_mask] = 0.0
        
        # Prevent out-of-bounds errors on target index scatter mapping
        safe_text_ids = batch_text_ids.clamp(0, self.vocab_size - 1)
        
        # Compute dynamic raw counts purely in parallel
        tokens_count = torch.zeros(batch_size, self.vocab_size, device=device)
        tokens_count.scatter_add_(1, safe_text_ids, increments)
        
        # Apply Augmented TF scaling across the matrix row-wise
        max_tf = tokens_count.max(dim=1, keepdim=True).values
        max_tf = torch.where(max_tf > 0, max_tf, torch.ones_like(max_tf)) # Guard against div-by-zero
        
        tf_matrix = torch.where(
            tokens_count > 0, 
            0.5 + 0.5 * (tokens_count / max_tf), 
            torch.zeros_like(tokens_count)
        )

        # =====================================================================
        # 2. COMBINE DYNAMIC TF WITH THE BLENDED HIERARCHICAL IDF
        # =====================================================================
        weighted_targets = tf_matrix * self.global_idf.unsqueeze(0)
        
        # Double-check rigid cleanup of system token frequencies 
        if self.pad_token_id < self.vocab_size:
            weighted_targets[:, self.pad_token_id] = 0.0
        if (self.pad_token_id - 1) < self.vocab_size:
            weighted_targets[:, self.pad_token_id - 1] = 0.0
            
        # Row-normalize to build an uncollapsible sparse target distribution
        target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
        # =====================================================================
        # 3. BINARY CROSS ENTROPY LOSS SETUP WITH GRADIENT CONTROL
        # =====================================================================
        vision_logits = self.classification_head(scaled_vision_features)
        
        classification_loss = F.binary_cross_entropy_with_logits(
            vision_logits, 
            target_distribution, 
            reduction='none'
        )
        
        # Average across vocabulary, then average across batch, scale dynamically
        mean_vocab_loss = classification_loss.mean(dim=1).mean()
        return mean_vocab_loss * self.loss_scale
    
    
# class VocabClassificationLoss(nn.Module):
#     def __init__(self, vision_dim=768, vocab_path="scene_graph_vocab.json", 
#                  image_idf_path="gsv_cities_image_idf.pt", target_initial_loss=4.0):
#         super().__init__()
        
#         with open(vocab_path, "r") as f:
#             self.vocab_size = len(json.load(f))
            
#         # Linear tracking classification head projects backbone features to vocab channels
#         self.classification_head = nn.Linear(vision_dim, self.vocab_size)
        
#         # Load and lock the static global image IDF matrix weights
#         img_idf = torch.load(image_idf_path, weights_only=True).clamp(min=0.0)
#         self.register_buffer("idf_weights", img_idf)
        
#         # Numerical scaling factor ensuring the loss scales gracefully alongside standard ranking losses
#         initial_mean_loss = -math.log(0.5) 
#         self.loss_scale = target_initial_loss / initial_mean_loss

#     def forward(self, vision_embeddings, batch_concept_ids):
#         batch_size = vision_embeddings.size(0)
#         device = vision_embeddings.device
        
#         # 1. Calculate the dynamic localized Term Frequency (TF) weights per batch index
#         tf_matrix = torch.zeros(batch_size, self.vocab_size, device=device)
        
#         for b in range(batch_size):
#             valid_tokens = batch_concept_ids[b][batch_concept_ids[b] != 0] # Exclude <PAD> tracking
#             if len(valid_tokens) > 0:
#                 tokens_count = torch.bincount(valid_tokens, minlength=self.vocab_size)
#                 max_tf = tokens_count.max()
#                 if max_tf > 0:
#                     # Augmented TF formula prevents highly repetitive items from dominating gradients
#                     tf_matrix[b] = torch.where(
#                         tokens_count > 0, 
#                         0.5 + 0.5 * (tokens_count.float() / max_tf), 
#                         0.0
#                     )

#         # 2. Construct probability targets by multiplying TF by static global IDF rarity bounds
#         weighted_targets = tf_matrix * self.idf_weights.unsqueeze(0)
#         target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
#         # 3. Model forward projection and multi-label BCE loss calculation
#         predicted_logits = self.classification_head(vision_embeddings)
        
#         classification_loss = F.binary_cross_entropy_with_logits(
#             predicted_logits, 
#             target_distribution, 
#             reduction='none'
#         )
        
#         return classification_loss.mean(dim=1).mean() * self.loss_scale 

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class VocabClassificationLoss(nn.Module):
    def __init__(self, vision_dim, vocab_path="scene_graph_vocab.json", 
                 image_idf_path="gsv_cities_image_idf.pt", target_initial_loss=4.0, grad_scale=0.05):
        super().__init__()
        
        # Load static global token Inverse Document Frequency trajectories
        img_idf = torch.load(image_idf_path, weights_only=True).clamp(min=0.0)
        self.register_buffer("idf_weights", img_idf)
        self.vocab_size = img_idf.size(0)
        
        # Linear tracking classification head projects backbone features to vocab channels
        self.classification_head = nn.Linear(vision_dim, self.vocab_size)
        
        # Balancing scale tracking constants
        initial_mean_loss = -torch.log(torch.tensor(0.5))
        self.loss_scale = target_initial_loss / initial_mean_loss
        self.loss_scale = 1
        self.grad_scale = grad_scale 

    def forward(self, vision_embeddings, batch_concept_ids):
        """
        Args:
            vision_embeddings (torch.Tensor): Raw model outputs 
            batch_concept_ids (torch.Tensor): LongTensor filled with word index targets. Shape: (N, Seq_Len)
        """
        if batch_concept_ids is None:
            return torch.tensor(0.0, device=vision_embeddings.device, requires_grad=True)
        
        scaled_vision_features = GradientScaleFunction.apply(vision_embeddings, self.grad_scale)
        logits = self.classification_head(scaled_vision_features)
            
        batch_size = logits.size(0)
        device = logits.device
        
        # --- 1. FULLY VECTORIZED TERM FREQUENCY (TF) CALCULATION ---
        # Initialize flat allocation allocation tensor maps
        tf_matrix = torch.zeros(batch_size, self.vocab_size, device=device, dtype=torch.float32)
        ones = torch.ones_like(batch_concept_ids, dtype=torch.float32, device=device)
        
        # Eliminate loop by accumulating word counts in parallel across the batch dim
        tf_matrix.scatter_add_(1, batch_concept_ids, ones)
        
        # Force ignore <PAD> tokens at index 0
        tf_matrix[:, 0] = 0.0
        
        # Extract individual row maxima to compute augmented structural frequencies
        max_tf = tf_matrix.max(dim=1, keepdim=True)[0]
        
        # Vectorized augmented TF mapping: 0.5 + 0.5 * (count / max)
        tf_matrix = torch.where(
            tf_matrix > 0,
            0.5 + 0.5 * (tf_matrix / (max_tf + 1e-8)),
            torch.zeros_like(tf_matrix)
        )

        # --- 2. LOGIT-SAFE TARGET DISTRIBUTION SCALING ---
        # Multiply our vectorized batch TF matrix by our registered global buffer weights
        weighted_targets = tf_matrix * self.idf_weights.unsqueeze(0)
        
        # CRITICAL REPAIR: Normalize by the max value per row instead of the sum.
        # This keeps prominent landmark class targets pinned at 1.0 so BCE functions normally.
        row_max = weighted_targets.max(dim=1, keepdim=True)[0]
        target_distribution = weighted_targets / (row_max + 1e-8)        
        
        # --- 3. MULTI-LABEL BINARY CROSS-ENTROPY EVALUATION ---
        classification_loss = F.binary_cross_entropy_with_logits(
            logits, 
            target_distribution, 
            reduction="mean"
        )
        
        return classification_loss * self.loss_scale