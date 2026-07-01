import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json

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
    def __init__(self, vision_dim=768, vocab_size=49408, idf_path="dataset_token_idf.pt", pad_token_id=49407, grad_scale=0.05, cls_adapter=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.grad_scale = grad_scale # שומר על משקולות ה-ViT מפני הצפה
        
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        
        self.cls_adapter = cls_adapter
        if cls_adapter:
            self.word_bridge_norm = nn.LayerNorm(vision_dim)
            self.word_bridge_adapter = nn.Sequential(
                nn.Linear(vision_dim, 256),
                nn.GELU(),
                nn.Linear(256, vision_dim)
            )
        
        try:
            idf_weights = torch.load(idf_path, weights_only=True)
            idf_weights = torch.clamp(idf_weights, min=0.0)
        except (FileNotFoundError, RuntimeError):
            print(f"Warning: {idf_path} not found. Defaulting to uniform weights.")
            idf_weights = torch.ones(vocab_size)
            
        self.register_buffer("idf_weights", idf_weights)

    def forward(self, vision_embeddings, batch_text_ids):
          
        # Normalize and pass through the adapter to cushion backward gradients
        if self.cls_adapter:
            normalized_words = self.word_bridge_norm(vision_embeddings)
            vision_embeddings = vision_embeddings + self.word_bridge_adapter(normalized_words)
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


class VocabClassificationLoss(nn.Module):
    def __init__(self, vision_dim, vocab_path="scene_graph_vocab.json", 
                 image_idf_path="gsv_cities_image_idf.pt", grad_scale=0.05, cls_adapter=0):
        super().__init__()
        
        # Load static global token Inverse Document Frequency trajectories
        img_idf = torch.load(image_idf_path, weights_only=True).clamp(min=0.0)
        self.register_buffer("idf_weights", img_idf)
        self.vocab_size = img_idf.size(0)
        
        # Linear tracking classification head projects backbone features to vocab channels
        self.classification_head = nn.Linear(vision_dim, self.vocab_size)
        
        # Balancing scale tracking constants
        self.grad_scale = grad_scale 
        self.cls_adapter = cls_adapter
        
        if cls_adapter:
            self.word_bridge_norm = nn.LayerNorm(vision_dim)
            self.word_bridge_adapter = nn.Sequential(
                nn.Linear(vision_dim, 256),
                nn.GELU(),
                nn.Linear(256, vision_dim)
            )

    def forward(self, vision_embeddings, batch_concept_ids):
        """
        Args:
            vision_embeddings (torch.Tensor): Raw model outputs 
            batch_concept_ids (torch.Tensor): LongTensor filled with word index targets. Shape: (N, Seq_Len)
        """
        if batch_concept_ids is None:
            return torch.tensor(0.0, device=vision_embeddings.device, requires_grad=True)
        
        # Normalize and pass through the adapter to cushion backward gradients
        if self.cls_adapter:
            normalized_words = self.word_bridge_norm(vision_embeddings)
            vision_embeddings = vision_embeddings + self.word_bridge_adapter(normalized_words)
        
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
        
        weighted_targets = tf_matrix * self.idf_weights.unsqueeze(0)
        target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)        
        
        log_probs = F.log_softmax(logits, dim=1)
        classification_loss = -torch.sum(target_distribution * log_probs, dim=1)        
        
        return classification_loss.mean()
    
# class VocabClassificationLoss(nn.Module):
#     def __init__(self, vision_dim, vocab_path="scene_graph_vocab.json", 
#                  image_idf_path="gsv_cities_image_idf.pt", target_initial_loss=4.0, grad_scale=0.05):
#         super().__init__()
        
#         # Load static global token Inverse Document Frequency trajectories
#         img_idf = torch.load(image_idf_path, weights_only=True).clamp(min=0.0)
#         self.register_buffer("idf_weights", img_idf)
#         self.vocab_size = img_idf.size(0)
        
#         # Linear tracking classification head projects backbone features to vocab channels
#         self.classification_head = nn.Linear(vision_dim, self.vocab_size)
        
#         # Balancing scale tracking constants
#         initial_mean_loss = -torch.log(torch.tensor(0.5))
#         self.loss_scale = target_initial_loss / initial_mean_loss
#         self.loss_scale = 1
#         self.grad_scale = grad_scale 

#     def forward(self, vision_embeddings, batch_concept_ids):
#         """
#         Args:
#             vision_embeddings (torch.Tensor): Raw model outputs 
#             batch_concept_ids (torch.Tensor): LongTensor filled with word index targets. Shape: (N, Seq_Len)
#         """
#         if batch_concept_ids is None:
#             return torch.tensor(0.0, device=vision_embeddings.device, requires_grad=True)
        
#         scaled_vision_features = GradientScaleFunction.apply(vision_embeddings, self.grad_scale)
#         logits = self.classification_head(scaled_vision_features)
            
#         batch_size = logits.size(0)
#         device = logits.device
        
#         # --- 1. FULLY VECTORIZED TERM FREQUENCY (TF) CALCULATION ---
#         # Initialize flat allocation allocation tensor maps
#         tf_matrix = torch.zeros(batch_size, self.vocab_size, device=device, dtype=torch.float32)
#         ones = torch.ones_like(batch_concept_ids, dtype=torch.float32, device=device)
        
#         # Eliminate loop by accumulating word counts in parallel across the batch dim
#         tf_matrix.scatter_add_(1, batch_concept_ids, ones)
        
#         # Force ignore <PAD> tokens at index 0
#         tf_matrix[:, 0] = 0.0
        
#         # Extract individual row maxima to compute augmented structural frequencies
#         max_tf = tf_matrix.max(dim=1, keepdim=True)[0]
        
#         # Vectorized augmented TF mapping: 0.5 + 0.5 * (count / max)
#         tf_matrix = torch.where(
#             tf_matrix > 0,
#             0.5 + 0.5 * (tf_matrix / (max_tf + 1e-8)),
#             torch.zeros_like(tf_matrix)
#         )

#         # --- 2. LOGIT-SAFE TARGET DISTRIBUTION SCALING ---
#         # Multiply our vectorized batch TF matrix by our registered global buffer weights
#         weighted_targets = tf_matrix * self.idf_weights.unsqueeze(0)
        
#         # CRITICAL REPAIR: Normalize by the max value per row instead of the sum.
#         # This keeps prominent landmark class targets pinned at 1.0 so BCE functions normally.
#         row_max = weighted_targets.max(dim=1, keepdim=True)[0]
#         target_distribution = weighted_targets / (row_max + 1e-8)        
        
#         # --- 3. MULTI-LABEL BINARY CROSS-ENTROPY EVALUATION ---
#         classification_loss = F.binary_cross_entropy_with_logits(
#             logits, 
#             target_distribution, 
#             reduction="mean"
#         )
        
#         return classification_loss * self.loss_scale