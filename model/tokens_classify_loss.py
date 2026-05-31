import torch
import torch.nn as nn
import torch.nn.functional as F


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