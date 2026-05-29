import torch
import torch.nn as nn
import torch.nn.functional as F

class TokensClassificationLoss(nn.Module):
    """
    Implements token-level classification supervision for CLIP's vision encoder
    conforming strictly to the SuperCLIP framework (arXiv:2512.14480).
    Fixed for proper multinomial distribution optimization and zero token-0 pollution.
    """
    def __init__(self, vision_dim=768, vocab_size=49408, idf_path="dataset_token_idf.pt", pad_token_id=49407):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # 1. lightweight linear projection head to vocabulary space
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        
        # 2. Load the pre-computed text token IDF weights
        try:
            idf_weights = torch.load(idf_path, weights_only=True)
            idf_weights = torch.clamp(idf_weights, min=0.0)
        except (FileNotFoundError, RuntimeError):
            print(f"Warning: {idf_path} not found or failed to load. Defaulting to uniform token weights.")
            idf_weights = torch.ones(vocab_size)
            
        self.register_buffer("idf_weights", idf_weights)

    def forward(self, vision_embeddings, batch_text_ids):
        """
        Args:
            vision_embeddings (Tensor): Pooled raw visual features [Batch, vision_dim]
            batch_text_ids (Tensor): Input token IDs directly from tokenizer [Batch, Seq_Len]
        """
        batch_size = vision_embeddings.size(0)
        device = vision_embeddings.device
        
        # =====================================================================
        # 1. BUILD DENSE TARGET WORD DISTRIBUTIONS (K-HOT + IDF)
        # =====================================================================
        # Construct raw binary matrix safely without index-0 modifications
        y_khot = torch.zeros(batch_size, self.vocab_size, dtype=vision_embeddings.dtype, device=device)
        y_khot.scatter_(1, batch_text_ids, 1.0)
        
        # CRITICAL FIX: Wipe out special structural tokens directly from their actual slots
        # This includes padding (49407) and optionally SOS (49406) if present in your tokenizer
        if self.pad_token_id < self.vocab_size:
            y_khot[:, self.pad_token_id] = 0.0
        if (self.pad_token_id - 1) < self.vocab_size: # Safely clears out SOS/EOS system tokens
            y_khot[:, self.pad_token_id - 1] = 0.0
            
        # Apply Inverse Document Frequency mask to suppress generic urban noise words
        weighted_targets = y_khot * self.idf_weights.unsqueeze(0)
        
        # Normalize each row into a valid probability distribution (Sums to 1.0 per sample)
        target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
        # =====================================================================
        # 2. COMPUTE VISION LOGITS & MULTI-LABEL MULTINOMIAL CROSS ENTROPY
        # =====================================================================
        # Project visual embeddings into vocabulary logits
        vision_logits = self.classification_head(vision_embeddings) # [Batch, Vocab_Size]
        
        # CRITICAL FIX: Revert to Cross-Entropy over Log-Softmax layout matching SuperCLIP
        # This forces the network to scale correct tokens relatively higher than wrong ones.
        log_probs = F.log_softmax(vision_logits, dim=1)
        
        # Sum strictly across vocabulary tokens first (retains whole-sample energy)
        # then apply the batch mean operation.
        classification_loss = -torch.sum(target_distribution * log_probs, dim=1)
        
        return classification_loss.mean()