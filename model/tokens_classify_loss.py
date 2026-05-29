# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class TokensClassificationLoss(nn.Module):
#     """
#     Implements token-level classification supervision for CLIP's vision encoder
#     conforming to the SuperCLIP framework (arXiv:2512.14480).
#     """
#     def __init__(self, vision_dim=768, vocab_size=49408, idf_path="dataset_token_idf.pt"):
#         super().__init__()
#         # 1. Add the lightweight linear projection head to the vision encoder
#         self.classification_head = nn.Linear(vision_dim, vocab_size)
        
#         # 2. Load the pre-computed text token IDF weights
#         try:
#             idf_weights = torch.load(idf_path)
#             # Clip any negative weights to 0.0 to prevent gradient reversal
#             idf_weights = torch.clamp(idf_weights, min=0.0)
#         except FileNotFoundError:
#             print(f"Warning: {idf_path} not found. Defaulting to uniform token weights.")
#             idf_weights = torch.ones(vocab_size)
            
#         # Register as a buffer so it moves to GPU automatically but remains untrainable
#         self.register_buffer("idf_weights", idf_weights)
#         self.vocab_size = vocab_size

#     def forward(self, vision_embeddings, batch_text_ids):
#         """
#         Args:
#             vision_embeddings (Tensor): Pooled visual features [Batch, vision_dim]
#             batch_text_ids (Tensor): Input token IDs from tokenizer [Batch, 77]
#         """
#         batch_size = vision_embeddings.size(0)
#         device = vision_embeddings.device
        
#         # =====================================================================
#         # 1. BUILD DENSE TARGET WORD DISTRIBUTIONS (K-HOT + IDF)
#         # =====================================================================
#         # Construct a binary presence matrix [Batch, Vocab_Size]
#         y_khot = torch.zeros(batch_size, self.vocab_size, device=device)
#         y_khot.scatter_(1, batch_text_ids, 1.0)
        
#         # Apply the Inverse Document Frequency mask
#         weighted_targets = y_khot * self.idf_weights.unsqueeze(0)
        
#         # Normalize each row to form a valid target probability distribution
#         target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
#         # =====================================================================
#         # 2. COMPUTE VISION LOGITS & MULTI-LABEL CROSS ENTRPY
#         # =====================================================================
#         # Project raw vision representations directly into vocabulary space
#         vision_logits = self.classification_head(vision_embeddings) # [Batch, Vocab_Size]
        
#         # Compute Log-Softmax activations safely across the vocabulary
#         log_probs = F.log_softmax(vision_logits, dim=1)
        
#         # Evaluate cross entropy objective (SuperCLIP Loss)
#         classification_loss = -torch.sum(target_distribution * log_probs, dim=1)
        
#         return classification_loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokensClassificationLoss(nn.Module):
    """
    Implements token-level classification supervision for CLIP's vision encoder
    conforming to the SuperCLIP framework (arXiv:2512.14480).
    """
    def __init__(self, vision_dim=768, vocab_size=49408, idf_path="dataset_token_idf.pt", pad_token_id=49407):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # 1. Add the lightweight linear projection head to the vision encoder
        self.classification_head = nn.Linear(vision_dim, vocab_size)
        
        # 2. Load the pre-computed text token IDF weights
        try:
            # Added weights_only=True for security/modern PyTorch compliance
            idf_weights = torch.load(idf_path, weights_only=True)
            idf_weights = torch.clamp(idf_weights, min=0.0)
        except (FileNotFoundError, RuntimeError):
            print(f"Warning: {idf_path} not found or failed to load. Defaulting to uniform token weights.")
            idf_weights = torch.ones(vocab_size)
            
        # Register as a buffer so it moves to GPU automatically but remains untrainable
        self.register_buffer("idf_weights", idf_weights)

    def forward(self, vision_embeddings, batch_text_ids):
        """
        Args:
            vision_embeddings (Tensor): Pooled raw visual features [Batch, vision_dim]
            batch_text_ids (Tensor): Input token IDs from tokenizer [Batch, Seq_Len]
        """
        batch_size = vision_embeddings.size(0)
        device = vision_embeddings.device
        
        # =====================================================================
        # 1. BUILD DENSE TARGET WORD DISTRIBUTIONS (K-HOT + IDF)
        # =====================================================================
        # Create a copy of text IDs to manipulate for padding masking
        clean_text_ids = batch_text_ids.clone()
        
        # Create a mask for valid tokens (ignoring pad tokens)
        # Note: If you also want to ignore EOS/SOS (e.g. 49406, 49407), add them to this mask
        valid_mask = (clean_text_ids != self.pad_token_id)
        
        # Temporarily replace invalid IDs with 0 so scatter_ doesn't break out of bounds,
        # we will zero them out of the target matrix immediately after.
        clean_text_ids[~valid_mask] = 0 
        
        # Construct a binary presence matrix [Batch, Vocab_Size]
        y_khot = torch.zeros(batch_size, self.vocab_size, device=device)
        y_khot.scatter_(1, clean_text_ids, 1.0)
        
        # Explicitly wipe out the 0-index padding target if it got scattered
        y_khot[:, self.pad_token_id] = 0.0
        
        # Apply the Inverse Document Frequency mask
        weighted_targets = y_khot * self.idf_weights.unsqueeze(0)
        
        # Normalize each row to form a valid target probability distribution
        target_distribution = weighted_targets / (weighted_targets.sum(dim=1, keepdim=True) + 1e-6)
        
        # =====================================================================
        # 2. COMPUTE VISION LOGITS & MULTI-LABEL BINARY CROSS ENTROPY
        # =====================================================================
        # Project raw vision representations directly into vocabulary space
        vision_logits = self.classification_head(vision_embeddings) # [Batch, Vocab_Size]
        
        # Compute multi-label binary cross entropy with logits.
        # This applies Sigmoid internally, ensuring non-mutually exclusive token prediction.
        classification_loss = F.binary_cross_entropy_with_logits(
            vision_logits, 
            target_distribution, 
            reduction='none'
        )
        
        # Sum across vocabulary tokens, then average across the batch dimension
        return classification_loss.sum(dim=1).mean()