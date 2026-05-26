import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMultiSimilarityLossCM(nn.Module):
    """
    Contrastive Consensus-Weighted Multi-Similarity (CCW-MS) Loss 
    for Cross-Modal Visual Place Recognition.
    
    Fully conforms to the mathematical formulation in cross_lavpr.pdf.
    """
    def __init__(self, alpha=2.0, beta=25.0, base=0.30, tau=0.07, eps=1e-5):
        """
        Args:
            alpha (float): Positive scale hyperparameter (alpha in Eq 3 & 5)[cite: 44, 54].
            beta (float): Negative scale hyperparameter (beta in Eq 4 & 5)[cite: 49, 54].
            base (float): Distance margin threshold (lambda in Eq 3 & 4).
            tau (float): Temperature hyperparameter for cross-location softmax (tau in Eq 2).
            eps (float): Guard constant to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base  # Maps to \lambda in the paper 
        self.tau = tau
        self.eps = eps

    def forward(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        """
        Args:
            embeddings (Tensor): Visual embeddings v_i [B_v, D] [cite: 18, 24]
            labels (Tensor): Location labels for images [B_v] [cite: 18]
            ref_emb (Tensor): Text embeddings t_{j,k} [B_t, D] [cite: 25, 26]
            ref_labels (Tensor): Location labels for text descriptions [B_t] [cite: 18]
        """
        v = embeddings
        v_labels = labels
        t = ref_emb if ref_emb is not None else embeddings
        t_labels = ref_labels if ref_labels is not None else labels

        # L2 Normalization to ensure true cosine similarities [cite: 21, 28]
        v = F.normalize(v, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)

        # =====================================================================
        # 3.1 Cross-Location Contrastive Consensus Weighting
        # =====================================================================
        # Pairwise text cosine similarity matrix: [B_t, B_t] (cos(t_{i,k}, t_{j,m})) [cite: 33]
        T_sim = torch.matmul(t, t.T) 

        # Identify unique locations in the text batch to form target profiles
        unique_labels, inverse_indices = torch.unique(t_labels, return_inverse=True)
        N_locs = unique_labels.size(0) # Total distinct locations N [cite: 18, 23]

        # Indicator mask mapping each text sample to its unique location profile: [N_locs, B_t]
        loc_indicator = (unique_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
        loc_counts = loc_indicator.sum(dim=1, keepdim=True) + self.eps
        
        # Profile aggregation matrix normalized by token counts per location (1 / |T_j|) [cite: 33]
        M = loc_indicator / loc_counts 

        # Equation (1): Context consensus matrix C(t_{i,k}, T_j) -> Shape: [B_t, N_locs] [cite: 33]
        C_matrix = torch.matmul(T_sim, M.T)

        # Equation (2): Cross-location softmax normalization over the N locations 
        W_softmax = F.softmax(C_matrix / self.tau, dim=1) # [B_t, N_locs]

        # Extract consensus weight for the true location identity (j = i numerator) [cite: 36, 39]
        true_loc_indices = inverse_indices
        W_text = W_softmax[torch.arange(t.size(0), device=t.device), true_loc_indices] # [B_t]

        # =====================================================================
        # 3.2 & 3.3 Cross-Modal Baseline Similarity & Masking
        # =====================================================================
        # S_VT(v_i, t_{j,k}) matrix -> Shape: [B_v, B_t] [cite: 21, 28]
        S_VT = torch.matmul(v, t.T) 
        
        # Ground-truth pair sets: Positive P(i) and Negative N(i) [cite: 20, 27]
        pos_mask = (v_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask

        # Online Internal Cross-Modal Hard Mining (Safeguard for b_acc stable updates)
        with torch.no_grad():
            max_neg = (S_VT - 2.0 * pos_mask).max(dim=1, keepdim=True)[0]
            min_pos = (S_VT + 2.0 * neg_mask).min(dim=1, keepdim=True)[0]
        
        mining_pos_mask = (S_VT > (max_neg - 0.1)).float() * pos_mask
        mining_neg_mask = (S_VT < (min_pos + 0.1)).float() * neg_mask

        # Broadcast text weights [B_t] across the visual batch dimension -> [1, B_t]
        W_expanded = W_text.unsqueeze(0)

        # =====================================================================
        # 3.2 Weighted Multi-Positive Pull Term
        # =====================================================================
        # Core mathematical wrapper matching Eq (3): -\alpha * W * (S_VT - \lambda) [cite: 44]
        pull_inputs = -self.alpha * W_expanded * (S_VT - self.base)
        
        # Numerically stable gating to prevent underflow/overflow under FP16 execution
        pull_inputs = torch.where(mining_pos_mask.bool(), pull_inputs, torch.tensor(-1e4, device=v.device))
        pull_exp = torch.exp(pull_inputs)
        
        # Pull(i) log-sum-exp pool per visual instance [cite: 44]
        pull_term = torch.log1p(pull_exp.sum(dim=1)) # [B_v]

        # =====================================================================
        # 3.3 Weighted Negative Push Term
        # =====================================================================
        # Core mathematical wrapper matching Eq (4): \beta * W * (S_VT - \lambda) [cite: 49]
        push_inputs = self.beta * W_expanded * (S_VT - self.base)
        
        # Pre-exponent masking to eliminate inf * 0.0 = NaN failure paths entirely
        push_inputs = torch.where(mining_neg_mask.bool(), push_inputs, torch.tensor(-1e4, device=v.device))
        push_exp = torch.exp(push_inputs)
        
        # Push(i) log-sum-exp pool per visual instance [cite: 49]
        push_term = torch.log1p(push_exp.sum(dim=1)) # [B_v]

        # =====================================================================
        # 3.4 Total Objective Function
        # =====================================================================
        # Equation (5): Combine terms weighted by inverse scale parameters [cite: 54]
        loss_per_instance = (1.0 / self.alpha) * pull_term + (1.0 / self.beta) * push_term # [B_v] [cite: 54]
        
        # Return average over the entire training batch [cite: 56]
        return loss_per_instance.mean()