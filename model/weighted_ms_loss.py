# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class WeightedMultiSimilarityLossCM(nn.Module):
#     """
#     Residual Consensus-Weighted Multi-Similarity Loss for Cross-Modal VPR.
#     Uses (1 + s * W) gating to prevent gradient underflow and accelerate learning
#     while prioritizing highly discriminative architectural descriptors.
#     """
#     def __init__(self, alpha=2.0, beta=25.0, base=0.30, tau=0.07, s=2.0, eps=1e-5):
#         """
#         Args:
#             alpha (float): Positive scale hyperparameter (Eq 3).
#             beta (float): Negative scale hyperparameter (Eq 4).
#             base (float): Distance margin threshold (lambda).
#             tau (float): Temperature for cross-location softmax (Eq 2).
#             s (float): Scaling factor for the residual consensus boost.
#             eps (float): Guard constant to avoid division by zero.
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.base = base
#         self.tau = tau
#         self.s = s  # Residual scale element factor
#         self.eps = eps

#     def forward(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
#         v = embeddings
#         v_labels = labels
#         t = ref_emb if ref_emb is not None else embeddings
#         t_labels = ref_labels if ref_labels is not None else labels

#         # L2 Normalization
#         v = F.normalize(v, p=2, dim=1)
#         t = F.normalize(t, p=2, dim=1)

#         # =====================================================================
#         # 1. CONTRASTIVE SOFTMAX CONSENSUS WEIGHTING (W_text) [cite: 29]
#         # =====================================================================
#         T_sim = torch.matmul(t, t.T) 

#         unique_labels, inverse_indices = torch.unique(t_labels, return_inverse=True)
        
#         loc_indicator = (unique_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
#         loc_counts = loc_indicator.sum(dim=1, keepdim=True) + self.eps
#         M = loc_indicator / loc_counts 

#         # Compute context consensus matrix C(t_{i,k}, T_j) [cite: 32, 33]
#         C_matrix = torch.matmul(T_sim, M.T)

#         # Cross-location Softmax [cite: 35, 36]
#         W_softmax = F.softmax(C_matrix / self.tau, dim=1) 

#         # Extract true identity weights [cite: 36, 39]
#         W_text = W_softmax[torch.arange(t.size(0), device=t.device), inverse_indices] 

#         # =====================================================================
#         # 2. IMPLEMENTING YOUR PROPOSAL: (1 + s * W) RESIDUAL HIGHWAY
#         # =====================================================================
#         # Bounds the multiplier between [1.0, 1.0 + s] instead of [0.0, 1.0]
#         W_residual = 1.0 + self.s * W_text
#         W_expanded = W_residual.unsqueeze(0) # [1, B_t]

#         # =====================================================================
#         # 3. CROSS-MODAL SIMILARITY & BUILT-IN MINING
#         # =====================================================================
#         S_VT = torch.matmul(v, t.T) # [B_v, B_t]
        
#         pos_mask = (v_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
#         neg_mask = 1.0 - pos_mask

#         with torch.no_grad():
#             max_neg = (S_VT - 2.0 * pos_mask).max(dim=1, keepdim=True)[0]
#             min_pos = (S_VT + 2.0 * neg_mask).min(dim=1, keepdim=True)[0]
        
#         mining_pos_mask = (S_VT > (max_neg - 0.1)).float() * pos_mask
#         mining_neg_mask = (S_VT < (min_pos + 0.1)).float() * neg_mask

#         # =====================================================================
#         # 4. STABLE OBJECTIVE FUNCTIONS WITH RESIDUAL WEIGHTS
#         # =====================================================================
#         # Pull Term [cite: 44]
#         pull_inputs = -self.alpha * W_expanded * (S_VT - self.base)
#         pull_inputs = torch.where(mining_pos_mask.bool(), pull_inputs, torch.tensor(-1e4, device=v.device))
#         pull_exp = torch.exp(pull_inputs)
#         pull_term = torch.log1p(pull_exp.sum(dim=1))

#         # Push Term [cite: 49]
#         push_inputs = self.beta * W_expanded * (S_VT - self.base)
#         push_inputs = torch.where(mining_neg_mask.bool(), push_inputs, torch.tensor(-1e4, device=v.device))
#         push_exp = torch.exp(push_inputs)
#         push_term = torch.log1p(push_exp.sum(dim=1))

#         # Combine inverse scales [cite: 54]
#         loss_per_instance = (1.0 / self.alpha) * pull_term + (1.0 / self.beta) * push_term
        
#         return loss_per_instance.mean()

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class WeightedMultiSimilarityLossCM(nn.Module):
#     """
#     Symmetric Contrastive Consensus-Weighted Multi-Similarity (SCCW-MS) Loss.
#     Enhanced with Active Cross-Modal Hard Negative Mining for CLS/EOS Pooling.
#     """
#     def __init__(self, alpha=2.0, beta=25.0, lambda_margin=0.30, gamma=0.5, tau_t=0.07, tau_v=0.07, eps=1e-5):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.lambda_margin = lambda_margin  
#         self.gamma = gamma
#         self.tau_t = tau_t
#         self.tau_v = tau_v
#         self.eps = eps

#     def forward(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
#         v = embeddings
#         v_labels = labels
#         t = ref_emb if ref_emb is not None else embeddings
#         t_labels = ref_labels if ref_labels is not None else labels

#         # Ensure true cosine similarities via L2 Normalization
#         v = F.normalize(v, p=2, dim=1)
#         t = F.normalize(t, p=2, dim=1)

#         # =====================================================================
#         # 3.1.1 Intra-Modal Textual Consensus Weighting (W_text)
#         # =====================================================================
#         T_sim = torch.matmul(t, t.T) 
#         unique_t_labels, inv_t_indices = torch.unique(t_labels, return_inverse=True)
#         loc_indicator_t = (unique_t_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
#         loc_counts_t = loc_indicator_t.sum(dim=1, keepdim=True) + self.eps
#         M_t = loc_indicator_t / loc_counts_t 

#         C_text = torch.matmul(T_sim, M_t.T) 
#         W_softmax_t = F.softmax(C_text / self.tau_t, dim=1) 
#         W_text = W_softmax_t[torch.arange(t.size(0), device=t.device), inv_t_indices] 

#         # =====================================================================
#         # 3.1.2 Intra-Modal Visual Consensus Weighting (W_vis)
#         # =====================================================================
#         V_sim = torch.matmul(v, v.T) 
#         unique_v_labels, inv_v_indices = torch.unique(v_labels, return_inverse=True)
#         loc_indicator_v = (unique_v_labels.unsqueeze(1) == v_labels.unsqueeze(0)).float()
#         loc_counts_v = loc_indicator_v.sum(dim=1, keepdim=True) + self.eps
#         M_v = loc_indicator_v / loc_counts_v

#         C_vis = torch.matmul(V_sim, M_v.T) 
#         W_softmax_v = F.softmax(C_vis / self.tau_v, dim=1) 
#         W_vis = W_softmax_v[torch.arange(v.size(0), device=v.device), inv_v_indices] 

#         # =====================================================================
#         # SAFETY SHIFT FOR CLS POOLING: BATCH-MAX SCALING
#         # =====================================================================
#         W_text_scaled = W_text / (W_text.max() + self.eps)
#         W_vis_scaled = W_vis / (W_vis.max() + self.eps)

#         # 3.1.3 Joint Pairwise Consensus Score (W_pair) -> [B_v, B_t]
#         W_pair = torch.sqrt(W_vis_scaled.unsqueeze(1) * W_text_scaled.unsqueeze(0)) 

#         # =====================================================================
#         # 3.2 & 3.3 Cross-Modal Baseline Similarity Matrix & Masks
#         # =====================================================================
#         S_VT = torch.matmul(v, t.T) 
#         pos_mask = (v_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float() 
#         neg_mask = 1.0 - pos_mask 

#         # =====================================================================
#         # 🔥 ACTIVE ONLINE CROSS-MODAL HARD MINING
#         # =====================================================================
#         # Find relative informative thresholds inside the current batch configuration
#         with torch.no_grad():
#             max_neg = (S_VT - 2.0 * pos_mask).max(dim=1, keepdim=True)[0] # Hardest negative per row
#             min_pos = (S_VT + 2.0 * neg_mask).min(dim=1, keepdim=True)[0] # Softest positive per row
        
#         # Filter masks: isolate only pairs that actively violate margins
#         mining_pos_mask = (S_VT < (min_pos + 0.1)).float() * pos_mask
#         mining_neg_mask = (S_VT > (max_neg - 0.1)).float() * neg_mask

#         # =====================================================================
#         # 3.2 Weighted Multi-Positive Pull Term
#         # =====================================================================
#         pull_inputs = -self.alpha * (1.0 + self.gamma * W_pair) * (S_VT - self.lambda_margin) 
#         # Restrict summation strictly to mining-selected positive pairs
#         pull_inputs = torch.where(mining_pos_mask.bool(), pull_inputs, torch.tensor(-1e4, device=v.device))
#         pull_exp = torch.exp(pull_inputs)
#         pull_term = torch.log1p(pull_exp.sum(dim=1)) 

#         # =====================================================================
#         # 3.3 Weighted Negative Push Term (With Residual Floor Protection)
#         # =====================================================================
#         push_inputs = self.beta * (1.0 + W_pair) * (S_VT - self.lambda_margin)
#         # Restrict summation strictly to mining-selected hard negative pairs
#         push_inputs = torch.where(mining_neg_mask.bool(), push_inputs, torch.tensor(-1e4, device=v.device))
#         push_exp = torch.exp(push_inputs)
#         push_term = torch.log1p(push_exp.sum(dim=1)) 

#         # =====================================================================
#         # 3.4 Total Objective Function
#         # =====================================================================
#         loss_per_instance = (1.0 / self.alpha) * pull_term + (1.0 / self.beta) * push_term
        
#         return loss_per_instance.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMultiSimilarityLossCM(nn.Module):
    """
    Symmetric Contrastive Consensus-Weighted Multi-Similarity (SCCW-MS) Loss.
    Upgraded with a Strict Harmonic Mean Joint Pairwise Consensus (AND-Gate Variant)
    and Active Cross-Modal Hard Negative Mining.
    """
    def __init__(self, alpha=2.0, beta=25.0, lambda_margin=0.30, gamma=0.5, tau_t=0.02, tau_v=0.02, eps=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_margin = lambda_margin  
        self.gamma = gamma
        self.tau_t = tau_t
        self.tau_v = tau_v
        self.eps = eps

    def forward(self, embeddings, labels, indices_tuple=None, ref_emb=None, ref_labels=None):
        v = embeddings
        v_labels = labels
        t = ref_emb if ref_emb is not None else embeddings
        t_labels = ref_labels if ref_labels is not None else labels

        # Ensure true cosine similarities via L2 Normalization
        v = F.normalize(v, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)

        # =====================================================================
        # 3.1.1 Intra-Modal Textual Consensus Weighting (W_text)
        # =====================================================================
        T_sim = torch.matmul(t, t.T) 
        unique_t_labels, inv_t_indices = torch.unique(t_labels, return_inverse=True)
        loc_indicator_t = (unique_t_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
        loc_counts_t = loc_indicator_t.sum(dim=1, keepdim=True) + self.eps
        M_t = loc_indicator_t / loc_counts_t 

        C_text = torch.matmul(T_sim, M_t.T) # Equation (1) [cite: 33]
        W_softmax_t = F.softmax(C_text / self.tau_t, dim=1) # Equation (2) [cite: 35]
        W_text = W_softmax_t[torch.arange(t.size(0), device=t.device), inv_t_indices] 

        # =====================================================================
        # 3.1.2 Intra-Modal Visual Consensus Weighting (W_vis)
        # =====================================================================
        V_sim = torch.matmul(v, v.T) 
        unique_v_labels, inv_v_indices = torch.unique(v_labels, return_inverse=True)
        loc_indicator_v = (unique_v_labels.unsqueeze(1) == v_labels.unsqueeze(0)).float()
        loc_counts_v = loc_indicator_v.sum(dim=1, keepdim=True) + self.eps
        M_v = loc_indicator_v / loc_counts_v

        C_vis = torch.matmul(V_sim, M_v.T) # Equation (3) [cite: 41]
        W_softmax_v = F.softmax(C_vis / self.tau_v, dim=1) # Equation (4) [cite: 41, 43]
        W_vis = W_softmax_v[torch.arange(v.size(0), device=v.device), inv_v_indices] 

        # =====================================================================
        # BATCH-MAX SCALING (Protects dynamic range before harmonic calculation)
        # =====================================================================
        W_text_scaled = W_text / (W_text.max() + self.eps)
        W_vis_scaled = W_vis / (W_vis.max() + self.eps)

        # =====================================================================
        # 🧪 NEW PROPOSAL: HARMONIC MEAN JOINT CONSENSUS (W_pair)
        # Formula: 2 * W_vis * W_text / (W_vis + W_text + epsilon)
        # =====================================================================
        W_v_expanded = W_vis_scaled.unsqueeze(1)  # [B_v, 1]
        W_t_expanded = W_text_scaled.unsqueeze(0)  # [1, B_t]
        
        numerator = 2.0 * W_v_expanded * W_t_expanded
        denominator = W_v_expanded + W_t_expanded + self.eps
        W_pair = numerator / denominator  # Grid matrix footprint: [B_v, B_t]

        # =====================================================================
        # 3.2 & 3.3 Cross-Modal Baseline Similarity Matrix & Masks
        # =====================================================================
        S_VT = torch.matmul(v, t.T)
        pos_mask = (v_labels.unsqueeze(1) == t_labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask

        # =====================================================================
        # ACTIVE ONLINE CROSS-MODAL HARD MINING
        # =====================================================================
        with torch.no_grad():
            max_neg = (S_VT - 2.0 * pos_mask).max(dim=1, keepdim=True)[0]
            min_pos = (S_VT + 2.0 * neg_mask).min(dim=1, keepdim=True)[0]
        
        mining_pos_mask = (S_VT < (min_pos + 0.1)).float() * pos_mask
        mining_neg_mask = (S_VT > (max_neg - 0.1)).float() * neg_mask

        # =====================================================================
        # 3.2 Weighted Multi-Positive Pull Term
        # =====================================================================
        pull_inputs = -self.alpha * (1.0 + self.gamma * W_pair) * (S_VT - self.lambda_margin)
        pull_inputs = torch.where(mining_pos_mask.bool(), pull_inputs, torch.tensor(-1e4, device=v.device))
        pull_exp = torch.exp(pull_inputs)
        pull_term = torch.log1p(pull_exp.sum(dim=1))

        # =====================================================================
        # 3.3 Weighted Negative Push Term (With Residual Highway Protection)
        # =====================================================================
        push_inputs = self.beta * (1.0 + W_pair) * (S_VT - self.lambda_margin)
        push_inputs = torch.where(mining_neg_mask.bool(), push_inputs, torch.tensor(-1e4, device=v.device))
        push_exp = torch.exp(push_inputs)
        push_term = torch.log1p(push_exp.sum(dim=1)) 

        # =====================================================================
        # 3.4 Total Objective Function
        # =====================================================================
        loss_per_instance = (1.0 / self.alpha) * pull_term + (1.0 / self.beta) * push_term
        
        return loss_per_instance.mean()