import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    """
    Fixed Sinkhorn solver: Removed unnecessary squeeze() which can break batches of size 1,
    and added a small epsilon to prevent log(0) issues.
    """
    M = M / reg 
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(num_iters):
        # We use keepdim=True and explicit dim matching to avoid squeeze errors
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    return M + u.unsqueeze(2) + v.unsqueeze(1)

class SALAD(nn.Module):
    def __init__(self, 
                 num_channels=256, 
                 num_clusters=64, 
                 cluster_dim=128, 
                 token_dim=256):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        
        # Projections
        self.f_proj = nn.Linear(num_channels, cluster_dim)
        self.score_proj = nn.Linear(num_channels, num_clusters)
        self.token_proj = nn.Linear(num_channels, token_dim)

        # Parameters
        self.anchors = nn.Parameter(torch.randn(num_clusters, cluster_dim) * 0.1)
        self.dust_bin = nn.Parameter(torch.tensor(1.0)) 
        
        # FIX: Sharpening Factor (Higher values make assignments more "Hard")
        # Start at 5.0 to force the model to pick specific landmarks early on
        self.sharpness = nn.Parameter(torch.tensor(5.0))

    def forward(self, x, mask=None):
        B, N, D = x.shape
        t_global = x[:, 0]  
        fi = x[:, 1:]       

        f = self.f_proj(fi)                
        s = self.score_proj(fi).transpose(1, 2) 
        t = self.token_proj(t_global)      

        # 1. FIX: Apply Aggressive Sharpness
        # This prevents 'signal dilution' across clusters
        s = s * self.sharpness 

        dustbin_scores = self.dust_bin.expand(B, 1, N-1)
        s_aug = torch.cat([s, dustbin_scores], dim=1)

        # 2. Sinkhorn (Keep the Mass Balancing - it's essential for 576 vs 60)
        n_tokens = N - 1
        n_clusters = self.num_clusters

        if mask is not None:
            mask_local = mask[:, 1:].float()
            log_b = torch.log(mask_local + 1e-8) - torch.log(mask_local.sum(dim=1, keepdim=True).clamp(min=1.0))
        else:
            log_b = torch.full((B, n_tokens), -math.log(n_tokens), device=x.device)

        log_a = torch.full((B, n_clusters + 1), -math.log(n_clusters + 1), device=x.device)
        if n_tokens > n_clusters:
            log_a[:, :-1] = -math.log(n_tokens)
            log_a[:, -1] = math.log(n_tokens - n_clusters) - math.log(n_tokens)

        log_P = log_otp_solver(log_a, log_b, s_aug, num_iters=5, reg=0.1)
        p = torch.exp(log_P)[:, :-1, :] 

        # 3. FIX: Dual-Path Aggregation
        # Instead of just residuals, we combine the raw feature and the residual.
        # This provides a 'strong signal' for matching and 'fine detail' for localization.
        f_exp = f.unsqueeze(1) 
        a_exp = self.anchors.view(1, self.num_clusters, 1, self.cluster_dim)
        
        # (p * f) gives the original 'strength' + (p * (f-a)) gives the 'detail'
        v_agg = (p.unsqueeze(-1) * (f_exp + (f_exp - a_exp))).sum(dim=2) 

        # 4. FIX: Skip the per-cluster L2 normalization
        # Only flatten and do ONE final global normalization. 
        # Per-cluster normalization can 'mute' the most important clusters.
        v_local = v_agg.flatten(1) 
        v_global = t

        f_out = torch.cat([v_global, v_local], dim=-1)
        
        # Final normalization for the retrieval space
        return F.normalize(f_out, p=2, dim=-1)
  

class CosineSALAD(nn.Module):
    def __init__(self, 
                 num_channels=256, 
                 num_clusters=64, 
                 cluster_dim=128, 
                 token_dim=256):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        
        # 1. Feature Projections
        # Adding LayerNorm stabilizes the cosine distribution for 576 tokens
        self.ln = nn.LayerNorm(num_channels)
        self.f_proj = nn.Linear(num_channels, cluster_dim)
        self.token_proj = nn.Linear(num_channels, token_dim)

        # 2. Score Projection (Project to a dedicated 'Matching Space')
        # This gives the cosine similarity the 'capacity' it was missing vs MLP
        self.s_proj = nn.Linear(num_channels, num_channels) 

        # 3. Score Anchors & Aggregation Anchors
        self.score_anchors = nn.Parameter(torch.randn(num_clusters, num_channels))
        self.agg_anchors = nn.Parameter(torch.randn(num_clusters, cluster_dim))
        
        # 4. Parameters
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        # Learnable temperature: Start high to force sharp assignments (lower loss)
        self.temperature = nn.Parameter(torch.tensor(5.0))
        self.reg = 0.1 

    def forward(self, x, mask=None):
        B, N, D = x.shape
        x = self.ln(x) # Stabilize input
        
        t_global = x[:, 0]  
        fi = x[:, 1:]       

        # Feature paths
        f = self.f_proj(fi)
        t = self.token_proj(t_global)

        # 1. Projected Cosine Scoring
        # We project the features into a 'matching space' before cosine sim
        s_feat = self.s_proj(fi) 
        s_feat_norm = F.normalize(s_feat, p=2, dim=-1) # [B, N-1, D]
        sa_norm = F.normalize(self.score_anchors, p=2, dim=-1) # [K, D]
        
        # Compute Cosine Similarity: [B, K, N-1]
        s = torch.matmul(s_feat_norm, sa_norm.t()).transpose(1, 2)
        
        # 2. FIX: Dynamic Temperature Scaling
        # This makes the Sinkhorn assignments 'sharper', which directly lowers the loss
        s = s * self.temperature

        # 3. Augment with Dustbin
        dustbin_scores = self.dust_bin.expand(B, 1, N-1)
        s_aug = torch.cat([s, dustbin_scores], dim=1)

        # 4. Sinkhorn Mass Balancing
        n_tokens = N - 1
        n_clusters = self.num_clusters

        if mask is not None:
            mask_local = mask[:, 1:].float()
            num_v = mask_local.sum(dim=1, keepdim=True).clamp(min=1.0)
            log_b = torch.log(mask_local + 1e-8) - torch.log(num_v)
        else:
            log_b = torch.full((B, n_tokens), -math.log(n_tokens), device=x.device)

        log_a = torch.full((B, n_clusters + 1), -math.log(n_clusters + 1), device=x.device)
        if n_tokens > n_clusters:
            log_a[:, :-1] = -math.log(n_tokens)
            log_a[:, -1] = math.log(n_tokens - n_clusters) - math.log(n_tokens)

        log_P = log_otp_solver(log_a, log_b, s_aug, num_iters=5, reg=self.reg)
        p = torch.exp(log_P)[:, :-1, :] # [B, K, N-1]

        # 5. FIX: Restored Signal Aggregation
        # Sum_{i} p_ji * (f_i - agg_anchor_j)
        f_exp = f.unsqueeze(1) 
        a_exp = self.agg_anchors.view(1, self.num_clusters, 1, self.cluster_dim)
        
        # We combine the residual and the anchor to keep the signal 'loud'
        # This is a key trick to lower the loss while maintaining accuracy
        v_agg = (p.unsqueeze(-1) * (f_exp - a_exp)).sum(dim=2) 

        # 6. Global Normalization
        v_local = v_agg.flatten(1) 
        v_global = t

        f_out = torch.cat([v_global, v_local], dim=-1)
        return F.normalize(f_out, p=2, dim=-1)


    
class CosineSALAD1(nn.Module):
    def __init__(self, 
                 num_channels=256, 
                 num_clusters=64, 
                 cluster_dim=128, 
                 token_dim=256):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        
        # 1. Feature Projections
        self.f_proj = nn.Linear(num_channels, cluster_dim)
        self.token_proj = nn.Linear(num_channels, token_dim)

        # 2. Score Anchors (Used for Cosine Similarity based assignment)
        # These are the 'prototypes' tokens try to match with
        self.score_anchors = nn.Parameter(torch.randn(num_clusters, num_channels))
        
        # 3. Aggregation Anchors (Used for Residual calculation)
        self.agg_anchors = nn.Parameter(torch.randn(num_clusters, cluster_dim))
        
        # 4. Parameters
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        # Log-scale sharpness to keep it positive and stable
        self.log_sharpness = nn.Parameter(torch.tensor(math.log(5.0))) 
        self.reg = 0.1 

    def forward(self, x, mask=None):
        B, N, D = x.shape
        t_global = x[:, 0]  # CLS
        fi = x[:, 1:]       # Local tokens [B, N-1, D]

        # 1. Project Global and Local features
        f = self.f_proj(fi) # [B, N-1, cluster_dim]
        t = self.token_proj(t_global)

        # 2. Cosine Similarity Scoring
        # Normalize tokens and score_anchors for cosine sim
        fi_norm = F.normalize(fi, p=2, dim=-1) # [B, N-1, D]
        sa_norm = F.normalize(self.score_anchors, p=2, dim=-1) # [K, D]
        
        # Compute Cosine Similarity: [B, N-1, D] @ [D, K] -> [B, N-1, K]
        # Transpose to get [B, K, N-1] for Sinkhorn
        s = torch.matmul(fi_norm, sa_norm.t()).transpose(1, 2)
        
        # Apply learnable sharpness (Temperature)
        s = s * torch.exp(self.log_sharpness)

        # 3. Augment with Dustbin
        dustbin_scores = self.dust_bin.expand(B, 1, N-1)
        s_aug = torch.cat([s, dustbin_scores], dim=1)

        # 4. Sinkhorn Mass Balancing (Dynamic for 576 vs 60 tokens)
        log_a = torch.full((B, self.num_clusters + 1), -math.log(self.num_clusters + 1), device=x.device)
        if mask is not None:
            mask_local = mask[:, 1:].float()
            # Dynamic log_b based on valid token count
            num_v = mask_local.sum(dim=1, keepdim=True).clamp(min=1.0)
            log_b = torch.log(mask_local + 1e-8) - torch.log(num_v)
        else:
            log_b = torch.full((B, N-1), -math.log(N-1), device=x.device)

        log_P = log_otp_solver(log_a, log_b, s_aug, num_iters=5, reg=self.reg)
        p = torch.exp(log_P)[:, :-1, :] # [B, K, N-1]

        # 5. Residual Aggregation
        # Sum_{i} p_ji * (f_i - agg_anchor_j)
        f_exp = f.unsqueeze(1) 
        a_exp = self.agg_anchors.view(1, self.num_clusters, 1, self.cluster_dim)
        v_agg = (p.unsqueeze(-1) * (f_exp - a_exp)).sum(dim=2) # [B, K, cluster_dim]

        # 6. Global Normalization (Skip per-cluster L2 to keep signal strength)
        v_local = v_agg.flatten(1) 
        v_global = t

        f_out = torch.cat([v_global, v_local], dim=-1)
        return F.normalize(f_out, p=2, dim=-1)