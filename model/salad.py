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
        log_a = torch.full((B, self.num_clusters + 1), -math.log(self.num_clusters + 1), device=x.device)
        if mask is not None:
            mask_local = mask[:, 1:].float()
            log_b = torch.log(mask_local + 1e-8) - torch.log(mask_local.sum(dim=1, keepdim=True).clamp(min=1.0))
        else:
            log_b = torch.full((B, N-1), -math.log(N-1), device=x.device)

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