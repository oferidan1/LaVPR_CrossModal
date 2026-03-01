import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalOTLoss(nn.Module):
    def __init__(self, epsilon=0.1, iterations=5, gamma=0.1):
        """
        gamma: Rejection cost. Low gamma = model easily 'gives up' on matching.
        epsilon: Entropy reg. Smaller = sparser, more 1-to-1 matching.
        """
        super().__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self.gamma = nn.Parameter(torch.tensor(gamma)) # Learnable rejection cost

    def forward(self, v, t, v_mask=None, t_mask=None):
        batch_size, n_v, dim = v.shape
        n_t = t.shape[1]

        # 1. Cosine Similarity Matrix
        A = torch.bmm(F.normalize(v, p=2, dim=-1), 
                      F.normalize(t, p=2, dim=-1).transpose(1, 2))

        # Cast to float32 to avoid overflow in Sinkhorn and masking
        A = A.float()

        # Keep a copy of unmasked similarities for loss computation
        A_raw = A.clone()

        # 2. Masking Padding: Force padding tokens to have 'infinite' matching cost
        if v_mask is not None:
            A = A.masked_fill(~v_mask.unsqueeze(2).bool(), -1e6)
        if t_mask is not None:
            A = A.masked_fill(~t_mask.unsqueeze(1).bool(), -1e6)

        # 3. Augment with Dustbins
        # Column for visual tokens to dump into; Row for text tokens to dump into
        gamma = self.gamma.to(dtype=A.dtype)
        dust_col = gamma.view(1, 1, 1).expand(batch_size, n_v, 1)
        A_aug = torch.cat([A, dust_col], dim=2) # [B, Nv, Nt+1]
        
        # Bottom-right corner (dustbin-to-dustbin) can be set to gamma or 0
        dust_row = gamma.view(1, 1, 1).expand(batch_size, 1, n_t + 1)
        A_aug = torch.cat([A_aug, dust_row], dim=1) # [B, Nv+1, Nt+1]

        # 4. Marginal Distributions (Mass)
        # mu: Distribution over visual tokens + 1 dustbin
        # nu: Distribution over text tokens + 1 dustbin
        if v_mask is not None:
            v_counts = v_mask.sum(dim=1, keepdim=True) + 1e-9
            mu_real = v_mask.float() / v_counts
        else:
            mu_real = torch.full((batch_size, n_v), 1.0/n_v, device=v.device)
            
        if t_mask is not None:
            t_counts = t_mask.sum(dim=1, keepdim=True) + 1e-9
            nu_real = t_mask.float() / t_counts
        else:
            nu_real = torch.full((batch_size, n_t), 1.0/n_t, device=v.device)

        # The dustbin tokens are given a mass of 1.0 to ensure they can 
        # absorb all real tokens if necessary.
        mu = torch.cat([mu_real, torch.ones((batch_size, 1), device=v.device)], dim=1)
        nu = torch.cat([nu_real, torch.ones((batch_size, 1), device=v.device)], dim=1)

        # 5. Stable Sinkhorn Iterations
        K = A_aug / self.epsilon
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)

        for _ in range(self.iterations):
            u = torch.log(mu + 1e-9) - torch.logsumexp(K + v.unsqueeze(1), dim=2)
            v = torch.log(nu + 1e-9) - torch.logsumexp(K + u.unsqueeze(2), dim=1)

        # 6. Final Transport Plan T*
        log_T = u.unsqueeze(2) + v.unsqueeze(1) + K
        T_star = torch.exp(log_T)

        # 7. Loss: Minimize Transport Cost (1 - Similarity)
        # This ensures the loss is positive and minimizes the distance between matched tokens.
        T_real = T_star[:, :n_v, :n_t]
        loss = torch.sum(T_real * (1.0 - A_raw), dim=(1, 2))

        return loss.mean()