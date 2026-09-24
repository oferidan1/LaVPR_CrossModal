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


# Replace the FILIPLoss class inside model/tokens_classify_loss.py with this.
# GradientScaleFunction is already defined in that file.


class FILIPLoss(nn.Module):
    """FILIP token-wise maximum similarity (late interaction).

        s(t->i) = sum_i w_i * max_j (t_i . v_j)
        s(i->t) = (1/m) sum_j max_i (v_j . t_i)

    loss_type
    ---------
    'infonce' : softmax cross-entropy over the batch, as in the FILIP paper.
                Needs very large batches - the paper uses 40,960 over 340M
                pairs. Degrades badly with few distinct negatives.
    'ms'      : Multi-Similarity pair weighting on the same MaxSim scores.
                Built for the small-batch regime, which is what place-labelled
                VPR batches are (~BS distinct places, since same-place items
                are positives, not BS*N).

    queue_size > 0 adds a MoCo-style FIFO of past image tokens (no gradient),
    so the text->image direction sees thousands of negative places instead of
    BS. Volume from the queue, selection from MS pair weighting.

    use_norm=False reproduces the original module exactly, for loading
    checkpoints trained before the LayerNorms were added.
    """

    def __init__(self,
                 text_dim=512,
                 vision_dim=768,
                 joint_dim=512,
                 num_patches=196,           # ViT-B/16 @224; needed for the queue
                 temperature=0.2,
                 chunk=8,
                 idf_path=None,
                 vocab_size=49408,
                 use_idf=True,
                 use_norm=True,
                 grad_scale=0.1,
                 warmup_steps=200,
                 max_logit_scale=30.0,
                 loss_type='ms',            # 'ms' | 'infonce'
                 ms_alpha=2.0,
                 ms_beta=50.0,
                 ms_base=0.5,
                 ms_dynamic_base=True,
                 queue_size=0):             # 0 disables the memory queue
        super().__init__()
        # HF CLIP returns vision last_hidden_state BEFORE post_layernorm but
        # text last_hidden_state AFTER final_layer_norm -> very different scales
        self.v_norm = nn.LayerNorm(vision_dim) if use_norm else None
        self.t_norm = nn.LayerNorm(text_dim) if use_norm else None

        self.vision_proj = nn.Linear(vision_dim, joint_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, joint_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())

        self.max_logit_scale = max_logit_scale
        self.chunk = chunk
        self.use_idf = use_idf
        self.grad_scale = grad_scale
        self.warmup_steps = warmup_steps

        self.loss_type = loss_type
        self.ms_alpha = ms_alpha
        self.ms_beta = ms_beta
        self.ms_base = ms_base
        self.ms_dynamic_base = ms_dynamic_base

        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

        # ---- cross-batch memory queue (image side only) ------------------
        self.queue_size = queue_size
        if queue_size:
            self.register_buffer("q_v", torch.zeros(queue_size, num_patches,
                                                    joint_dim, dtype=torch.half))
            self.register_buffer("q_lab", torch.full((queue_size,), -1, dtype=torch.long))
            self.register_buffer("q_ptr", torch.zeros(1, dtype=torch.long))

        if use_idf and idf_path is not None:
            try:
                w = torch.load(idf_path, weights_only=True).clamp(min=0.0)
                # temper: raw inverse frequency is unstable on large vocabularies
                w = torch.log1p(w)
                w = w / w.mean().clamp(min=1e-6)
            except (FileNotFoundError, RuntimeError):
                print(f"FILIPLoss: {idf_path} not found, uniform weights.")
                w = torch.ones(vocab_size)
        else:
            w = torch.ones(vocab_size)
        self.register_buffer("idf_weights", w)

        self.reset_parameters()

    def reset_parameters(self):
        """Call again after any global kaiming/relu init in the parent module."""
        nn.init.normal_(self.vision_proj.weight, std=0.02)
        nn.init.normal_(self.text_proj.weight, std=0.02)

    # ------------------------------------------------------------------
    def _weights(self, text_tokens, t_mask):
        if self.use_idf:
            idx = text_tokens.clamp(0, self.idf_weights.numel() - 1)
            w = self.idf_weights[idx].to(t_mask.dtype)
        else:
            w = torch.ones_like(t_mask)
        w = w * t_mask
        return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    def _scores(self, t, v, w, t_mask, both=True):
        """t: (Bt, n, d) | v: (Bv, m, d) | w, t_mask: (Bt, n)

        Chunked over the text batch - the full (Bt, Bv, n, m) tensor is far too
        large to materialise. Padded text positions are pushed to -1e4 so they
        never win the image->text argmax and contribute nothing to the
        text->image weighted sum.
        """
        t2i, i2t = [], []
        for s in range(0, t.size(0), self.chunk):
            tc = t[s:s + self.chunk]
            wc = w[s:s + self.chunk]
            mc = t_mask[s:s + self.chunk]
            sim = torch.einsum('cnd,bmd->cbnm', tc, v)
            sim = sim + (1.0 - mc)[:, None, :, None] * (-1e4)
            t2i.append((sim.max(dim=-1).values * wc[:, None, :]).sum(-1))
            if both:
                i2t.append(sim.max(dim=-2).values.mean(-1))
            del sim
        t2i = torch.cat(t2i, dim=0)
        i2t = torch.cat(i2t, dim=0) if both else None
        return t2i, i2t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _enqueue(self, v, labels):
        B = v.shape[0]
        p = int(self.q_ptr.item())
        idx = (torch.arange(B, device=v.device) + p) % self.queue_size
        self.q_v[idx] = v.detach().half()
        self.q_lab[idx] = labels.detach()
        self.q_ptr[0] = (p + B) % self.queue_size

    # ------------------------------------------------------------------
    def _ms_loss(self, S, same, base):
        """Multi-Similarity pair weighting on a CROSS-MODAL score matrix.

        S: [Bt, Bv], same: [Bt, Bv] boolean positive mask.
        The diagonal is a genuine positive here (text i describes image i), so
        unlike unimodal MS it is NOT excluded. S need not be square once the
        memory queue is in use.
        """
        neg = ~same
        lp = torch.where(same, torch.exp(-self.ms_alpha * (S - base)),
                         torch.zeros_like(S)).sum(dim=1)
        ln = torch.where(neg, torch.exp(self.ms_beta * (S - base)),
                         torch.zeros_like(S)).sum(dim=1)
        return (torch.log1p(lp) / self.ms_alpha
                + torch.log1p(ln) / self.ms_beta).mean()

    def _base_for(self, S, same):
        """MaxSim scores are averages of per-token maxima over ~196 patches, so
        they sit well above plain cosine and a fixed margin transfers badly.
        Track the mean positive score instead, clamped to a sane band.
        """
        if not self.ms_dynamic_base:
            return self.ms_base
        with torch.no_grad():
            pos = S[same]
            if pos.numel() == 0:
                return self.ms_base
            return float(pos.mean().clamp(0.2, 0.9))

    def _infonce(self, s_t2i, s_i2t, same_t2i, same_i2t, dtype):
        scale = self.logit_scale.exp().clamp(max=self.max_logit_scale)

        def ce(S, same):
            q = same.to(dtype)
            q = q / q.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return -(q * F.log_softmax(S * scale, dim=-1)).sum(-1).mean()

        if s_i2t is None:
            return ce(s_t2i, same_t2i)
        return 0.5 * (ce(s_t2i, same_t2i) + ce(s_i2t.t(), same_i2t))

    # ------------------------------------------------------------------
    def forward(self, img_local, text_local, t_mask, text_tokens, labels):
        """
        img_local  : (B, m, vision_dim)  patch tokens, CLS already stripped
        text_local : (B, n, text_dim)    text tokens
        t_mask     : (B, n)              1 for real tokens, 0 for padding
        text_tokens: (B, n)              token ids, for the IDF lookup
        labels     : (B,)                place labels
        """
        if self.training:
            self._step += 1
            if self._step.item() <= self.warmup_steps:
                # train the projections alone first; a random head otherwise
                # floods the encoders with noise
                img_local, text_local = img_local.detach(), text_local.detach()
            elif self.grad_scale < 1.0:
                img_local = GradientScaleFunction.apply(img_local, self.grad_scale)
                text_local = GradientScaleFunction.apply(text_local, self.grad_scale)

        vi = self.v_norm(img_local) if self.v_norm is not None else img_local
        te = self.t_norm(text_local) if self.t_norm is not None else text_local
        v = F.normalize(self.vision_proj(vi), dim=-1)
        t = F.normalize(self.text_proj(te), dim=-1)

        t_mask = t_mask.to(t.dtype)
        w = self._weights(text_tokens, t_mask)

        use_q = self.queue_size > 0 and bool((self.q_lab >= 0).any())
        if use_q:
            valid = self.q_lab >= 0
            v_all = torch.cat([v, self.q_v[valid].to(v.dtype)], dim=0)
            lab_img = torch.cat([labels, self.q_lab[valid]], dim=0)
            # image->text is not defined for queued images (no queued text),
            # so with a queue we optimise the text->image direction only -
            # which is the direction the benchmark evaluates anyway.
            s_t2i, s_i2t = self._scores(t, v_all, w, t_mask, both=False)
        else:
            v_all, lab_img = v, labels
            s_t2i, s_i2t = self._scores(t, v_all, w, t_mask, both=True)

        # a queued image of the same place is a POSITIVE, not a negative -
        # treating it as negative trains the model to reject correct matches
        same_t2i = labels.view(-1, 1) == lab_img.view(1, -1)      # [B, B+Q]
        same_i2t = labels.view(-1, 1) == labels.view(1, -1)       # [B, B]

        if self.loss_type == 'ms':
            base = self._base_for(s_t2i, same_t2i)
            loss = self._ms_loss(s_t2i, same_t2i, base)
            if s_i2t is not None:
                loss = 0.5 * (loss + self._ms_loss(s_i2t.t(), same_i2t, base))
        else:
            loss = self._infonce(s_t2i, s_i2t, same_t2i, same_i2t, t.dtype)

        if self.training and self.queue_size:
            self._enqueue(v, labels)
        return loss