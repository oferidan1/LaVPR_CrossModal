import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


class TokensClassificationLoss(nn.Module):
    """Predict, from the pooled VISION embedding, which words its own caption
    emphasises.

    The image has no access to the sentence, so it must infer from pixels which
    properties a describer would foreground. That is a cross-modal transfer,
    not self-distillation - which is what makes an attention target
    non-circular here.
    """

    def __init__(self,
                 vision_dim=768,
                 vocab_size=49408,
                 idf_path=None,
                 weight_mode='attn',        # uniform | idf | attn | idf_attn
                 grad_scale=0.05,
                 warmup_steps=200,
                 target_temp=1.0,
                 label_smooth=0.0,
                 drop_specials=True):
        super().__init__()
        self.head = nn.Linear(vision_dim, vocab_size)
        self.weight_mode = weight_mode
        self.grad_scale = grad_scale
        self.warmup_steps = warmup_steps
        self.target_temp = target_temp
        self.label_smooth = label_smooth
        self.drop_specials = drop_specials
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))

        w = torch.ones(vocab_size)
        if idf_path is not None:
            try:
                raw = torch.load(idf_path, weights_only=True).clamp(min=0.0)
                raw = torch.log1p(raw)                # raw inverse frequency is
                w = raw / raw.mean().clamp(min=1e-6)  # far too heavy-tailed
            except (FileNotFoundError, RuntimeError):
                print(f"TokensClassificationLoss: {idf_path} missing, uniform IDF.")
        self.register_buffer("idf", w)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # -- target -------------------------------------------------------------
    def build_target(self, ids, mask, attn_w=None):
        """(B, vocab) soft distribution over the pair's own caption."""
        B, L = ids.shape
        keep = mask.float().clone()
        if self.drop_specials:
            b = torch.arange(B, device=ids.device)
            keep[:, 0] = 0.0
            keep[b, ids.argmax(-1)] = 0.0

        if self.weight_mode == 'uniform':
            w = keep
        elif self.weight_mode == 'idf':
            w = self.idf[ids.clamp(0, self.idf.numel() - 1)].float() * keep
        elif self.weight_mode == 'attn':
            assert attn_w is not None, "weight_mode='attn' needs attn_w"
            w = attn_w.float() * keep
        elif self.weight_mode == 'idf_attn':
            assert attn_w is not None, "weight_mode='idf_attn' needs attn_w"
            w = self.idf[ids.clamp(0, self.idf.numel() - 1)].float() * attn_w.float() * keep
        else:
            raise ValueError(self.weight_mode)

        if self.target_temp != 1.0:
            # < 1 sharpens toward the peak token, > 1 flattens. Attention can be
            # peaky or near-uniform depending on the sentence; this is the knob
            # that makes the two weightings comparable in entropy.
            w = w.clamp(min=0).pow(1.0 / self.target_temp) * keep

        w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)

        tgt = torch.zeros(B, self.idf.numel(), device=ids.device, dtype=w.dtype)
        # scatter_add, not scatter: a word appearing twice should carry twice
        # the mass
        tgt.scatter_add_(1, ids.clamp(0, self.idf.numel() - 1), w)
        if self.label_smooth > 0:
            tgt = (1 - self.label_smooth) * tgt + self.label_smooth / tgt.shape[1]
        return tgt.detach()

    # -- forward ------------------------------------------------------------
    def forward(self, vision_embeddings, ids, mask, attn_w=None):
        """
        vision_embeddings : (B, vision_dim)  pooled patch tokens of the image
        ids, mask         : (B, L)           the pair's OWN caption
        attn_w            : (B, L)           from eot_attention_weights(), or None

        The i-th image is supervised by the i-th caption. No place-level mask
        anywhere in this function - that is deliberate.
        """
        if self.training:
            self._step += 1
            if self._step.item() <= self.warmup_steps:
                # a random head otherwise floods the vision tower with noise
                vision_embeddings = vision_embeddings.detach()
            elif self.grad_scale < 1.0:
                vision_embeddings = GradientScaleFunction.apply(
                    vision_embeddings, self.grad_scale)

        tgt = self.build_target(ids, mask, attn_w)
        logp = F.log_softmax(self.head(vision_embeddings).float(), dim=-1)
        loss = -(tgt * logp).sum(-1).mean()

        with torch.no_grad():
            # log this. if the two weightings differ wildly in entropy you are
            # comparing target sharpness, not target quality - use target_temp
            # to match them before believing the ablation.
            self.last_target_entropy = -(tgt * (tgt + 1e-9).log()).sum(-1).mean()
        return loss


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
    
    
@torch.no_grad()
def attention_stats(w):
    """Sanity numbers for any attention target. Log these every epoch.
 
    A target that collapses onto one patch/token, or flattens to uniform, is
    teaching nothing - and both look identical in the loss curve.
    """
    n = w.shape[-1]
    p = w.clamp(min=1e-9)
    ent = -(p * p.log()).sum(-1).mean()
    return {'entropy': float(ent),
            'entropy_norm': float(ent / torch.log(torch.tensor(float(n)))),
            'max_weight': float(w.max(-1).values.mean()),
            'top5_mass': float(w.topk(min(5, n), dim=-1).values.sum(-1).mean())}


"""
Attention-weighted token supervision - the training-side counterpart to the
deletion probe.

The probe established, on two checkpoints, that last-layer EOT attention
identifies the tokens the retrieval score depends on significantly better than
corpus TF-IDF (p = 0.0005 at k=10, Bonferroni-corrected). This replaces the
TF-IDF weighting in the SuperCLIP-style token/vocab loss with exactly the same
quantity the probe measured, so the thing that was validated is the thing that
gets trained.

Two design points that are not optional:

  STOP-GRADIENT. The target comes from the text tower. Without detach the model
  can minimise the loss by moving the target, and the objective collapses. The
  gradient must flow only into the prediction head and the vision tower.

  INSTANCE-LEVEL. The target is the attention of the pair's OWN text, never a
  same-place text from another view. Place-level supervision is correct for the
  global metric loss and wrong here: forcing an image to predict the salient
  words of a description written from a different viewpoint is wrong-label
  supervision at exactly the granularity this loss operates on.

weight_mode gives the ablation ladder measured by the probe:
    'uniform'  - presence only, no weighting          (floor)
    'idf'      - corpus prior                         (current baseline)
    'attn'     - EOT attention                        (the proposal)
    'idf_attn' - product of the two, renormalised     (does IDF add anything?)
"""


# ---------------------------------------------------------------------------
# the attention target
# ---------------------------------------------------------------------------
@torch.no_grad()
def eot_attention_weights(text_model, hidden_states, ids, mask=None, drop_specials=True):
    """
    Computes head-averaged attention weights from the EOT token to prior tokens
    in the final layer of openai/clip-vit-base-patch16.
    
    Args:
        text_model: model.text_model (CLIPTextTransformer)
        hidden_states: tuple of hidden states from output_hidden_states=True
        ids: LongTensor of token IDs (B, L)
        mask: optional attention mask (B, L)
        drop_specials: whether to zero-out attention to BOS and EOT itself
    """
    layer = text_model.encoder.layers[-1]
    x = layer.layer_norm1(hidden_states[-2])
    attn = layer.self_attn
    
    B, L, D = x.shape
    nh = attn.num_heads if hasattr(attn, "num_heads") else text_model.config.num_attention_heads
    hd = D // nh

    # Project Q and K: (B, num_heads, L, head_dim)
    q = attn.q_proj(x).view(B, L, nh, hd).transpose(1, 2)
    k = attn.k_proj(x).view(B, L, nh, hd).transpose(1, 2)

    # Locate EOT token (id=49407 for openai/clip-vit-base-patch16)
    eos_id = getattr(text_model.config, "eos_token_id", 49407)
    is_eos = (ids == eos_id)
    if is_eos.any():
        eot = is_eos.int().argmax(dim=-1)
    elif mask is not None:
        eot = (mask.long().sum(dim=-1) - 1).clamp(min=0)
    else:
        eot = ids.argmax(dim=-1)

    # Extract EOT's queries: (B, num_heads, head_dim)
    b = torch.arange(B, device=ids.device)
    q_eot = q[b, :, eot]

    # Attention logits: (B, num_heads, L)
    logits = torch.einsum('bhd,bhld->bhl', q_eot, k) * (hd ** -0.5)

    # Causal & padding mask (EOT only attends to pos <= eot)
    pos = torch.arange(L, device=ids.device).unsqueeze(0)
    allow = (pos <= eot.unsqueeze(1))
    if mask is not None:
        allow = allow & mask.bool()

    if drop_specials:
        # Exclude BOS (index 0) and self-attention (index == eot)
        content_mask = allow & (pos != eot.unsqueeze(1)) & (pos != 0)
        # Prevent completely empty masks on empty/short sentences like "<BOS> <EOS>"
        has_content = content_mask.any(dim=-1, keepdim=True)
        allow = torch.where(has_content, content_mask, allow)

    logits = logits.masked_fill(~allow.unsqueeze(1), float('-inf'))
    
    # Softmax over sequence length, average across heads
    w = torch.softmax(logits.float(), dim=-1).mean(dim=1)
    return torch.nan_to_num(w, nan=0.0)

@torch.no_grad()
def cls_attention_weights(vision_model, hidden_states, has_cls=True, drop_cls=True, return_grid=False):
    """(B, num_patches) last-layer CLS attention over patches, head-averaged."""
    layer = vision_model.encoder.layers[-1]
    x = layer.layer_norm1(hidden_states[-2])
    attn = layer.self_attn
    B, L, D = x.shape
    nh = getattr(attn, 'num_heads', None) or vision_model.config.num_attention_heads
    hd = D // nh

    q = attn.q_proj(x).view(B, L, nh, hd).transpose(1, 2)      # (B, nh, L, hd)
    k = attn.k_proj(x).view(B, L, nh, hd).transpose(1, 2)

    q_row = q[:, :, 0] if has_cls else q.mean(dim=2)           # (B, nh, hd)
    logits = torch.einsum('bhd,bhld->bhl', q_row, k) * (hd ** -0.5)

    if has_cls and drop_cls:
        # masking the CLS self-attention BEFORE the softmax means the remaining
        # patch weights already sum to 1
        logits[..., 0] = float('-inf')

    w = torch.softmax(logits.float(), dim=-1).mean(dim=1)      # (B, L)
    if has_cls:
        w = w[:, 1:]                                           # patches only
        if not drop_cls:
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
    w = torch.nan_to_num(w, nan=0.0).detach()                  # stop-gradient

    if return_grid:
        n = w.shape[-1]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(f"{n} patches is not a square grid")
        return w, w.view(-1, side, side)
    return w
    
     
class PatchAttentionLoss(nn.Module):
    """From the pooled TEXT embedding, predict which image regions the vision
    CLS attends to.
 
    THE TRAP in the obvious version. The target is a spatial distribution over
    patch positions, and most of its structure is a generic street-scene layout
    prior - sky at the top, road at the bottom, facades in the middle band -
    shared by every image in the dataset. A head predicting that prior gets the
    direct cross-entropy most of the way down while learning nothing
    text-specific. 'direct' mode is kept only as the ablation that demonstrates
    this.
 
    'contrastive' mode fixes it: the predicted map must match ITS OWN image's
    map better than the other images' maps in the batch. Note this is invariant
    to the shared prior by construction - subtracting any map m common to all
    targets shifts every logit in a row by the same p_i . m, which the softmax
    cancels. So the loss can only be reduced by predicting image-SPECIFIC
    deviation, which is the only part the text could plausibly know.
 
    Instance-level, not place-level: a different view of the same place has a
    different layout, and the text has no way to know which view it is paired
    with.
    """
 
    def __init__(self, text_dim=512, num_patches=196, mode='contrastive',
                 temperature=0.07, grad_scale=0.05, warmup_steps=200):
        super().__init__()
        self.head = nn.Linear(text_dim, num_patches)
        self.mode = mode
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature).log())
        self.grad_scale = grad_scale
        self.warmup_steps = warmup_steps
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long))
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
 
    def forward(self, text_embeddings, patch_attn):
        """text_embeddings (B, text_dim) | patch_attn (B, num_patches), detached"""
        if self.training:
            self._step += 1
            if self._step.item() <= self.warmup_steps:
                text_embeddings = text_embeddings.detach()
            elif self.grad_scale < 1.0:
                text_embeddings = GradientScaleFunction.apply(
                    text_embeddings, self.grad_scale)
 
        logp = F.log_softmax(self.head(text_embeddings).float(), dim=-1)
 
        if self.mode == 'direct':
            return -(patch_attn * logp).sum(-1).mean()
 
        # S[i, j] = -CE(target map of image j, predicted map of text i)
        S = logp @ patch_attn.float().t()                      # (B_text, B_img)
        scale = self.logit_scale.exp().clamp(max=100.0)
        tgt = torch.arange(S.shape[0], device=S.device)        # instance-level
        return 0.5 * (F.cross_entropy(S * scale, tgt)
                      + F.cross_entropy(S.t() * scale, tgt))
 
    @torch.no_grad()
    def match_acc(self, text_embeddings, patch_attn):
        """Fraction of texts whose predicted map matches its own image best.
        1/B is chance. If this sits at chance the head has learned the layout
        prior and nothing else."""
        logp = F.log_softmax(self.head(text_embeddings).float(), dim=-1)
        S = logp @ patch_attn.float().t()
        tgt = torch.arange(S.shape[0], device=S.device)
        return (S.argmax(-1) == tgt).float().mean()