"""
reranker_module.py
==================
Stage-2 reranker on top of a frozen LaVPR_cross backbone.

Two reranker architectures available via --model_type:

  cross_attn  (default)
  ─────────────────────
  text_tokens (Q) → CrossAttnBlock × n_layers ← img_patches (K,V)
  CLS of attended text → score_head → scalar

  qformer
  ───────
  learned queries → self-attn
                  → cross-attn ← img_patches
                  → cross-attn ← text_tokens
  mean-pool queries → score_head → scalar

Both share the same:
  - Frozen LaVPR_cross backbone (DINOv2 + BGE)
  - Precomputed local token input (no backbone at train time)
  - Margin / BCE loss
  - Training loop and validation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.LaVPR_cross import LaVPR_cross, _IMAGE_ENCODER_DIM, _TEXT_ENCODER_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_dims(backbone: LaVPR_cross):
    img_key = backbone.image_model_name
    txt_key = backbone.text_model_name.lower()
    img_dim = _IMAGE_ENCODER_DIM.get(img_key)
    if img_dim is None:
        raise ValueError(f"Unknown image backbone '{img_key}'")
    txt_dim = None
    for key, dim in _TEXT_ENCODER_DIM.items():
        if key in txt_key:
            txt_dim = dim
            break
    if txt_dim is None:
        raise ValueError(f"Unknown text backbone '{txt_key}'")
    return img_dim, txt_dim


# ---------------------------------------------------------------------------
# Shared score head builder
# ---------------------------------------------------------------------------

def _build_score_head(d_model: int, deep: bool) -> nn.Sequential:
    """
    shallow (deep=False): LayerNorm → Linear(d→1)
    deep    (deep=True):  LayerNorm → Linear(d→d/2) → GELU → Linear(d/2→1)
    Last linear always zero-initialised for stable early training.
    """
    if deep:
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2, bias=True),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, bias=True),
        )
        nn.init.kaiming_uniform_(head[1].weight, nonlinearity='relu')
    else:
        head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1, bias=True),
        )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


# ===========================================================================
# Architecture 1 — Cross-Attention Reranker
# ===========================================================================

class CrossAttnBlock(nn.Module):
    """
    Pre-norm block:
      1. Cross-attention  — text tokens (Q) attend to image patches (K,V)
      2. Self-attention   — text tokens refine among themselves
      3. FFN
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q     = nn.LayerNorm(d_model)
        self.norm_kv    = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_cross = nn.Dropout(dropout)

        self.norm_self  = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_self  = nn.Dropout(dropout)

        self.norm_ffn   = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv, query_pad_mask=None):
        q       = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        ca, _   = self.cross_attn(q, kv_norm, kv_norm)
        query   = query + self.drop_cross(ca)

        q2 = self.norm_self(query)
        sa, _ = self.self_attn(q2, q2, q2, key_padding_mask=query_pad_mask)
        query = query + self.drop_self(sa)

        query = query + self.ffn(self.norm_ffn(query))
        return query


class CrossModalReranker(nn.Module):
    """
    Text tokens attend to image patches via n_layers of CrossAttnBlock.
    CLS token of the attended sequence → score_head → scalar.

    img_local  (B, N, D_img) → img_proj → (B, N, d)
    text_local (B, L, D_txt) → txt_proj → (B, L, d)
                                            ↓ CrossAttnBlock × n_layers
    CLS → score_head → (B,)
    """

    def __init__(
        self,
        img_dim:         int   = 768,
        txt_dim:         int   = 1024,
        d_model:         int   = 128,
        n_heads:         int   = 4,
        n_layers:        int   = 1,
        dropout:         float = 0.1,
        out_dim:         int   = 768,
        deep_score_head: bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.img_proj = nn.Linear(img_dim, d_model, bias=False)
        self.txt_proj = nn.Linear(txt_dim, d_model, bias=False)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim, bias=False),
        )
        self.score_head = _build_score_head(d_model, deep_score_head)

        nn.init.kaiming_uniform_(self.img_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.txt_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj[1].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, img_local, text_local, text_mask=None):
        dtype       = self.img_proj.weight.dtype
        img_tokens  = self.img_proj(img_local.to(dtype))
        text_tokens = self.txt_proj(text_local.to(dtype))

        pad_mask = text_mask.eq(0) if text_mask is not None else None

        x = text_tokens
        for block in self.blocks:
            x = block(query=x, kv=img_tokens, query_pad_mask=pad_mask)

        cls   = x[:, 0]
        score = self.score_head(cls).squeeze(-1)             # (B,)
        out   = self.out_proj(cls) if not self.training else None
        emb   = F.normalize(out, p=2, dim=1) if out is not None else None
        return emb, score


# ===========================================================================
# Architecture 2 — QFormer Reranker
# ===========================================================================

class QFormerLayer(nn.Module):
    """
    One QFormer block:
      1. Self-attention  among learned queries
      2. Cross-attention queries → image patches
      3. Cross-attention queries → text tokens  (with text pad mask)
      4. FFN
    All pre-norm.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        # 1. Self-attn
        self.norm_sa   = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_sa   = nn.Dropout(dropout)

        # 2. Cross-attn → image
        self.norm_img    = nn.LayerNorm(d_model)
        self.norm_img_kv = nn.LayerNorm(d_model)
        self.img_attn    = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_img    = nn.Dropout(dropout)

        # 3. Cross-attn → text
        self.norm_txt    = nn.LayerNorm(d_model)
        self.norm_txt_kv = nn.LayerNorm(d_model)
        self.txt_attn    = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_txt    = nn.Dropout(dropout)

        # 4. FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, img_tokens, text_tokens, txt_pad_mask=None):
        # 1. Self-attention among queries
        q  = self.norm_sa(queries)
        sa, _ = self.self_attn(q, q, q)
        queries = queries + self.drop_sa(sa)

        # 2. Cross-attention → image patches (no padding)
        q   = self.norm_img(queries)
        kv  = self.norm_img_kv(img_tokens)
        ca, _ = self.img_attn(q, kv, kv)
        queries = queries + self.drop_img(ca)

        # 3. Cross-attention → text tokens (with pad mask)
        q   = self.norm_txt(queries)
        kv  = self.norm_txt_kv(text_tokens)
        ca, _ = self.txt_attn(q, kv, kv, key_padding_mask=txt_pad_mask)
        queries = queries + self.drop_txt(ca)

        # 4. FFN
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class QFormerReranker(nn.Module):
    """
    Learned query vectors attend to both image patches and text tokens
    independently in each layer, producing a fused representation.

    img_local  (B, N, D_img) → img_proj → (B, N, d)
    text_local (B, L, D_txt) → txt_proj → (B, L, d)
    learned queries (1, Q, d) → QFormerLayer × n_layers
    mean-pool → score_head → (B,)

    Advantage over CrossModalReranker:
      Learned queries are not anchored to either modality — they learn
      task-specific patterns (e.g. query 1→signage, query 2→facades).
      Both image and text are keys/values per layer, enabling richer fusion.
    """

    def __init__(
        self,
        img_dim:         int   = 768,
        txt_dim:         int   = 1024,
        d_model:         int   = 128,
        n_heads:         int   = 4,
        n_layers:        int   = 2,
        num_queries:     int   = 32,
        dropout:         float = 0.1,
        out_dim:         int   = 768,
        deep_score_head: bool  = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        # Learned query vectors — the key innovation
        self.queries  = nn.Parameter(torch.zeros(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.img_proj = nn.Linear(img_dim, d_model, bias=False)
        self.txt_proj = nn.Linear(txt_dim, d_model, bias=False)

        self.layers = nn.ModuleList([
            QFormerLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim, bias=False),
        )
        self.score_head = _build_score_head(d_model, deep_score_head)

        nn.init.kaiming_uniform_(self.img_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.txt_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj[1].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, img_local, text_local, text_mask=None):
        B           = img_local.shape[0]
        dtype       = self.img_proj.weight.dtype
        img_tokens  = self.img_proj(img_local.to(dtype))     # (B, N, d)
        text_tokens = self.txt_proj(text_local.to(dtype))    # (B, L, d)

        pad_mask = text_mask.eq(0) if text_mask is not None else None

        # Expand learned queries to batch
        queries = self.queries.expand(B, -1, -1).contiguous()  # (B, Q, d)

        for layer in self.layers:
            queries = layer(queries, img_tokens, text_tokens, pad_mask)

        # Mean-pool queries → scalar score
        pooled = queries.mean(dim=1)                           # (B, d)
        score  = self.score_head(pooled).squeeze(-1)           # (B,)
        out    = self.out_proj(pooled) if not self.training else None
        emb    = F.normalize(out, p=2, dim=1) if out is not None else None
        return emb, score


# ===========================================================================
# Factory
# ===========================================================================

def build_reranker(
    model_type:      str,
    img_dim:         int,
    txt_dim:         int,
    d_model:         int,
    n_heads:         int,
    n_layers:        int,
    dropout:         float,
    out_dim:         int,
    deep_score_head: bool,
    num_queries:     int   = 32,   # QFormer only
) -> nn.Module:
    """Return the appropriate reranker network."""
    if model_type == 'cross_attn':
        return CrossModalReranker(
            img_dim=img_dim, txt_dim=txt_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dropout=dropout, out_dim=out_dim,
            deep_score_head=deep_score_head,
        )
    elif model_type == 'qformer':
        return QFormerReranker(
            img_dim=img_dim, txt_dim=txt_dim,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            num_queries=num_queries,
            dropout=dropout, out_dim=out_dim,
            deep_score_head=deep_score_head,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose 'cross_attn' or 'qformer'.")


# ===========================================================================
# Lightning Module (shared by both architectures)
# ===========================================================================

class CrossModalRerankModule(pl.LightningModule):
    """
    Stage-2 reranker Lightning module.

    model_type='cross_attn' → CrossModalReranker (text queries image patches)
    model_type='qformer'    → QFormerReranker    (learned queries fuse both)

    Training uses precomputed local tokens — backbone never runs at train time.
    Validation runs backbone live on Pitts30k and reports R@k before/after reranking.
    """

    def __init__(
        self,
        backbone_ckpt:   str,

        # ---- Architecture
        model_type:      str   = 'cross_attn',   # 'cross_attn' | 'qformer'
        d_model:         int   = 128,
        n_heads:         int   = 4,
        n_layers:        int   = 1,
        dropout:         float = 0.1,
        out_dim:         int   = 768,
        deep_score_head: bool  = False,
        num_queries:     int   = 32,              # QFormer only

        # ---- Validation
        rerank_k:        int   = 20,
        faiss_gpu:       bool  = False,

        # ---- Loss
        loss_name:       str   = 'margin',        # 'margin' | 'bce'
        margin:          float = 0.3,

        # ---- Optimiser
        lr:              float = 1e-4,
        weight_decay:    float = 1e-3,
        warmup_steps:    int   = 200,
        epochs:          int   = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- Frozen backbone ----
        self.backbone = LaVPR_cross.load_from_checkpoint(
            backbone_ckpt, map_location='cpu')
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        img_dim, txt_dim = _infer_dims(self.backbone)

        # ---- Reranker network ----
        self.reranker = build_reranker(
            model_type      = model_type,
            img_dim         = img_dim,
            txt_dim         = txt_dim,
            d_model         = d_model,
            n_heads         = n_heads,
            n_layers        = n_layers,
            dropout         = dropout,
            out_dim         = out_dim,
            deep_score_head = deep_score_head,
            num_queries     = num_queries,
        )

        # ---- Store params ----
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.epochs       = epochs
        self.loss_name    = loss_name
        self.margin       = margin

        n_params = sum(p.numel() for p in self.reranker.parameters())
        print(f"Reranker: model_type={model_type}  "
              f"d_model={d_model}  n_layers={n_layers}  "
              f"params={n_params:,}")
        print(f"Loss: {loss_name}"
              + (f"  margin={margin}" if loss_name == 'margin' else ""))

    # ------------------------------------------------------------------
    # Feature extraction (backbone frozen)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_local(self, images, texts):
        self.backbone.eval()
        _, img_local               = self.backbone.encode_image(images)
        _, text_local, text_mask, _ = self.backbone.encode_text(texts)
        if img_local is None:
            raise RuntimeError("Backbone returned no patch tokens. Use DINOv2.")
        return img_local, text_local, text_mask

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_pairs(self, img_local, text_local, text_mask):
        """Returns (emb, score) — score is (M,) scalar."""
        return self.reranker(img_local, text_local, text_mask)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, scores, place_ids, query_place_id):
        if self.loss_name == 'margin':
            return self._margin_loss(scores, place_ids, query_place_id)
        elif self.loss_name == 'bce':
            return self._bce_loss(scores, place_ids, query_place_id)
        else:
            raise ValueError(f"Unknown loss '{self.loss_name}'. "
                             f"Choose 'margin' or 'bce'.")

    def _margin_loss(self, scores, place_ids, query_place_id, margin=None):
        """
        Margin ranking loss — best positive must score ≥ margin above all negatives.
        Directly optimises R@1. Ignores easy negatives already below margin.
        """
        m         = margin if margin is not None else self.margin
        query_pid = query_place_id.unsqueeze(1)
        is_pos    = place_ids.eq(query_pid)
        is_valid  = place_ids >= 0

        total = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        valid = 0
        for b in range(scores.shape[0]):
            pm = is_pos[b] & is_valid[b]
            nm = ~is_pos[b] & is_valid[b]
            if not pm.any() or not nm.any():
                continue
            best_pos   = scores[b][pm].max()
            losses     = F.relu(m - best_pos + scores[b][nm])
            total      = total + losses.mean()
            valid     += 1
        return total / max(valid, 1)

    def _bce_loss(self, scores, place_ids, query_place_id):
        """
        BCE loss — treats each candidate independently.
        All positives (all GSV views) contribute equally.
        Good for R@5/10 but doesn't model ranking directly.
        """
        query_pid  = query_place_id.unsqueeze(1)
        is_pos     = place_ids.eq(query_pid)
        is_valid   = place_ids >= 0
        labels     = is_pos.float()
        valid_mask = is_valid.float()
        bce        = F.binary_cross_entropy_with_logits(
            scores, labels, reduction='none')
        return (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Backbone never runs — tokens are precomputed and loaded from disk.
        All B*K pairs processed in one batched forward pass.

        Batch keys:
            query_text_local : (B, L, D_txt)
            query_mask       : (B, L)
            cand_img_locals  : (B, K, N, D_img)
            labels           : (B, K)  place IDs  -1=padding
            query_place_id   : (B,)    true place ID of query
        """
        text_local     = batch['query_text_local'].to(self.device, dtype=torch.bfloat16)
        text_mask      = batch['query_mask'].to(self.device)
        img_locals     = batch['cand_img_locals'].to(self.device, dtype=torch.bfloat16)
        place_ids      = batch['labels'].to(self.device)
        query_place_id = batch['query_place_id'].to(self.device)

        B, K, N, D_img = img_locals.shape
        L              = text_local.shape[1]

        # Single batched forward — all B*K pairs at once
        imgs_flat = img_locals.reshape(B * K, N, D_img)
        txt_flat  = text_local.unsqueeze(1)\
                              .expand(-1, K, -1, -1)\
                              .reshape(B * K, L, -1).contiguous()
        msk_flat  = text_mask.unsqueeze(1)\
                             .expand(-1, K, -1)\
                             .reshape(B * K, -1).contiguous()

        _, scores_flat = self._score_pairs(imgs_flat, txt_flat, msk_flat)
        scores = scores_flat.view(B, K)

        loss = self._compute_loss(scores, place_ids, query_place_id)
        self.log('train/loss', loss, prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Validation — real Pitts30k / MSLS evaluation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        with torch.no_grad():
            img_embeds,  img_local           = self.backbone.encode_image(
                places.to(self.device))
            text_embeds, text_local, mask, _ = self.backbone.encode_text(texts)
        return {
            'descriptors': img_embeds .detach().cpu(),
            'text_embeds': text_embeds.detach().cpu(),
            'img_local'  : img_local  .detach().cpu().half(),
            'text_local' : text_local .detach().cpu().half(),
            'text_mask'  : mask       .detach().cpu(),
        }

    def validation_epoch_end(self, val_step_outputs):
        import faiss
        import utils

        dm       = self.trainer.datamodule
        val_dm   = getattr(dm, 'val_datamodule', dm)
        k_values = [1, 5, 10, 15, 20, 50, 100]
        rerank_k = self.hparams.rerank_k

        if len(val_dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for val_set_name, val_dataset, step_outputs in zip(
                val_dm.val_set_names, val_dm.val_datasets, val_step_outputs):

            img_feats   = torch.cat([o['descriptors'] for o in step_outputs], dim=0)
            text_feats  = torch.cat([o['text_embeds']  for o in step_outputs], dim=0)
            all_img_loc = torch.cat([o['img_local']    for o in step_outputs], dim=0)

            # Pad variable-length text tokens
            tl_list  = [tl for o in step_outputs for tl in o['text_local']]
            tm_list  = [tm for o in step_outputs for tm in o['text_mask']]
            max_L    = max(t.shape[0] for t in tl_list)
            D_txt    = tl_list[0].shape[1]
            N_total  = len(tl_list)

            all_txt_loc = torch.zeros(N_total, max_L, D_txt, dtype=torch.float16)
            all_txt_msk = torch.zeros(N_total, max_L, dtype=torch.bool)
            for i, (tl, tm) in enumerate(zip(tl_list, tm_list)):
                L = tl.shape[0]
                all_txt_loc[i, :L] = tl
                all_txt_msk[i, :L] = tm

            if 'pitts' in val_set_name:
                num_references = val_dataset.num_db
                positives      = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives      = val_dataset.pIdx
            else:
                raise NotImplementedError(f'Add validation for {val_set_name}')

            r_feats    = img_feats   [:num_references]
            q_feats    = text_feats  [num_references:]
            db_img_loc = all_img_loc [:num_references]
            q_txt_loc  = all_txt_loc [num_references:]
            q_txt_msk  = all_txt_msk [num_references:]

            N_db = r_feats.shape[0]
            N_q  = q_feats.shape[0]
            D    = r_feats.shape[1]

            # Stage-1: FAISS retrieval — L2-normalise for cosine similarity
            print(f"\n[{val_set_name}] Stage-1  N_db={N_db:,}  N_q={N_q:,}")
            r_np = r_feats.numpy().astype('float32')
            q_np = q_feats.numpy().astype('float32')
            r_np /= np.linalg.norm(r_np, axis=1, keepdims=True).clip(min=1e-8)
            q_np /= np.linalg.norm(q_np, axis=1, keepdims=True).clip(min=1e-8)
            index = faiss.IndexFlatIP(D)
            index.add(r_np)
            _, I_stage1 = index.search(q_np, max(k_values))

            recalls_before = utils.get_validation_recalls(
                r_list=r_feats, q_list=q_feats, k_values=k_values,
                gt=positives, print_results=True,
                dataset_name=f'{val_set_name}_stage1',
                faiss_gpu=self.hparams.faiss_gpu)

            # Stage-2: reranking
            print(f"[{val_set_name}] Reranking top-{rerank_k}...")
            self.reranker.eval()
            rerank_batch = 64   # queries per GPU batch during val reranking
            I_reranked   = I_stage1[:, :rerank_k].copy()

            for q_start in range(0, N_q, rerank_batch):
                q_end        = min(q_start + rerank_batch, N_q)
                bsz          = q_end - q_start
                cand_indices = I_stage1[q_start:q_end, :rerank_k]

                with torch.no_grad():
                    img_loc_bsz = torch.stack([
                        db_img_loc[cand_indices[b]] for b in range(bsz)
                    ]).to(self.device).float()

                    txt_loc_bsz = q_txt_loc[q_start:q_end].to(self.device).float()
                    txt_msk_bsz = q_txt_msk[q_start:q_end].to(self.device)

                    bsz_K = bsz * rerank_k
                    N_p   = img_loc_bsz.shape[2]
                    L_    = txt_loc_bsz.shape[1]
                    D_t_  = txt_loc_bsz.shape[2]

                    imgs_flat = img_loc_bsz.view(bsz_K, N_p, -1)
                    txt_flat  = txt_loc_bsz.unsqueeze(1)\
                                           .expand(-1, rerank_k, -1, -1)\
                                           .reshape(bsz_K, L_, D_t_).contiguous()
                    msk_flat  = txt_msk_bsz.unsqueeze(1)\
                                           .expand(-1, rerank_k, -1)\
                                           .reshape(bsz_K, L_).contiguous()

                    _, scores_flat = self._score_pairs(imgs_flat, txt_flat, msk_flat)
                    ranking = scores_flat.view(bsz, rerank_k)\
                                        .argsort(dim=1, descending=True).cpu().numpy()

                for b in range(bsz):
                    I_reranked[q_start + b] = cand_indices[b][ranking[b]]

            recalls_after = self._recalls_from_index(
                I_reranked, I_stage1, positives, k_values, rerank_k)

            # Print table
            print(f"\n{'':=<65}")
            _mtype = getattr(self.hparams, 'model_type', 'cross_attn')
            print(f"  {val_set_name}  —  Stage-1 vs Reranked  [{_mtype}]")
            print(f"{'':=<65}")
            print(f"  {'':5s}  {'Stage-1':>10s}  {'Reranked':>10s}  {'Gain':>8s}")
            for k in [1, 5, 10, 20]:
                b = recalls_before.get(k, 0.0)
                a = recalls_after.get(k, 0.0)
                print(f"  R@{k:<3d}  {b:>10.4f}  {a:>10.4f}  {a-b:>+8.4f}")
            print(f"{'':=<65}\n")

            prog_bar_ks = {1, 5}
            for k in k_values:
                self.log(f'{val_set_name}/R{k}_stage1',
                         recalls_before.get(k, 0.0),
                         prog_bar=(k in prog_bar_ks), logger=True)
                self.log(f'{val_set_name}/R{k}_reranked',
                         recalls_after.get(k, 0.0),
                         prog_bar=(k in prog_bar_ks), logger=True)

            self.log(f'{val_set_name}/R1_gain',
                     recalls_after.get(1, 0.0) - recalls_before.get(1, 0.0),
                     prog_bar=True, logger=True)

    @staticmethod
    def _recalls_from_index(I_reranked, I_stage1, positives, k_values, rerank_k):
        N_q  = I_reranked.shape[0]
        tail = I_stage1[:, rerank_k:] if I_stage1.shape[1] > rerank_k else None
        I_full = np.concatenate([I_reranked, tail], axis=1) \
                 if tail is not None else I_reranked
        n_cols  = I_full.shape[1]
        recalls = {}
        for k in k_values:
            k_eff = min(k, n_cols)
            hits  = sum(
                1 for q in range(N_q)
                if set(positives[q]) & set(I_full[q, :k_eff].tolist())
            )
            recalls[k] = hits / N_q
        return recalls

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt   = torch.optim.AdamW(
            self.reranker.parameters(),
            lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        step = self.trainer.global_step
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = scale * self.lr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rerank(self, query_texts, candidate_images):
        """
        Re-score and re-sort stage-1 candidates.
        candidate_images : (B, K, C, H, W)
        Returns          : (B, K) sorted indices best→worst
        """
        self.backbone.eval()
        self.reranker.eval()
        B, K, C, H, W = candidate_images.shape
        scores_list = []
        for b in range(B):
            imgs_b  = candidate_images[b].to(self.device)
            texts_b = [query_texts[b]] * K
            img_local, text_local, text_mask = self._extract_local(imgs_b, texts_b)
            _, score_b = self._score_pairs(img_local, text_local, text_mask)
            scores_list.append(score_b.cpu())
        scores = torch.stack(scores_list, dim=0)
        return scores.argsort(dim=1, descending=True)