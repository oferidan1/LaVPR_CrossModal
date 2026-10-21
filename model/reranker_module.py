"""
Cross-Modal Reranker for VPR (Visual Place Recognition)
========================================================

Stage-2 reranker on top of a frozen LaVPR_cross backbone.

Pipeline
--------
    Stage-1 (frozen LaVPR_cross, global embeds)
        → top-K image candidates per text query

    Stage-2 (this module — only CrossModalReranker trains)
        → cross-attention over local patch / text tokens
        → scalar relevance score per (query, candidate) pair via score_head
        → listwise cross-entropy loss (no miner needed — all negatives are hard)
        → at inference: re-sort the top-K by scalar score

Architecture
------------
    text_local  (K, L, D_txt)  ─→  txt_proj  ─→  (K, L, d)
                                                        │
                                                  CrossAttnBlock × n_layers
                                                  (text queries image patches)
                                                        │
    img_local   (K, N, D_img)  ─→  img_proj  ─→  (K, N, d)

    CLS of attended text  →  score_head  →  scalar score  (K,)

Training data (GSV)
-------------------
    - 4 GT positives per place
    - Stage-1 top-100 shortlist mined offline
    - Precomputed local tokens stored in HDF5 or .pt files
    - Backbone never runs at training time
"""

import os
import random

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
    """Read native hidden dims straight from the frozen backbone."""
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
# Cross-attention block
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    """
    Pre-norm block:
        1. Cross-attention  — text tokens (Q) attend to image patches (K, V)
        2. Self-attention   — text tokens refine among themselves
        3. FFN

    Parameters
    ----------
    d_model : inner dimension (both modalities projected here before entering)
    n_heads : attention heads
    dropout : applied inside MHA and FFN
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        # --- cross-attention ---
        self.norm_q    = nn.LayerNorm(d_model)
        self.norm_kv   = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_cross = nn.Dropout(dropout)

        # --- self-attention on text ---
        self.norm_self  = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop_self  = nn.Dropout(dropout)

        # --- FFN ---
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,            # (B, L, d)  text tokens
        kv:    torch.Tensor,            # (B, N, d)  image patch tokens
        query_pad_mask: torch.Tensor = None,  # (B, L) bool, True = pad position
    ) -> torch.Tensor:                  # (B, L, d)

        # 1. Cross-attention
        q       = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        ca, _   = self.cross_attn(q, kv_norm, kv_norm)    # image has no padding
        query   = query + self.drop_cross(ca)

        # 2. Self-attention  (mask: MHA expects True = ignore)
        q2 = self.norm_self(query)
        sa, _ = self.self_attn(
            q2, q2, q2,
            key_padding_mask=query_pad_mask,        # (B, L) True = pad
        )
        query = query + self.drop_self(sa)

        # 3. FFN
        query = query + self.ffn(self.norm_ffn(query))
        return query


# ---------------------------------------------------------------------------
# Reranker network
# ---------------------------------------------------------------------------

class CrossModalReranker(nn.Module):
    """
    Projects both modalities to d_model, runs n_layers of cross-attention,
    then projects the attended CLS token to a shared out_dim embedding.

    Output is L2-normalised so cosine similarity == dot product.

    Parameters
    ----------
    img_dim  : native DINOv2 patch dim  (768 for ViT-B)
    txt_dim  : native BGE hidden dim    (1024 for bge-large, 768 for bge-base)
    d_model  : inner attention dim
    n_heads  : attention heads  (d_model must be divisible by n_heads)
    n_layers : number of CrossAttnBlocks
    dropout  : dropout rate
    out_dim  : final embedding dim (should match backbone embeds_dim for
               compatibility with existing MS-Loss / FAISS infrastructure)
    """

    def __init__(
        self,
        img_dim:         int   = 768,
        txt_dim:         int   = 1024,
        d_model:         int   = 256,
        n_heads:         int   = 8,
        n_layers:        int   = 2,
        dropout:         float = 0.1,
        out_dim:         int   = 512,
        deep_score_head: bool  = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.img_proj = nn.Linear(img_dim, d_model, bias=False)
        self.txt_proj = nn.Linear(txt_dim, d_model, bias=False)

        self.blocks = nn.ModuleList([
            CrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Project attended CLS → shared embedding space
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim, bias=False),
        )

        # Scalar relevance score head — used at inference
        # deep_score_head=False: Linear(d→1)       — shallow, fewer params
        # deep_score_head=True:  Linear(d→d/2)+GELU+Linear(d/2→1) — non-linear
        if deep_score_head:
            self.score_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2, bias=True),
                nn.GELU(),
                nn.Linear(d_model // 2, 1, bias=True),
            )
        else:
            self.score_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1, bias=True),
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.img_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.txt_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj[1].weight, mode='fan_in', nonlinearity='relu')
        # score_head: last linear init to near-zero for stable early training
        last = self.score_head[-1]   # Linear layer regardless of depth
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(
        self,
        img_local:  torch.Tensor,           # (B, N, D_img)
        text_local: torch.Tensor,           # (B, L, D_txt)
        text_mask:  torch.Tensor = None,    # (B, L)  1=real token, 0=pad
    ) -> torch.Tensor:                      # (B, out_dim)  L2-normed
        """
        Returns a refined L2-normalised embedding for each (image, text) pair.
        text_mask follows HuggingFace convention: 1 = keep, 0 = pad.
        Internally converted to True=ignore for nn.MultiheadAttention.
        """
        dtype       = self.img_proj.weight.dtype
        img_tokens  = self.img_proj(img_local.to(dtype))    # (B, N, d)
        text_tokens = self.txt_proj(text_local.to(dtype))   # (B, L, d)

        # Convert HF mask (1=keep) → MHA pad mask (True=ignore)
        pad_mask = None
        if text_mask is not None:
            pad_mask = text_mask.eq(0)            # (B, L)  True = pad

        x = text_tokens
        for block in self.blocks:
            x = block(query=x, kv=img_tokens, query_pad_mask=pad_mask)

        cls   = x[:, 0]                           # attended CLS  (B, d)
        score = self.score_head(cls).squeeze(-1)  # (B,)  scalar relevance
        # out_proj only needed at inference (embedding for FAISS-style retrieval)
        out   = self.out_proj(cls) if not self.training else None
        emb   = F.normalize(out, p=2, dim=1) if out is not None else None
        return emb, score



# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class CrossModalRerankModule(pl.LightningModule):
    """
    Stage-2 cross-modal reranker.

    Only CrossModalReranker parameters are trained.
    LaVPR_cross backbone is loaded from a checkpoint and fully frozen.

    Training data expectations (batch dict keys)
    ---------------------------------------------
        query_texts : List[str]           length B
        images      : (B, K, C, H, W)    K candidates from stage-1 shortlist
        labels      : (B, K)             place IDs (long)  — same ID = positive pair
        is_positive : (B, K)             float 1.0/0.0     — for logging only

    GSV specifics
    -------------
        - 4 GT positives per place → MS Loss sees multiple (pos, pos) pairs
          within a query's shortlist, pulling them together in embedding space.
        - Training shortlist K=100 (richer mining than inference top-20).
        - Inference shortlist K=20  (latency constraint — rerank() respects this).
    """

    def __init__(
        self,
        backbone_ckpt:  str,

        # ---- Reranker arch
        d_model:   int   = 256,
        n_heads:   int   = 8,
        n_layers:  int   = 2,
        dropout:   float = 0.1,
        out_dim:   int   = 512,     # should match backbone embeds_dim

        # ---- Score head
        deep_score_head: bool  = False,  # False=Linear(d→1)  True=Linear(d→d/2)+GELU+Linear

        # ---- Validation
        rerank_k:        int   = 20,
        faiss_gpu:       bool  = False,

        # ---- Loss
        loss_name:       str   = 'margin',   # 'margin' | 'bce'
        margin:          float = 0.3,        # used by margin loss only

        # ---- Optimiser
        lr:           float = 1e-4,
        weight_decay: float = 1e-3,
        warmup_steps: int   = 200,
        epochs:       int   = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ----------------------------------------------------------------
        # Frozen backbone
        # ----------------------------------------------------------------
        self.backbone = LaVPR_cross.load_from_checkpoint(
            backbone_ckpt, map_location='cpu')
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        img_dim, txt_dim = _infer_dims(self.backbone)

        # ----------------------------------------------------------------
        # Trainable reranker
        # ----------------------------------------------------------------
        self.reranker = CrossModalReranker(
            img_dim          = img_dim,
            txt_dim          = txt_dim,
            d_model          = d_model,
            n_heads          = n_heads,
            n_layers         = n_layers,
            dropout          = dropout,
            out_dim          = out_dim,
            deep_score_head  = deep_score_head,
        )



        # ----------------------------------------------------------------
        # Optimiser params
        # ----------------------------------------------------------------
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.epochs       = epochs
        self.loss_name    = loss_name
        self.margin       = margin
        print(f"Loss: {loss_name}"
              + (f"  margin={margin}" if loss_name == 'margin' else ""))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Feature extraction (backbone is frozen — always no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_local(self, images: torch.Tensor, texts: list):
        """
        Extract local tokens from frozen backbone.

        Parameters
        ----------
        images : (B, C, H, W)
        texts  : List[str] length B

        Returns
        -------
        img_local  : (B, N, D_img)   DINOv2 patch tokens
        text_local : (B, L, D_txt)   BGE last hidden state (all tokens)
        text_mask  : (B, L)          HF attention mask  1=real, 0=pad
        """
        self.backbone.eval()
        _, img_local               = self.backbone.encode_image(images)
        _, text_local, text_mask, _ = self.backbone.encode_text(texts)

        if img_local is None:
            raise RuntimeError(
                "Image backbone returned no local tokens. "
                "CrossModalReranker requires DINOv2 (not SelaVPR)."
            )
        return img_local, text_local, text_mask

    # ------------------------------------------------------------------
    # Core scoring: given local tokens for B*(query,candidate) pairs
    # ------------------------------------------------------------------

    def _score_pairs(
        self,
        img_local:  torch.Tensor,   # (M, N, D_img)
        text_local: torch.Tensor,   # (M, L, D_txt)
        text_mask:  torch.Tensor,   # (M, L)
    ):
        """Returns (embeddings (M, out_dim), scores (M,))"""
        return self.reranker(img_local, text_local, text_mask)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        scores:         torch.Tensor,   # (B, K)
        place_ids:      torch.Tensor,   # (B, K)
        query_place_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """Dispatch to the configured loss function."""
        if self.loss_name == 'margin':
            return self._margin_loss(scores, place_ids, query_place_id)
        elif self.loss_name == 'bce':
            return self._bce_loss(scores, place_ids, query_place_id)
        else:
            raise ValueError(f"Unknown loss_name '{self.loss_name}'. "
                             f"Choose 'margin' or 'bce'.")

    def _margin_loss(
        self,
        scores:         torch.Tensor,   # (B, K)
        place_ids:      torch.Tensor,   # (B, K)
        query_place_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Margin ranking loss: best positive must score at least `margin`
        above every negative. Directly optimises R@1.

        loss = mean( max(0, margin - best_pos + neg_score) ) over all negatives.

        Pros: focuses on hard negatives, directly targets R@1, no softmax saturation.
        Cons: only uses the best positive — other views contribute nothing if
              best_pos already clears the margin.
        """
        query_pid  = query_place_id.unsqueeze(1)
        is_pos     = place_ids.eq(query_pid)
        is_valid   = place_ids >= 0

        total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        valid      = 0

        for b in range(scores.shape[0]):
            pos_mask = is_pos[b] & is_valid[b]
            neg_mask = ~is_pos[b] & is_valid[b]

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_scores = scores[b][pos_mask]             # (P,)
            neg_scores = scores[b][neg_mask]             # (N,)

            best_pos   = pos_scores.max()
            losses     = F.relu(self.margin - best_pos + neg_scores)  # (N,)
            total_loss = total_loss + losses.mean()
            valid     += 1

        return total_loss / max(valid, 1)

    def _bce_loss(
        self,
        scores:         torch.Tensor,   # (B, K)
        place_ids:      torch.Tensor,   # (B, K)
        query_place_id: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Binary cross-entropy loss: treat each candidate independently.
        Positive → label 1, negative → label 0.

        loss = -mean( y·log(σ(s)) + (1-y)·log(1-σ(s)) ) over valid candidates.

        Pros: all positives (all 4 GSV views) contribute gradient equally,
              stable training, well-understood.
        Cons: treats candidates independently — does not model ranking directly.
              A negative at 0.49 and positive at 0.51 both get low loss even
              though the gap is dangerously small.
        """
        query_pid = query_place_id.unsqueeze(1)
        is_pos    = place_ids.eq(query_pid)
        is_valid  = place_ids >= 0

        # Build labels: 1.0 for positives, 0.0 for negatives, ignore padding
        labels = is_pos.float()                          # (B, K)

        # Mask out padding entries
        valid_mask = is_valid.float()                    # (B, K)

        # BCE per element, then mean over valid only
        bce = F.binary_cross_entropy_with_logits(
            scores, labels, reduction='none')            # (B, K)
        loss = (bce * valid_mask).sum() / valid_mask.sum().clamp(min=1)
        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Backbone never runs here — tokens are precomputed.
        Peak memory = B * K * token_tensors (tiny vs raw images).
        Batch size can now be large (64+) without OOM.

        Batch keys (from RerankDataset):
            query_text_local : (B, L, D_txt)
            query_mask       : (B, L)
            cand_img_locals  : (B, K, N, D_img)
            labels           : (B, K)  long
        """
        # Keep in bf16/fp16 to halve GPU memory — projections handle precision
        text_local      = batch['query_text_local'].to(self.device, dtype=torch.bfloat16)
        text_mask       = batch['query_mask'].to(self.device)
        img_locals      = batch['cand_img_locals'].to(self.device, dtype=torch.bfloat16)
        place_ids       = batch['labels'].to(self.device)
        query_place_id  = batch['query_place_id'].to(self.device)

        B, K, N, D_img = img_locals.shape
        L              = text_local.shape[1]

        # Chunked: process chunk_size queries at a time.
        # - Larger than 1 → GPU sees real workload, no per-kernel-launch overhead
        # - Smaller than B → activation memory stays bounded
        # chunk_size=4: 4×50=200 pairs per forward → tiny memory, 16x fewer kernel launches vs loop
        chunk_size = B   # process all queries at once — tune down if OOM
        scores_list = []

        for start in range(0, B, chunk_size):
            end      = min(start + chunk_size, B)
            c        = end - start                           # actual chunk size

            img_c    = img_locals[start:end].reshape(c * K, N, D_img)
            txt_c    = text_local[start:end].unsqueeze(1)\
                                            .expand(-1, K, -1, -1)\
                                            .reshape(c * K, L, -1).contiguous()
            msk_c    = text_mask[start:end].unsqueeze(1)\
                                           .expand(-1, K, -1)\
                                           .reshape(c * K, -1).contiguous()

            _, s_c   = self._score_pairs(img_c, txt_c, msk_c)   # (c*K,)
            scores_list.append(s_c.view(c, K))

        scores = torch.cat(scores_list, dim=0)                   # (B, K)

        loss = self._compute_loss(scores, place_ids, query_place_id)
        self.log('train/loss', loss, prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Validation — real Pitts30k / MSLS evaluation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Collect global embeddings + local tokens from frozen backbone.
        Store local tokens (not raw images) to avoid holding 14GB of images in RAM.
        """
        places, _, texts = batch
        with torch.no_grad():
            img_embeds,  img_local           = self.backbone.encode_image(
                places.to(self.device))
            text_embeds, text_local, mask, _ = self.backbone.encode_text(texts)
        return {
            'descriptors' : img_embeds .detach().cpu(),          # (B, D)
            'text_embeds' : text_embeds.detach().cpu(),          # (B, D)
            'img_local'   : img_local  .detach().cpu().half(),   # (B, N, D_img) fp16
            'text_local'  : text_local .detach().cpu().half(),   # (B, L, D_txt) fp16
            'text_mask'   : mask       .detach().cpu(),          # (B, L)
        }

    def validation_epoch_end(self, val_step_outputs):
        """
        Two-stage evaluation on Pitts30k / MSLS:

        Stage-1: FAISS text→image retrieval with global embeddings (backbone only).
        Stage-2: reranker re-scores the top-K shortlist per query.

        Logs R@k before and after reranking so the gain is visible directly.
        """
        import faiss
        import utils  # get_validation_recalls from your codebase

        # val_datamodule is the GSVCitiesDataModule attached in train_reranker.py
        dm       = self.trainer.datamodule
        val_dm   = getattr(dm, 'val_datamodule', dm)   # fallback to dm itself
        k_values = [1, 5, 10, 15, 20, 50, 100]
        rerank_k = self.hparams.rerank_k

        if len(val_dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for val_set_name, val_dataset, step_outputs in zip(
                val_dm.val_set_names, val_dm.val_datasets, val_step_outputs):

            # ---- Collect embeddings and local tokens ----
            img_feats  = torch.cat([o['descriptors'] for o in step_outputs], dim=0)
            text_feats = torch.cat([o['text_embeds'] for o in step_outputs], dim=0)
            all_img_loc = torch.cat([o['img_local']  for o in step_outputs], dim=0)

            # Text local tokens have variable L — pad to max length in epoch
            all_txt_loc_list = [tl for o in step_outputs for tl in o['text_local']]
            all_txt_msk_list = [tm for o in step_outputs for tm in o['text_mask']]
            max_L   = max(t.shape[0] for t in all_txt_loc_list)
            D_txt   = all_txt_loc_list[0].shape[1]
            N_total = len(all_txt_loc_list)

            all_txt_loc = torch.zeros(N_total, max_L, D_txt, dtype=torch.float16)
            all_txt_msk = torch.zeros(N_total, max_L, dtype=torch.bool)
            for i, (tl, tm) in enumerate(zip(all_txt_loc_list, all_txt_msk_list)):
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
                raise NotImplementedError(
                    f'Add validation logic for {val_set_name}')

            # Split DB (image gallery) and queries (text)
            r_feats      = img_feats   [:num_references]   # (N_db, D)
            q_feats      = text_feats  [num_references:]   # (N_q,  D)
            db_img_loc   = all_img_loc [:num_references]   # (N_db, N_p, D_img)
            q_txt_loc    = all_txt_loc [num_references:]   # (N_q,  L,  D_txt)
            q_txt_msk    = all_txt_msk [num_references:]   # (N_q,  L)

            N_db = r_feats.shape[0]
            N_q  = q_feats.shape[0]
            D    = r_feats.shape[1]

            # ---- Stage-1: FAISS retrieval ----
            print(f"\n[{val_set_name}] Stage-1 FAISS retrieval  "
                  f"N_db={N_db:,}  N_q={N_q:,}  top-k={max(k_values)}")
            index = faiss.IndexFlatIP(D)
            index.add(r_feats.numpy().astype('float32'))
            _, I_stage1 = index.search(
                q_feats.numpy().astype('float32'), max(k_values))  # (N_q, 100)

            # Stage-1 recalls (before reranking)
            recalls_before = utils.get_validation_recalls(
                r_list        = r_feats,
                q_list        = q_feats,
                k_values      = k_values,
                gt            = positives,
                print_results = True,
                dataset_name  = f'{val_set_name}_stage1',
                faiss_gpu     = self.hparams.faiss_gpu,
            )

            # ---- Stage-2: reranker on top-rerank_k shortlist ----
            print(f"[{val_set_name}] Reranking top-{rerank_k} shortlist...")
            self.reranker.eval()

            # Build reranked index: for each query, re-score top-rerank_k images
            # Process queries in mini-batches to avoid OOM
            rerank_batch = getattr(self.hparams, 'val_rerank_batch', 8)
            I_reranked   = I_stage1[:, :rerank_k].copy()   # (N_q, rerank_k)

            for q_start in range(0, N_q, rerank_batch):
                q_end   = min(q_start + rerank_batch, N_q)
                bsz     = q_end - q_start

                # Gather precomputed local tokens for shortlist
                cand_indices = I_stage1[q_start:q_end, :rerank_k]  # (bsz, K)

                with torch.no_grad():
                    # Gather all candidate image tokens: (bsz, K, N_p, D_img)
                    img_loc_bsz = torch.stack([
                        db_img_loc[cand_indices[b]] for b in range(bsz)
                    ]).to(self.device).float()                       # (bsz, K, N_p, D_img)

                    # Query text tokens: (bsz, L, D_txt) → expand to (bsz*K, L, D_txt)
                    txt_loc_bsz = q_txt_loc[q_start:q_end]\
                        .to(self.device).float()                     # (bsz, L, D_txt)
                    txt_msk_bsz = q_txt_msk[q_start:q_end]\
                        .to(self.device)                             # (bsz, L)

                    bsz_K   = bsz * rerank_k
                    N_p     = img_loc_bsz.shape[2]
                    L_      = txt_loc_bsz.shape[1]
                    D_txt_  = txt_loc_bsz.shape[2]

                    imgs_flat = img_loc_bsz.view(bsz_K, N_p, -1)    # (bsz*K, N_p, D_img)
                    txt_flat  = txt_loc_bsz.unsqueeze(1)\
                                           .expand(-1, rerank_k, -1, -1)\
                                           .reshape(bsz_K, L_, D_txt_)\
                                           .contiguous()             # (bsz*K, L, D_txt)
                    msk_flat  = txt_msk_bsz.unsqueeze(1)\
                                           .expand(-1, rerank_k, -1)\
                                           .reshape(bsz_K, L_)\
                                           .contiguous()             # (bsz*K, L)

                    _, scores_flat = self._score_pairs(imgs_flat, txt_flat, msk_flat)
                    scores  = scores_flat.view(bsz, rerank_k).cpu()  # (bsz, K)
                    ranking = scores.argsort(dim=1, descending=True).numpy()

                # Reorder shortlist indices by reranker scores
                for b in range(bsz):
                    I_reranked[q_start + b] = cand_indices[b][ranking[b]]

            # Compute recalls after reranking
            # Build a fake r_list/q_list using reranked order for get_validation_recalls
            # Instead, compute recalls directly from I_reranked
            recalls_after = self._recalls_from_index(
                I_reranked, I_stage1, positives, k_values, rerank_k)

            # ---- Log and print ----
            print(f"\n{'':=<65}")
            print(f"  {val_set_name}  —  Stage-1 vs Reranked")
            print(f"{'':=<65}")
            print(f"  {'':5s}  {'Stage-1':>10s}  {'Reranked':>10s}  {'Gain':>8s}")
            for k in [1, 5, 10, 20]:
                b = recalls_before.get(k, 0.0)
                a = recalls_after.get(k, 0.0)
                print(f"  R@{k:<3d}  {b:>10.4f}  {a:>10.4f}  {a-b:>+8.4f}")
            print(f"{'':=<65}\n")

            # Log all k values — single call per metric, logger=True for all
            prog_bar_ks = {1, 5}   # show R@1 and R@5 on progress bar
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
    def _recalls_from_index(
        I_reranked:  np.ndarray,   # (N_q, rerank_k)  reranked top-K indices
        I_stage1:    np.ndarray,   # (N_q, max_k)     full stage-1 ranking
        positives:   list,         # GT positives per query
        k_values:    list,
        rerank_k:    int,
    ) -> dict:
        """
        Compute recall@k from a reranked shortlist.

        For k <= rerank_k : use I_reranked order.
        For k >  rerank_k : append stage-1 tail (ranks rerank_k..k) unchanged,
                            since the reranker only touched the top-rerank_k.
        """
        import numpy as np

        N_q     = I_reranked.shape[0]
        max_k   = max(k_values)
        recalls = {}

        # Build full reranked ranking: reranked head + stage-1 tail
        # The tail is everything stage-1 ranked beyond rerank_k (untouched by reranker)
        if I_stage1.shape[1] > rerank_k:
            tail   = I_stage1[:, rerank_k:]                 # (N_q, stage1_k - rerank_k)
            I_full = np.concatenate([I_reranked, tail], axis=1)  # (N_q, rerank_k + tail)
        else:
            # Stage-1 only retrieved rerank_k or fewer — no tail
            I_full = I_reranked

        n_cols = I_full.shape[1]
        for k in k_values:
            k_eff = min(k, n_cols)   # guard against I_full having fewer cols than k
            hits  = 0
            for q_idx in range(N_q):
                gt    = set(positives[q_idx])
                top_k = set(I_full[q_idx, :k_eff].tolist())
                if gt & top_k:
                    hits += 1
            recalls[k] = hits / N_q

        return recalls

    # ------------------------------------------------------------------
    # Optimiser  (only reranker params — backbone is frozen)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.reranker.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Linear LR warmup matching LaVPR_cross convention."""
        step = self.trainer.global_step
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = scale * self.lr

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rerank(
        self,
        query_texts:      list,            # List[str]  length B
        candidate_images: torch.Tensor,    # (B, K, C, H, W)
    ) -> torch.Tensor:                     # (B, K)  indices sorted best → worst
        """
        Re-score and re-sort stage-1 candidates using score_head.
        Process one query at a time to match training memory pattern.
        """
        self.backbone.eval()
        self.reranker.eval()

        B, K, C, H, W = candidate_images.shape
        scores_list = []

        for b in range(B):
            imgs_b = candidate_images[b].to(self.device)    # (K, C, H, W)
            texts_b = [query_texts[b]] * K

            img_local, text_local, text_mask = self._extract_local(imgs_b, texts_b)
            _, score_b = self._score_pairs(img_local, text_local, text_mask)  # (K,)
            scores_list.append(score_b.cpu())

        scores  = torch.stack(scores_list, dim=0)            # (B, K)
        ranking = scores.argsort(dim=1, descending=True)     # (B, K)
        return ranking

    # ------------------------------------------------------------------
    # Checkpoint — handled entirely by ModelCheckpoint callback.
    # Load reranker weights from .ckpt with:
    #   model = CrossModalRerankModule.load_from_checkpoint('path/to.ckpt')
    #   reranker_weights = model.reranker.state_dict()
    # ------------------------------------------------------------------