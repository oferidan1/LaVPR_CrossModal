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
        → refined L2-normalised embedding per (query, candidate) pair
        → MS Loss over batch + hard-negative queue
        → at inference: re-sort the top-K by refined similarity

Architecture
------------
    text_local  (B*K, L, D_txt)  ─→  txt_proj  ─→  (B*K, L, d)
                                                          │
                                                    CrossAttnBlock × n_layers
                                                    (text queries image patches)
                                                          │
    img_local   (B*K, N, D_img)  ─→  img_proj  ─→  (B*K, N, d)

    CLS token of attended text  →  out_proj  →  L2-norm  →  (B*K, out_dim)

Training data (GSV)
-------------------
    - 4 GT positives per place
    - Stage-1 top-100 shortlist as training candidates (richer mining)
    - Stage-1 top-20  shortlist at inference (latency constraint)
    - Hard-negative queue (4 096 entries) accumulates confusing cases
      across batches so MS Loss sees sufficient pair diversity
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
        q  = self.norm_q(query)
        k  = self.norm_kv(kv)
        ca, _ = self.cross_attn(q, k, k)           # image has no padding
        query = query + self.drop_cross(ca)

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
        img_dim:  int   = 768,
        txt_dim:  int   = 1024,
        d_model:  int   = 256,
        n_heads:  int   = 8,
        n_layers: int   = 2,
        dropout:  float = 0.1,
        out_dim:  int   = 512,
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

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.img_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.txt_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.out_proj[1].weight, mode='fan_in', nonlinearity='relu')

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
        img_tokens  = self.img_proj(img_local)    # (B, N, d)
        text_tokens = self.txt_proj(text_local)   # (B, L, d)

        # Convert HF mask (1=keep) → MHA pad mask (True=ignore)
        pad_mask = None
        if text_mask is not None:
            pad_mask = text_mask.eq(0)            # (B, L)  True = pad

        x = text_tokens
        for block in self.blocks:
            x = block(query=x, kv=img_tokens, query_pad_mask=pad_mask)

        cls = x[:, 0]                             # attended CLS  (B, d)
        out = self.out_proj(cls)                  # (B, out_dim)
        return F.normalize(out, p=2, dim=1)



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

        # ---- Validation
        rerank_k:        int   = 20,    # shortlist size at val/inference
        faiss_gpu:       bool  = False,

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
            img_dim  = img_dim,
            txt_dim  = txt_dim,
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_layers,
            dropout  = dropout,
            out_dim  = out_dim,
        )



        # ----------------------------------------------------------------
        # Optimiser params
        # ----------------------------------------------------------------
        self.lr           = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.epochs       = epochs

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
    ) -> torch.Tensor:              # (M, out_dim)  L2-normed
        """Run reranker forward on M (image, text) pairs."""
        return self.reranker(img_local, text_local, text_mask)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _listwise_loss(
        self,
        refined:   torch.Tensor,   # (B*K, out_dim)  L2-normed refined embeds
        place_ids: torch.Tensor,   # (B, K)           place IDs, -1 = padding
    ) -> torch.Tensor:
        """
        Per-query softmax cross-entropy over the shortlist.

        For each query:
          - scores  = cosine sim of each candidate to the mean positive anchor
          - targets = uniform distribution over positives (soft label)
          - loss    = KL( targets || softmax(scores) )

        O(B*K) — no pairwise matrix, no miner.
        This is correct here because every negative in the shortlist is already
        a hard negative (pre-mined from stage-1), so mining is redundant.
        """
        B, K   = place_ids.shape
        refined_bk = refined.view(B, K, -1)          # (B, K, out_dim)

        # Identify positives per query using place_id of first entry
        # (first entry is always a positive — guaranteed by RerankDataset)
        query_pid = place_ids[:, 0].unsqueeze(1)     # (B, 1)
        is_pos    = place_ids.eq(query_pid)          # (B, K)
        is_valid  = place_ids >= 0                   # (B, K)  exclude padding

        total_loss = 0.0
        valid      = 0

        for b in range(B):
            pos_mask = is_pos[b]  & is_valid[b]      # (K,)
            neg_mask = ~is_pos[b] & is_valid[b]

            if not pos_mask.any() or not neg_mask.any():
                continue

            cand_embs = refined_bk[b]                # (K, out_dim)

            # Anchor = mean of positive embeddings (detached — not a learned param)
            pos_anchor = F.normalize(
                cand_embs[pos_mask].mean(dim=0, keepdim=True).detach(),
                p=2, dim=1)                          # (1, out_dim)

            # Scores for all valid candidates
            valid_mask_b = pos_mask | neg_mask       # (K,)
            scores  = (cand_embs[valid_mask_b] * pos_anchor).sum(dim=1)  # (P+N,)
            is_pos_valid = pos_mask[valid_mask_b]    # (P+N,) bool

            # Soft targets: uniform over positives
            targets   = is_pos_valid.float()
            targets   = targets / targets.sum().clamp(min=1)
            log_probs = F.log_softmax(scores, dim=0)
            total_loss += -(targets * log_probs).sum()
            valid += 1

        return total_loss / max(valid, 1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        """
        Batch keys
        ----------
        query_texts : List[str]          length B
        images      : (B, K, C, H, W)   K candidates (train: up to 100, val: 20)
        labels      : (B, K)  long      place IDs — same ID = positive pair
        is_positive : (B, K)  float     1.0 for GT positives (for logging)
        """
        query_texts = batch['query_texts']              # List[str]
        images      = batch['images'].to(self.device)  # (B, K, C, H, W)
        place_ids   = batch['labels'].to(self.device)  # (B, K)  long

        B, K, C, H, W = images.shape

        # ---- 1. Extract local tokens from frozen backbone ----
        # Flatten candidates into a single batch of M = B*K images
        imgs_flat = images.view(B * K, C, H, W)

        # Text: repeat each query K times to pair with each candidate
        texts_rep = [t for t in query_texts for _ in range(K)]  # length B*K

        img_local, text_local, text_mask = self._extract_local(imgs_flat, texts_rep)
        # img_local  : (B*K, N, D_img)
        # text_local : (B*K, L, D_txt)
        # text_mask  : (B*K, L)

        # ---- 2. Refined embeddings (reranker — grad flows here) ----
        refined = self._score_pairs(img_local, text_local, text_mask)  # (B*K, out_dim)

        flat_labels = place_ids.view(-1)   # (B*K,)

        # ---- 3. Listwise loss per query ----
        loss = self._listwise_loss(refined, place_ids)

        # ---- Logging ----
        self.log('train/loss', loss, prog_bar=True, logger=True)

        return loss

    # ------------------------------------------------------------------
    # Validation — real Pitts30k / MSLS evaluation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """
        Collect global embeddings from the frozen backbone.
        Mirrors LaVPR_cross.validation_step exactly.
        """
        places, _, texts = batch
        with torch.no_grad():
            img_embeds,  _ = self.backbone.encode_image(places)
            text_embeds, _, _, _ = self.backbone.encode_text(texts)
        return {
            'descriptors': img_embeds.detach().cpu(),   # global image embed
            'text_embeds': text_embeds.detach().cpu(),  # global text  embed
            'images'     : places.cpu(),                # raw images for reranker
            'texts'      : list(texts),                 # raw texts  for reranker
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
        rerank_k = self.hparams.get('rerank_k', 20)

        if len(val_dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        for val_set_name, val_dataset, step_outputs in zip(
                val_dm.val_set_names, val_dm.val_datasets, val_step_outputs):

            # ---- Collect embeddings ----
            img_feats   = torch.cat([o['descriptors'] for o in step_outputs], dim=0)
            text_feats  = torch.cat([o['text_embeds']  for o in step_outputs], dim=0)
            all_images  = torch.cat([o['images']        for o in step_outputs], dim=0)
            all_texts   = [t for o in step_outputs for t in o['texts']]

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
            r_feats = img_feats[:num_references]    # (N_db, D)  image gallery
            q_feats = text_feats[num_references:]   # (N_q,  D)  text queries
            q_images_all = all_images[num_references:]  # (N_q, C, H, W)
            q_texts_all  = all_texts[num_references:]   # List[str] len N_q
            db_images_all = all_images[:num_references] # (N_db, C, H, W)

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
                faiss_gpu     = self.hparams.get('faiss_gpu', False),
            )

            # ---- Stage-2: reranker on top-rerank_k shortlist ----
            print(f"[{val_set_name}] Reranking top-{rerank_k} shortlist...")
            self.reranker.eval()

            # Build reranked index: for each query, re-score top-rerank_k images
            # Process queries in mini-batches to avoid OOM
            rerank_batch = 8
            I_reranked   = I_stage1[:, :rerank_k].copy()   # (N_q, rerank_k)

            for q_start in range(0, N_q, rerank_batch):
                q_end   = min(q_start + rerank_batch, N_q)
                bsz     = q_end - q_start

                # Gather shortlist images for this query batch
                cand_indices = I_stage1[q_start:q_end, :rerank_k]  # (bsz, K)
                cand_images  = torch.stack([
                    db_images_all[cand_indices[b]]   # (K, C, H, W)
                    for b in range(bsz)
                ]).to(self.device)                   # (bsz, K, C, H, W)

                batch_texts = q_texts_all[q_start:q_end]  # List[str] len bsz

                with torch.no_grad():
                    K   = rerank_k
                    C, H, W = cand_images.shape[2:]
                    imgs_flat  = cand_images.view(bsz * K, C, H, W)
                    texts_rep  = [t for t in batch_texts for _ in range(K)]

                    img_local, text_local, text_mask = self._extract_local(
                        imgs_flat, texts_rep)
                    refined = self._score_pairs(
                        img_local, text_local, text_mask)   # (bsz*K, out_dim)
                    refined_bk = refined.view(bsz, K, -1)   # (bsz, K, out_dim)

                    # Text anchor: encode query once, project to out_dim
                    _, text_local_q, mask_q, _ = self.backbone.encode_text(batch_texts)
                    txt_tokens = self.reranker.txt_proj(text_local_q)  # (bsz, L, d)
                    txt_cls    = txt_tokens[:, 0]                      # (bsz, d)
                    txt_anchor = F.normalize(
                        self.reranker.out_proj(txt_cls), p=2, dim=1)   # (bsz, out_dim)

                    sim     = (refined_bk * txt_anchor.unsqueeze(1)).sum(dim=-1)  # (bsz, K)
                    ranking = sim.argsort(dim=1, descending=True).cpu().numpy()   # (bsz, K)

                # Reorder shortlist indices by reranker scores
                for b in range(bsz):
                    I_reranked[q_start + b] = cand_indices[b][ranking[b]]

            # Compute recalls after reranking
            # Build a fake r_list/q_list using reranked order for get_validation_recalls
            # Instead, compute recalls directly from I_reranked
            recalls_after = self._recalls_from_index(
                I_reranked, I_stage1, positives, k_values, rerank_k)

            # ---- Log and print ----
            print(f"\n{'':=<60}")
            print(f"  {val_set_name}  —  Stage-1 vs Reranked")
            print(f"{'':=<60}")
            for k in [1, 5, 10]:
                b = recalls_before.get(k, 0.0)
                a = recalls_after.get(k, 0.0)
                print(f"  R@{k:<3d}  before={b:.4f}  after={a:.4f}  "
                      f"gain={a - b:+.4f}")
            print(f"{'':=<60}\n")

            for k in k_values:
                self.log(f'{val_set_name}/R{k}_stage1',
                         recalls_before.get(k, 0.0), prog_bar=False, logger=True)
                self.log(f'{val_set_name}/R{k}_reranked',
                         recalls_after.get(k, 0.0),  prog_bar=False, logger=True)

            self.log(f'{val_set_name}/R1_stage1',
                     recalls_before.get(1, 0.0), prog_bar=True, logger=True)
            self.log(f'{val_set_name}/R1_reranked',
                     recalls_after.get(1, 0.0),  prog_bar=True, logger=True)
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
        if I_stage1.shape[1] >= max_k:
            tail     = I_stage1[:, rerank_k:max_k]          # (N_q, max_k - rerank_k)
            I_full   = np.concatenate([I_reranked, tail], axis=1)  # (N_q, max_k)
        else:
            I_full   = I_reranked

        for k in k_values:
            hits = 0
            for q_idx in range(N_q):
                gt      = set(positives[q_idx])
                top_k   = set(I_full[q_idx, :k].tolist())
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
        candidate_images: torch.Tensor,    # (B, K, C, H, W)   K = top-20 from stage-1
    ) -> torch.Tensor:                     # (B, K)  indices sorted best → worst
        """
        Re-score and re-sort stage-1 candidates.

        Returns argsort indices so callers can reorder their candidate lists:
            sorted_images = candidate_images[b, ranking[b]]
        """
        self.backbone.eval()
        self.reranker.eval()

        B, K, C, H, W = candidate_images.shape

        imgs_flat = candidate_images.view(B * K, C, H, W).to(self.device)
        texts_rep = [t for t in query_texts for _ in range(K)]

        img_local, text_local, text_mask = self._extract_local(imgs_flat, texts_rep)
        refined = self._score_pairs(img_local, text_local, text_mask)   # (B*K, out_dim)
        refined_bk = refined.view(B, K, -1)                             # (B, K, out_dim)

        # Score each candidate against a shared text anchor.
        # Re-encode text once (without repeating K times) for efficiency.
        _, text_local_q, mask_q, _ = self.backbone.encode_text(query_texts)
        # Text anchor: project through txt_proj → take CLS → out_proj
        txt_tokens  = self.reranker.txt_proj(text_local_q)       # (B, L, d)
        txt_cls     = txt_tokens[:, 0]                           # (B, d)   CLS before blocks
        txt_anchor  = F.normalize(
            self.reranker.out_proj(txt_cls), p=2, dim=1)         # (B, out_dim)

        # Cosine similarity between refined candidate embeds and text anchor
        sim = (refined_bk * txt_anchor.unsqueeze(1)).sum(dim=-1)  # (B, K)
        ranking = sim.argsort(dim=1, descending=True)             # (B, K)

        return ranking

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        """Save reranker weights separately for easy standalone loading."""
        ckpt_cb = next(
            (cb for cb in self.trainer.checkpoint_callbacks
             if isinstance(cb, pl.callbacks.ModelCheckpoint)), None)
        if ckpt_cb is None:
            return
        save_dir = os.path.dirname(ckpt_cb.dirpath)
        save_path = os.path.join(save_dir, 'reranker_weights.pt')
        torch.save(self.reranker.state_dict(), save_path)
        print(f"Saved reranker weights to: {save_path}")