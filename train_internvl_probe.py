"""
train_internvl_probe.py
=======================
Train a linear probe on top of frozen InternVL2-2B for VPR reranking.

Architecture:
    InternVL2-2B (frozen) — processes raw image + text jointly
    Linear probe  (trained) — LayerNorm → Linear(hidden→1) → scalar score

Input:
    Raw images (224×224) + raw text descriptions
    Uses gsv_triplets.json for hard negative pairs
    Does NOT use precomputed token cache — runs full VLM forward

Loss:
    BCE (default) or margin ranking loss

Why a separate script:
    InternVL2 processes raw images, not precomputed patch tokens.
    The dataloader, forward pass, and memory management are all
    different from the slim reranker — keeping them separate avoids
    complicating reranker_module.py.

Example:
    python train_internvl_probe.py \\
        --internvl_path OpenGVLab/InternVL2-2B \\
        --triplets_json cache/gsv_triplets.json \\
        --train_csv     datasets/descriptions/gsv_cities_places.csv \\
        --image_root    /home/shared/datasets/gsv_cities/ \\
        --loss_name     bce \\
        --max_cands     10  \\
        --subset        20000 \\
        --batch_size    8   \\
        --epochs        5   \\
        --gpu           0
"""

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InternVLRerankDataset(Dataset):
    """
    Loads (query_text, candidate_image, label) triplets for probe training.

    Each __getitem__ returns one (text, image, label) pair — NOT a full
    shortlist. The probe scores pairs independently so we flatten triplets
    into individual pairs. The loss is computed per query batch in
    collate_fn to preserve the listwise structure for margin loss.

    Returns
    -------
    dict with:
        image       : (3, H, W) tensor
        text        : str  — query description
        label       : float  1.0=positive 0.0=negative
        query_idx   : int  — groups pairs belonging to same query
        place_id    : int  — query place ID
        cand_place  : int  — candidate place ID
    """

    def __init__(
        self,
        triplets_json: str,
        train_csv:     str,
        image_root:    str,
        max_cands:     int   = 10,
        subset:        int   = 0,     # 0 = use all triplets
        image_size:    int   = 224,
        hard_only:     bool  = True,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.max_cands  = max_cands

        # Load CSV
        df = pd.read_csv(train_csv)
        df = df[df['place_id'] >= 0].reset_index(drop=True)
        self.img_paths   = df['image_path'].tolist()
        self.place_ids_  = df['place_id'].tolist()

        # Load triplets
        with open(triplets_json) as f:
            triplets = json.load(f)

        if hard_only:
            triplets = [t for t in triplets if t.get('top1_is_wrong', True)]
            print(f"Hard-only: {len(triplets):,} triplets")

        if subset > 0 and subset < len(triplets):
            random.shuffle(triplets)
            triplets = triplets[:subset]
            print(f"Subset: {len(triplets):,} triplets")

        # Flatten triplets → individual (query, candidate, label) pairs
        # Group by query_idx so collate can reconstruct per-query batches
        self.pairs = []
        for t in triplets:
            q_idx    = t['query_idx']
            place_id = t['place_id']
            pos_set  = set(t['pos_idxs'])

            # Sample candidates: guarantee at least one positive,
            # fill remaining slots with negatives
            pos  = t['pos_idxs']
            negs = t['neg_idxs']

            # Pick one positive guaranteed + sample rest
            anchor_pos = [random.choice(pos)]
            extra_pos  = [p for p in pos if p != anchor_pos[0]]
            neg_sample = random.sample(negs, min(max_cands - 1, len(negs)))
            pool       = extra_pos + neg_sample
            random.shuffle(pool)
            cands      = anchor_pos + pool[:max_cands - 1]

            for c_idx in cands:
                label = 1.0 if c_idx in pos_set else 0.0
                self.pairs.append({
                    'query_idx'  : q_idx,
                    'cand_idx'   : c_idx,
                    'label'      : label,
                    'place_id'   : place_id,
                    'cand_place' : self.place_ids_[c_idx],
                })

        print(f"Total pairs: {len(self.pairs):,}  "
              f"({len(triplets):,} queries × ~{max_cands} cands)")

        # Image transforms — InternVL2 uses its own normalisation
        # but we use IMAGENET stats as a reasonable default
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

    def _load_image(self, idx):
        path = self.img_paths[idx]
        if not path.startswith('/'):
            path = str(self.image_root / path)
        return self.transform(Image.open(path).convert('RGB'))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair    = self.pairs[idx]
        q_idx   = pair['query_idx']
        c_idx   = pair['cand_idx']

        # Placeholder text — overridden by InternVLRerankDatasetWithDesc
        text = ''

        # Candidate image
        image   = self._load_image(c_idx)

        return {
            'image'      : image,
            'text'       : text,
            'label'      : torch.tensor(pair['label'],       dtype=torch.float32),
            'query_idx'  : torch.tensor(pair['query_idx'],   dtype=torch.long),
            'place_id'   : torch.tensor(pair['place_id'],    dtype=torch.long),
            'cand_place' : torch.tensor(pair['cand_place'],  dtype=torch.long),
        }


class InternVLRerankDatasetWithDesc(InternVLRerankDataset):
    """
    Extends InternVLRerankDataset to load actual text descriptions
    from the CSV (requires a 'description' column).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # train_csv is 2nd positional arg or keyword arg
        if 'train_csv' in kwargs:
            train_csv = kwargs['train_csv']
        elif len(args) > 1:
            train_csv = args[1]
        else:
            raise ValueError("train_csv not found in args or kwargs")
        df = pd.read_csv(train_csv)
        df = df[df['place_id'] >= 0].reset_index(drop=True)
        self.descriptions = df['description'].fillna('').tolist()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        q_idx = self.pairs[idx]['query_idx']
        item['text'] = self.descriptions[q_idx]
        return item


def collate_fn(batch):
    """Stack images and labels; keep texts as list."""
    return {
        'image'      : torch.stack([b['image']      for b in batch]),
        'text'       : [b['text']                   for b in batch],
        'label'      : torch.stack([b['label']      for b in batch]),
        'query_idx'  : torch.stack([b['query_idx']  for b in batch]),
        'place_id'   : torch.stack([b['place_id']   for b in batch]),
        'cand_place' : torch.stack([b['cand_place'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# InternVL2 Probe Model
# ---------------------------------------------------------------------------

class InternVLProbe(nn.Module):
    """
    Frozen InternVL2-2B + trainable linear probe.

    The VLM processes each (image, text) pair jointly and produces
    a joint hidden state. We pool the last hidden state and pass it
    through a linear probe to get a scalar relevance score.

    Only the probe parameters are trained (~2050 parameters total).
    The VLM backbone is completely frozen.

    Parameters
    ----------
    internvl_path : HuggingFace model ID or local path
    pool_method   : 'cls' | 'mean' — how to pool the hidden states
    prompt_template : format string with {description} placeholder
    """

    def __init__(
        self,
        internvl_path:    str   = 'OpenGVLab/InternVL2-2B',
        pool_method:      str   = 'cls',
        prompt_template:  str   = '<image>\n{description}',
        lora_r:           int   = 0,    # 0=frozen probe only, >0=LoRA
    ):
        super().__init__()
        self.pool_method     = pool_method
        self.prompt_template = prompt_template

        print(f"Loading InternVL2 from {internvl_path}...")
        from transformers import AutoTokenizer
        from transformers import AutoModel, PreTrainedModel

        # Compatibility patch for newer transformers (e.g. 5.x) with remote InternVL code
        if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
            PreTrainedModel.all_tied_weights_keys = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            internvl_path, trust_remote_code=True)

        self.vlm = AutoModel.from_pretrained(
            internvl_path,
            torch_dtype        = torch.bfloat16,
            trust_remote_code  = True,
            low_cpu_mem_usage  = True,
            use_flash_attn     = False,   # disable flash_attn for compatibility
        )

        if lora_r > 0:
            # Apply LoRA to language model attention layers
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                r              = lora_r,
                lora_alpha     = lora_r * 2,
                target_modules = ['q_proj', 'v_proj',
                                  'k_proj', 'o_proj'],
                lora_dropout   = 0.05,
                bias           = 'none',
            )
            self.vlm = get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()
            self.use_lora = True
        else:
            # Freeze everything — probe only
            for p in self.vlm.parameters():
                p.requires_grad = False
            self.vlm.eval()
            self.use_lora = False
            print(f"InternVL2 frozen: "
                  f"{sum(p.numel() for p in self.vlm.parameters()):,} params")

        # Get dims by running a dummy forward through each encoder
        with torch.no_grad():
            dummy_img = torch.zeros(1, 3, 224, 224,
                                    dtype=torch.bfloat16)
            dummy_vis = self.vlm.extract_feature(dummy_img)  # (1, n_patches, D_vis)
            D_vis     = dummy_vis.shape[-1]

            dummy_tok = self.vlm.language_model.get_input_embeddings()(
                torch.zeros(1, 4, dtype=torch.long))          # (1, 4, D_llm)
            D_llm     = dummy_tok.shape[-1]

        print(f"D_vis={D_vis}  D_llm={D_llm}")

        # Fused dim: D_vis//2 + D_llm//2
        probe_dim = D_vis // 2 + D_llm // 2
        print(f"D_vis={D_vis}  D_llm={D_llm}  probe_dim={probe_dim}")

        # Projection heads — project visual and text to same space before fusion
        self._vis_proj = nn.Linear(D_vis, D_vis // 2, bias=False)
        self._txt_proj = nn.Linear(D_llm, D_llm // 2, bias=False)
        nn.init.xavier_uniform_(self._vis_proj.weight)
        nn.init.xavier_uniform_(self._txt_proj.weight)

        # Probe — LayerNorm + Linear on fused representation
        self.probe = nn.Sequential(
            nn.LayerNorm(probe_dim),
            nn.Linear(probe_dim, 1, bias=True),
        )
        nn.init.zeros_(self.probe[1].weight)
        nn.init.zeros_(self.probe[1].bias)
        print(f"Probe params: {sum(p.numel() for p in self.probe.parameters()) + sum(p.numel() for p in self._vis_proj.parameters()) + sum(p.numel() for p in self._txt_proj.parameters())}")

    def _tokenize(self, texts, device):
        """Tokenize text descriptions."""
        prompts = [self.prompt_template.format(description=t) for t in texts]
        tokens  = self.tokenizer(
            prompts,
            return_tensors  = 'pt',
            padding         = True,
            truncation      = True,
            max_length      = 512,
        ).to(device)
        return tokens

    def _pool(self, hidden_states, attention_mask):
        """Pool last hidden state → (B, D).

        For LLM-based models (InternVL2), the last non-padding token
        is more meaningful than position 0 which may be an image token.
        'cls'  → last non-padding token (EOS / last real token)
        'mean' → mean over all real tokens
        """
        last = hidden_states[-1]   # (B, L, D)
        if self.pool_method == 'cls':
            # Last non-padding token — robust for both encoder and decoder LLMs
            seq_lens = attention_mask.sum(dim=1) - 1   # (B,) index of last real token
            seq_lens = seq_lens.clamp(min=0)
            batch_idx = torch.arange(last.shape[0], device=last.device)
            return last[batch_idx, seq_lens]            # (B, D)
        elif self.pool_method == 'mean':
            mask = attention_mask.unsqueeze(-1).float()
            return (last * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            raise ValueError(f"Unknown pool_method '{self.pool_method}'")

    def forward(self, images, texts):
        """
        Parameters
        ----------
        images : (B, 3, H, W) float tensor
        texts  : List[str] length B

        Returns
        -------
        scores : (B,) scalar relevance scores
        """
        device  = images.device
        pixel   = images.to(torch.bfloat16)
        tokens  = self._tokenize(texts, device)

        # Vision encoder — no LoRA here, safe to use no_grad
        with torch.no_grad():
            visual_feats = self.vlm.extract_feature(pixel)  # (B, n_patches, D_vis)
            vis_pooled   = visual_feats.mean(dim=1).float() # (B, D_vis)

        # LLM forward — LoRA is here, must run with grad during training
        # Use the language model directly with text tokens only
        no_grad_ctx = torch.enable_grad() if (self.use_lora and self.training) else torch.no_grad()
        with no_grad_ctx:
            llm_out = self.vlm.language_model(
                input_ids      = tokens['input_ids'],
                attention_mask = tokens['attention_mask'],
                output_hidden_states = True,
                return_dict    = True,
            )
            # Last hidden state, mean pool over real tokens
            last_hidden = llm_out.hidden_states[-1]          # (B, L, D_llm)
            mask        = tokens['attention_mask'].unsqueeze(-1).float()
            txt_pooled  = (last_hidden.float() * mask).sum(1) \
                          / mask.sum(1).clamp(min=1)          # (B, D_llm)

        vis_proj = self._vis_proj(vis_pooled)             # (B, D_vis/2)
        txt_proj = self._txt_proj(txt_pooled)             # (B, D/2)
        fused    = torch.cat([vis_proj, txt_proj], dim=-1) # (B, D)

        return self.probe(fused).squeeze(-1)               # (B,)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class InternVLProbeLightning(pl.LightningModule):
    """
    Lightning wrapper for InternVL2 probe training.

    Training: BCE or margin loss on (image, text, label) pairs.
    Validation: R@1 on Pitts30k val using the probe scores to rerank
                stage-1 predictions from the LaVPR_cross backbone.
    """

    def __init__(
        self,
        internvl_path:    str   = 'OpenGVLab/InternVL2-2B',
        backbone_ckpt:    str   = None,
        pool_method:      str   = 'cls',
        prompt_template:  str   = '<image>\n{description}',
        lora_r:           int   = 0,
        loss_name:        str   = 'bce',
        margin:           float = 0.3,
        lr:               float = 1e-3,   # higher LR — only probe trains
        weight_decay:     float = 1e-4,
        warmup_steps:     int   = 100,
        epochs:           int   = 5,
        rerank_k:         int   = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = InternVLProbe(
            internvl_path   = internvl_path,
            pool_method     = pool_method,
            prompt_template = prompt_template,
            lora_r          = lora_r,
        )

        self.loss_name   = loss_name
        self.margin      = margin
        self.lr          = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.epochs      = epochs
        self.rerank_k    = rerank_k

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _compute_loss(self, scores, labels, query_idx, place_ids, cand_places):
        if self.loss_name == 'bce':
            return F.binary_cross_entropy_with_logits(scores, labels)
        elif self.loss_name == 'margin':
            return self._margin_loss(scores, labels, query_idx)
        else:
            raise ValueError(f"Unknown loss '{self.loss_name}'")

    def _margin_loss(self, scores, labels, query_idx):
        """
        Per-query margin loss.
        Groups pairs by query_idx and applies:
            best_pos_score > all_neg_scores + margin
        """
        total = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        valid = 0

        unique_queries = query_idx.unique()
        for qid in unique_queries:
            mask     = query_idx.eq(qid)
            s_q      = scores[mask]
            l_q      = labels[mask]

            pos_mask = l_q > 0.5
            neg_mask = l_q <= 0.5

            if not pos_mask.any() or not neg_mask.any():
                continue

            best_pos   = s_q[pos_mask].max()
            losses     = F.relu(self.margin - best_pos + s_q[neg_mask])
            total      = total + losses.mean()
            valid     += 1

        return total / max(valid, 1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        images     = batch['image'].to(self.device)
        texts      = batch['text']
        labels     = batch['label'].to(self.device)
        query_idx  = batch['query_idx'].to(self.device)
        place_ids  = batch['place_id'].to(self.device)
        cand_places = batch['cand_place'].to(self.device)

        scores = self.model(images, texts)
        loss   = self._compute_loss(
            scores, labels, query_idx, place_ids, cand_places)

        self.log('train/loss', loss, prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Validation — rerank Pitts30k val using probe scores
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        """Offload InternVL2 to CPU, free all GPU memory, load backbone."""
        from model.LaVPR_cross import LaVPR_cross
        import gc

        print(f"GPU before val offload: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

        # Move InternVL2 to CPU — frees weights + optimizer states
        self.model.vlm.cpu()
        self.model.probe.cpu()
        self.model._vis_proj.cpu()
        self.model._txt_proj.cpu()

        # Force garbage collection and cache clear
        gc.collect()
        torch.cuda.empty_cache()

        print(f"GPU after offload: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

        # Load backbone fresh — no caching to avoid shared tensor state
        # with self.model.vlm which may share InternVL2 components
        print("Loading LaVPR backbone fresh for validation...")
        self._backbone = LaVPR_cross.load_from_checkpoint(
            self.hparams.backbone_ckpt,
            map_location=self.device,
            strict=True,
        )
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad = False

        # Verify all backbone params are on GPU
        cpu_params = [n for n, p in self._backbone.named_parameters()
                      if p.device.type != 'cuda']
        if cpu_params:
            print(f"WARNING: {len(cpu_params)} backbone params still on CPU: "
                  f"{cpu_params[:3]}")
            # Force move
            self._backbone = self._backbone.cuda()

        print(f"GPU after backbone load: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    def _restore_internvl_after_val(self):
        """Delete backbone and restore InternVL2 to GPU after validation."""
        import gc
        # Delete backbone entirely — loaded fresh next val epoch
        if hasattr(self, '_backbone') and self._backbone is not None:
            del self._backbone
            self._backbone = None
        gc.collect()
        torch.cuda.empty_cache()
        # Restore InternVL2
        self.model.to(self.device)
        self.model.vlm.to(self.device)
        self.model.probe.to(self.device)
        self.model._vis_proj.to(self.device)
        self.model._txt_proj.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        print("InternVL2 restored to GPU.")

    def validation_step(self, batch, batch_idx):
        """
        Run backbone on each batch immediately — no raw image storage.
        Only compact embeddings (~768-dim floats) kept on CPU.
        Val batch size is controlled by the val DataLoader (bs=8).
        """
        places, _, texts = batch
        backbone_device = next(self._backbone.parameters()).device
        with torch.no_grad():
            emb_i, _       = self._backbone.encode_image(
                places.to(backbone_device))
            emb_t, _, _, _ = self._backbone.encode_text(texts)
        return {
            'img_emb'  : emb_i.detach().cpu(),
            'text_emb' : emb_t.detach().cpu(),
            'texts'    : list(texts),
        }

    def validation_epoch_end(self, outputs):
        """Full Pitts30k val evaluation."""
        import faiss
        from model.LaVPR_cross import LaVPR_cross

        if not self.hparams.backbone_ckpt:
            print("No backbone_ckpt — skipping validation reranking.")
            self._restore_internvl_after_val()
            return

        if not outputs:
            print("No validation outputs — skipping.")
            self._restore_internvl_after_val()
            return

        # Embeddings already computed in validation_step — just cat them
        all_texts  = [t for o in outputs for t in o['texts']]
        img_feats  = torch.cat([o['img_emb']  for o in outputs], dim=0).numpy()
        text_feats = torch.cat([o['text_emb'] for o in outputs], dim=0).numpy()
        print(f"Validation: {len(all_texts)} samples")

        # Stage-1 FAISS
        dm          = self.trainer.datamodule
        val_dataset = dm.val_datasets[0] if dm.val_datasets else None
        if val_dataset is None:
            print("No val dataset found — skipping reranking.")
            self._restore_internvl_after_val()
            return
        # Handle both attribute name conventions
        num_db = getattr(val_dataset, 'num_db',
                getattr(val_dataset, 'num_database', None))
        if num_db is None:
            raise AttributeError(
                "val_dataset has neither 'num_db' nor 'num_database'. "
                f"Available: {[a for a in dir(val_dataset) if 'num' in a.lower()]}")

        r_np = img_feats [:num_db].astype('float32')
        q_np = text_feats[num_db:].astype('float32')
        r_np /= np.linalg.norm(r_np, axis=1, keepdims=True).clip(min=1e-8)
        q_np /= np.linalg.norm(q_np, axis=1, keepdims=True).clip(min=1e-8)

        index = faiss.IndexFlatIP(r_np.shape[1])
        index.add(r_np)
        _, I_stage1 = index.search(q_np, max(20, self.rerank_k))

        # Handle both camelCase and snake_case API
        if hasattr(val_dataset, 'getPositives'):
            positives = val_dataset.getPositives()
        elif hasattr(val_dataset, 'get_positives'):
            positives = val_dataset.get_positives()
        else:
            raise AttributeError("val_dataset has no getPositives/get_positives method")
        N_q       = q_np.shape[0]

        def recall_at_k(I, k):
            hits = sum(1 for q in range(N_q)
                      if set(positives[q]) & set(I[q, :k].tolist()))
            return hits / N_q

        r1_before = recall_at_k(I_stage1, 1)
        r5_before = recall_at_k(I_stage1, 5)

        # Free backbone from GPU memory and restore InternVL2 to GPU for reranking
        self._restore_internvl_after_val()

        # Reranking with InternVL2 probe
        self.model.eval()
        I_reranked = I_stage1[:, :self.rerank_k].copy()

        # Get database image paths and query texts
        db_img_paths = val_dataset.images_paths[:num_db]
        # Query texts: val_dataset.descriptions contains all samples,
        # queries start at index num_db
        if hasattr(val_dataset, 'descriptions') and \
                len(val_dataset.descriptions) > num_db:
            q_texts = val_dataset.descriptions[num_db:]
        else:
            # Fallback: use texts collected in validation_step
            # all_texts order matches dataloader order (DB first, queries after)
            q_texts = all_texts[num_db:]
            if len(q_texts) == 0:
                print("Warning: no query texts found — reranking skipped")
                return

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

        rerank_batch = 16   # small — InternVL2 is memory intensive

        with torch.no_grad():
            for q_start in tqdm(range(0, N_q, rerank_batch),
                                desc="InternVL probe reranking"):
                q_end = min(q_start + rerank_batch, N_q)
                bsz   = q_end - q_start

                cand_idx = I_stage1[q_start:q_end, :self.rerank_k]  # (bsz, K)
                K        = self.rerank_k

                # Flatten to (bsz*K,) pairs
                images_flat = []
                texts_flat  = []
                for b in range(bsz):
                    q_text = q_texts[q_start + b]
                    for k in range(K):
                        db_idx = cand_idx[b, k]
                        img    = transform(
                            Image.open(db_img_paths[db_idx]).convert('RGB'))
                        images_flat.append(img)
                        texts_flat.append(q_text)

                images_t = torch.stack(images_flat).to(self.device)
                scores_flat = self.model(images_t, texts_flat)   # (bsz*K,)
                scores = scores_flat.view(bsz, K).cpu().numpy()

                for b in range(bsz):
                    order = np.argsort(-scores[b])
                    I_reranked[q_start + b] = cand_idx[b][order]

        r1_after = recall_at_k(I_reranked, 1)
        r5_after = recall_at_k(I_reranked, 5)

        print(f"\n{'':=<55}")
        print(f"  InternVL2 Probe — Stage-1 vs Reranked")
        print(f"{'':=<55}")
        print(f"  {'':5}  {'Stage-1':>10}  {'Reranked':>10}  {'Gain':>8}")
        print(f"  R@1    {r1_before:>10.4f}  {r1_after:>10.4f}  "
              f"{r1_after-r1_before:>+8.4f}")
        print(f"  R@5    {r5_before:>10.4f}  {r5_after:>10.4f}  "
              f"{r5_after-r5_before:>+8.4f}")
        print(f"{'':=<55}\n")

        self.log('val/R1_stage1',   r1_before, prog_bar=True,  logger=True)
        self.log('val/R1_reranked', r1_after,  prog_bar=True,  logger=True)
        self.log('val/R1_gain',
                 r1_after - r1_before, prog_bar=True, logger=True)

        # Empty GPU cache after reranking
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        """Safety hook ensuring backbone is cleaned up and InternVL is on GPU."""
        if hasattr(self, '_backbone') and self._backbone is not None:
            self._restore_internvl_after_val()

    # ------------------------------------------------------------------
    # Optimiser — only probe parameters
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # Train probe + projections + LoRA params (if enabled)
        probe_params = (list(self.model.probe.parameters()) +
                        list(self.model._vis_proj.parameters()) +
                        list(self.model._txt_proj.parameters()))
        if self.model.use_lora:
            lora_params  = [p for p in self.model.vlm.parameters()
                            if p.requires_grad]
            probe_params = probe_params + lora_params
            print(f"Training {sum(p.numel() for p in lora_params):,} LoRA params + "
                  f"{sum(p.numel() for p in probe_params)-sum(p.numel() for p in lora_params)} probe params")
        else:
            print(f"Training {sum(p.numel() for p in probe_params)} probe params only")

        opt   = torch.optim.AdamW(
            probe_params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        step = self.trainer.global_step
        if step < self.warmup_steps:
            scale = (step + 1) / self.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = scale * self.lr


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class InternVLRerankDataModule(pl.LightningDataModule):
    def __init__(
        self,
        triplets_json: str,
        train_csv:     str,
        image_root:    str,
        max_cands:     int   = 10,
        subset:        int   = 20000,
        batch_size:    int   = 8,
        num_workers:   int   = 4,
        image_size:    int   = 224,
        hard_only:     bool  = True,
    ):
        super().__init__()
        self.triplets_json = triplets_json
        self.train_csv     = train_csv
        self.image_root    = image_root
        self.max_cands     = max_cands
        self.subset        = subset
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.image_size    = image_size
        self.hard_only     = hard_only
        self.val_datamodule  = None
        self._val_dataset    = None
        self._val_dataloader = None
        self._val_set_name   = None

    def setup(self, stage=None):
        self.train_dataset = InternVLRerankDatasetWithDesc(
            triplets_json = self.triplets_json,
            train_csv     = self.train_csv,
            image_root    = self.image_root,
            max_cands     = self.max_cands,
            subset        = self.subset,
            image_size    = self.image_size,
            hard_only     = self.hard_only,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size       = self.batch_size,
            shuffle          = True,
            num_workers      = self.num_workers,
            pin_memory       = True,
            drop_last        = True,
            collate_fn       = collate_fn,
            persistent_workers = self.num_workers > 0,
            prefetch_factor    = 2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return self._val_dataloader

    @property
    def val_datasets(self):
        return [self._val_dataset] if self._val_dataset else []

    @property
    def val_set_names(self):
        return [self._val_set_name] if self._val_set_name else []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument

    # ---- InternVL2 ----
    a('--internvl_path',    type=str,
      default='OpenGVLab/InternVL2-2B',
      help='HuggingFace model ID or local path for InternVL2-2B')
    a('--pool_method',      type=str, default='cls',
      choices=['cls', 'mean'],
      help='How to pool InternVL2 hidden states')
    a('--lora_r',           type=int, default=0,
      help='LoRA rank — 0=frozen probe only, 16=light LoRA, 64=full LoRA')
    a('--prompt_template',  type=str,
      default='<image>\n{description}',
      help='Prompt template — {description} is replaced with query text')

    # ---- Backbone (for validation only) ----
    a('--backbone_ckpt',    type=str,
      default='checkpoints/dinov2-bge-ft-lora-ms/checkpoints/'
              'lavpr_epoch(19)_step(15640)_R1[0.5663]_R5[0.8237].ckpt',
      help='LaVPR_cross backbone ckpt — used for stage-1 val only')

    # ---- Data ----
    a('--triplets_json',    type=str, default='cache/gsv_triplets.json')
    a('--train_csv',        type=str,
      default='datasets/descriptions/gsv_cities_places.csv')
    a('--image_root',       type=str,
      default='/home/shared/datasets/gsv_cities/')
    a('--image_size',       type=int, default=224)
    a('--max_cands',        type=int, default=10,
      help='Candidates per query — keep small (InternVL2 is slow)')
    a('--subset',           type=int, default=20000,
      help='Number of training queries (0=all). Start small for speed.')
    a('--no_hard_only',     action='store_true', default=False,
      help='Use all queries including easy ones (default: hard_only=True)')

    # ---- Validation ----
    a('--val_image_root',   type=str,
      default='/home/shared/datasets/pitts30k/images/val')
    a('--val_csv',          type=str,
      default='datasets/descriptions/pitts30k_val_800_queries.csv')
    a('--val_set_names',    type=str, default=['pitts30k_val'], nargs='+')
    a('--rerank_k',         type=int, default=20)

    # ---- Loss ----
    a('--loss_name',        type=str, default='bce',
      choices=['bce', 'margin'])
    a('--margin',           type=float, default=0.3)

    # ---- Optimiser ----
    a('--lr',               type=float, default=1e-4,
      help='1e-3 for probe only, 1e-4 for LoRA')
    a('--weight_decay',     type=float, default=1e-4)
    a('--warmup_steps',     type=int,   default=100)
    a('--epochs',           type=int,   default=5)
    a('--batch_size',       type=int,   default=8,
      help='Small batch — InternVL2 forward is memory intensive')
    a('--grad_accum',       type=int,   default=4,
      help='Accumulate gradients — effective batch = batch_size × grad_accum')

    # ---- Hardware ----
    a('--gpu',              type=str,   default='0')
    a('--num_workers',      type=int,   default=4)
    a('--precision',        type=str,   default='bf16',
      choices=['32', '16', 'bf16'])

    # ---- Resume ----
    a('--resume',           type=str,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    pl.utilities.seed.seed_everything(seed=42)
    torch.set_float32_matmul_precision('high')

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nInternVL2 Probe Training")
    print(f"  internvl_path : {args.internvl_path}")
    print(f"  pool_method   : {args.pool_method}")
    print(f"  loss          : {args.loss_name}")
    print(f"  subset        : {args.subset:,} queries")
    print(f"  max_cands     : {args.max_cands}")
    print(f"  effective batch: {eff_batch} "
          f"({args.batch_size} × grad_accum={args.grad_accum})")

    # ------------------------------------------------------------------
    # DataModule
    # ------------------------------------------------------------------
    from dataloaders.GSVCitiesDataloader import IMAGENET_MEAN_STD
    from dataloaders.test_dataset import TestDataset

    hard_only = not args.no_hard_only   # True by default
    print(f"  hard_only     : {hard_only}")

    train_dm = InternVLRerankDataModule(
        triplets_json = args.triplets_json,
        train_csv     = args.train_csv,
        image_root    = args.image_root,
        max_cands     = args.max_cands,
        subset        = args.subset,
        batch_size    = args.batch_size,
        num_workers   = args.num_workers,
        image_size    = args.image_size,
        hard_only     = hard_only,
    )

    # Build val dataset directly — avoids GSVCitiesDataModule.setup()
    # which requires a train dataset base_path we don't have here
    val_image_root  = args.val_image_root
    database_folder = os.path.join(val_image_root, 'database')
    queries_folder  = os.path.join(val_image_root, 'queries')

    val_dataset = TestDataset(
        database_folder,
        queries_folder,
        args.val_csv,
        val_image_root,
        mean_std                = IMAGENET_MEAN_STD,
        positive_dist_threshold = 25,
        image_size              = args.image_size,
        use_labels              = True,
    )

    val_dl = DataLoader(
        val_dataset,
        batch_size  = 8,     # small — backbone runs during val_step
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # Attach directly to train_dm so Lightning can access them
    train_dm._val_dataset   = val_dataset
    train_dm._val_dataloader = val_dl
    train_dm._val_set_name  = args.val_set_names[0]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = InternVLProbeLightning.load_from_checkpoint(args.resume)
    else:
        model = InternVLProbeLightning(
            internvl_path   = args.internvl_path,
            backbone_ckpt   = args.backbone_ckpt,
            pool_method     = args.pool_method,
            prompt_template = args.prompt_template,
            lora_r          = args.lora_r,
            loss_name       = args.loss_name,
            margin          = args.margin,
            lr              = args.lr,
            weight_decay    = args.weight_decay,
            warmup_steps    = args.warmup_steps,
            epochs          = args.epochs,
            rerank_k        = args.rerank_k,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_name = (
        'internvl_probe_'
        'ep({epoch:02d})_'
        'R1rr[{val/R1_reranked:.4f}]_'
        'gain[{val/R1_gain:.4f}]'
    )
    checkpoint_cb = ModelCheckpoint(
        monitor                 = 'val/R1_reranked',
        filename                = ckpt_name,
        auto_insert_metric_name = False,
        save_weights_only       = True,
        save_top_k              = 3,
        mode                    = 'max',
        save_last               = True,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator             = 'gpu',
        devices                 = [0],
        default_root_dir        = './LOGS/internvl_probe',
        num_sanity_val_steps    = 0,   # -1 = run full val before training
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 1,
        callbacks               = [checkpoint_cb],
        log_every_n_steps       = 20,
        gradient_clip_val       = 1.0,
        gradient_clip_algorithm = 'norm',
        precision               = args.precision,
        accumulate_grad_batches = args.grad_accum,
    )

    trainer.fit(model=model, datamodule=train_dm)