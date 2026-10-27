"""
train_internvl_probe.py
=======================
Train InternVL2-2B as a cross-encoder reranker for VPR.

Architecture (LoRA mode):
    InternVL2-2B processes image + text JOINTLY in a single forward pass.
    The image tokens are inserted into the LLM context via InternVL2's
    native <image> token mechanism. LoRA adapts the LLM cross-attention
    layers so the model learns to ground text tokens in image regions.
    Score = Linear(last_hidden_state[last_real_token]) → scalar.

    This is the correct use of a VLM as a cross-encoder reranker,
    analogous to MonoBERT / RankLLaMA in text retrieval.

Architecture (probe-only mode, lora_r=0):
    InternVL2 frozen. Same joint forward, only score_head trained.

Why joint forward matters:
    Previous version ran vision and text through SEPARATE paths and
    concatenated representations. LoRA never saw image+text together,
    so it learned nothing useful for cross-modal alignment.
    Joint forward lets every LLM attention layer attend over both
    image patch tokens and text tokens simultaneously.

Example:
    python train_internvl_probe.py \\
        --internvl_path OpenGVLab/InternVL2-1B \\
        --lora_r        16  \\
        --loss_name     bce \\
        --max_cands     20  \\
        --subset        50000 \\
        --batch_size    8   \\
        --grad_accum    4   \\
        --lr            1e-4 \\
        --epochs        10  \\
        --gpu           1
"""

import os
import gc
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
    (query_text, candidate_image, label) pairs for cross-encoder training.

    Flattens triplets into individual pairs. Guarantees at least one
    positive per query in each sampled shortlist.
    """

    def __init__(
        self,
        triplets_json: str,
        train_csv:     str,
        image_root:    str,
        max_cands:     int  = 20,
        subset:        int  = 0,
        image_size:    int  = 224,
        hard_only:     bool = True,
    ):
        super().__init__()
        self.image_root = Path(image_root)
        self.max_cands  = max_cands

        df = pd.read_csv(train_csv)
        df = df[df['place_id'] >= 0].reset_index(drop=True)
        self.img_paths  = df['image_path'].tolist()
        self.place_ids_ = df['place_id'].tolist()
        self.descriptions = (df['description'].fillna('').tolist()
                             if 'description' in df.columns else
                             [''] * len(df))

        with open(triplets_json) as f:
            triplets = json.load(f)

        if hard_only:
            triplets = [t for t in triplets if t.get('top1_is_wrong', True)]
            print(f"Hard-only: {len(triplets):,} triplets")

        if 0 < subset < len(triplets):
            random.shuffle(triplets)
            triplets = triplets[:subset]
            print(f"Subset: {len(triplets):,} triplets")

        self.pairs = []
        for t in triplets:
            q_idx   = t['query_idx']
            pos_set = set(t['pos_idxs'])
            pos     = t['pos_idxs']
            negs    = t['neg_idxs']

            anchor  = [random.choice(pos)]
            extra   = [p for p in pos if p != anchor[0]]
            neg_s   = random.sample(negs, min(max_cands - 1, len(negs)))
            pool    = extra + neg_s
            random.shuffle(pool)
            cands   = anchor + pool[:max_cands - 1]

            for c_idx in cands:
                self.pairs.append({
                    'query_idx'  : q_idx,
                    'cand_idx'   : c_idx,
                    'label'      : 1.0 if c_idx in pos_set else 0.0,
                    'place_id'   : t['place_id'],
                    'cand_place' : self.place_ids_[c_idx],
                })

        print(f"Total pairs: {len(self.pairs):,} "
              f"({len(triplets):,} queries × ~{max_cands} cands)")

        # InternVL2 vision encoder uses its own normalisation stats,
        # NOT standard ImageNet. Using wrong stats degrades vision features.
        # These are the values from InternVL2's build_transform():
        #   mean = (0.485, 0.456, 0.406)  — OpenAI CLIP stats
        #   std  = (0.229, 0.224, 0.225)
        # InternVL2 actually uses IMAGENET_MEAN/STD from its own config.
        # Confirmed from: https://huggingface.co/OpenGVLab/InternVL2-1B
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std =(0.229, 0.224, 0.225)),
        ])

    def _load_image(self, idx):
        path = self.img_paths[idx]
        if not path.startswith('/'):
            path = str(self.image_root / path)
        return self.transform(Image.open(path).convert('RGB'))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair  = self.pairs[idx]
        q_idx = pair['query_idx']
        c_idx = pair['cand_idx']
        return {
            'image'      : self._load_image(c_idx),
            'text'       : self.descriptions[q_idx],
            'label'      : torch.tensor(pair['label'],      dtype=torch.float32),
            'query_idx'  : torch.tensor(pair['query_idx'],  dtype=torch.long),
            'place_id'   : torch.tensor(pair['place_id'],   dtype=torch.long),
            'cand_place' : torch.tensor(pair['cand_place'], dtype=torch.long),
        }


def collate_fn(batch):
    return {
        'image'      : torch.stack([b['image']      for b in batch]),
        'text'       : [b['text']                   for b in batch],
        'label'      : torch.stack([b['label']      for b in batch]),
        'query_idx'  : torch.stack([b['query_idx']  for b in batch]),
        'place_id'   : torch.stack([b['place_id']   for b in batch]),
        'cand_place' : torch.stack([b['cand_place'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# InternVL2 Cross-Encoder Reranker
# ---------------------------------------------------------------------------

class InternVLReranker(nn.Module):
    """
    InternVL2 as a cross-encoder reranker.

    Image and text are processed JOINTLY in a single LLM forward pass:
        1. InternVL2 vision encoder extracts patch tokens from the image.
        2. Patch tokens are inserted into the LLM input sequence at the
           <image> token position (InternVL2's native mechanism).
        3. The LLM attention layers process patch tokens + text tokens
           together — every layer attends over the full multimodal context.
        4. Score = score_head(last_real_token_hidden_state) → scalar.

    With LoRA (lora_r > 0):
        LoRA adapts q,k,v,o projections in the LLM attention layers.
        Because the forward is joint, LoRA gradients flow through
        cross-modal attention — the model learns to ground text tokens
        in image patch regions.

    With lora_r = 0:
        VLM fully frozen. Only score_head is trained (~D params).
        Useful as a zero-shot or fast baseline.
    """

    def __init__(
        self,
        internvl_path:   str  = 'OpenGVLab/InternVL2-1B',
        lora_r:          int  = 16,
        lora_alpha:      int  = 32,
        lora_dropout:    float = 0.05,
        lora_targets:    str  = 'q_proj,v_proj,k_proj,o_proj,qkv,proj',
    ):
        super().__init__()

        print(f"Loading InternVL2 from {internvl_path}...")
        from transformers import AutoTokenizer, AutoModel

        self.tokenizer = AutoTokenizer.from_pretrained(
            internvl_path, trust_remote_code=True)
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache image context token id — stored on the model, not config.
        # Try model attribute first, then tokenizer special tokens.
        # Must be cached before forward() is ever called.
        self.img_context_token_id = None  # set after vlm is loaded below

        self.vlm = AutoModel.from_pretrained(
            internvl_path,
            torch_dtype       = torch.bfloat16,
            trust_remote_code = True,
            low_cpu_mem_usage = True,
            use_flash_attn    = False,
        )

        # Resolve img_context_token_id — try several locations InternVL2
        # may store it depending on version:
        self.img_context_token_id = (
            getattr(self.vlm, 'img_context_token_id', None)
            or getattr(self.vlm.config, 'img_context_token_id', None)
            or self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        )
        if (self.img_context_token_id is None
                or self.img_context_token_id
                   == self.tokenizer.unk_token_id):
            raise RuntimeError(
                "Could not find img_context_token_id. "
                "Check that '<IMG_CONTEXT>' is in the tokenizer vocab.")
        print(f"img_context_token_id: {self.img_context_token_id}")

        # Cache BOS/EOS ids — resolved once, used every forward call
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = self.tokenizer.convert_tokens_to_ids('<s>')
            if bos == self.tokenizer.unk_token_id:
                bos = 1
        self.bos_token_id = bos

        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = self.tokenizer.convert_tokens_to_ids('</s>')
            if eos == self.tokenizer.unk_token_id:
                eos = 2
        self.eos_token_id = eos

        nl = self.tokenizer.encode('\n', add_special_tokens=False)
        self.nl_token_id  = nl[0] if nl else 13

        print(f"bos={self.bos_token_id}  "
              f"eos={self.eos_token_id}  "
              f"nl={self.nl_token_id}")

        # Detect LLM hidden dim BEFORE LoRA wrapping (safer)
        with torch.no_grad():
            dummy     = torch.zeros(1, 4, dtype=torch.long)
            self.D_llm = self.vlm.language_model.get_input_embeddings()(
                dummy).shape[-1]
        D_llm = self.D_llm
        print(f"D_llm={D_llm}")

        self.use_lora = lora_r > 0
        if self.use_lora:
            from peft import get_peft_model, LoraConfig
            if lora_targets == 'all-linear':
                # Find every nn.Linear in the model manually —
                # 'all-linear' string shortcut requires newer PEFT versions.
                targets = set()
                for name, module in self.vlm.named_modules():
                    if isinstance(module, nn.Linear):
                        # Use the leaf module name (last segment)
                        targets.add(name.split('.')[-1])
                targets = list(targets)
                print(f"LoRA all-linear: {len(targets)} unique layer names: "
                      f"{sorted(targets)}")
            else:
                targets = [t.strip() for t in lora_targets.split(',')]
            lora_cfg = LoraConfig(
                r              = lora_r,
                lora_alpha     = lora_alpha,
                target_modules = targets,
                lora_dropout   = lora_dropout,
                bias           = 'none',
            )
            self.vlm = get_peft_model(self.vlm, lora_cfg)
            self.vlm.print_trainable_parameters()
        else:
            for p in self.vlm.parameters():
                p.requires_grad = False
            self.vlm.eval()
            n = sum(p.numel() for p in self.vlm.parameters())
            print(f"VLM frozen: {n:,} params")

        # Score head: LayerNorm + Linear(D_llm → 1)
        # Zero-init linear so initial scores are uniform (stable training)
        self.score_head = nn.Sequential(
            nn.LayerNorm(D_llm),
            nn.Linear(D_llm, 1, bias=True),
        )
        nn.init.zeros_(self.score_head[1].weight)
        nn.init.zeros_(self.score_head[1].bias)
        # Note: zero init means all initial scores = 0, so argsort
        # preserves stage-1 order at epoch 0 (zero gain, not negative).
        # If gain is negative after training, check pos/neg score debug.
        n_head = sum(p.numel() for p in self.score_head.parameters())
        print(f"Score head params: {n_head:,}")

    def forward(self, pixel_values, texts):
        """
        Joint image+text forward through InternVL2 using
        prepare_inputs_embeds — the correct internal API.

        InternVL2's forward works by:
          1. Vision encoder extracts patch embeddings.
          2. patch embeddings are projected to LLM dim via mlp1.
          3. Projected patches replace <img> placeholder tokens in the
             LLM input embedding sequence.
          4. LLM runs over the fused sequence — every attention layer
             sees both patch tokens and text tokens.

        This is identical to what chat() does internally, but
        differentiable and without running generation.

        Parameters
        ----------
        pixel_values : (B, 3, H, W) float tensor on device
        texts        : List[str] length B

        Returns
        -------
        scores : (B,) float — relevance scores (logits, fp32)
        """
        device = pixel_values.device
        B      = pixel_values.shape[0]
        bf16   = torch.bfloat16

        # ── 1. Vision encoder ──────────────────────────────────────────
        # InternVL2's extract_feature() runs:
        #   ViT → pixel_shuffle (reduces tokens) → mlp1 (projects to D_llm)
        # It returns (B, n_patches, D_llm) ready to inject into the LLM.
        # Do NOT call mlp1 again — extract_feature already includes it.
        # extract_feature is defined on the base InternVLChatModel.
        # When LoRA-wrapped, call via get_base_model() to be safe.
        base_vlm   = (self.vlm.get_base_model()
                      if hasattr(self.vlm, 'get_base_model')
                      else self.vlm)
        img_embeds = base_vlm.extract_feature(
            pixel_values.to(bf16))                    # (B, n_patches, D_llm)
        n_patches  = img_embeds.shape[1]              # 64 for 224px input

        # ── 2. Build input_ids with img_context_token placeholder ────────
        # Tokenize text part only (no image placeholder string)
        img_token_id = self.img_context_token_id  # cached in __init__

        text_enc = self.tokenizer(
            texts,
            return_tensors     = 'pt',
            padding            = True,
            truncation         = True,
            max_length         = 256,
            add_special_tokens = True,
        ).to(device)                                   # (B, L_text)

        # Build img placeholder: [img_token_id] × n_patches for each item
        # Shape: (B, n_patches)
        img_ids = torch.full(
            (B, n_patches), img_token_id,
            dtype=torch.long, device=device)

        # Use cached token ids from __init__
        bos_id = self.bos_token_id
        eos_id = self.eos_token_id
        nl_id  = self.nl_token_id

        nl_ids  = torch.full((B, 1), nl_id,  dtype=torch.long, device=device)
        bos_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        eos_ids = torch.full((B, 1), eos_id, dtype=torch.long, device=device)

        text_ids  = text_enc['input_ids']
        text_mask = text_enc['attention_mask']

        # Remove leading BOS from tokenized text — we prepend our own
        if text_ids.shape[1] > 0 and (text_ids[:, 0] == bos_id).all():
            text_ids  = text_ids[:, 1:]
            text_mask = text_mask[:, 1:]

        ones1 = torch.ones((B, 1),        dtype=torch.long, device=device)
        onesN = torch.ones((B, n_patches), dtype=torch.long, device=device)

        input_ids      = torch.cat([bos_ids, img_ids, nl_ids,
                                    text_ids, eos_ids], dim=1)
        attention_mask = torch.cat([ones1, onesN, ones1,
                                    text_mask, ones1], dim=1)

        # ── 3. Build input embeddings, replace img positions with patches ─
        embed_fn     = base_vlm.language_model.get_input_embeddings()
        # Clone so in-place assignment does not corrupt autograd graph
        input_embeds = embed_fn(input_ids).clone()    # (B, L, D_llm)

        # Positions 1 … n_patches (after BOS) hold img_token_id
        # Replace them with projected visual patch embeddings
        img_positions = slice(1, 1 + n_patches)
        input_embeds[:, img_positions, :] = img_embeds.to(input_embeds.dtype)

        # ── 4. LLM forward over fused sequence ────────────────────────
        # InternVLChatModel.forward() does not accept inputs_embeds —
        # we must call the language_model directly.
        # LoRA wraps the language_model's attention layers internally,
        # so calling base_vlm.language_model() still activates LoRA
        # adapters because they are registered on the layer modules,
        # not on the top-level PeftModel forward.
        out = base_vlm.language_model(
            inputs_embeds        = input_embeds,
            attention_mask       = attention_mask,
            output_hidden_states = True,
            return_dict          = True,
        )

        if out.hidden_states is None:
            raise RuntimeError(
                "output_hidden_states=True but hidden_states is None.")

        last_hidden = out.hidden_states[-1].float()   # (B, L, D_llm) fp32

        # ── 5. Pool at last real token → score ─────────────────────────
        # Last real token = EOS or last non-padding text token.
        # For a causal LLM this position has attended over the full
        # sequence (image patches + text) so it is the correct summary
        # representation for scoring.
        seq_lens  = attention_mask.sum(dim=1) - 1     # (B,) 0-indexed
        seq_lens  = seq_lens.clamp(min=0)
        batch_idx = torch.arange(B, device=device)
        pooled    = last_hidden[batch_idx, seq_lens]  # (B, D_llm)

        return self.score_head(pooled).squeeze(-1)    # (B,)


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class InternVLRerankerLightning(pl.LightningModule):

    def __init__(
        self,
        # ---- Model ----
        internvl_path:   str   = 'OpenGVLab/InternVL2-1B',
        lora_r:          int   = 16,
        lora_alpha:      int   = 32,
        lora_dropout:    float = 0.05,
        lora_targets:    str   = 'q_proj,v_proj,k_proj,o_proj,qkv,proj',
        # ---- Backbone (val only) ----
        backbone_ckpt:   str   = None,
        # ---- Loss ----
        loss_name:       str   = 'bce',
        margin:          float = 0.3,
        # ---- Optimiser ----
        lr:              float = 1e-4,
        weight_decay:    float = 1e-4,
        warmup_steps:    int   = 100,
        epochs:          int   = 10,
        grad_accum:      int   = 4,
        # ---- Data ----
        triplets_json:   str   = '',
        train_csv:       str   = '',
        image_root:      str   = '',
        val_csv:         str   = '',
        val_image_root:  str   = '',
        val_set_name:    str   = 'pitts30k_val',
        max_cands:       int   = 20,
        subset:          int   = 50000,
        batch_size:      int   = 8,
        num_workers:     int   = 4,
        image_size:      int   = 224,
        hard_only:       bool  = True,
        # ---- Eval ----
        rerank_k:        int   = 20,
    ):
        super().__init__()
        self.save_hyperparameters()   # saves ALL params to hparams.yaml

        self.model = InternVLReranker(
            internvl_path = internvl_path,
            lora_r        = lora_r,
            lora_alpha    = lora_alpha,
            lora_dropout  = lora_dropout,
            lora_targets  = lora_targets,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _bce_loss(self, scores, labels):
        return F.binary_cross_entropy_with_logits(scores, labels)

    def _margin_loss(self, scores, labels, query_idx):
        total = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        valid = 0
        for qid in query_idx.unique():
            mask = query_idx.eq(qid)
            s, l = scores[mask], labels[mask]
            pos, neg = l > 0.5, l <= 0.5
            if not pos.any() or not neg.any():
                continue
            best_pos = s[pos].max()
            total   += F.relu(self.hparams.margin - best_pos + s[neg]).mean()
            valid   += 1
        return total / max(valid, 1)

    def _compute_loss(self, scores, labels, query_idx):
        if self.hparams.loss_name == 'bce':
            return self._bce_loss(scores, labels)
        elif self.hparams.loss_name == 'margin':
            return self._margin_loss(scores, labels, query_idx)
        raise ValueError(f"Unknown loss '{self.hparams.loss_name}'")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        scores = self.model(batch['image'].to(self.device), batch['text'])
        labels = batch['label'].to(self.device)
        loss   = self._compute_loss(
            scores, labels,
            batch['query_idx'].to(self.device),
        )
        self.log('train/loss', loss, prog_bar=True, logger=True)

        # Debug every 100 steps — confirm pos scores > neg scores
        if batch_idx % 100 == 0:
            pos_mask = labels > 0.5
            neg_mask = labels <= 0.5
            if pos_mask.any() and neg_mask.any():
                pos_mean = scores[pos_mask].mean().item()
                neg_mean = scores[neg_mask].mean().item()
                gap = pos_mean - neg_mean
                if gap > 0.01:
                    status = 'OK'
                elif gap < -0.01:
                    status = 'INVERTED — check labels!'
                else:
                    status = 'no signal yet'
                print(f"\n[DEBUG step {self.global_step}] "
                      f"pos_score={pos_mean:.4f}  "
                      f"neg_score={neg_mean:.4f}  "
                      f"gap={gap:.4f}  ({status})")
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self):
        """Offload InternVL2 to CPU, load LaVPR backbone on GPU."""
        from model.LaVPR_cross import LaVPR_cross

        print(f"GPU before val offload: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB")
        self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"GPU after offload: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB")

        print("Loading LaVPR backbone for stage-1 val...")
        self._backbone = LaVPR_cross.load_from_checkpoint(
            self.hparams.backbone_ckpt,
            map_location = self.device,
            strict       = True,
        )
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad = False

        # Verify on GPU
        cpu_p = [n for n, p in self._backbone.named_parameters()
                 if p.device.type != 'cuda']
        if cpu_p:
            print(f"WARNING: {len(cpu_p)} backbone params on CPU — forcing cuda")
            self._backbone = self._backbone.cuda()

        print(f"GPU after backbone load: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB")

    def _restore_internvl(self):
        """Delete backbone, restore InternVL2 to GPU."""
        if getattr(self, '_backbone', None) is not None:
            del self._backbone
            self._backbone = None
        gc.collect()
        torch.cuda.empty_cache()
        self.model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()
        print("InternVL2 restored to GPU.")

    def validation_step(self, batch, batch_idx):
        """Run backbone per batch — no raw image storage.
        TestDataset returns (images, labels, texts) tuple."""
        if isinstance(batch, (list, tuple)):
            places, _, texts = batch
        else:
            places = batch['image']
            texts  = batch['text']
        dev = next(self._backbone.parameters()).device
        with torch.no_grad():
            emb_i, _       = self._backbone.encode_image(places.to(dev))
            emb_t, _, _, _ = self._backbone.encode_text(texts)
        return {
            'img_emb'  : emb_i.detach().cpu(),
            'text_emb' : emb_t.detach().cpu(),
            'texts'    : list(texts),
        }

    def validation_epoch_end(self, outputs):
        import faiss

        if not self.hparams.backbone_ckpt:
            print("No backbone_ckpt — skipping val.")
            self._restore_internvl()
            return
        if not outputs:
            print("No val outputs.")
            self._restore_internvl()
            return

        all_texts  = [t for o in outputs for t in o['texts']]
        img_feats  = torch.cat([o['img_emb']  for o in outputs]).numpy()
        text_feats = torch.cat([o['text_emb'] for o in outputs]).numpy()
        print(f"Val samples: {len(all_texts)}")

        dm          = self.trainer.datamodule
        val_dataset = dm.val_datasets[0] if dm.val_datasets else None
        if val_dataset is None:
            print("No val dataset.")
            self._restore_internvl()
            return

        num_db = getattr(val_dataset, 'num_db',
                 getattr(val_dataset, 'num_database', None))
        if num_db is None:
            raise AttributeError("val_dataset missing num_db / num_database")

        r_np = img_feats[:num_db].astype('float32')
        q_np = text_feats[num_db:].astype('float32')
        r_np /= np.linalg.norm(r_np, axis=1, keepdims=True).clip(min=1e-8)
        q_np /= np.linalg.norm(q_np, axis=1, keepdims=True).clip(min=1e-8)

        index = faiss.IndexFlatIP(r_np.shape[1])
        index.add(r_np)
        K_search    = max(self.hparams.rerank_k, 20)
        _, I_stage1 = index.search(q_np, K_search)

        if hasattr(val_dataset, 'getPositives'):
            positives = val_dataset.getPositives()
        elif hasattr(val_dataset, 'get_positives'):
            positives = val_dataset.get_positives()
        else:
            raise AttributeError("val_dataset missing getPositives/get_positives")

        N_q = q_np.shape[0]

        def recall_at_k(I, k):
            return sum(1 for q in range(N_q)
                       if set(positives[q]) & set(I[q, :k].tolist())) / N_q

        r1_s1  = recall_at_k(I_stage1, 1)
        r5_s1  = recall_at_k(I_stage1, 5)
        r10_s1 = recall_at_k(I_stage1, 10)
        r20_s1 = recall_at_k(I_stage1, 20)

        # Restore InternVL2 for reranking
        self._restore_internvl()

        # ---- Reranking ----
        self.model.eval()
        K          = self.hparams.rerank_k
        I_reranked = I_stage1[:, :K].copy()

        db_img_paths = val_dataset.images_paths[:num_db]

        # q_texts: always use all_texts[num_db:] — collected from
        # validation_step in dataloader order (db first, queries after).
        # This is guaranteed correct regardless of val_dataset attributes.
        q_texts = all_texts[num_db:]
        print(f"[DEBUG val] len(all_texts)={len(all_texts)}  "
              f"num_db={num_db}  N_q={N_q}  len(q_texts)={len(q_texts)}")

        if len(q_texts) != N_q:
            print(f"ERROR: len(q_texts)={len(q_texts)} != N_q={N_q}. "
                  f"all_texts has wrong length — check val dataloader order.")
            self._log_results(r1_s1, r5_s1, r10_s1, r20_s1,
                              r1_s1, r5_s1, r10_s1, r20_s1)
            self._restore_internvl()
            return

        if len(q_texts) == 0:
            print("No query texts — skipping reranking.")
            self._log_results(r1_s1, r5_s1, r10_s1, r20_s1,
                              r1_s1, r5_s1, r10_s1, r20_s1)
            return

        transform = transforms.Compose([
            transforms.Resize((self.hparams.image_size,
                               self.hparams.image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std =(0.229, 0.224, 0.225)),
        ])

        # Debug — confirm query/image pairing is correct
        print(f"\n[DEBUG val] N_q={N_q}  num_db={num_db}  "
              f"len(q_texts)={len(q_texts)}")
        print(f"[DEBUG val] q_texts[0][:80]: {q_texts[0][:80]}")
        print(f"[DEBUG val] db_img_paths[I_stage1[0,0]]: "
              f"{db_img_paths[I_stage1[0, 0]]}")
        print(f"[DEBUG val] positives[0]: {list(positives[0])[:5]}")
        print(f"[DEBUG val] I_stage1[0,:5]: {I_stage1[0,:5].tolist()}")

        rerank_bs = 8

        with torch.no_grad():
            for q0 in tqdm(range(0, N_q, rerank_bs),
                           desc="InternVL2 reranking"):
                q1  = min(q0 + rerank_bs, N_q)
                bsz = q1 - q0
                cand_idx = I_stage1[q0:q1, :K]   # (bsz, K)

                imgs_flat, texts_flat = [], []
                for b in range(bsz):
                    qt = q_texts[q0 + b]
                    for k in range(K):
                        img = transform(
                            Image.open(
                                db_img_paths[cand_idx[b, k]]
                            ).convert('RGB'))
                        imgs_flat.append(img)
                        texts_flat.append(qt)

                pv     = torch.stack(imgs_flat).to(self.device)
                scores = self.model(pv, texts_flat).view(bsz, K).cpu().numpy()

                # Debug score stats on first batch only
                if q0 == 0:
                    print(f"\n[DEBUG val scores] mean={scores.mean():.4f} "
                          f"std={scores.std():.4f} "
                          f"min={scores.min():.4f} max={scores.max():.4f}")
                    # Check if stage-1 rank-1 candidate scores higher than others
                    # scores[b] is (K,) — index 0 is stage-1 top-1
                    for b in range(min(3, bsz)):
                        q_idx_abs  = q0 + b
                        pos_set    = set(positives[q_idx_abs])
                        cands_b    = cand_idx[b].tolist()
                        pos_scores = [scores[b,k] for k,c in enumerate(cands_b)
                                      if c in pos_set]
                        neg_scores = [scores[b,k] for k,c in enumerate(cands_b)
                                      if c not in pos_set]
                        if pos_scores and neg_scores:
                            print(f"  q{b}: pos_score={np.mean(pos_scores):.4f}  "
                                  f"neg_score={np.mean(neg_scores):.4f}  "
                                  f"gap={np.mean(pos_scores)-np.mean(neg_scores):.4f}"
                                  f"  {'OK' if np.mean(pos_scores) > np.mean(neg_scores) else 'INVERTED'}")

                for b in range(bsz):
                    order              = np.argsort(-scores[b])
                    I_reranked[q0 + b] = cand_idx[b][order]

        r1_rr  = recall_at_k(I_reranked, 1)
        r5_rr  = recall_at_k(I_reranked, 5)
        r10_rr = recall_at_k(I_reranked, 10)
        r20_rr = recall_at_k(I_reranked, 20)

        self._log_results(r1_s1, r5_s1, r10_s1, r20_s1,
                          r1_rr,  r5_rr,  r10_rr,  r20_rr)
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if getattr(self, '_backbone', None) is not None:
            self._restore_internvl()

    def _log_results(self, r1_s1, r5_s1, r10_s1, r20_s1,
                           r1_rr,  r5_rr,  r10_rr,  r20_rr):
        sn = self.hparams.val_set_name
        print(f"\n{'':=<60}")
        print(f"  InternVL2 Cross-Encoder [{sn}]")
        print(f"{'':=<60}")
        print(f"  {'':5}  {'Stage-1':>10}  {'Reranked':>10}  {'Gain':>8}")
        for tag, b, a in [('R@1',  r1_s1,  r1_rr),
                           ('R@5',  r5_s1,  r5_rr),
                           ('R@10', r10_s1, r10_rr),
                           ('R@20', r20_s1, r20_rr)]:
            print(f"  {tag:<5}  {b:>10.4f}  {a:>10.4f}  {a-b:>+8.4f}")
        print(f"{'':=<60}\n")

        kw = dict(prog_bar=True, logger=True)
        nk = dict(prog_bar=False, logger=True)
        self.log(f'{sn}/R1_stage1',   r1_s1,  **kw)
        self.log(f'{sn}/R5_stage1',   r5_s1,  **nk)
        self.log(f'{sn}/R10_stage1',  r10_s1, **nk)
        self.log(f'{sn}/R20_stage1',  r20_s1, **nk)
        self.log(f'{sn}/R1_reranked', r1_rr,  **kw)
        self.log(f'{sn}/R5_reranked', r5_rr,  **nk)
        self.log(f'{sn}/R10_reranked',r10_rr, **nk)
        self.log(f'{sn}/R20_reranked',r20_rr, **nk)
        self.log(f'{sn}/R1_gain',     r1_rr - r1_s1, **kw)
        # Alias without '/' for ModelCheckpoint filename template.
        # on_epoch=True and sync_dist=True ensure PL can read these
        # when it formats the checkpoint filename after validation.
        ckpt_kw = dict(prog_bar=False, logger=True,
                       on_epoch=True, sync_dist=False)
        self.log('val_R1_reranked', r1_rr,         **ckpt_kw)
        self.log('val_R1_gain',     r1_rr - r1_s1, **ckpt_kw)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        # score_head always trainable; LoRA params trainable when lora_r>0
        params = [p for p in self.model.parameters() if p.requires_grad]
        # Sanity check — every score_head param must be in trainable params
        assert all(
            any(p is q for q in params)
            for p in self.model.score_head.parameters()
        ), "score_head params not in trainable params — check requires_grad"
        n_train = sum(p.numel() for p in params)
        print(f"Trainable params: {n_train:,}")

        opt   = torch.optim.AdamW(
            params,
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.hparams.epochs)
        return [opt], [{'scheduler': sched, 'interval': 'epoch'}]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        step = self.trainer.global_step
        ws   = self.hparams.warmup_steps
        if step < ws:
            scale = (step + 1) / ws
            for pg in optimizer.param_groups:
                pg['lr'] = scale * self.hparams.lr


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class InternVLRerankDataModule(pl.LightningDataModule):
    def __init__(
        self,
        triplets_json: str,
        train_csv:     str,
        image_root:    str,
        max_cands:     int  = 20,
        subset:        int  = 50000,
        batch_size:    int  = 8,
        num_workers:   int  = 4,
        image_size:    int  = 224,
        hard_only:     bool = True,
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
        self._val_dataset    = None
        self._val_dataloader = None
        self._val_set_name   = None

    def setup(self, stage=None):
        self.train_dataset = InternVLRerankDataset(
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
            batch_size         = self.batch_size,
            shuffle            = True,
            num_workers        = self.num_workers,
            pin_memory         = True,
            drop_last          = True,
            collate_fn         = collate_fn,
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

    # Model
    a('--internvl_path',  type=str,   default='OpenGVLab/InternVL2-1B')
    a('--lora_r',         type=int,   default=16,
      help='LoRA rank. 0 = frozen VLM, probe-only.')
    a('--lora_targets',   type=str,
      default='q_proj,v_proj,k_proj,o_proj,qkv,proj',
      help='"all-linear" targets every nn.Linear in the full model '
           '(vision encoder, mlp1, LLM attention + FFN). '
           'Pass a comma-separated list to restrict to specific layers.')

    # Backbone
    a('--backbone_ckpt',  type=str,
      default='checkpoints/dinov2-bge-ft-lora-ms/checkpoints/'
              'lavpr_epoch(19)_step(15640)_R1[0.5663]_R5[0.8237].ckpt')

    # Data
    a('--triplets_json',  type=str,   default='cache/gsv_triplets.json')
    a('--train_csv',      type=str,
      default='datasets/descriptions/gsv_cities_places.csv')
    a('--image_root',     type=str,
      default='/home/shared/datasets/gsv_cities/')
    a('--image_size',     type=int,   default=224)
    a('--max_cands',      type=int,   default=20)
    a('--subset',         type=int,   default=50000)
    a('--no_hard_only',   action='store_true', default=False)

    # Validation
    a('--val_image_root', type=str,
      default='/home/shared/datasets/pitts30k/images/val')
    a('--val_csv',        type=str,
      default='datasets/descriptions/pitts30k_val_800_queries.csv')
    a('--val_set_names',  type=str,   default=['pitts30k_val'], nargs='+')
    a('--rerank_k',       type=int,   default=20)

    # Loss
    a('--loss_name',      type=str,   default='bce',
      choices=['bce', 'margin'])
    a('--margin',         type=float, default=0.3)

    # Optimiser
    a('--lr',             type=float, default=1e-4)
    a('--weight_decay',   type=float, default=1e-4)
    a('--warmup_steps',   type=int,   default=100)
    a('--epochs',         type=int,   default=10)
    a('--batch_size',     type=int,   default=8)
    a('--grad_accum',     type=int,   default=4)

    # Hardware
    a('--gpu',            type=str,   default='0')
    a('--num_workers',    type=int,   default=4)
    a('--precision',      type=str,   default='bf16',
      choices=['32', '16', 'bf16'])

    # Resume
    a('--resume',         type=str,   default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    pl.utilities.seed.seed_everything(seed=42)
    torch.set_float32_matmul_precision('high')

    args      = parse_args()
    hard_only = not args.no_hard_only
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print(f"\nInternVL2 Cross-Encoder Reranker Training")
    print(f"  model       : {args.internvl_path}")
    print(f"  lora_r      : {args.lora_r}")
    print(f"  loss        : {args.loss_name}")
    print(f"  max_cands   : {args.max_cands}")
    print(f"  subset      : {args.subset:,}")
    print(f"  eff. batch  : {args.batch_size * args.grad_accum}")

    # DataModule
    from dataloaders.GSVCitiesDataloader import IMAGENET_MEAN_STD
    from dataloaders.test_dataset import TestDataset

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

    val_image_root  = args.val_image_root
    val_dataset = TestDataset(
        os.path.join(val_image_root, 'database'),
        os.path.join(val_image_root, 'queries'),
        args.val_csv,
        val_image_root,
        mean_std                = IMAGENET_MEAN_STD,
        positive_dist_threshold = 25,
        image_size              = args.image_size,
        use_labels              = True,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size  = 8,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    train_dm._val_dataset    = val_dataset
    train_dm._val_dataloader = val_dl
    train_dm._val_set_name   = args.val_set_names[0]

    # Model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = InternVLRerankerLightning.load_from_checkpoint(args.resume)
    else:
        model = InternVLRerankerLightning(
            internvl_path  = args.internvl_path,
            lora_r         = args.lora_r,
            lora_alpha     = args.lora_r * 2,
            lora_dropout   = 0.05,
            lora_targets   = args.lora_targets,
            backbone_ckpt  = args.backbone_ckpt,
            loss_name      = args.loss_name,
            margin         = args.margin,
            lr             = args.lr,
            weight_decay   = args.weight_decay,
            warmup_steps   = args.warmup_steps,
            epochs         = args.epochs,
            grad_accum     = args.grad_accum,
            triplets_json  = args.triplets_json,
            train_csv      = args.train_csv,
            image_root     = args.image_root,
            val_csv        = args.val_csv,
            val_image_root = args.val_image_root,
            val_set_name   = args.val_set_names[0],
            max_cands      = args.max_cands,
            subset         = args.subset,
            batch_size     = args.batch_size,
            num_workers    = args.num_workers,
            image_size     = args.image_size,
            hard_only      = hard_only,
            rerank_k       = args.rerank_k,
        )

    # Checkpoint
    val_set_name = args.val_set_names[0]
    # PL ModelCheckpoint uses {metric_name:.4f} syntax.
    # Metric names with '/' must be escaped as '{val_set_name/metric}'
    # but '/' in filenames causes path issues on Linux.
    # Workaround: use a fixed filename template with epoch only,
    # and rely on monitor + save_top_k=3 to keep the best checkpoints.
    checkpoint_cb = ModelCheckpoint(
        monitor                 = f'{val_set_name}/R1_reranked',
        filename                = 'internvl_reranker_ep{epoch:02d}_R1rr{val_R1_reranked:.4f}_gain{val_R1_gain:.4f}',
        auto_insert_metric_name = False,
        save_weights_only       = False,  # must be False for LoRA — True
                                          # skips optimizer state but also
                                          # risks missing PEFT adapter weights
                                          # in some PL+PEFT version combos.
        save_top_k              = 3,
        mode                    = 'max',
        save_last               = True,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator             = 'gpu',
        devices                 = [0],
        default_root_dir        = './LOGS/internvl_reranker',
        num_sanity_val_steps    = 0,   # set to -1 to run full val before train
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