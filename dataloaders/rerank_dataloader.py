"""
rerank_dataloader.py
====================
Dataset and DataModule for stage-2 reranker training.

Reads precomputed local tokens from a single HDF5 file — O(1) random access,
no per-file Python overhead, no filesystem metadata lookups.

HDF5 layout (cache_dir/local_tokens.h5):
    /img/{idx}   : (N_patches, D_img)  float16
    /text/{idx}  : (L, D_txt)          float16
    /mask/{idx}  : (L,)                uint8
"""

import os
import json
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# ---------------------------------------------------------------------------
# Place-ID accessor
# ---------------------------------------------------------------------------

class PlaceIDAccessor:
    """Just place_ids — no images, no transforms."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        invalid = df['place_id'].astype(int) < 0
        if invalid.any():
            df = df[~invalid].reset_index(drop=True)
        self.place_ids = df['place_id'].astype(int).tolist()

    def get_place_id(self, idx: int) -> int:
        return self.place_ids[idx]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    Reads precomputed local tokens from a single HDF5 file.
    One HDF5 handle is opened lazily per DataLoader worker — safe for multiprocessing.
    No backbone, no image loading, no transforms at training time.
    """

    def __init__(
        self,
        accessor       : PlaceIDAccessor,
        triplets       : List[dict],
        h5_path        : str,           # path to local_tokens.h5 OR token directory
        shortlist_size : int = 100,
        max_cands      : int = 50,
        max_pos        : int = 4,
    ):
        self.accessor       = accessor
        self.triplets       = triplets
        self.shortlist_size = shortlist_size
        self.max_cands      = max_cands
        self.max_pos        = max_pos
        self._h5            = None

        # Auto-detect format: HDF5 file or directory of .pt files
        if os.path.isfile(h5_path) and h5_path.endswith('.h5'):
            self.h5_path  = h5_path
            self.token_dir = None
            print(f"  RerankDataset: using HDF5  {h5_path}")
        else:
            # Treat as directory of .pt files — use path as-is
            self.h5_path   = None
            self.token_dir = h5_path
            print(f"  RerankDataset: using .pt files  {self.token_dir}")

    def __len__(self):
        return len(self.triplets)

    def _h5file(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5

    def _load_img(self, idx: int) -> torch.Tensor:
        if self.h5_path:
            return torch.from_numpy(
                self._h5file()['img'][str(idx)][:]).float()
        return torch.load(
            os.path.join(self.token_dir, f'img_{idx}.pt'),
            map_location='cpu', weights_only=True).float()

    def _load_text(self, idx: int):
        if self.h5_path:
            f          = self._h5file()
            text_local = torch.from_numpy(f['text'][str(idx)][:]).float()
            mask       = torch.from_numpy(f['mask'][str(idx)][:].astype('bool'))
        else:
            text_local = torch.load(
                os.path.join(self.token_dir, f'text_{idx}.pt'),
                map_location='cpu', weights_only=True).float()
            mask       = torch.load(
                os.path.join(self.token_dir, f'mask_{idx}.pt'),
                map_location='cpu', weights_only=True).bool()
        return text_local, mask

    def __getitem__(self, idx):
        t        = self.triplets[idx]
        q_idx    = t['query_idx']
        place_id = t['place_id']

        # Sample positives
        pos_sample = random.sample(
            t['pos_idxs'], min(self.max_pos, len(t['pos_idxs'])))

        # Sample hard negatives — capped at max_cands to control memory
        n_neg_slots = self.max_cands - len(pos_sample)
        neg_sample  = random.sample(
            t['neg_idxs'], min(n_neg_slots, len(t['neg_idxs'])))

        # Build shortlist
        img_indices  = pos_sample + neg_sample
        place_labels = (
            [place_id] * len(pos_sample) +
            [self.accessor.get_place_id(i) for i in neg_sample]
        )
        is_pos_flags = [1.0] * len(pos_sample) + [0.0] * len(neg_sample)

        # Shuffle
        order        = list(range(len(img_indices)))
        random.shuffle(order)
        img_indices  = [img_indices[i]  for i in order]
        place_labels = [place_labels[i] for i in order]
        is_pos_flags = [is_pos_flags[i] for i in order]

        # Pad to fixed size
        while len(img_indices) < self.max_cands:
            img_indices .append(img_indices[-1])
            place_labels.append(-1)
            is_pos_flags.append(0.0)

        # Load query text tokens
        q_text_local, q_mask = self._load_text(q_idx)

        # Load candidate image tokens
        cand_img_locals = torch.stack([
            self._load_img(i) for i in img_indices
        ])   # (K, N, D_img)

        return {
            'query_text_local' : q_text_local,
            'query_mask'       : q_mask,
            'cand_img_locals'  : cand_img_locals,
            'labels'           : torch.tensor(place_labels, dtype=torch.long),
            'is_positive'      : torch.tensor(is_pos_flags, dtype=torch.float),
            'query_place_id'   : torch.tensor(place_id, dtype=torch.long),  # explicit query place ID
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def rerank_collate(batch):
    # Text length varies per sample — pad to longest in batch
    max_L  = max(b['query_text_local'].shape[0] for b in batch)
    D_txt  = batch[0]['query_text_local'].shape[1]

    text_padded = torch.zeros(len(batch), max_L, D_txt)
    mask_padded = torch.zeros(len(batch), max_L, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b['query_text_local'].shape[0]
        text_padded[i, :L] = b['query_text_local']
        mask_padded[i, :L] = b['query_mask']

    return {
        'query_text_local' : text_padded,                                           # (B, L, D_txt)
        'query_mask'       : mask_padded,                                           # (B, L)
        'cand_img_locals'  : torch.stack([b['cand_img_locals']  for b in batch]),  # (B, K, N, D_img)
        'labels'           : torch.stack([b['labels']           for b in batch]),  # (B, K)
        'is_positive'      : torch.stack([b['is_positive']      for b in batch]),  # (B, K)
        'query_place_id'   : torch.stack([b['query_place_id']   for b in batch]),  # (B,)
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class RerankDataModule(pl.LightningDataModule):

    def __init__(
        self,
        triplets_json  : str,
        token_path     : str,   # either local_tokens.h5 or local_tokens/ directory
        train_csv      : str,
        top_k_train    : int  = 100,   # mined shortlist size
        max_cands      : int  = 50,    # candidates loaded per step
        max_pos        : int  = 4,
        batch_size     : int  = 32,
        num_workers    : int  = 4,
    ):
        super().__init__()
        self.triplets_json = triplets_json
        self.token_path    = token_path
        self.train_csv     = train_csv
        self.top_k_train   = top_k_train
        self.max_cands     = max_cands
        self.max_pos       = max_pos
        self.batch_size    = batch_size
        self.num_workers   = num_workers

        self.train_dataset  : Optional[RerankDataset] = None
        self.val_datamodule = None   # attached externally in train_reranker.py

    def setup(self, stage=None):
        print("RerankDataModule.setup(): loading triplets...")
        with open(self.triplets_json) as f:
            triplets = json.load(f)

        accessor = PlaceIDAccessor(self.train_csv)

        self.train_dataset = RerankDataset(
            accessor       = accessor,
            triplets       = triplets,
            h5_path        = self.token_path,
            shortlist_size = self.top_k_train,
            max_cands      = self.max_cands,
            max_pos        = self.max_pos,
        )
        print(f"  Train queries : {len(self.train_dataset):>8,}  "
              f"shortlist_size={self.top_k_train}  "
              f"tokens={self.token_path}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = False,
            drop_last   = True,
            collate_fn  = rerank_collate,
        )

    def _ensure_val_setup(self):
        if self.val_datamodule is not None and (
                not hasattr(self.val_datamodule, 'val_datasets') or
                not self.val_datamodule.val_datasets):
            self.val_datamodule.setup('fit')

    @property
    def val_datasets(self):
        self._ensure_val_setup()
        return self.val_datamodule.val_datasets if self.val_datamodule else []

    @property
    def val_set_names(self):
        self._ensure_val_setup()
        return self.val_datamodule.val_set_names if self.val_datamodule else []

    def val_dataloader(self):
        if self.val_datamodule is None:
            return None
        self._ensure_val_setup()
        return self.val_datamodule.val_dataloader()

    def reload_triplets(self):
        print("RerankDataModule: reloading triplets...")
        with open(self.triplets_json) as f:
            triplets = json.load(f)
        self.train_dataset.triplets = triplets
        print(f"  Triplets: {len(triplets):,}")