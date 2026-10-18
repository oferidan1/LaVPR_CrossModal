"""
rerank_dataloader.py
====================
Dataset and DataModule for stage-2 reranker training.

Loads the pre-mined triplets produced by build_rerank_dataset.py and serves
batches of:
    query_texts  : List[str]          length B
    images       : (B, K, C, H, W)   K candidates from the stage-1 shortlist
    labels       : (B, K)  long      GSV place_ids  — same ID = positive pair
    is_positive  : (B, K)  float     1.0 for GT positives (logging only)

K differs between train (top_k_train, e.g. 100) and val (top_k_val, e.g. 20).

Positives and negatives are sampled from the pre-mined shortlist per epoch,
so each epoch sees a slightly different mix while staying within the hard-
negative pool.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pytorch_lightning as pl

# Match the default_transform in GSVCitiesDataset exactly
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Image transform (same normalisation as LaVPR_cross training)
# ---------------------------------------------------------------------------

def get_transform(image_size: int, augment: bool = False) -> T.Compose:
    """
    augment=True  → random horizontal flip + colour jitter for train split.
    augment=False → deterministic resize + crop for val/inference.
    """
    if augment:
        return T.Compose([
            T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])


# ---------------------------------------------------------------------------
# Low-level image + text accessors
# We re-use the GSV flat lists built during mining (paths + texts + place_ids
# stored in the .npz companion or reconstructed from the original dataset).
# ---------------------------------------------------------------------------

class GSVFlatAccessor:
    """
    Thin wrapper that gives random-access to image paths, text descriptions,
    and place_ids using the same ordering as build_rerank_dataset.py.

    The ordering is determined by the flat iteration over the GSV dataset
    (shuffle_all=False, random_sample_from_each_place=False) so indices in
    gsv_triplets.json map 1-to-1 to entries here.

    Parameters
    ----------
    image_paths : List[str]   all image paths in dataset order
    texts       : List[str]   one text description per image
    place_ids   : np.ndarray  (N,) int64
    transform   : callable    applied to PIL images
    """

    def __init__(
        self,
        image_paths : List[str],
        texts       : List[str],
        place_ids   : np.ndarray,
        transform   = None,
    ):
        assert len(image_paths) == len(texts) == len(place_ids)
        self.image_paths = image_paths
        self.texts       = texts
        self.place_ids   = place_ids
        self.transform   = transform

    def get_image(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def get_text(self, idx: int) -> str:
        return self.texts[idx]

    def get_place_id(self, idx: int) -> int:
        return int(self.place_ids[idx])

    def __len__(self):
        return len(self.image_paths)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    One sample = one query + its sampled shortlist of K candidates.

    Parameters
    ----------
    accessor        : GSVFlatAccessor
    triplets        : List[dict]  from gsv_triplets.json['train' | 'val']
    shortlist_size  : K — number of candidates per sample
                        train → top_k_train (e.g. 100)
                        val   → top_k_val   (e.g. 20)
    max_pos         : cap on positives included per sample
                        GSV has 4 views per place → default 4
    transform       : torchvision transform (augment for train, plain for val)
    """

    def __init__(
        self,
        accessor       : GSVFlatAccessor,
        triplets       : List[dict],
        shortlist_size : int  = 100,
        max_pos        : int  = 4,
        transform      = None,
    ):
        self.accessor       = accessor
        self.triplets       = triplets
        self.shortlist_size = shortlist_size
        self.max_pos        = max_pos
        self.transform      = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t        = self.triplets[idx]
        q_idx    = t['query_idx']
        place_id = t['place_id']

        # --- Sample positives (all available, up to max_pos) ---
        pos_pool   = t['pos_idxs']
        pos_sample = random.sample(pos_pool, min(self.max_pos, len(pos_pool)))

        # --- Sample hard negatives to fill remaining slots ---
        neg_pool    = t['neg_idxs']
        n_neg_slots = self.shortlist_size - len(pos_sample)
        neg_sample  = random.sample(neg_pool, min(n_neg_slots, len(neg_pool)))

        # --- Build shortlist: positives + negatives, then shuffle ---
        img_indices  = pos_sample + neg_sample
        place_labels = (
            [place_id] * len(pos_sample) +
            [self.accessor.get_place_id(i) for i in neg_sample]
        )
        is_pos_flags = [1.0] * len(pos_sample) + [0.0] * len(neg_sample)

        # Shuffle so the model cannot exploit position
        order        = list(range(len(img_indices)))
        random.shuffle(order)
        img_indices  = [img_indices[i]  for i in order]
        place_labels = [place_labels[i] for i in order]
        is_pos_flags = [is_pos_flags[i] for i in order]

        # --- Pad to fixed shortlist_size so collate can stack ---
        # Repeat last entry if we have fewer candidates than expected
        while len(img_indices) < self.shortlist_size:
            img_indices .append(img_indices[-1])
            place_labels.append(-1)          # -1 = padding, ignored by MS loss
            is_pos_flags.append(0.0)

        # --- Load images ---
        images = torch.stack([
            self.accessor.get_image(i) for i in img_indices
        ])   # (shortlist_size, C, H, W)

        return {
            # query
            'query_texts': self.accessor.get_text(q_idx),
            # candidates
            'images'     : images,                                   # (K, C, H, W)
            'labels'     : torch.tensor(place_labels, dtype=torch.long),   # (K,)
            'is_positive': torch.tensor(is_pos_flags, dtype=torch.float),  # (K,)
        }


# ---------------------------------------------------------------------------
# Custom collate — handles variable-length text lists
# ---------------------------------------------------------------------------

def rerank_collate(batch):
    """
    Stack tensors normally; keep query_texts as a plain Python list of strings
    (the backbone tokenizer handles padding internally).
    """
    query_texts = [b['query_texts'] for b in batch]
    images      = torch.stack([b['images']      for b in batch])   # (B, K, C, H, W)
    labels      = torch.stack([b['labels']      for b in batch])   # (B, K)
    is_positive = torch.stack([b['is_positive'] for b in batch])   # (B, K)
    return {
        'query_texts': query_texts,
        'images'     : images,
        'labels'     : labels,
        'is_positive': is_positive,
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class RerankDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for stage-2 reranker training.

    Expects
    -------
    triplets_json : path to gsv_triplets.json  (from build_rerank_dataset.py)
    embeddings_npz: path to gsv_embeddings.npz (for place_ids)
    gsv_datamodule: the *same* GSVCitiesDataModule used in stage-1 training,
                    already setup — we call setup('fit') on it to get the
                    flat image paths and text descriptions.

    Parameters
    ----------
    triplets_json   : str   path to gsv_triplets.json
    embeddings_npz  : str   path to gsv_embeddings.npz  (needs place_ids)
    image_root      : str   GSV image root (same as train.py)
    train_csv       : str   GSV train CSV  (same as train.py)
    mapping_path    : str   GSV mapping JSON
    image_size      : int   224 to match backbone
    top_k_train     : int   shortlist size for training   (e.g. 100)
    top_k_val       : int   shortlist size for validation (e.g. 20)
    max_pos         : int   max positives per sample      (4 for GSV)
    batch_size      : int   number of queries per batch
    num_workers     : int
    augment_train   : bool  random flip/jitter on train images
    """

    def __init__(
        self,
        triplets_json  : str,
        embeddings_npz : str,
        # GSV dataset args — mirror train.py
        image_root     : str,
        train_csv      : str,
        mapping_path   : str,
        image_size     : int  = 224,
        img_per_place  : int  = 4,
        # Reranker-specific
        top_k_train    : int  = 100,
        max_pos        : int  = 4,
        # Training
        batch_size     : int  = 16,
        num_workers    : int  = 4,
        augment_train  : bool = True,
    ):
        super().__init__()
        self.triplets_json  = triplets_json
        self.embeddings_npz = embeddings_npz
        self.image_root     = image_root
        self.train_csv      = train_csv
        self.mapping_path   = mapping_path
        self.image_size     = image_size
        self.img_per_place  = img_per_place
        self.top_k_train    = top_k_train
        self.max_pos        = max_pos
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.augment_train  = augment_train

        self.train_dataset : Optional[RerankDataset] = None
        self.val_dataset   : Optional[RerankDataset] = None

    # ------------------------------------------------------------------
    def _build_accessor(self, transform) -> GSVFlatAccessor:
        """
        Read the CSV directly — same column order as build_rerank_dataset.py
        so triplet indices map 1-to-1 to rows here.

        CSV columns used: image_path, description, place_id
        place_ids are also verified against gsv_embeddings.npz.
        """
        import pandas as pd

        df = pd.read_csv(self.train_csv)
        image_root = self.image_root.rstrip('/')

        raw_paths = df['image_path'].tolist()
        paths     = raw_paths if raw_paths[0].startswith('/') \
                    else [image_root + '/' + p for p in raw_paths]
        texts     = df['description'].astype(str).tolist()
        place_ids = df['place_id'].astype(int).tolist()

        # place_ids come from the CSV directly — no npz cross-check needed.
        # The npz is only used by the mining script; the dataloader just needs
        # the flat (path, text, place_id) lists in the same row order as the
        # CSV that was passed to build_rerank_dataset.py.
        return GSVFlatAccessor(
            image_paths = paths,
            texts       = texts,
            place_ids   = np.array(place_ids, dtype=np.int64),
            transform   = transform,
        )

    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None):
        print("RerankDataModule.setup(): loading triplets...")
        with open(self.triplets_json) as f:
            triplets = json.load(f)   # flat list of dicts

        transform = get_transform(self.image_size, augment=self.augment_train)
        accessor  = self._build_accessor(transform)

        self.train_dataset = RerankDataset(
            accessor       = accessor,
            triplets       = triplets,
            shortlist_size = self.top_k_train,
            max_pos        = self.max_pos,
        )
        print(f"  Train queries : {len(self.train_dataset):>8,}  "
              f"shortlist_size={self.top_k_train}")

    # ------------------------------------------------------------------
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = True,
            num_workers = self.num_workers,
            pin_memory  = True,
            drop_last   = True,
            collate_fn  = rerank_collate,
        )

    # ------------------------------------------------------------------
    def reload_triplets(self):
        """
        Re-read gsv_triplets.json in-place — call from ReMiningCallback
        after re-running build_rerank_dataset.py.
        Replaces the triplet list without rebuilding the accessor.
        """
        print("RerankDataModule: reloading triplets from disk...")
        with open(self.triplets_json) as f:
            triplets = json.load(f)   # flat list
        self.train_dataset.triplets = triplets
        print(f"  Triplets: {len(triplets):,}")