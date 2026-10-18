"""
build_rerank_dataset.py
=======================
Offline script that builds the reranker training dataset from a frozen
LaVPR_cross checkpoint and your GSV CSV.

CSV format expected
-------------------
    image_path, description, flip, change_color, place_id

What it does
------------
1. Reads the CSV directly — no GSVCitiesDataset needed.
2. Embeds everything with the frozen backbone:
       text (description col) → global text embedding  (N, embeds_dim)
       image (image_path col) → global image embedding (N, embeds_dim)
3. FAISS text→image search: for each query retrieve top-K images.
4. Labels each candidate: positive (same place_id) or hard negative.
5. Saves two files:
       cache/gsv_embeddings.npz  — embeddings + place_ids (reusable)
       cache/gsv_triplets.json   — per-query shortlists with labels

Triplet format (one entry per query)
-------------------------------------
{
  "query_idx" : int,        # row index in the CSV
  "place_id"  : int,
  "shortlist" : [int, ...], # top-K image indices (stage-1 order)
  "pos_idxs"  : [int, ...], # shortlist entries with same place_id
  "neg_idxs"  : [int, ...]  # shortlist entries with different place_id
}

Usage
-----
python build_rerank_dataset.py \
    --backbone_ckpt  checkpoints/lavpr_cross_best.ckpt \
    --train_csv      datasets/descriptions/gsv_cities_descriptions.csv \
    --image_root     /home/shared/datasets/gsv_cities/ \
    --top_k_train    100 \
    --top_k_val      20  \
    --cache_dir      cache/ \
    --batch_size     256 \
    --gpu            0
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import faiss
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from model.LaVPR_cross import LaVPR_cross


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    a = p.add_argument

    a('--backbone_ckpt',  type=str, required=True)
    a('--train_csv',      type=str,
      default='datasets/descriptions/gsv_cities_places.csv',
      help='CSV with columns: image_path, description, place_id')
    a('--image_root',     type=str,
      default='/home/shared/datasets/gsv_cities/',
      help='Prepended to image_path column to get the full path')
    a('--image_size',     type=int, default=224)
    a('--top_k_train',   type=int, default=100,
      help='Shortlist size — how many stage-1 candidates to retrieve per query')
    a('--cache_dir',     type=str, default='cache/')
    a('--batch_size',    type=int, default=256)
    a('--num_workers',   type=int, default=4)
    a('--gpu',           type=str, default='0')
    a('--faiss_gpu',          action='store_true')
    a('--skip_local_tokens',  action='store_true',
      help='Skip local token extraction (if you only want to re-mine)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — flat dataset straight from the CSV
# ---------------------------------------------------------------------------

class CSVImageDataset(Dataset):
    """
    Flat dataset built directly from the GSV CSV.

    Columns used
    ------------
    image_path  : relative path; joined with image_root to get full path
    description : text query for this image
    place_id    : integer place identifier (same ID = positive pair)

    Row order is the same as the CSV — this is the canonical index used
    throughout: in embeddings.npz, triplets.json, and the rerank dataloader.
    """

    def __init__(
        self,
        csv_path:   str,
        image_root: str,
        image_size: int = 224,
    ):
        df = pd.read_csv(csv_path)

        # Validate required columns
        for col in ('image_path', 'description', 'place_id'):
            assert col in df.columns, \
                f"Column '{col}' not found in {csv_path}. " \
                f"Found: {list(df.columns)}"

        # Drop rows with invalid place_id (e.g. -1 means no label)
        invalid = df['place_id'].astype(int) < 0
        if invalid.any():
            print(f"  WARNING: dropping {invalid.sum():,} rows with place_id < 0")
            df = df[~invalid].reset_index(drop=True)

        self.descriptions = df['description'].astype(str).tolist()
        self.place_ids    = df['place_id'].astype(int).tolist()
        self.image_root   = image_root.rstrip('/')

        # Resolve paths: if the CSV already has absolute paths use them directly.
        # If relative, join with image_root. If doubled somehow, strip and rejoin.
        raw_paths  = df['image_path'].tolist()
        image_root = self.image_root  # already stripped of trailing /

        def resolve(p):
            if p.startswith('/'):
                return p                          # already absolute
            return image_root + '/' + p           # relative → join

        self.image_paths = [resolve(p) for p in raw_paths]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"CSVImageDataset: {len(self):,} rows  "
              f"| {len(set(self.place_ids)):,} unique places  "
              f"| CSV: {csv_path}")
        print(f"  Sample path [0]: {self.image_paths[0]}")
        print(f"  Sample path [-1]: {self.image_paths[-1]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        full_path = self.image_paths[idx]   # already fully resolved in __init__
        img = Image.open(full_path).convert('RGB')
        img = self.transform(img)
        return {
            'image'    : img,
            'text'     : self.descriptions[idx],
            'place_id' : self.place_ids[idx],
            'idx'      : idx,
        }


# ---------------------------------------------------------------------------
# Step 2 — embed with frozen backbone
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_all(
    backbone: LaVPR_cross,
    dataset:  CSVImageDataset,
    batch_size:  int,
    num_workers: int,
    device:      torch.device,
):
    """
    Pass 1: global embeddings for FAISS mining.

    Returns
    -------
    img_feats  : (N, embeds_dim)  float32  L2-normed
    text_feats : (N, embeds_dim)  float32  L2-normed
    place_ids  : (N,)             int64
    """
    backbone.eval()

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    img_list, text_list, pid_list = [], [], []

    for batch in tqdm(loader, desc='Embedding (global)'):
        images    = batch['image'].to(device)
        texts     = batch['text']
        place_ids = batch['place_id']

        img_e,  _       = backbone.encode_image(images)
        text_e, _, _, _ = backbone.encode_text(texts)

        img_list .append(img_e .cpu().float().numpy())
        text_list.append(text_e.cpu().float().numpy())
        pid_list .append(np.array(place_ids, dtype=np.int64))

    return (
        np.concatenate(img_list,  axis=0),
        np.concatenate(text_list, axis=0),
        np.concatenate(pid_list,  axis=0),
    )


@torch.no_grad()
def extract_local_tokens(
    backbone:    LaVPR_cross,
    dataset:     CSVImageDataset,
    triplets:    list,
    cache_dir:   str,
    batch_size:  int,
    num_workers: int,
    device:      torch.device,
):
    """
    Pass 2: extract and save local tokens only for images that appear
    in the mined shortlists — not the full dataset.

    Saves per-index .pt files under cache_dir/local_tokens/:
        img_{idx}.pt   : (N_patches, D_img)  float16
        text_{idx}.pt  : (L, D_txt)          float16
        mask_{idx}.pt  : (L,)                bool

    Using float16 halves disk usage with negligible precision loss
    for downstream cosine similarity.
    """
    token_dir = os.path.join(cache_dir, 'local_tokens')
    os.makedirs(token_dir, exist_ok=True)

    # Collect unique indices needed — queries + all shortlist images
    needed = set()
    for t in triplets:
        needed.add(t['query_idx'])
        needed.update(t['shortlist'])
    needed = sorted(needed)

    print(f"  Unique indices to extract: {len(needed):,}  "
          f"(out of {len(dataset):,} total)")

    # Check which are already cached
    todo = [i for i in needed
            if not os.path.exists(os.path.join(token_dir, f'img_{i}.pt'))]
    print(f"  Already cached: {len(needed) - len(todo):,}  "
          f"To extract: {len(todo):,}")

    if not todo:
        print("  All tokens already cached — skipping extraction.")
        return

    # Use a Subset loader over only the needed indices
    from torch.utils.data import Subset
    subset  = Subset(dataset, todo)
    loader  = DataLoader(
        subset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    backbone.eval()
    saved = 0

    for batch in tqdm(loader, desc='Extracting local tokens'):
        images    = batch['image'].to(device)   # (B, C, H, W)
        texts     = batch['text']               # List[str]
        idxs      = batch['idx']               # (B,)  original dataset indices

        _, img_local               = backbone.encode_image(images)
        _, text_local, mask, _     = backbone.encode_text(texts)
        # img_local  : (B, N, D_img)
        # text_local : (B, L, D_txt)
        # mask       : (B, L)

        for b in range(len(idxs)):
            idx = int(idxs[b])
            torch.save(img_local [b].half().cpu(),
                       os.path.join(token_dir, f'img_{idx}.pt'))
            torch.save(text_local[b].half().cpu(),
                       os.path.join(token_dir, f'text_{idx}.pt'))
            torch.save(mask      [b].cpu(),
                       os.path.join(token_dir, f'mask_{idx}.pt'))
            saved += 1

    print(f"  Saved {saved:,} token files → {token_dir}")


# ---------------------------------------------------------------------------
# Step 3 — FAISS search + label
# ---------------------------------------------------------------------------

def mine_triplets(
    img_feats:   np.ndarray,   # (N, D)  image gallery
    text_feats:  np.ndarray,   # (N, D)  one text query per image
    place_ids:   np.ndarray,   # (N,)    int64
    top_k_train: int,
    faiss_gpu:   bool,
) -> list:
    """
    For each text query retrieve top-K images, then label each as
    positive (same place_id) or hard negative.

    Returns a flat list of triplet dicts — one per query that has at least
    one positive and one hard negative in its shortlist.
    """
    N, D = img_feats.shape

    # --- Build FAISS index ---
    print(f"Building FAISS IndexFlatIP  N={N:,}  D={D}")
    index = faiss.IndexFlatIP(D)   # feats are already L2-normed → IP = cosine
    if faiss_gpu:
        res   = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(img_feats)

    # Retrieve top-(K+1): +1 because the query's own image is in the gallery
    print(f"Searching top-{top_k_train + 1} for {N:,} queries...")
    _, I = index.search(text_feats, top_k_train + 1)   # (N, K+1)

    # --- Label ---
    triplets = []
    no_pos   = 0

    for q_idx in tqdm(range(N), desc='Labelling'):
        q_place   = int(place_ids[q_idx])
        retrieved = [int(i) for i in I[q_idx] if int(i) != q_idx]  # exclude self

        pos = [i for i in retrieved if int(place_ids[i]) == q_place]
        neg = [i for i in retrieved if int(place_ids[i]) != q_place]

        if pos and neg:
            triplets.append({
                'query_idx' : q_idx,
                'place_id'  : q_place,
                'shortlist' : retrieved[:top_k_train],
                'pos_idxs'  : pos,
                'neg_idxs'  : neg,
            })
        else:
            no_pos += 1

    # --- Summary ---
    avg_pos = np.mean([len(t['pos_idxs']) for t in triplets])
    avg_neg = np.mean([len(t['neg_idxs']) for t in triplets])
    print(f"\nMining summary")
    print(f"  Total queries        : {N:,}")
    print(f"  Triplets built       : {len(triplets):,}")
    print(f"  Skipped (no pos/neg) : {no_pos:,}")
    print(f"  Avg pos in top-{top_k_train:<3d}  : {avg_pos:.2f}  "
          f"(expect ~3 for GSV 4-views/place)")
    print(f"  Avg neg in top-{top_k_train:<3d}  : {avg_neg:.2f}")

    return triplets


# ---------------------------------------------------------------------------
# Step 4 — sanity check
# ---------------------------------------------------------------------------

def sanity_check(triplets: list, place_ids: np.ndarray, n: int = 5):
    print(f"\n--- Sanity check (first {n} triplets) ---")
    for t in triplets[:n]:
        q_place  = t['place_id']
        pos_pids = [int(place_ids[i]) for i in t['pos_idxs']]
        neg_pids = [int(place_ids[i]) for i in t['neg_idxs'][:3]]
        assert all(p == q_place for p in pos_pids), \
            f"Positive label mismatch! query={q_place} pos_pids={pos_pids}"
        assert all(p != q_place for p in neg_pids), \
            f"Negative label mismatch! query={q_place} neg_pids={neg_pids}"
        print(f"  query_idx={t['query_idx']:7,}  place_id={q_place:8,}  "
              f"pos={len(t['pos_idxs'])}  neg={len(t['neg_idxs'])}  "
              f"neg_pids_sample={neg_pids}")
    print("  All checks passed.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.cache_dir, exist_ok=True)
    emb_path     = os.path.join(args.cache_dir, 'gsv_embeddings.npz')
    triplet_path = os.path.join(args.cache_dir, 'gsv_triplets.json')

    # ------------------------------------------------------------------
    # 1. Flat dataset — read CSV first (fast, no GPU needed)
    # ------------------------------------------------------------------
    print(f"\n[1/4] Reading CSV  {args.train_csv}")
    dataset = CSVImageDataset(
        csv_path   = args.train_csv,
        image_root = args.image_root,
        image_size = args.image_size,
    )

    # ------------------------------------------------------------------
    # 2. Embed (or load from cache)
    #    Load the backbone only if the cache does not already exist —
    #    re-runs after the first embed pass skip the model entirely.
    # ------------------------------------------------------------------
    if os.path.exists(emb_path):
        print(f"\n[2/4] Loading cached embeddings  {emb_path}")
        data       = np.load(emb_path)
        img_feats  = data['img_feats']
        text_feats = data['text_feats']
        place_ids  = data['place_ids']
        assert len(img_feats) == len(dataset), (
            f"Cached embeddings ({len(img_feats):,}) != "
            f"CSV rows ({len(dataset):,}). "
            f"Delete {emb_path} and re-run to rebuild.")
    else:
        print(f"\n[2/4] Loading backbone  {args.backbone_ckpt}")
        backbone = LaVPR_cross.load_from_checkpoint(
            args.backbone_ckpt, map_location='cpu')
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone = backbone.to(device)
        print(f"      embeds_dim={backbone.embeds_dim}  "
              f"image={backbone.image_model_name}  "
              f"text={backbone.text_model_name}")

        print(f"\n[3/4] Embedding {len(dataset):,} rows  "
              f"batch_size={args.batch_size}")
        img_feats, text_feats, place_ids = embed_all(
            backbone    = backbone,
            dataset     = dataset,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            device      = device,
        )
        np.savez(emb_path,
                 img_feats  = img_feats,
                 text_feats = text_feats,
                 place_ids  = place_ids)
        print(f"      Saved → {emb_path}  "
              f"({os.path.getsize(emb_path)/1e6:.0f} MB)")

        # Free GPU memory before FAISS
        del backbone
        torch.cuda.empty_cache()

    print(f"      shapes — img_feats {img_feats.shape}  "
          f"text_feats {text_feats.shape}  "
          f"place_ids {place_ids.shape}")

    # ------------------------------------------------------------------
    # 4. Mine
    # ------------------------------------------------------------------
    print(f"\n[4/4] Mining hard negatives  top_k_train={args.top_k_train}")
    triplets = mine_triplets(
        img_feats   = img_feats,
        text_feats  = text_feats,
        place_ids   = place_ids,
        top_k_train = args.top_k_train,
        faiss_gpu   = args.faiss_gpu,
    )

    sanity_check(triplets, place_ids)

    with open(triplet_path, 'w') as f:
        json.dump(triplets, f)

    triplet_mb = os.path.getsize(triplet_path) / 1e6
    print(f"Saved → {triplet_path}  ({triplet_mb:.0f} MB)")

    # ------------------------------------------------------------------
    # 5. Extract local tokens for all shortlist images
    #    Backbone needed again — reload if it was freed after embedding.
    # ------------------------------------------------------------------
    if not args.skip_local_tokens:
        print(f"\n[5/5] Extracting local tokens for shortlist images")
        if 'backbone' not in dir():
            print(f"      Reloading backbone for local token extraction...")
            backbone = LaVPR_cross.load_from_checkpoint(
                args.backbone_ckpt, map_location='cpu')
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            backbone = backbone.to(device)

        extract_local_tokens(
            backbone    = backbone,
            dataset     = dataset,
            triplets    = triplets,
            cache_dir   = args.cache_dir,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            device      = device,
        )
    else:
        print("\nSkipping local token extraction (--skip_local_tokens).")

    print("\nDone.  Run train_reranker.py to start stage-2 training.")