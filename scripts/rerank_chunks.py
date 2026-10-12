"""
Chunk-based BF reranking for text VPR.

Pipeline:
  1. Load single CSV (db + queries) matching TestDataset split convention
  2. Build global shortlist (top-K) from precomputed or computed descriptors
  3. Encode all chunks with merged BGE
  4. BF match query chunks vs candidate chunks (cosine similarity)
  5. Score = mean of best-match cosine similarities per query chunk
  6. Re-rank shortlist, evaluate Recall@K

CSV format:
    image_path, description, chunks   (chunks is a JSON list of strings)
    db paths contain --db_folder string, query paths contain --query_folder string

Usage:
    python rerank_chunks.py \
        --csv          datasets/descriptions/amstertime.csv \
        --image_root   /data/amstertime \
        --db_folder    database \
        --query_folder queries \
        --model_name   path/to/merged_bge \
        --global_feats_path amstertime_global.pt \
        --topk 20 --topk_out 5
"""

import argparse
import ast
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import cv2
import posixpath
from dataloaders.test_dataset import TestDataset


# ---------------------------------------------------------------------------
# BGE encoder (merged LoRA — plain HF model)
# ---------------------------------------------------------------------------

def load_encoder(model_name: str, device: str = 'cuda'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
    model     = model.eval().to(device)
    return tokenizer, model


@torch.no_grad()
def encode_texts(texts: list, tokenizer, model, device: str,
                 batch_size: int = 256) -> torch.Tensor:
    """Encode list of strings → L2-normalised (N, D) cpu tensor."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i: i + batch_size]
        tokens = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors='pt').to(device)
        out    = model(**tokens, return_dict=True)
        embeds = F.normalize(out.last_hidden_state[:, 0], p=2, dim=1)
        all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)



# ---------------------------------------------------------------------------
# BF chunk matching
# ---------------------------------------------------------------------------

def bf_chunk_score(q_emb: np.ndarray, db_emb: np.ndarray) -> float:
    """
    For each query chunk find its best matching DB chunk (no crossCheck —
    too strict with few chunks). Score = mean of per-query-chunk best cosine
    similarities.

    cv2.BFMatcher args:
        normType   = cv2.NORM_L2   — correct for L2-normalised vectors
        crossCheck = False         — one-sided NN; crossCheck drops too many
                                     matches when chunk count is small (2-5)

    cos_sim = 1 - dist^2 / 2  (exact for unit vectors)
    """
    matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # knnMatch with k=1: one best DB chunk per query chunk
    matches = matcher.knnMatch(q_emb.astype(np.float32),
                               db_emb.astype(np.float32), k=1)
    sims = [1.0 - (m[0].distance ** 2) / 2.0 for m in matches if m]
    return float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Recall evaluation
# ---------------------------------------------------------------------------

def compute_recalls(indices: np.ndarray, positives, k_values: list) -> dict:
    """
    indices   : (N_q, topk) int array of predicted db indices
    positives : list/array of positive db index sets per query
                (TestDataset.get_positives() returns arrays of arrays)
    """
    N_q = indices.shape[0]
    # pre-convert each query's positives to a set once
    pos_sets = [set(np.asarray(positives[q]).ravel().tolist())
                for q in range(N_q)]
    return {
        k: 100.0 * sum(
            bool(set(indices[q, :k].tolist()) & pos_sets[q])
            for q in range(N_q)
        ) / N_q
        for k in k_values
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def rerank(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Build dataset — reuses TestDataset for paths, positives and UTM labels
    print("Loading dataset...")
    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_ds = TestDataset(
        database_folder=args.db_folder,
        queries_folder=args.query_folder,
        csv_path=args.csv,
        image_root=args.image_root,
        mean_std=IMAGENET_MEAN_STD,
        positive_dist_threshold=args.positive_dist_threshold,
        use_labels=True,
    )
    db_paths = test_ds.database_paths
    q_paths  = test_ds.queries_paths
    N_db, N_q = test_ds.num_database, test_ds.num_queries
    positives = test_ds.get_positives()   # list of arrays, one per query
    print(f"DB: {N_db}  |  Queries: {N_q}")

    # ---- Short-circuit: load precomputed reranked results and just print recalls
    if args.load_reranked:
        print(f"Loading precomputed reranked results from {args.load_reranked}")
        saved    = torch.load(args.load_reranked)
        reranked = saved['reranked_indices']
        topk_idx = saved['topk_indices']
        positives = test_ds.get_positives()
        k_values  = [1, 5, 10, 20]
        header    = f"{'Method':<30}" + "".join(f"  R@{k:<5}" for k in k_values)
        sep       = "-" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        def _pr(label, idx):
            r   = compute_recalls(idx, positives, k_values)
            print(f"{label:<30}" + "".join(f"  {r[k]:<7.1f}" for k in k_values))
        _pr("Global (before rerank)", topk_idx)
        _pr("Chunk BF rerank       ", reranked)
        print(sep)
        return

    # ---- Load chunks + descriptions from chunked CSV
    print("Loading chunks from CSV...")
    chunks_map, desc_map = {}, {}
    df = pd.read_csv(args.csv, engine='python', encoding='utf-8',
                     on_bad_lines='skip', quotechar='"', skipinitialspace=True)
    for row in df.itertuples():
        abs_path = posixpath.join(args.image_root, row.image_path)
        raw = row.chunks
        if isinstance(raw, str):
            try:
                chunks = json.loads(raw)
            except json.JSONDecodeError:
                chunks = ast.literal_eval(raw)
        else:
            chunks = [row.description]
        chunks_map[abs_path] = chunks
        desc_map[abs_path]   = row.description

    # ---- Load encoder
    print(f"Loading encoder from {args.model_name}...")
    tokenizer, model = load_encoder(args.model_name, device)

    # ---- Global descriptors — order matches TestDataset: [db..., queries...]
    if args.global_feats_path:
        print(f"Loading global features from {args.global_feats_path}")
        saved = torch.load(args.global_feats_path)
        if 'db_feats' in saved and 'q_feats' in saved:
            db_global = saved['db_feats']       # (N_db, D)
            q_global  = saved['q_feats']        # (N_q,  D)
        else:
            # combined tensor ordered [db..., queries...] as in eval pipeline
            all_feats = saved['descriptors']
            db_global = all_feats[:N_db]
            q_global  = all_feats[N_db:]
    else:
        print("Computing global descriptors from full descriptions...")
        db_global = encode_texts([desc_map[p] for p in db_paths],
                                 tokenizer, model, device, args.batch_size)
        q_global  = encode_texts([desc_map[p] for p in q_paths],
                                 tokenizer, model, device, args.batch_size)

    # ---- Initial top-K shortlist
    print(f"Building top-{args.topk} shortlist...")
    sim_matrix      = torch.mm(q_global, db_global.T)       # (N_q, N_db)
    _, topk_indices = sim_matrix.topk(args.topk, dim=1)     # (N_q, topk)

    # ---- Encode only chunks needed: all queries + topk db candidates
    topk_db_indices = set(topk_indices.numpy().flatten().tolist())

    if args.chunk_feats_path and os.path.exists(args.chunk_feats_path):
        print(f"Loading cached chunk embeddings from {args.chunk_feats_path}")
        chunk_cache     = torch.load(args.chunk_feats_path)
        chunk_emb_map   = chunk_cache['chunk_emb_map']   # {abs_path: (n_chunks, D) tensor}
    else:
        print("Collecting unique chunks for queries and topk candidates...")
        paths_to_encode = q_paths + [db_paths[i] for i in topk_db_indices]

        all_chunk_texts, chunk_idx = [], {}
        for path in paths_to_encode:
            for c in chunks_map.get(path, [desc_map[path]]):
                if c not in chunk_idx:
                    chunk_idx[c] = len(all_chunk_texts)
                    all_chunk_texts.append(c)

        print(f"Encoding {len(all_chunk_texts)} unique chunks "
              f"({len(topk_db_indices)} unique DB candidates + {N_q} queries)...")
        all_chunk_embeds = encode_texts(all_chunk_texts, tokenizer, model,
                                        device, args.batch_size)

        # Build per-path embedding map
        chunk_emb_map = {}
        for path in paths_to_encode:
            chunks = chunks_map.get(path, [desc_map[path]])
            chunk_emb_map[path] = all_chunk_embeds[[chunk_idx[c] for c in chunks]]

        if args.chunk_feats_path:
            print(f"Saving chunk embeddings to {args.chunk_feats_path}")
            torch.save({'chunk_emb_map': chunk_emb_map}, args.chunk_feats_path)

    def get_chunk_emb(path):
        return chunk_emb_map[path].numpy()

    # ---- Rerank
    print("Reranking...")
    reranked = np.zeros((N_q, args.topk), dtype=np.int64)

    # global cosine scores for the shortlist — used for blending
    global_scores = torch.mm(q_global, db_global.T)    # (N_q, N_db)

    for q_idx in range(N_q):
        q_emb     = get_chunk_emb(q_paths[q_idx])
        shortlist = topk_indices[q_idx].tolist()

        chunk_sc  = np.array([bf_chunk_score(q_emb, get_chunk_emb(db_paths[i]))
                               for i in shortlist], dtype=np.float32)
        global_sc = global_scores[q_idx, shortlist].numpy()

        # z-normalise each signal then blend
        def znorm(x):
            s = x.std()
            return (x - x.mean()) / s if s > 1e-6 else x - x.mean()

        blended = args.global_weight * znorm(global_sc) + \
                  (1 - args.global_weight) * znorm(chunk_sc)

        order           = np.argsort(-blended)
        reranked[q_idx] = [shortlist[i] for i in order]

        if q_idx % 100 == 0:
            print(f"  {q_idx}/{N_q}")

    # ---- Recall before/after (positives from TestDataset UTM labels)
    k_values = [1, 5, 10, 20]
    header   = f"{'Method':<30}" + "".join(f"  R@{k:<5}" for k in k_values)
    sep      = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    def print_recalls(label, indices):
        r   = compute_recalls(indices, positives, k_values)
        row = f"{label:<30}" + "".join(f"  {r[k]:<7.1f}" for k in k_values)
        print(row)

    print_recalls("Global (before rerank)", topk_indices.numpy())
    print_recalls("Chunk BF rerank       ", reranked)
    print(sep)

    # ---- Save
    torch.save({
        'reranked_indices': reranked,
        'topk_indices':     topk_indices.numpy(),
        'db_paths':         db_paths,
        'q_paths':          q_paths,
    }, args.output)
    print(f"\nSaved to {args.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--csv",                      type=str,   default="datasets/descriptions/pitts30k_test_chunked.csv",          help="CSV with db+query rows, description and chunks columns")
    a("--image_root",               type=str,   default="/home/shared/datasets/pitts30k/images/test",              help="root directory prepended to image_path column")
    a("--db_folder",                type=str,   default="/home/shared/datasets/pitts30k/images/test/database",     help="database folder path (passed to TestDataset)")
    a("--query_folder",             type=str,   default="/home/shared/datasets/pitts30k/images/test/queries",      help="queries folder path (passed to TestDataset)")
    a("--positive_dist_threshold",  type=int,   default=25,                                                    help="UTM radius in meters for positive match")
    a("--model_name",               type=str,   default="checkpoints/bge-l-lora-ms-merged",                   help="merged BGE model path or HF name")
    a("--global_feats_path",        type=str,   default=None,                                                  help=".pt with db_feats+q_feats or combined descriptors[db..q]")
    a("--chunk_feats_path",         type=str,   default=None,                                                  help=".pt to save/load chunk embeddings cache (skips encoding if exists)")
    a("--global_weight",            type=float, default=0.5,                                                   help="blend weight for global score (0=chunk only, 1=global only, 0.5=equal)")
    a("--topk",                     type=int,   default=20,                                                    help="shortlist size for reranking")
    a("--topk_out",                 type=int,   default=5,                                                     help="k for final recall reporting")
    a("--output",                   type=str,   default="reranked.pt",                                        help="output file")
    a("--load_reranked",            type=str,   default=None,                                                  help="load precomputed reranked .pt and print recalls (skips all encoding)")
    a("--batch_size",               type=int,   default=256,                                                   help="encoding batch size")
    args = p.parse_args()
    rerank(args)