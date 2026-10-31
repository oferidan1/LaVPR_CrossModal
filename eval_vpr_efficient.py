"""
eval_vpr_efficient.py  —  Memory-efficient evaluation for large galleries (e.g. SF-XL 2.8M)

Pipeline
--------
1. Encode ALL db images + query texts → global vectors only  (no local tokens stored)
2. FAISS top-K retrieval                                      (~40 MB for 2.8M × 512-d)
3. For each query, re-run the encoder on its top-K db images  to get local tokens
4. Rerank with the LaVPR CrossModalRerankModule
5. Compute Recall@K

Memory comparison (2.8M images, 256 patches, 768-d, float32)
  Old approach  : 2.8M × 256 × 768 × 4  ≈ 2.0 TiB  ← OOM
  New approach  : 2.8M × 512 × 4        ≈  5.4 GB   ← fine
"""

import argparse
import sys
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
import torch.nn.functional as F

import eval_parser
from model.LaVPR_wrapper import LaVPR_wrapper
from dataloaders.test_dataset import TestDataset


# ---------------------------------------------------------------------------
# Timing stats (same as eval_vpr.py)
# ---------------------------------------------------------------------------

class TimingStats:
    """Accumulate wall-clock timings and print a summary table.

    ms/item meaning per stage:
      encode_global_*   → per image or query encoded
      faiss_retrieval   → per query searched
      rerank_*          → per candidate image scored (bsz × K)
      encode_local_*    → per image re-encoded for local tokens
    """

    def __init__(self):
        self._data = {}

    def record(self, name: str, n_items: int, elapsed_s: float):
        self._data.setdefault(name, []).append((n_items, elapsed_s))

    def summary(self) -> str:
        if not self._data:
            return "(no timing data)"
        lines = [
            f"\n{'='*72}",
            f"  {'Stage':<32}  {'Calls':>5}  {'Items':>8}  "
            f"{'Total(s)':>9}  {'ms/item':>8}",
            f"  {'(items = images for encode, cand-images for rerank)':<68}",
            f"  {'-'*32}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}",
        ]
        for name, records in self._data.items():
            n_calls = len(records)
            n_items = sum(r[0] for r in records)
            total_s = sum(r[1] for r in records)
            ms_per  = (total_s / n_items * 1000) if n_items > 0 else 0
            lines.append(
                f"  {name:<32}  {n_calls:>5}  {n_items:>8}  "
                f"{total_s:>9.2f}  {ms_per:>8.2f}")
        lines.append(f"{'='*72}\n")
        return "\n".join(lines)


TIMING = TimingStats()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _maybe(mod, name, x):
    layer = getattr(mod, name, None)
    return layer(x) if layer is not None else x


# ---------------------------------------------------------------------------
# Stage 1 – global encoding (no local tokens)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_global(model, dataloader, total_len, encoder_dim, device,
                  bfloat16=False, is_image=True, cross_modal=3,
                  desc="Encoding", timing_key="encode_global"):
    """
    Returns two arrays of shape (total_len, encoder_dim):
      vision_descriptors  – image global embeddings  (zeros for query slots)
      text_descriptors    – text  global embeddings  (zeros for db slots)
    """
    vision_descriptors = np.zeros((total_len, encoder_dim), dtype=np.float32)
    text_descriptors   = np.zeros((total_len, encoder_dim), dtype=np.float32)

    for images, indices, texts in tqdm(dataloader, desc=desc):
        t0 = time.perf_counter()
        if bfloat16:
            images = images.bfloat16()

        if cross_modal >= 2:
            out = model.encode_single(images.to(device), texts)
            img_global, txt_global = out[0], out[1]
            if is_image and img_global is not None:
                vision_descriptors[indices.numpy()] = img_global.cpu().float().numpy()
            text_descriptors[indices.numpy()] = txt_global.cpu().float().numpy()
        else:
            if is_image:
                img_global = model.encode_image(images.to(device))
                vision_descriptors[indices.numpy()] = img_global.cpu().float().numpy()
            txt_global = model.encode_text(texts)
            text_descriptors[indices.numpy()] = txt_global.cpu().float().numpy()

        TIMING.record(timing_key, len(indices), time.perf_counter() - t0)

    return vision_descriptors, text_descriptors


# ---------------------------------------------------------------------------
# Stage 2 – FAISS retrieval
# ---------------------------------------------------------------------------

def faiss_search(db_vecs: np.ndarray, q_vecs: np.ndarray,
                 top_k: int) -> tuple[np.ndarray, np.ndarray]:
    t0      = time.perf_counter()
    db_vecs = normalize(db_vecs)
    q_vecs  = normalize(q_vecs)
    index   = faiss.IndexFlatIP(db_vecs.shape[1])
    index.add(db_vecs)
    scores, preds = index.search(q_vecs, top_k)
    TIMING.record('faiss_retrieval', len(q_vecs), time.perf_counter() - t0)
    return scores, preds


# ---------------------------------------------------------------------------
# Stage 3+4 – lazy local encoding + reranking
# ---------------------------------------------------------------------------

@torch.no_grad()
def rerank_topk_batched(
    model,
    dataset,
    predictions:        np.ndarray,
    num_database:       int,
    img_local_dim:      int,
    num_img_tokens:     int,
    txt_local_dim:      int,
    num_text_tokens:    int,
    vision_descriptors: np.ndarray,
    text_descriptors:   np.ndarray,
    max_rerank_k:       int   = 20,
    query_batch:        int   = 8,
    encode_batch:       int   = 128,
    alpha:              float = 0.0,
    device:             str   = "cuda",
    bfloat16:           bool  = False,
) -> np.ndarray:
    reranker = model.reranker_network
    if reranker is None:
        raise RuntimeError("model.reranker_network is None.")
    reranker.eval()
    w_dtype = next(reranker.parameters()).dtype

    # Validate reranker input dims
    img_proj_in = reranker.img_proj.weight.shape[1]
    txt_proj_in = reranker.txt_proj.weight.shape[1]
    if img_local_dim != img_proj_in:
        raise RuntimeError(
            f"img local dim mismatch: reranker expects {img_proj_in}, "
            f"buffer has {img_local_dim}.")
    if txt_local_dim != txt_proj_in:
        raise RuntimeError(
            f"txt local dim mismatch: reranker expects {txt_proj_in}, "
            f"buffer has {txt_local_dim}.")

    num_q    = predictions.shape[0]
    n_cand   = min(max_rerank_k, predictions.shape[1])
    reranked = predictions.copy()

    logger.info(f"Batched lazy reranking: {num_q} queries × top-{n_cand} "
                f"(query_batch={query_batch}, encode_batch={encode_batch}, "
                f"alpha={alpha}, dtype={w_dtype})")

    def _encode_indices(ds_indices: list):
        subset = Subset(dataset, ds_indices)
        loader = DataLoader(subset, batch_size=encode_batch,
                            num_workers=4, shuffle=False, pin_memory=True)
        img_parts, txt_parts, att_parts = [], [], []
        for images, _, texts in loader:
            t0 = time.perf_counter()
            if bfloat16:
                images = images.bfloat16()
            out = model.encode_single(images.to(device), texts)
            img_local, txt_local, attn = out[2], out[3], out[5]
            if img_local is None or txt_local is None:
                raise RuntimeError("encode_single returned None local tokens.")
            img_parts.append(img_local.cpu().float())
            txt_parts.append(txt_local.cpu().float())
            att_parts.append(
                attn.cpu().long() if attn is not None
                else torch.ones(txt_local.shape[:2], dtype=torch.long))
            TIMING.record('encode_local_combined_batch',
                          len(images), time.perf_counter() - t0)

        # Pad text tokens to global max L across batches
        max_L  = max(t.shape[1] for t in txt_parts)
        D_txt  = txt_parts[0].shape[2]
        txt_padded, att_padded = [], []
        for t, a in zip(txt_parts, att_parts):
            pad = max_L - t.shape[1]
            if pad > 0:
                t = torch.cat([t, torch.zeros(t.shape[0], pad, D_txt)], dim=1)
                a = torch.cat([a, torch.zeros(a.shape[0], pad, dtype=torch.long)], dim=1)
            txt_padded.append(t)
            att_padded.append(a)

        return (torch.cat(img_parts,  dim=0),
                torch.cat(txt_padded, dim=0),
                torch.cat(att_padded, dim=0))

    def _zn(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else x

    for q_start in tqdm(range(0, num_q, query_batch), desc="Batched rerank"):
        q_end = min(q_start + query_batch, num_q)
        bsz   = q_end - q_start

        q_ds_idxs   = [num_database + q for q in range(q_start, q_end)]
        cand_matrix = predictions[q_start:q_end, :n_cand]  # (bsz, K)

        # Deduplicate DB candidates
        flat_db_idxs        = cand_matrix.ravel()
        unique_db_idxs, inv = np.unique(flat_db_idxs, return_inverse=True)

        # Encode unique DB images and queries
        all_img_local, _, _        = _encode_indices(unique_db_idxs.tolist())
        _, all_txt_local, all_attn = _encode_indices(q_ds_idxs)

        # Score every (query, candidate) pair
        t0_rerank    = time.perf_counter()
        local_scores = np.zeros((bsz, n_cand), dtype=np.float32)

        for b in range(bsz):
            cand_inv  = inv[b * n_cand : (b + 1) * n_cand]
            img_cands = all_img_local[cand_inv]              # (K, N_p, D_img)

            real_len  = max(int(all_attn[b].sum().item()), 1)
            txt_local = all_txt_local[b, :real_len].unsqueeze(0)    # (1, L, D_txt)
            txt_mask  = all_attn[b, :real_len].unsqueeze(0).bool()  # (1, L)

            K = img_cands.shape[0]
            L = txt_local.shape[1]
            D_txt_ = txt_local.shape[2]

            img_t = img_cands.to(device).to(w_dtype)
            txt_t = txt_local.expand(K, -1, -1).contiguous().to(device).to(w_dtype)
            msk_t = txt_mask.expand(K, -1).contiguous().to(device)

            _, scores = reranker(img_t, txt_t, msk_t)
            local_scores[b] = scores.float().cpu().numpy()

        TIMING.record('rerank_lavpr_batch',
                      bsz * n_cand, time.perf_counter() - t0_rerank)

        # Optional global cosine fusion
        if alpha > 0.0:
            for b in range(bsz):
                q_ds_idx   = q_ds_idxs[b]
                q_global   = text_descriptors[q_ds_idx]
                db_globals = vision_descriptors[cand_matrix[b]]
                q_norm     = q_global  / (np.linalg.norm(q_global) + 1e-8)
                db_norm    = db_globals / (np.linalg.norm(
                    db_globals, axis=1, keepdims=True) + 1e-8)
                global_sim = db_norm @ q_norm
                local_scores[b] = (alpha * _zn(global_sim)
                                   + (1.0 - alpha) * _zn(local_scores[b]))

        for b in range(bsz):
            order = np.argsort(-local_scores[b])
            reranked[q_start + b, :n_cand] = cand_matrix[b][order]

    logger.info("Batched reranking complete.")
    return reranked


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recalls(predictions: np.ndarray, positives_per_query: list,
                    recall_values: list) -> tuple:
    recalls = np.zeros(len(recall_values))
    for q_idx, preds in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.isin(preds[:n], positives_per_query[q_idx])):
                recalls[i:] += 1
                break
    recalls = recalls / len(predictions) * 100
    r_str   = ", ".join([f"R@{v}: {r:.1f}" for v, r in zip(recall_values, recalls)])
    return recalls, r_str


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def run_condition(args, model, test_ds, cond_name,
                  img_local_dim, num_img_tokens,
                  txt_local_dim, num_text_tokens,
                  db_vis_descs, db_txt_descs,
                  device, log_dir):
    """
    Run stage-2 FAISS + optional stage-3/4 reranking for one query condition.
    DB global descriptors are passed in (pre-encoded once) to avoid re-encoding
    the gallery for every SVOX / SF-XL condition.
    """
    num_db    = test_ds.num_database
    num_q     = test_ds.num_queries
    total_len = len(test_ds)
    max_top_k = max(args.recall_values)

    prefix = f"[{cond_name}] " if cond_name else ""

    # ── Query global encoding ────────────────────────────────────────────────
    vis_descs = np.zeros((total_len, model.encoder_dim), dtype=np.float32)
    txt_descs = np.zeros((total_len, model.encoder_dim), dtype=np.float32)
    vis_descs[:num_db] = db_vis_descs
    txt_descs[:num_db] = db_txt_descs

    with torch.inference_mode():
        q_subset = Subset(test_ds, list(range(num_db, total_len)))
        q_loader = DataLoader(q_subset, num_workers=args.num_workers,
                              batch_size=args.batch_size, shuffle=False)
        vis_q, txt_q = encode_global(
            model, q_loader, total_len, model.encoder_dim, device,
            bfloat16=args.bfloat16, is_image=args.is_image,
            cross_modal=args.cross_modal,
            desc=f"{prefix}Encoding queries (global)",
            timing_key="encode_global_query")
    vis_descs[num_db:] = vis_q[num_db:]
    txt_descs[num_db:] = txt_q[num_db:]

    # ── FAISS retrieval ──────────────────────────────────────────────────────
    if args.cross_modal and not args.is_image:
        db_vecs = txt_descs[:num_db]
    else:
        db_vecs = vis_descs[:num_db]
    q_vecs = txt_descs[num_db:]

    scores, retrieval_preds = faiss_search(db_vecs, q_vecs, max_top_k)
    logger.info(f"{prefix}Retrieval done. Shape: {retrieval_preds.shape}")

    # ── Reranking ────────────────────────────────────────────────────────────
    reranking_preds   = retrieval_preds.copy()
    reranking_enabled = False

    do_rerank = (getattr(args, 'reranker', False)
                 and args.cross_modal == 3
                 and model.reranker_network is not None)

    if do_rerank:
        reranking_enabled = True
        max_rerank_k      = min(args.max_rerank, max_top_k)
        logger.info(f"{prefix}Stage 3+4: Lazy reranking top-{max_rerank_k}")

        t_rerank = time.perf_counter()
        reranking_preds = rerank_topk_batched(
            model              = model,
            dataset            = test_ds,
            predictions        = retrieval_preds,
            num_database       = num_db,
            img_local_dim      = img_local_dim,
            num_img_tokens     = num_img_tokens,
            txt_local_dim      = txt_local_dim,
            num_text_tokens    = num_text_tokens,
            vision_descriptors = vis_descs,
            text_descriptors   = txt_descs,
            max_rerank_k       = max_rerank_k,
            query_batch        = getattr(args, 'rerank_query_batch', 8),
            encode_batch       = getattr(args, 'rerank_batch', 128),
            alpha              = getattr(args, 'rerank_alpha', 0.0),
            device             = device,
            bfloat16           = args.bfloat16,
        )
        TIMING.record('stage34_total', num_q, time.perf_counter() - t_rerank)
        logger.info(f"{prefix}Reranking done in "
                    f"{time.perf_counter() - t_rerank:.1f}s")

    # ── Recalls ──────────────────────────────────────────────────────────────
    ret_recalls, ret_str, rer_recalls, rer_str = None, None, None, None
    if args.use_labels:
        positives = test_ds.get_positives()
        ret_recalls, ret_str = compute_recalls(
            retrieval_preds, positives, args.recall_values)
        logger.info(f"{prefix}Retrieval : {ret_str}")
        print(f"{prefix}Retrieval : {ret_str}")

        if reranking_enabled:
            rer_recalls, rer_str = compute_recalls(
                reranking_preds, positives, args.recall_values)
            logger.info(f"{prefix}Reranking : {rer_str}")
            print(f"{prefix}Reranking : {rer_str}")

        model_path = args.lora_path if args.lora_path else args.model_path
        with open("eval_vpr_results.csv", "a") as f:
            tag = f"{args.model_name} [{cond_name}]" if cond_name else args.model_name
            if reranking_enabled:
                f.write(f"{model_path},{tag},"
                        f"Retrieval: {ret_str} | Reranking: {rer_str}\n")
            else:
                f.write(f"{model_path},{tag},{ret_str}\n")

    return ret_recalls, ret_str, rer_recalls, rer_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    start_time = datetime.now()
    logger.remove()
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(f"Arguments: {args}")

    IMAGENET = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP     = {'mean': [0.48145466, 0.4578275, 0.40821073],
                'std':  [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP   = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    name = args.model_name.lower()
    if any(k in name for k in ('blip', 'clip', 'eva')):
        mean_std = BLIP
    elif 'siglip' in name:
        mean_std = SIGLIP
    else:
        mean_std = IMAGENET

    # ── Dataset routing ──────────────────────────────────────────────────────
    eval_conditions = None   # None = single condition

    if args.data == 'sfxl':
        args.database_folder = "/home/shared/datasets/SF-XL/processed/test/database"
        args.image_root      = "/home/shared/datasets/SF-XL/processed/test"
        eval_conditions = [
            ("queries_v1", f"{args.image_root}/queries_v1",
             "datasets/descriptions/sfxl_queries_v1.csv"),
            ("queries_v2", f"{args.image_root}/queries_v2",
             "datasets/descriptions/sfxl_queries_v2.csv"),
            ("occlusion",  f"{args.image_root}/queries_occlusion",
             "datasets/descriptions/sfxl_queries_occlusion.csv"),
            ("night",      f"{args.image_root}/queries_night",
             "datasets/descriptions/sfxl_queries_night.csv"),
        ]
    elif args.data in ('pitts', 'pitts_test', 'pitts30k'):
        args.database_folder = "/home/shared/datasets/pitts30k/images/test/database"
        args.queries_folder  = "/home/shared/datasets/pitts30k/images/test/queries"
        args.image_root      = "/home/shared/datasets/pitts30k/images/test"
        args.queries_csv     = "datasets/descriptions/pitts30k_test_descriptions.csv"
    elif args.data in ('pitts_val', 'pitts30k_val'):
        args.database_folder = "/home/shared/datasets/pitts30k/images/val/database"
        args.queries_folder  = "/home/shared/datasets/pitts30k/images/val/queries"
        args.image_root      = "/home/shared/datasets/pitts30k/images/val"
        args.queries_csv     = "datasets/descriptions/pitts30k_val_800_queries.csv"
    elif args.data in ('msls', 'msls_val'):
        args.database_folder = "/home/shared/datasets/msls/val/database"
        args.queries_folder  = "/home/shared/datasets/msls/val/query"
        args.image_root      = "/home/shared/datasets/msls/val/"
        args.queries_csv     = "datasets/descriptions/msls_val_descriptions.csv"
    elif args.data in ('svox', 'svox_all'):
        args.database_folder = "/home/shared/datasets/svox/images/test/gallery"
        args.image_root      = "/home/shared/datasets/svox/images/test"
        eval_conditions = [
            ("Night",    f"{args.image_root}/queries_night",
             "datasets/descriptions/svox_night.csv"),
            ("Sun",      f"{args.image_root}/queries_sun",
             "datasets/descriptions/svox_sun.csv"),
            ("Rain",     f"{args.image_root}/queries_rain",
             "datasets/descriptions/svox_rain.csv"),
            ("Snow",     f"{args.image_root}/queries_snow",
             "datasets/descriptions/svox_snow.csv"),
            ("Overcast", f"{args.image_root}/queries_overcast",
             "datasets/descriptions/svox_overcast.csv"),
        ]

    # ── Model ────────────────────────────────────────────────────────────────
    model  = LaVPR_wrapper(args)
    device = args.device
    logger.info(f"Encoder dim: {model.encoder_dim}")

    num_img_tokens  = model.img_local_tokens
    num_text_tokens = model.txt_local_tokens
    img_local_dim   = model.img_local_dim
    txt_local_dim   = model.txt_local_dim

    # ── Build the one shared TestDataset for the gallery ─────────────────────
    # For multi-condition datasets the gallery is always the same;
    # we build a temporary dataset just to encode the DB, then swap
    # queries_folder / queries_csv per condition.
    if eval_conditions is not None:
        # Use first condition's queries to build an initial dataset
        # (only the DB portion is used here)
        first_q_folder = eval_conditions[0][1]
        first_q_csv    = eval_conditions[0][2]
    else:
        first_q_folder = args.queries_folder
        first_q_csv    = args.queries_csv

    db_dataset = TestDataset(
        args.database_folder, first_q_folder, first_q_csv,
        args.image_root, mean_std=mean_std,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size, use_labels=args.use_labels)
    num_db = db_dataset.num_database
    logger.info(f"Gallery: {num_db} images")

    # ── Encode gallery once ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1: Encoding gallery (global, once for all conditions)")
    logger.info("=" * 60)

    # Allocate a buffer sized for db only
    db_vis_buf = np.zeros((num_db, model.encoder_dim), dtype=np.float32)
    db_txt_buf = np.zeros((num_db, model.encoder_dim), dtype=np.float32)
    # Temporary full buffer so encode_global indices work correctly
    total_tmp  = num_db + db_dataset.num_queries

    t_gallery = time.perf_counter()
    with torch.inference_mode():
        db_subset = Subset(db_dataset, list(range(num_db)))
        db_loader = DataLoader(db_subset, num_workers=args.num_workers,
                               batch_size=args.batch_size, shuffle=False)
        vis_tmp, txt_tmp = encode_global(
            model, db_loader, total_tmp, model.encoder_dim, device,
            bfloat16=args.bfloat16, is_image=args.is_image,
            cross_modal=args.cross_modal,
            desc="Encoding gallery (global)", timing_key="encode_global_db")
    db_vis_buf[:] = vis_tmp[:num_db]
    db_txt_buf[:] = txt_tmp[:num_db]
    logger.info(f"Gallery encoded in {time.perf_counter() - t_gallery:.1f}s")

    # ── Per-condition loop ───────────────────────────────────────────────────
    if eval_conditions is None:
        # Single condition
        test_ds = db_dataset   # already built
        run_condition(
            args, model, test_ds, cond_name=None,
            img_local_dim=img_local_dim, num_img_tokens=num_img_tokens,
            txt_local_dim=txt_local_dim, num_text_tokens=num_text_tokens,
            db_vis_descs=db_vis_buf, db_txt_descs=db_txt_buf,
            device=device, log_dir=log_dir)
    else:
        all_ret_recalls = []
        all_rer_recalls = []
        all_cond_results = []

        for cond_idx, (cond_name, q_folder, q_csv) in enumerate(eval_conditions):
            banner = (f"\n{'='*75}\n  Condition: {cond_name} "
                      f"({cond_idx+1}/{len(eval_conditions)})\n"
                      f"  Queries: {q_folder}\n  CSV: {q_csv}\n{'='*75}")
            logger.info(banner)
            print(banner)

            test_ds = TestDataset(
                args.database_folder, q_folder, q_csv,
                args.image_root, mean_std=mean_std,
                positive_dist_threshold=args.positive_dist_threshold,
                image_size=args.image_size, use_labels=args.use_labels)
            logger.info(f"  Queries: {test_ds.num_queries}")

            ret_rec, ret_str, rer_rec, rer_str = run_condition(
                args, model, test_ds, cond_name=cond_name,
                img_local_dim=img_local_dim, num_img_tokens=num_img_tokens,
                txt_local_dim=txt_local_dim, num_text_tokens=num_text_tokens,
                db_vis_descs=db_vis_buf, db_txt_descs=db_txt_buf,
                device=device, log_dir=log_dir)

            if ret_rec is not None:
                all_ret_recalls.append(ret_rec)
                all_cond_results.append((cond_name, ret_str, rer_str))
            if rer_rec is not None:
                all_rer_recalls.append(rer_rec)

        # ── Summary ─────────────────────────────────────────────────────────
        if all_ret_recalls:
            mean_ret     = np.mean(all_ret_recalls, axis=0)
            mean_ret_str = ", ".join([f"R@{v}: {r:.1f}"
                                      for v, r in zip(args.recall_values, mean_ret)])
            header = (f"\n{'='*75}\n  Summary ({len(eval_conditions)} conditions)"
                      f"\n{'='*75}")
            logger.info(header); print(header)
            for c_name, r_str, rr_str in all_cond_results:
                line = (f"  {c_name:<12} | Retrieval: {r_str}"
                        + (f" | Reranking: {rr_str}" if rr_str else ""))
                logger.info(line); print(line)
            sep = f"{'-'*75}"
            logger.info(sep); print(sep)

            model_path = args.lora_path if args.lora_path else args.model_path
            if all_rer_recalls:
                mean_rer     = np.mean(all_rer_recalls, axis=0)
                mean_rer_str = ", ".join([f"R@{v}: {r:.1f}"
                                          for v, r in zip(args.recall_values, mean_rer)])
                mean_line = (f"  {'MEAN':<12} | Retrieval: {mean_ret_str} "
                             f"| Reranking: {mean_rer_str}")
                logger.info(mean_line); print(mean_line)
                with open("eval_vpr_results.csv", "a") as f:
                    f.write(f"{model_path},{args.model_name} [Mean],"
                            f"Retrieval: {mean_ret_str} | "
                            f"Reranking: {mean_rer_str}\n")
            else:
                mean_line = f"  {'MEAN':<12} | Retrieval: {mean_ret_str}"
                logger.info(mean_line); print(mean_line)
                with open("eval_vpr_results.csv", "a") as f:
                    f.write(f"{model_path},{args.model_name} [Mean],"
                            f"{mean_ret_str}\n")
            footer = f"{'='*75}\n"
            logger.info(footer); print(footer)

    # ── Timing summary ───────────────────────────────────────────────────────
    timing_str = TIMING.summary()
    logger.info(timing_str)
    print(timing_str)


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)