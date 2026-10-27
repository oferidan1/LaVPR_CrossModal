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
def encode_global(model, dataloader, total_len, encoder_dim, device, bfloat16=False,
                  is_image=True, cross_modal=3, desc="Encoding"):
    """
    Returns two arrays of shape (total_len, encoder_dim):
      vision_descriptors  – image global embeddings  (zeros for query slots)
      text_descriptors    – text  global embeddings  (zeros for db slots)
    Only the relevant modality is filled for each split.
    """
    vision_descriptors = np.zeros((total_len, encoder_dim), dtype=np.float32)
    text_descriptors   = np.zeros((total_len, encoder_dim), dtype=np.float32)

    for images, indices, texts in tqdm(dataloader, desc=desc):
        if bfloat16:
            images = images.bfloat16()

        if cross_modal >= 2:
            # encode_single returns (global_img, global_txt, img_local, txt_local, ...)
            # We discard local tokens here.
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

    return vision_descriptors, text_descriptors


# ---------------------------------------------------------------------------
# Stage 2 – FAISS retrieval
# ---------------------------------------------------------------------------

def faiss_search(db_vecs: np.ndarray, q_vecs: np.ndarray,
                 top_k: int) -> tuple[np.ndarray, np.ndarray]:
    db_vecs = normalize(db_vecs)
    q_vecs  = normalize(q_vecs)
    index   = faiss.IndexFlatIP(db_vecs.shape[1])
    index.add(db_vecs)
    scores, preds = index.search(q_vecs, top_k)
    return scores, preds   # (num_q, top_k)


# ---------------------------------------------------------------------------
# Stage 3+4 – lazy local encoding + reranking
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_local_tokens(model, dataset, db_indices: np.ndarray,
                        img_local_dim: int, num_img_tokens: int,
                        device, bfloat16=False, batch_size=32):
    """
    Encode a specific set of database images and return their local tokens.

    Parameters
    ----------
    db_indices : 1-D array of database indices to encode (e.g. top-K for one query)

    Returns
    -------
    img_local : np.ndarray  shape (len(db_indices), num_img_tokens, img_local_dim)
    """
    subset     = Subset(dataset, db_indices.tolist())
    loader     = DataLoader(subset, batch_size=batch_size, num_workers=4, shuffle=False)
    collected  = []

    for images, _, texts in loader:
        if bfloat16:
            images = images.bfloat16()
        out = model.encode_single(images.to(device), texts)
        # out[2] = img_local  shape (B, N_patches, D)
        img_local = out[2]
        if img_local is None:
            raise RuntimeError("Model returned no local image tokens. "
                               "Make sure cross_modal >= 2.")
        collected.append(img_local.cpu().float().numpy())

    return np.concatenate(collected, axis=0)   # (K, N_patches, D)


@torch.no_grad()
def encode_query_local_tokens(model, dataset, q_ds_indices: list[int],
                               txt_local_dim: int, num_text_tokens: int,
                               device, bfloat16=False, batch_size=32):
    """
    Encode a batch of queries and return their local text tokens + attention masks.

    Returns
    -------
    txt_local   : np.ndarray  (len(q_ds_indices), num_text_tokens, txt_local_dim)
    txt_attents : np.ndarray  (len(q_ds_indices), num_text_tokens)  int64
    """
    subset    = Subset(dataset, q_ds_indices)
    loader    = DataLoader(subset, batch_size=batch_size, num_workers=4, shuffle=False)
    txt_parts, att_parts = [], []

    for images, _, texts in loader:
        if bfloat16:
            images = images.bfloat16()
        out = model.encode_single(images.to(device), texts)
        # out[3] = txt_local, out[5] = attention_mask
        txt_local, attn = out[3], out[5]
        if txt_local is None:
            raise RuntimeError("Model returned no local text tokens.")
        txt_parts.append(txt_local.cpu().float().numpy())
        if attn is not None:
            att_parts.append(attn.cpu().long().numpy())
        else:
            att_parts.append(np.ones(txt_local.shape[:2], dtype=np.int64))

    return (np.concatenate(txt_parts, axis=0),
            np.concatenate(att_parts, axis=0))


@torch.no_grad()
def rerank_topk_batched(
    model,
    dataset,
    predictions:        np.ndarray,   # (num_q, K)  stage-1 predictions
    num_database:       int,
    img_local_dim:      int,
    num_img_tokens:     int,
    txt_local_dim:      int,
    num_text_tokens:    int,
    vision_descriptors: np.ndarray,   # (total, D)
    text_descriptors:   np.ndarray,   # (total, D)
    max_rerank_k:       int   = 20,
    query_batch:        int   = 8,    # queries per outer step
    encode_batch:       int   = 128,  # images/texts per DataLoader batch
    alpha:              float = 0.0,
    device:             str   = "cuda",
    bfloat16:           bool  = False,
) -> np.ndarray:
    """
    Batched lazy reranking — encodes Q queries and their unique top-K DB
    candidates together in one DataLoader pass per outer step, so GPU
    utilisation stays high and DataLoader overhead is amortised over Q×K items.

    Speedup over the per-query version
    -----------------------------------
    Old: num_q  DataLoader launches  (1 query  at a time → tiny batches, lots of overhead)
    New: num_q // query_batch  launches  (Q queries → Q×K unique DB images in one shot)

    Memory per outer step (worst case, no deduplication)
    ------------------------------------------------------
      Q × K × N_patches × D_img × 4 bytes
      e.g. Q=8, K=20, N=256, D=768  →  ~120 MB  — easily fits in GPU VRAM
    """
    reranker = model.reranker_network
    if reranker is None:
        raise RuntimeError("model.reranker_network is None.")
    reranker.eval()
    w_dtype = next(reranker.parameters()).dtype

    num_q  = predictions.shape[0]
    n_cand = min(max_rerank_k, predictions.shape[1])
    reranked = predictions.copy()

    logger.info(f"Batched lazy reranking: {num_q} queries × top-{n_cand} "
                f"(query_batch={query_batch}, encode_batch={encode_batch}, "
                f"alpha={alpha}, dtype={w_dtype})")

    def _encode_indices(ds_indices: list[int]):
        """Encode a flat list of dataset indices; return (img_local, txt_local, txt_attn)."""
        subset = Subset(dataset, ds_indices)
        loader = DataLoader(subset, batch_size=encode_batch,
                            num_workers=4, shuffle=False, pin_memory=True)
        img_parts, txt_parts, att_parts = [], [], []
        for images, _, texts in loader:
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

        # Text tokens can have different L per batch (tokenizer pads to the
        # longest sequence *within* each batch).  Pad everything to the global
        # maximum length before concatenating across batches.
        max_L  = max(t.shape[1] for t in txt_parts)
        D_txt  = txt_parts[0].shape[2]
        txt_padded = []
        att_padded = []
        for t, a in zip(txt_parts, att_parts):
            pad = max_L - t.shape[1]
            if pad > 0:
                t = torch.cat([t, torch.zeros(t.shape[0], pad, D_txt)], dim=1)
                a = torch.cat([a, torch.zeros(a.shape[0], pad, dtype=torch.long)], dim=1)
            txt_padded.append(t)
            att_padded.append(a)

        return (torch.cat(img_parts,  dim=0),   # (N, N_p, D_img)
                torch.cat(txt_padded, dim=0),   # (N, max_L, D_txt)
                torch.cat(att_padded, dim=0))   # (N, max_L)

    def _z(x: np.ndarray) -> np.ndarray:
        std = x.std()
        return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else x

    for q_start in tqdm(range(0, num_q, query_batch), desc="Batched rerank"):
        q_end = min(q_start + query_batch, num_q)
        bsz   = q_end - q_start

        # Dataset indices for this batch of queries
        q_ds_idxs = [num_database + q for q in range(q_start, q_end)]

        # DB candidate indices: (bsz, K)
        cand_matrix = predictions[q_start:q_end, :n_cand]   # (bsz, K)

        # ----------------------------------------------------------------
        # Deduplicate DB candidates across queries in this batch so each
        # unique image is encoded only once even when queries share candidates.
        # ----------------------------------------------------------------
        flat_db_idxs          = cand_matrix.ravel()           # (bsz*K,)
        unique_db_idxs, inv   = np.unique(flat_db_idxs, return_inverse=True)
        # inv : (bsz*K,) maps flat position → position in unique_db_idxs

        # Encode unique DB images + all queries in ONE DataLoader call each.
        # (Two separate calls keeps DB images and query texts cleanly separated.)
        all_img_local, _, _         = _encode_indices(unique_db_idxs.tolist())
        _, all_txt_local, all_attn  = _encode_indices(q_ds_idxs)
        # all_img_local : (|unique_db|, N_p, D_img)
        # all_txt_local : (bsz,        L,   D_txt)
        # all_attn      : (bsz,        L)

        # ----------------------------------------------------------------
        # Score every (query, candidate) pair with the reranker
        # ----------------------------------------------------------------
        local_scores = np.zeros((bsz, n_cand), dtype=np.float32)

        for b in range(bsz):
            # Gather this query's K image tokens via the inverse index
            cand_inv   = inv[b * n_cand : (b + 1) * n_cand]   # (K,) → unique positions
            img_cands  = all_img_local[cand_inv]               # (K, N_p, D_img)

            # Trim text padding
            real_len  = max(int(all_attn[b].sum().item()), 1)
            txt_local = all_txt_local[b, :real_len].unsqueeze(0)   # (1, real_len, D_txt)
            txt_mask  = all_attn[b, :real_len].unsqueeze(0).bool() # (1, real_len)

            K     = img_cands.shape[0]
            L     = txt_local.shape[1]
            D_txt = txt_local.shape[2]

            img_t = img_cands.to(device).to(w_dtype)                        # (K, N_p, D_img)
            txt_t = txt_local.expand(K, -1, -1).contiguous().to(device).to(w_dtype)  # (K, L, D_txt)
            msk_t = txt_mask.expand(K, -1).contiguous().to(device)          # (K, L)

            _, scores = reranker(img_t, txt_t, msk_t)
            local_scores[b] = scores.float().cpu().numpy()

        # ----------------------------------------------------------------
        # Optional global cosine fusion
        # ----------------------------------------------------------------
        if alpha > 0.0:
            for b in range(bsz):
                q_ds_idx   = q_ds_idxs[b]
                q_global   = text_descriptors[q_ds_idx]
                db_globals = vision_descriptors[cand_matrix[b]]     # (K, D)
                q_norm     = q_global  / (np.linalg.norm(q_global)  + 1e-8)
                db_norm    = db_globals / (np.linalg.norm(db_globals, axis=1, keepdims=True) + 1e-8)
                global_sim = db_norm @ q_norm                        # (K,)
                local_scores[b] = alpha * _z(global_sim) + (1.0 - alpha) * _z(local_scores[b])

        # ----------------------------------------------------------------
        # Reorder shortlists
        # ----------------------------------------------------------------
        for b in range(bsz):
            order = np.argsort(-local_scores[b])
            reranked[q_start + b, :n_cand] = cand_matrix[b][order]

    logger.info("Batched reranking complete.")
    return reranked


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def compute_recalls(predictions: np.ndarray, positives_per_query: list,
                    recall_values: list[int]) -> tuple[np.ndarray, str]:
    recalls = np.zeros(len(recall_values))
    for q_idx, preds in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.isin(preds[:n], positives_per_query[q_idx])):
                recalls[i:] += 1
                break
    recalls = recalls / len(predictions) * 100
    r_str = ", ".join([f"R@{v}: {r:.1f}" for v, r in zip(recall_values, recalls)])
    return recalls, r_str


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

    # ---- Dataset normalization stats ----
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

    # ---- Dataset paths ----
    if args.data in ('sfxl',):
        args.database_folder = "/home/shared/datasets/SF-XL/processed/test/database"
        args.queries_folder  = "/home/shared/datasets/SF-XL/processed/test/queries_v1"
        args.image_root      = "/home/shared/datasets/SF-XL/processed/test"
        args.queries_csv     = "datasets/descriptions/sfxl_queries_v1.csv"
    elif args.data in ('pitts', 'pitts_test', 'pitts30k'):
        args.database_folder = "/home/shared/datasets/pitts30k/images/test/database"
        args.queries_folder  = "/home/shared/datasets/pitts30k/images/test/queries"
        args.image_root      = "/home/shared/datasets/pitts30k/images/test"
        args.queries_csv     = "datasets/descriptions/pitts30k_test_descriptions.csv"
    elif args.data in ('msls', 'msls_val'):
        args.database_folder = "/home/shared/datasets/msls/val/database"
        args.queries_folder  = "/home/shared/datasets/msls/val/query"
        args.image_root      = "/home/shared/datasets/msls/val/"
        args.queries_csv     = "datasets/descriptions/msls_val_descriptions.csv"
    # add more datasets as needed

    # ---- Model ----
    model = LaVPR_wrapper(args)
    device = args.device
    logger.info(f"Encoder dim: {model.encoder_dim}")

    # ---- Local token dimensions (cross_modal=3: DINOv2+BGE) ----
    num_img_tokens  = model.img_local_tokens   # e.g. 256
    num_text_tokens = model.txt_local_tokens   # e.g. 512
    img_local_dim   = model.img_local_dim      # e.g. 768
    txt_local_dim   = model.txt_local_dim      # e.g. 1024

    # ---- Build dataset ----
    test_ds = TestDataset(
        args.database_folder, args.queries_folder, args.queries_csv,
        args.image_root, mean_std=mean_std,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size, use_labels=args.use_labels)
    logger.info(f"Dataset: {test_ds}")

    num_db    = test_ds.num_database
    num_q     = test_ds.num_queries
    total_len = len(test_ds)
    logger.info(f"DB: {num_db}  Queries: {num_q}  Total: {total_len}")

    max_top_k = max(args.recall_values)

    # =========================================================================
    # STAGE 1 — Global encoding (no local tokens → tiny memory footprint)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 1: Global encoding (no local tokens)")
    logger.info(f"  Memory for global vectors: "
                f"{total_len * model.encoder_dim * 4 / 1e9:.2f} GB")
    logger.info("=" * 60)

    with torch.inference_mode():
        # --- Database ---
        db_subset  = Subset(test_ds, list(range(num_db)))
        db_loader  = DataLoader(db_subset,  num_workers=args.num_workers,
                                batch_size=args.batch_size, shuffle=False)
        vis_descs, txt_descs = encode_global(
            model, db_loader, total_len, model.encoder_dim, device,
            bfloat16=args.bfloat16, is_image=args.is_image,
            cross_modal=args.cross_modal, desc="Encoding DB (global)")

        # --- Queries ---
        q_subset  = Subset(test_ds, list(range(num_db, total_len)))
        q_loader  = DataLoader(q_subset, num_workers=args.num_workers,
                               batch_size=args.batch_size, shuffle=False)
        vis_descs_q, txt_descs_q = encode_global(
            model, q_loader, total_len, model.encoder_dim, device,
            bfloat16=args.bfloat16, is_image=args.is_image,
            cross_modal=args.cross_modal, desc="Encoding Queries (global)")

    # Merge into one array (db slots have vis_descs, query slots have txt_descs)
    vis_descs[num_db:] = vis_descs_q[num_db:]
    txt_descs[num_db:] = txt_descs_q[num_db:]

    # =========================================================================
    # STAGE 2 — FAISS top-K retrieval
    # =========================================================================
    logger.info("Stage 2: FAISS retrieval")

    if args.cross_modal and not args.is_image:
        db_vecs = txt_descs[:num_db]
    else:
        db_vecs = vis_descs[:num_db]
    q_vecs = txt_descs[num_db:]

    scores, retrieval_preds = faiss_search(db_vecs, q_vecs, max_top_k)
    logger.info(f"  Retrieval done. Predictions shape: {retrieval_preds.shape}")

    # =========================================================================
    # STAGE 3+4 — Lazy local-token extraction + reranking
    # =========================================================================
    reranking_preds   = retrieval_preds.copy()
    reranking_enabled = False

    do_rerank = (getattr(args, 'reranker', False) and args.cross_modal == 3
                 and model.reranker_network is not None)

    if do_rerank:
        reranking_enabled = True
        max_rerank_k      = min(args.max_rerank, max_top_k)
        logger.info("=" * 60)
        logger.info(f"Stage 3+4: Lazy local encoding + LaVPR reranking (top-{max_rerank_k})")
        logger.info(f"  Peak extra memory per query ≈ "
                    f"{max_rerank_k * num_img_tokens * img_local_dim * 4 / 1e6:.1f} MB")
        logger.info("=" * 60)

        reranking_preds = rerank_topk_batched(
            model           = model,
            dataset         = test_ds,
            predictions     = retrieval_preds,
            num_database    = num_db,
            img_local_dim   = img_local_dim,
            num_img_tokens  = num_img_tokens,
            txt_local_dim   = txt_local_dim,
            num_text_tokens = num_text_tokens,
            vision_descriptors = vis_descs,
            text_descriptors   = txt_descs,
            max_rerank_k    = max_rerank_k,
            query_batch     = getattr(args, 'rerank_query_batch', 8),
            encode_batch    = getattr(args, 'rerank_batch', 128),
            alpha           = getattr(args, 'rerank_alpha', 0.0),
            device          = device,
            bfloat16        = args.bfloat16,
        )

    # =========================================================================
    # STAGE 5 — Recall metrics
    # =========================================================================
    if args.use_labels:
        positives = test_ds.get_positives()

        ret_recalls, ret_str = compute_recalls(retrieval_preds, positives, args.recall_values)
        logger.info(f"Retrieval : {ret_str}")
        print(f"Retrieval : {ret_str}")

        if reranking_enabled:
            rer_recalls, rer_str = compute_recalls(reranking_preds, positives, args.recall_values)
            logger.info(f"Reranking : {rer_str}")
            print(f"Reranking : {rer_str}")

        model_path = args.lora_path if args.lora_path else args.model_path
        with open("eval_vpr_results.csv", "a") as f:
            if reranking_enabled:
                f.write(f"{model_path},{args.model_name},"
                        f"Retrieval: {ret_str} | Reranking: {rer_str}\n")
            else:
                f.write(f"{model_path},{args.model_name},{ret_str}\n")


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)