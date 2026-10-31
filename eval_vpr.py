import argparse
import eval_parser
from argparse import Namespace
import sys
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm
from model.LaVPR_wrapper import LaVPR_wrapper
import os
from dataloaders.test_dataset import TestDataset
from dataloaders.MapillaryTestDataset import MSLSTest
import utils.visualizations as visualizations
from sklearn.decomposition import PCA
import asyncio
import base64
import json
import mimetypes
import re
from openai import AsyncOpenAI
import torch.nn.functional as F
import csv


def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _maybe(mod, name, x):
    layer = getattr(mod, name, None)
    return layer(x) if layer is not None else x


def _z(x):
    return (x - x.mean()) / x.std().clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Timing stats accumulator
# ---------------------------------------------------------------------------

class TimingStats:
    """Accumulate wall-clock timings and print a summary table."""

    def __init__(self):
        self._data = {}   # name -> list of (n_items, elapsed_s)

    def record(self, name: str, n_items: int, elapsed_s: float):
        self._data.setdefault(name, []).append((n_items, elapsed_s))

    # What "items" means per stage — ms/item is always per candidate image:
    #   encode_batch          → images in that dataloader batch
    #   encode_gallery_total  → total gallery images
    #   encode_queries_total  → total query images
    #   faiss_stage1          → queries searched
    #   rerank_*_batch        → candidate images scored (bsz × K)
    #   rerank_jev_batch      → (query, candidate) pairs in that chunk
    #   filip_retrieval_*     → queries processed (each queries full gallery)

    def summary(self) -> str:
        if not self._data:
            return "(no timing data)"
        lines = [
            f"\n{'='*72}",
            f"  {'Stage':<30}  {'Calls':>5}  {'Items':>8}  "
            f"{'Total(s)':>9}  {'ms/item':>8}",
            f"  {'(items = images for encode, queries for rerank)':<68}",
            f"  {'-'*30}  {'-'*5}  {'-'*8}  {'-'*9}  {'-'*8}",
        ]
        for name, records in self._data.items():
            n_calls = len(records)
            n_items = sum(r[0] for r in records)
            total_s = sum(r[1] for r in records)
            ms_per  = (total_s / n_items * 1000) if n_items > 0 else 0
            lines.append(
                f"  {name:<30}  {n_calls:>5}  {n_items:>8}  "
                f"{total_s:>9.2f}  {ms_per:>8.2f}")
        lines.append(f"{'='*72}\n")
        return "\n".join(lines)


TIMING = TimingStats()


# ---------------------------------------------------------------------------
# Stage-1 retrieval cache: save / load top-K results as CSV
# ---------------------------------------------------------------------------
STAGE1_TOPK_SAVE = 50


def save_stage1_cache(save_path, predictions, scores,
                      query_filenames, db_filenames,
                      topk=STAGE1_TOPK_SAVE):
    k      = min(topk, predictions.shape[1])
    db_arr = np.array(db_filenames)
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['query_path']
        for r in range(1, k + 1):
            header += [f'db_path_{r}', f'score_{r}']
        writer.writerow(header)
        for q_idx, q_path in enumerate(query_filenames):
            row = [q_path]
            for r in range(k):
                db_idx = int(predictions[q_idx, r])
                row.append(db_arr[db_idx])
                row.append(f'{scores[q_idx, r]:.6f}')
            writer.writerow(row)
    logger.info(
        f"Stage-1 cache saved → {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")


def load_stage1_cache(save_path, db_filenames):
    db_path_to_idx = {p: i for i, p in enumerate(db_filenames)}
    query_filenames  = []
    top_db_filenames = []
    predictions_list = []
    scores_list      = []
    with open(save_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        k      = (len(header) - 1) // 2
        for row in reader:
            query_filenames.append(row[0])
            db_paths, scs, idxs = [], [], []
            for r in range(k):
                db_path = row[1 + r * 2]
                sc      = float(row[2 + r * 2])
                db_idx  = db_path_to_idx.get(db_path, -1)
                db_paths.append(db_path)
                scs.append(sc)
                idxs.append(db_idx)
            top_db_filenames.append(db_paths)
            predictions_list.append(idxs)
            scores_list.append(scs)
    predictions = np.array(predictions_list, dtype=np.int64)
    scores      = np.array(scores_list,      dtype=np.float32)
    n_missing   = (predictions == -1).sum()
    if n_missing:
        logger.warning(
            f"Stage-1 cache: {n_missing} db paths not found in current dataset.")
    logger.info(
        f"Stage-1 cache loaded ← {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")
    return {
        'query_filenames':  query_filenames,
        'top_db_filenames': top_db_filenames,
        'predictions':      predictions,
        'scores':           scores,
    }


def save_rerank_cache(save_path, predictions, scores,
                      query_filenames, db_filenames,
                      topk=STAGE1_TOPK_SAVE):
    k      = min(topk, predictions.shape[1])
    db_arr = np.array(db_filenames)
    if scores is None:
        scores = np.tile(
            np.arange(k, 0, -1, dtype=np.float32), (predictions.shape[0], 1))
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['query_path']
        for r in range(1, k + 1):
            header += [f'db_path_{r}', f'score_{r}']
        writer.writerow(header)
        for q_idx, q_path in enumerate(query_filenames):
            row = [q_path]
            for r in range(k):
                db_idx = int(predictions[q_idx, r])
                row.append(db_arr[db_idx])
                row.append(f'{scores[q_idx, r]:.6f}')
            writer.writerow(row)
    logger.info(
        f"Stage-2 cache saved → {save_path}  "
        f"(queries={len(query_filenames)}, topk={k})")


def verify_cache_query_order(cache, test_ds):
    cached  = cache['query_filenames']
    current = test_ds.images_paths[test_ds.num_database:]
    if len(cached) != len(current):
        logger.warning(
            f"Cache has {len(cached)} queries but dataset has {len(current)} "
            "— predictions may be misaligned!")
        return
    n_bad = sum(c != d for c, d in zip(cached, current))
    if n_bad:
        logger.warning(
            f"Cache query filenames: {n_bad}/{len(cached)} mismatches.")
    else:
        logger.info("Cache query filenames match current dataset ✓")


# ---------------------------------------------------------------------------

def encode_batch(model, args, images, texts, indices, all_descriptors,
                 vision_descriptors, text_descriptors, img_local_descs,
                 text_local_descs, text_tokens_all, text_attentions):
    if args.bfloat16:
        images = images.bfloat16()

    t0 = time.perf_counter()

    if args.cross_modal <= 1:
        if args.is_image:
            image_features = model.encode_image(images.to(args.device))
            vision_descriptors[indices.numpy(), :] = image_features.cpu().float().numpy()
        text_features = model.encode_text(texts)
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
    else:
        descriptors, text_features, img_local, text_local, text_tokens, text_attention_mask = \
            model.encode_single(images.to(args.device), texts)
        if args.is_image and descriptors is not None:
            vision_descriptors[indices.numpy(), :] = descriptors.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
        if img_local is not None:
            N = min(img_local.shape[1], img_local_descs.shape[1])
            img_local_descs[indices.numpy(), :N, :] = img_local.cpu().float().numpy()[:, :N, :]
        if text_local is not None:
            L = min(text_local.shape[1], text_local_descs.shape[1])
            text_local_descs[indices.numpy(), :L, :] = text_local.cpu().float().numpy()[:, :L, :]
        if text_attention_mask is not None:
            L = min(text_attention_mask.shape[1], text_attentions.shape[1])
            text_attentions[indices.numpy(), :L] = text_attention_mask.cpu().long().numpy()[:, :L]

    elapsed = time.perf_counter() - t0
    TIMING.record('encode_batch', len(indices), elapsed)


def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors,
                            queries_descriptors, max_results):
    t0 = time.perf_counter()
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    database_descriptors = normlize_features(database_descriptors)
    queries_descriptors  = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    elapsed = time.perf_counter() - t0
    TIMING.record('faiss_stage1', len(queries_descriptors), elapsed)
    return scores, predictions


# ---------------------------------------------------------------------------
# FILIP: full-gallery retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def filip_full_retrieval(model, test_ds, vision_descriptors, text_descriptors,
                         img_local_descs, text_local_desc, text_tokens_all,
                         max_results=100, db_chunk=2048, alpha=0.0,
                         db_on_gpu=True, device="cuda"):
    head = getattr(model.single_encoder, "filip_loss", None)
    if head is None:
        raise RuntimeError("Model has no filip_loss head.")
    head.to(device).eval()
    has_cls = getattr(model.single_encoder, "has_cls", True)
    use_idf = getattr(head, "use_idf", False)
    hdt     = next(head.parameters()).dtype
    num_db, num_q = test_ds.num_database, test_ds.num_queries
    logger.info(f"FILIP full-gallery retrieval over {num_db} database images")
    parts = []
    for s in tqdm(range(0, num_db, db_chunk), desc="projecting database"):
        e = min(s + db_chunk, num_db)
        v = torch.from_numpy(img_local_descs[s:e]).to(device)
        if has_cls:
            v = v[:, 1:]
        v = head.vision_proj(_maybe(head, "v_norm", v.to(hdt)))
        v = F.normalize(v, dim=-1).half()
        parts.append(v if db_on_gpu else v.cpu())
    db_v = torch.cat(parts, dim=0)
    del parts
    torch.cuda.empty_cache()
    db_g = F.normalize(torch.from_numpy(vision_descriptors[:num_db]).to(device).float(), dim=-1)
    q_g  = F.normalize(torch.from_numpy(text_descriptors[num_db:]).to(device).float(), dim=-1)
    topk        = min(max_results, num_db)
    predictions = np.zeros((num_q, topk), dtype=np.int64)
    out_scores  = np.zeros((num_q, topk), dtype=np.float32)
    for q in tqdm(range(num_q), desc="FILIP retrieval"):
        t0  = time.perf_counter()
        gi   = num_db + q
        raw  = torch.from_numpy(text_local_desc[gi]).to(device)
        mask = raw.abs().sum(-1) > 0
        if mask.sum() == 0:
            predictions[q] = np.arange(topk)
            continue
        raw = raw[mask].unsqueeze(0)
        t   = head.text_proj(_maybe(head, "t_norm", raw.to(hdt)))
        t   = F.normalize(t, dim=-1).squeeze(0).half()
        if use_idf:
            ids = torch.from_numpy(text_tokens_all[gi]).to(device)[mask]
            w   = head.idf_weights[ids.clamp(0, head.idf_weights.numel() - 1)].half()
        else:
            w = torch.ones(t.shape[0], device=device, dtype=torch.half)
        w = w / w.sum().clamp(min=1e-6)
        sc = torch.empty(num_db, device=device, dtype=torch.half)
        for s in range(0, num_db, db_chunk):
            v = db_v[s:s + db_chunk]
            if not db_on_gpu:
                v = v.to(device, non_blocking=True)
            sim = torch.einsum('nd,bmd->bnm', t, v)
            sc[s:s + v.shape[0]] = (sim.max(dim=-1).values * w).sum(-1)
            del sim
        s_final = sc.float()
        if alpha > 0:
            g       = (q_g[q].unsqueeze(0) * db_g).sum(-1)
            s_final = alpha * _z(g) + (1.0 - alpha) * _z(s_final)
        vals, idx       = torch.topk(s_final, topk)
        predictions[q]  = idx.cpu().numpy()
        out_scores[q]   = vals.cpu().numpy()
        TIMING.record('filip_retrieval_per_query', 1, time.perf_counter() - t0)
    del db_v
    torch.cuda.empty_cache()
    return out_scores, predictions


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------

def rerank_predictions(model, test_ds, predictions, vision_descriptors,
                       text_descriptors, img_local_descs, text_local_desc,
                       text_attentions=None, max_rerank_k=25, device="cuda"):
    logger.info(f"Reranking top-{max_rerank_k} candidates using Cross-Attention...")
    rerank_model = model.single_encoder
    rerank_model.to(device)
    rerank_model.eval()
    reranked_predictions      = predictions.copy()
    img_local_descs_tensor    = torch.from_numpy(img_local_descs).to(device)
    text_local_desc_tensor    = torch.from_numpy(text_local_desc).to(device)
    vision_descriptors_tensor = torch.from_numpy(vision_descriptors).to(device)
    text_descriptors_tensor   = torch.from_numpy(text_descriptors).to(device)
    text_attn_tensor = (torch.from_numpy(text_attentions).to(device)
                        if text_attentions is not None else None)
    if text_attn_tensor is None:
        logger.warning("rerank_predictions: no attention mask supplied.")
    n_cand = int(max_rerank_k)
    with torch.no_grad():
        for q_idx in tqdm(range(test_ds.num_queries), desc="Reranking queries"):
            t0 = time.perf_counter()
            actual_q_ds_idx  = test_ds.num_database + q_idx
            query_text_global = text_descriptors_tensor[actual_q_ds_idx].unsqueeze(0)
            raw_text_local    = text_local_desc_tensor[actual_q_ds_idx]
            if text_attn_tensor is not None:
                mask_row = text_attn_tensor[actual_q_ds_idx]
            else:
                mask_row = raw_text_local.any(dim=-1).long()
            if mask_row.sum() == 0:
                mask_row = torch.ones_like(mask_row)
            true_len         = int(mask_row.nonzero()[-1].item()) + 1
            query_text_local = raw_text_local[:true_len].unsqueeze(0)
            query_mask       = mask_row[:true_len].unsqueeze(0)
            candidate_db_indices = predictions[q_idx, :n_cand]
            B                    = len(candidate_db_indices)
            candidate_img_local  = img_local_descs_tensor[candidate_db_indices]
            candidate_img_global = vision_descriptors_tensor[candidate_db_indices]
            L, D_dim             = query_text_local.shape[1], query_text_local.shape[2]
            text_local_expanded  = query_text_local.expand(B, L, D_dim)
            text_global_expanded = query_text_global.expand(B, -1)
            text_mask_expanded   = query_mask.expand(B, L)
            if next(rerank_model.parameters()).dtype == torch.bfloat16:
                candidate_img_local  = candidate_img_local.bfloat16()
                text_local_expanded  = text_local_expanded.bfloat16()
                candidate_img_global = candidate_img_global.bfloat16()
                text_global_expanded = text_global_expanded.bfloat16()
            scores = rerank_model.cross_attn_classifier(
                candidate_img_local, text_local_expanded,
                candidate_img_global, text_global_expanded)
            scores = scores.float().cpu().numpy()
            reranked_order = np.argsort(-scores)
            reranked_predictions[q_idx, :n_cand] = candidate_db_indices[reranked_order]
            TIMING.record('rerank_cross_attn_per_query', n_cand, time.perf_counter() - t0)
    return reranked_predictions


@torch.no_grad()
def rerank_by_lavpr(
    model, test_ds, predictions, img_local_descs, text_local_desc,
    text_attentions, vision_descs=None, text_descs=None,
    max_rerank_k=20, rerank_batch=64, alpha=0.0, device='cuda',
) -> np.ndarray:
    reranker = model.reranker_network
    if reranker is None:
        raise RuntimeError("reranker_network is None.")
    reranker.eval()
    w_dtype = next(reranker.parameters()).dtype
    num_db  = test_ds.num_database
    num_q   = test_ds.num_queries
    n_cand  = min(max_rerank_k, predictions.shape[1])

    # Detect reranker's expected input dims from its projection layers.
    # reranker.img_proj / txt_proj are Linear(D_in, D_hidden).
    img_proj_in = reranker.img_proj.weight.shape[1]
    txt_proj_in = reranker.txt_proj.weight.shape[1]
    got_img_dim = img_local_descs.shape[-1]
    got_txt_dim = text_local_desc.shape[-1]

    logger.info(f"LaVPR reranker  img_proj expects {img_proj_in}  "
                f"got {got_img_dim}  |  "
                f"txt_proj expects {txt_proj_in}  got {got_txt_dim}")

    if got_img_dim != img_proj_in:
        raise RuntimeError(
            f"rerank_by_lavpr: img local dim mismatch — "
            f"reranker expects {img_proj_in}, buffer has {got_img_dim}. "
            "Make sure --cross_modal=3 and the model was trained with the "
            "same img_local_dim as the reranker (check model.img_local_tokens "
            "and model.embeds_dim).")
    if got_txt_dim != txt_proj_in:
        raise RuntimeError(
            f"rerank_by_lavpr: text local dim mismatch — "
            f"reranker expects {txt_proj_in}, buffer has {got_txt_dim}. "
            "The reranker was likely trained with raw BGE dim "
            f"({txt_proj_in}) but the eval buffer holds projected "
            f"embeds_dim ({got_txt_dim}). Pass the raw local text "
            "descriptors (before image_proj_local / text_proj_local).")

    logger.info(f"LaVPR reranking top-{n_cand} for {num_q} queries  "
                f"(batch={rerank_batch}  alpha={alpha}  dtype={w_dtype})")
    reranked = predictions.copy()

    for q_start in tqdm(range(0, num_q, rerank_batch), desc="LaVPR rerank"):
        t0    = time.perf_counter()
        q_end = min(q_start + rerank_batch, num_q)
        bsz   = q_end - q_start
        q_ds_idxs = [num_db + q for q in range(q_start, q_end)]

        txt_locals, txt_masks = [], []
        for i in q_ds_idxs:
            full_mask = text_attentions[i]
            real_len  = max(int(full_mask.sum()), 1)
            txt_locals.append(text_local_desc[i][:real_len])
            txt_masks .append(full_mask         [:real_len])

        max_L = max(t.shape[0] for t in txt_locals)
        D_txt = txt_locals[0].shape[1]
        txt_np = np.zeros((bsz, max_L, D_txt), dtype=np.float32)
        msk_np = np.zeros((bsz, max_L),        dtype=np.float32)
        for b, (tl, tm) in enumerate(zip(txt_locals, txt_masks)):
            L = tl.shape[0]; txt_np[b, :L] = tl; msk_np[b, :L] = tm

        txt_t = torch.from_numpy(txt_np).to(device)
        msk_t = torch.from_numpy(msk_np).bool().to(device)

        cand_idx = predictions[q_start:q_end, :n_cand]
        img_np   = np.stack([img_local_descs[cand_idx[b]]
                             for b in range(bsz)], axis=0)
        img_t    = torch.from_numpy(img_np).to(device)

        K, N_p, D_img = n_cand, img_t.shape[2], img_t.shape[3]
        L_   = txt_t.shape[1]
        D_t_ = txt_t.shape[2]

        imgs_flat = img_t.reshape(bsz * K, N_p, D_img).to(w_dtype)
        txt_flat  = (txt_t.unsqueeze(1)
                         .expand(-1, K, -1, -1)
                         .reshape(bsz * K, L_, D_t_)
                         .contiguous().to(w_dtype))
        msk_flat  = (msk_t.unsqueeze(1)
                          .expand(-1, K, -1)
                          .reshape(bsz * K, L_)
                          .contiguous())

        _, scores_flat = reranker(imgs_flat, txt_flat, msk_flat)
        local_scores   = scores_flat.float().view(bsz, K).cpu().numpy()

        if alpha > 0.0 and vision_descs is not None and text_descs is not None:
            for b in range(bsz):
                q_ds_idx   = num_db + q_start + b
                q_global   = text_descs[q_ds_idx]
                db_globals = vision_descs[cand_idx[b]]
                q_norm     = q_global   / (np.linalg.norm(q_global) + 1e-8)
                db_norm    = db_globals / (np.linalg.norm(
                    db_globals, axis=1, keepdims=True) + 1e-8)
                global_sim = db_norm @ q_norm
                def _zn(x):
                    std = x.std()
                    return (x - x.mean()) / (std + 1e-8) if std > 1e-8 else x
                local_scores[b] = (alpha * _zn(global_sim)
                                   + (1.0 - alpha) * _zn(local_scores[b]))

        for b in range(bsz):
            order = np.argsort(-local_scores[b])
            reranked[q_start + b, :n_cand] = cand_idx[b][order]

        TIMING.record('rerank_lavpr_batch', bsz * n_cand, time.perf_counter() - t0)

    logger.info("LaVPR reranking complete.")
    return reranked


def rerank_by_jev(test_ds, predictions, max_rerank_k=25, batch_size=64,
                  match_weight=0.6, device="cuda"):
    from huggingface_hub import snapshot_download
    from PIL import Image
    logger.info(f"Reranking top-{max_rerank_k} candidates using JEV...")
    path = snapshot_download("Cloudflare/clef-flash")
    sys.path.insert(0, path)
    from joint_schema_model import collate_records, encode_record, load_release_model
    jev_model, jev_processor = load_release_model(path, device=device)
    jev_model.eval()
    QUESTION_SCHEMA = {
        "place_match": {
            "type": "choice",
            "instructions": "Does the image show the same place as the query description?",
            "criteria": {
                "match": "Same location.", "no_match": "Different location.",
                "uncertain": "Cannot determine.",
            },
        },
        "confidence": {
            "type": "noul",
            "instructions": "The image and query description refer to the exact same geographic location.",
        },
    }
    db_paths = np.array(test_ds.images_paths[:test_ds.num_database])
    q_texts  = test_ds.descriptions[test_ds.num_database:]
    n_cand   = min(max_rerank_k, predictions.shape[1])
    reranked = predictions.copy()
    all_pairs = []
    for q_idx in range(test_ds.num_queries):
        cand_indices = predictions[q_idx, :n_cand]
        for rank, db_idx in enumerate(cand_indices):
            all_pairs.append((q_idx, rank, q_texts[q_idx], db_paths[db_idx]))
    all_scores = {}
    for start in tqdm(range(0, len(all_pairs), batch_size), desc="JEV rerank"):
        t0    = time.perf_counter()
        chunk = all_pairs[start: start + batch_size]
        records, keys = [], []
        for q_idx, rank, query_text, img_path in chunk:
            try:
                image  = Image.open(img_path).convert("RGB")
                record = {"state": {"query": query_text}, "images": [image],
                          "questions": QUESTION_SCHEMA}
                encoded = encode_record(jev_processor.tokenizer, record,
                                        processor=jev_processor)
                records.append(encoded); keys.append((q_idx, rank))
            except Exception as e:
                logger.warning(f"JEV: failed {img_path}: {e}")
                all_scores[(q_idx, rank)] = 0.0
        if not records:
            continue
        batch = collate_records(records, jev_processor.tokenizer.pad_token_id,
                                torch.device(device))
        with torch.inference_mode():
            batch_logits = jev_model(batch)
        for (q_idx, rank), encoded, logits in zip(keys, records, batch_logits):
            results = {}
            for question, q_logits in zip(encoded.questions, logits):
                probs = q_logits.float().softmax(-1).tolist()
                results[question.question_id] = dict(zip(question.option_ids, probs))
            match_prob    = results["place_match"].get("match", 0.0)
            location_prob = list(results["confidence"].values())[0]
            all_scores[(q_idx, rank)] = (match_weight*match_prob
                                         + (1-match_weight)*location_prob)
        TIMING.record('rerank_jev_batch', len(chunk), time.perf_counter() - t0)
    for q_idx in range(test_ds.num_queries):
        cand_indices = predictions[q_idx, :n_cand]
        sc    = np.array([all_scores.get((q_idx, r), 0.0) for r in range(n_cand)])
        order = np.argsort(-sc)
        reranked[q_idx, :n_cand] = cand_indices[order]
    logger.info("JEV reranking complete.")
    return reranked


def rerank_by_mllm(image_paths, target_text, predictions,
                   max_concurrent_requests=8, debug=True):
    if len(image_paths) != len(predictions):
        raise ValueError(f"Length mismatch.")

    async def encode_image_async(image_path):
        def _read_and_b64(path):
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type: mime_type = "image/jpeg"
            with open(path, "rb") as f:
                return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"
        return await asyncio.to_thread(_read_and_b64, image_path)

    async def evaluate_single_image(client, semaphore, orig_rank, pred_index, path):
        async with semaphore:
            if not os.path.exists(path):
                return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}
            try:
                img_data_url = await encode_image_async(path)
                prompt_text = f"""You are an expert visual place recognition and OCR assistant.
Analyze this image to evaluate if it shows the location described by the target text query.
TARGET QUERY: "{target_text}"
SCORING CRITERIA (0 to 100): 85-100 strong match, 50-84 partial, 0-49 no match.
OUTPUT FORMAT: Respond ONLY with a JSON object: {{"score": <number 0-100>, "reason": "<brief justification>"}}"""
                response = await client.chat.completions.create(
                    model="neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt_text.strip()},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ]}], temperature=0.0, max_tokens=80)
                raw = response.choices[0].message.content.strip()
                score = 0.0
                try:
                    clean = re.sub(r"```[a-zA-Z]*", "", raw).strip()
                    score = float(json.loads(clean).get("score", 0.0))
                except Exception:
                    m = re.search(r'"(?:score)"\s*:\s*(\d+(?:\.\d+)?)', raw)
                    if m: score = float(m.group(1))
                return {"pred_index": pred_index, "score": score, "orig_rank": orig_rank}
            except Exception:
                return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

    async def run_batch():
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with AsyncOpenAI(base_url="http://localhost:8000/v1",
                               api_key="not-needed") as client:
            tasks = [evaluate_single_image(client, semaphore, oi, pi, path)
                     for oi, (pi, path) in enumerate(zip(predictions, image_paths))]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import nest_asyncio; nest_asyncio.apply()
        results = loop.run_until_complete(run_batch())
    else:
        results = asyncio.run(run_batch())
    return [r["pred_index"] for r in sorted(results, key=lambda x: (-x["score"], x["orig_rank"]))]


@torch.no_grad()
def rerank_by_filip(
    test_ds, predictions, img_local_descs, text_local_descs, text_attentions,
    vision_descs=None, text_descs=None, max_rerank_k=20, rerank_batch=64,
    alpha=0.0, device="cuda", skip_cls=True,
) -> np.ndarray:
    num_db = test_ds.num_database; num_q = test_ds.num_queries
    n_cand = min(max_rerank_k, predictions.shape[1])
    D_img  = img_local_descs.shape[-1]; D_txt = text_local_descs.shape[-1]
    if D_img != D_txt:
        raise ValueError(f"FILIP: img dim ({D_img}) != text dim ({D_txt}).")
    logger.info(f"FILIP reranking top-{n_cand} for {num_q} queries")
    reranked = predictions.copy()
    for q_start in tqdm(range(0, num_q, rerank_batch), desc="FILIP rerank"):
        t0    = time.perf_counter()
        q_end = min(q_start + rerank_batch, num_q); bsz = q_end - q_start
        txt_locals, txt_masks = [], []
        for q in range(q_start, q_end):
            ds_idx = num_db+q; mask_row = text_attentions[ds_idx]
            real_len = max(int(mask_row.sum()), 1)
            txt_locals.append(text_local_descs[ds_idx][:real_len])
            txt_masks .append(mask_row         [:real_len])
        max_L = max(t.shape[0] for t in txt_locals)
        txt_np = np.zeros((bsz,max_L,D_txt),dtype=np.float32)
        msk_np = np.zeros((bsz,max_L),      dtype=np.float32)
        for b,(tl,tm) in enumerate(zip(txt_locals,txt_masks)):
            L=tl.shape[0]; txt_np[b,:L]=tl; msk_np[b,:L]=tm
        txt_t = torch.from_numpy(txt_np).to(device)
        msk_t = torch.from_numpy(msk_np).bool().to(device)
        cand_idx = predictions[q_start:q_end,:n_cand]
        img_np   = np.stack([img_local_descs[cand_idx[b]] for b in range(bsz)],axis=0)
        if skip_cls: img_np = img_np[:,:,1:,:]
        img_t    = torch.from_numpy(img_np).to(device); K=img_t.shape[1]
        img_norm = F.normalize(img_t, dim=-1); txt_norm = F.normalize(txt_t, dim=-1)
        txt_exp  = txt_norm.unsqueeze(1).expand(-1,K,-1,-1)
        sim      = torch.einsum('bkld,bkpd->bklp', txt_exp, img_norm)
        max_sim  = sim.max(dim=-1).values
        mask_exp = msk_t.unsqueeze(1).expand(-1,K,-1)
        max_sim  = max_sim * mask_exp
        n_real   = mask_exp.sum(dim=-1).clamp(min=1)
        filip_scores = (max_sim.sum(dim=-1)/n_real).cpu().numpy()
        if alpha > 0.0 and vision_descs is not None and text_descs is not None:
            for b in range(bsz):
                q_ds_idx=num_db+q_start+b
                q_global=text_descs[q_ds_idx]; db_globals=vision_descs[cand_idx[b]]
                q_n=q_global/(np.linalg.norm(q_global)+1e-8)
                db_n=db_globals/(np.linalg.norm(db_globals,axis=1,keepdims=True)+1e-8)
                global_sim=db_n@q_n
                def _zn(x): std=x.std(); return (x-x.mean())/(std+1e-8) if std>1e-8 else x
                filip_scores[b]=alpha*_zn(global_sim)+(1.0-alpha)*_zn(filip_scores[b])
        for b in range(bsz):
            order=np.argsort(-filip_scores[b])
            reranked[q_start+b,:n_cand]=cand_idx[b][order]
        TIMING.record('rerank_filip_batch', bsz * n_cand, time.perf_counter() - t0)
    logger.info("FILIP reranking complete.")
    return reranked


@torch.no_grad()
def rerank_by_max_sim(
    test_ds, predictions, img_local_descs, text_local_descs, text_attentions,
    vision_descs=None, text_descs=None, max_rerank_k=20, rerank_batch=64,
    alpha=0.0, device="cuda", skip_cls=True,
) -> np.ndarray:
    num_db = test_ds.num_database; num_q = test_ds.num_queries
    n_cand = min(max_rerank_k, predictions.shape[1])
    D_img  = img_local_descs.shape[-1]; D_txt = text_local_descs.shape[-1]
    if D_img != D_txt:
        raise ValueError(f"MaxSim requires matching dims, got img={D_img} txt={D_txt}.")
    logger.info(f"MaxSim reranking top-{n_cand} for {num_q} queries")
    reranked = predictions.copy()
    for q_start in tqdm(range(0, num_q, rerank_batch), desc="MaxSim rerank"):
        t0    = time.perf_counter()
        q_end = min(q_start+rerank_batch,num_q); bsz=q_end-q_start
        txt_locals, txt_masks = [], []
        for q in range(q_start,q_end):
            ds_idx=num_db+q; mask_row=text_attentions[ds_idx]
            real_len=max(int(mask_row.sum()),1)
            txt_locals.append(text_local_descs[ds_idx][:real_len])
            txt_masks .append(mask_row         [:real_len])
        max_L=max(t.shape[0] for t in txt_locals)
        txt_np=np.zeros((bsz,max_L,D_txt),dtype=np.float32)
        msk_np=np.zeros((bsz,max_L),      dtype=np.float32)
        for b,(tl,tm) in enumerate(zip(txt_locals,txt_masks)):
            L=tl.shape[0]; txt_np[b,:L]=tl; msk_np[b,:L]=tm
        txt_t=torch.from_numpy(txt_np).to(device)
        msk_t=torch.from_numpy(msk_np).bool().to(device)
        cand_idx=predictions[q_start:q_end,:n_cand]
        img_np=np.stack([img_local_descs[cand_idx[b]] for b in range(bsz)],axis=0)
        if skip_cls: img_np=img_np[:,:,1:,:]
        img_t=torch.from_numpy(img_np).to(device); K=img_t.shape[1]
        img_norm=F.normalize(img_t,dim=-1); txt_norm=F.normalize(txt_t,dim=-1)
        txt_exp=txt_norm.unsqueeze(1).expand(-1,K,-1,-1)
        sim=torch.einsum('bkld,bkpd->bklp',txt_exp,img_norm)
        max_sim=sim.max(dim=-1).values
        mask_exp=msk_t.unsqueeze(1).expand(-1,K,-1)
        max_sim=max_sim*mask_exp; n_real=mask_exp.sum(dim=-1).clamp(min=1)
        local_scores=(max_sim.sum(dim=-1)/n_real).cpu().numpy()
        if alpha>0.0 and vision_descs is not None and text_descs is not None:
            for b in range(bsz):
                q_ds_idx=num_db+q_start+b
                q_global=text_descs[q_ds_idx]; db_globals=vision_descs[cand_idx[b]]
                q_n=q_global/(np.linalg.norm(q_global)+1e-8)
                db_n=db_globals/(np.linalg.norm(db_globals,axis=1,keepdims=True)+1e-8)
                global_sim=db_n@q_n
                def _zn(x): std=x.std(); return (x-x.mean())/(std+1e-8) if std>1e-8 else x
                local_scores[b]=alpha*_zn(global_sim)+(1.0-alpha)*_zn(local_scores[b])
        for b in range(bsz):
            order=np.argsort(-local_scores[b])
            reranked[q_start+b,:n_cand]=cand_idx[b][order]
        TIMING.record('rerank_maxsim_batch', bsz * n_cand, time.perf_counter() - t0)
    logger.info("MaxSim reranking complete.")
    return reranked


@torch.no_grad()
def filip_vs_global_diagnostic(model, test_ds, predictions, scores,
                               vision_descriptors, text_descriptors,
                               img_local_descs, text_local_desc,
                               text_tokens_all=None, text_attentions=None,
                               K=20, device="cuda"):
    head = getattr(model.single_encoder, "filip_loss", None)
    if head is None:
        logger.warning("No filip_loss head; skipping diagnostic."); return
    head.to(device).eval()
    has_cls=getattr(model.single_encoder,"has_cls",True)
    use_idf=getattr(head,"use_idf",False) and text_tokens_all is not None
    dtype=next(head.parameters()).dtype
    nd,nq=test_ds.num_database,test_ds.num_queries
    K=min(K,predictions.shape[1])
    img_l=torch.from_numpy(img_local_descs).to(device)
    txt_l=torch.from_numpy(text_local_desc).to(device)
    G=scores[:,:K].astype(np.float64); Fm=np.zeros((nq,K),dtype=np.float64)
    for q in tqdm(range(nq),desc="filip-vs-global"):
        gi=nd+q
        if text_attentions is not None:
            m=torch.from_numpy(text_attentions[gi]).to(device).bool()
        else:
            m=txt_l[gi].abs().sum(-1)>0
        if m.sum()==0: continue
        t=head.text_proj(_maybe(head,"t_norm",txt_l[gi][m].unsqueeze(0).to(dtype)))
        t=F.normalize(t,dim=-1).squeeze(0)
        if use_idf:
            ids=torch.from_numpy(text_tokens_all[gi]).to(device)[m]
            w=head.idf_weights[ids.clamp(0,head.idf_weights.numel()-1)].to(t.dtype)
        else:
            w=torch.ones(t.shape[0],device=device,dtype=t.dtype)
        w=w/w.sum().clamp(min=1e-6)
        cand=predictions[q,:K]; v=img_l[cand]
        if has_cls: v=v[:,1:]
        v=F.normalize(head.vision_proj(_maybe(head,"v_norm",v.to(dtype))),dim=-1)
        sim=torch.einsum('nd,kmd->knm',t,v)
        Fm[q]=(sim.max(dim=-1).values*w.unsqueeze(0)).sum(-1).float().cpu().numpy()
    def _rank(a): return np.argsort(np.argsort(a,axis=1),axis=1).astype(np.float64)
    def _rowcorr(a,b):
        a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True)
        den=(np.sqrt((a**2).sum(1))*np.sqrt((b**2).sum(1))); ok=den>1e-12
        return ((a*b).sum(1)[ok]/den[ok]).mean()
    logger.info("="*62)
    logger.info(f"FILIP vs GLOBAL on the same top-{K} shortlist, {nq} queries")
    logger.info(f"  pearson  (within-query) : {_rowcorr(Fm,G):+.4f}")
    logger.info(f"  spearman (within-query) : {_rowcorr(_rank(Fm),_rank(G)):+.4f}")
    logger.info(f"  top-1 agreement         : {(Fm.argmax(1)==0).mean():.3f}")
    if getattr(test_ds,'get_positives',None) is not None:
        pos=[set(np.asarray(p).ravel().tolist()) for p in test_ds.get_positives()]
        g_ok=np.array([predictions[q,0] in pos[q] for q in range(nq)])
        f_ok=np.array([predictions[q,int(Fm[q].argmax())] in pos[q] for q in range(nq)])
        in_k=np.array([len(pos[q]&set(predictions[q,:K].tolist()))>0 for q in range(nq)])
        repairable=(~g_ok)&in_k
        logger.info(f"  R@1 global    : {100*g_ok.mean():.2f}")
        logger.info(f"  R@1 filip     : {100*f_ok.mean():.2f}")
        logger.info(f"  repaired      : {100*(f_ok&repairable).sum()/max(repairable.sum(),1):.2f}%")
        logger.info(f"  damaged       : {100*((~f_ok)&g_ok).sum()/max(g_ok.sum(),1):.2f}%")
    logger.info("="*62)


@torch.no_grad()
def rerank_by_internvl(
    test_ds, predictions, retrieval_scores, internvl_ckpt,
    max_rerank_k=20, rerank_batch=8, alpha=0.5,
    image_size=224, device='cuda',
) -> np.ndarray:
    from train_internvl_probe import InternVLRerankerLightning
    from torchvision import transforms
    from PIL import Image as PILImage
    logger.info(f"Loading InternVL2 reranker from {internvl_ckpt}...")
    pl_model = InternVLRerankerLightning.load_from_checkpoint(
        internvl_ckpt, map_location='cpu')
    reranker = pl_model.model
    reranker.eval().to(device)
    logger.info("InternVL2 reranker loaded.")
    num_db = test_ds.num_database; num_q = test_ds.num_queries
    n_cand = min(max_rerank_k, predictions.shape[1])
    db_img_paths = test_ds.images_paths[:num_db]
    q_texts      = test_ds.descriptions[num_db:]
    if len(q_texts) == 0:
        logger.warning("rerank_by_internvl: no query descriptions — skipping.")
        return predictions.copy()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    logger.info(f"InternVL2 reranking top-{n_cand} for {num_q} queries "
                f"(batch={rerank_batch}, alpha={alpha})")
    reranked = predictions.copy()
    with torch.no_grad():
        for q_start in tqdm(range(0, num_q, rerank_batch), desc="InternVL2 rerank"):
            t0    = time.perf_counter()
            q_end = min(q_start+rerank_batch, num_q); bsz = q_end-q_start
            cand_idx = predictions[q_start:q_end, :n_cand]
            imgs_flat, texts_flat = [], []
            for b in range(bsz):
                qt = q_texts[q_start+b]
                for k in range(n_cand):
                    img = transform(PILImage.open(
                        db_img_paths[cand_idx[b,k]]).convert('RGB'))
                    imgs_flat.append(img); texts_flat.append(qt)
            pv     = torch.stack(imgs_flat).to(device)
            scores = reranker(pv, texts_flat).float().view(bsz, n_cand).cpu().numpy()
            if alpha > 0.0:
                global_scores = retrieval_scores[q_start:q_end, :n_cand]
                for b in range(bsz):
                    def _zn(x): std=x.std(); return (x-x.mean())/(std+1e-8) if std>1e-8 else x
                    scores[b] = (alpha*_zn(global_scores[b].astype(np.float32))
                                 + (1.0-alpha)*_zn(scores[b]))
            for b in range(bsz):
                order=np.argsort(-scores[b])
                reranked[q_start+b,:n_cand]=cand_idx[b][order]
            TIMING.record('rerank_internvl_batch', bsz * n_cand, time.perf_counter()-t0)
    logger.info("InternVL2 reranking complete.")
    return reranked


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
    logger.add(log_dir / "info.log",
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD     = {'mean': [0.48145466, 0.4578275, 0.40821073],
                         'std':  [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD   = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    if 'blip' in args.model_name.lower() or 'clip' in args.model_name.lower() \
            or 'eva' in args.model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD

    model = LaVPR_wrapper(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")

    if args.cross_modal == 3:
        num_img_tokens  = model.img_local_tokens
        num_text_tokens = model.txt_local_tokens
        txt_local_dim   = model.txt_local_dim   # raw BGE dim (1024 large / 768 base)
        img_local_dim   = model.img_local_dim   # raw DINOv2 dim (768 for ViT-B)
    else:
        num_img_tokens  = 197
        num_text_tokens = 77
        txt_local_dim   = model.encoder_dim
        img_local_dim   = 768

    if args.data in ['amster', 'amstertime']:
        args.database_folder = "/home/shared/datasets/amstertime/test/database"
        args.queries_folder  = "/home/shared/datasets/amstertime/test/queries"
        args.image_root      = "/home/shared/datasets/amstertime/test"
        args.queries_csv     = "datasets/descriptions/amstertime_descriptions.csv"
    elif args.data in ['pitts', 'pitts_test', 'pitts30k']:
        args.database_folder = "/home/shared/datasets/pitts30k/images/test/database"
        args.queries_folder  = "/home/shared/datasets/pitts30k/images/test/queries"
        args.image_root      = "/home/shared/datasets/pitts30k/images/test"
        args.queries_csv     = "datasets/descriptions/pitts30k_test_descriptions.csv"
    elif args.data in ['pitts_val', 'pitts30k_val']:
        args.database_folder = "/home/shared/datasets/pitts30k/images/val/database"
        args.queries_folder  = "/home/shared/datasets/pitts30k/images/val/queries"
        args.image_root      = "/home/shared/datasets/pitts30k/images/val"
        args.queries_csv     = "datasets/descriptions/pitts30k_val_800_queries.csv"
    elif args.data in ['msls', 'msls_val']:
        args.database_folder = "/home/shared/datasets/msls/val/database"
        args.queries_folder  = "/home/shared/datasets/msls/val/query"
        args.image_root      = "/home/shared/datasets/msls/val/"
        args.queries_csv     = "datasets/descriptions/msls_val_descriptions.csv"
    elif args.data == 'msls_challenge':
        args.database_folder = "/home/shared/datasets/msls_challenge"
        args.queries_folder  = None
        args.image_root      = "/home/shared/datasets/msls_challenge/test"
        args.queries_csv     = "datasets/descriptions/msls_challenge_descriptions.csv"
    elif args.data == 'nordland':
        args.database_folder = "/home/shared/datasets/nordland/test/database"
        args.queries_folder  = "/home/shared/datasets/nordland/test/queries"
        args.image_root      = "/home/shared/datasets/nordland/test"
        args.queries_csv     = "datasets/descriptions/nordland_descriptions.csv"
    elif args.data == 'sfxl':
        args.database_folder = "/home/shared/datasets/SF-XL/processed/test/database"        
        args.image_root      = "/home/shared/datasets/SF-XL/processed/test"
        # args.queries_folder  = "/home/shared/datasets/SF-XL/processed/test/queries_v1"
        # args.queries_csv     = "datasets/descriptions/sfxl_queries_v1.csv"
        eval_conditions = [
            ("queries_v1", f"{args.image_root}/queries_v1", "datasets/descriptions/sfxl_queries_v1.csv"),
            ("queries_v2", f"{args.image_root}/queries_v2", "datasets/descriptions/sfxl_queries_v2.csv"),
            ("occlusion",  f"{args.image_root}/queries_occlusion", "datasets/descriptions/sfxl_queries_occlusion.csv"),
            ("night",    f"{args.image_root}/queries_night", "datasets/descriptions/sfxl_queries_night.csv"),                    
        ]
    elif args.data in ['svox', 'svox_all']:
        args.database_folder = "/home/shared/datasets/svox/images/test/gallery"
        args.image_root      = "/home/shared/datasets/svox/images/test"
        eval_conditions = [
            ("Night",    f"{args.image_root}/queries_night",    "datasets/descriptions/svox_night.csv"),
            ("Sun",      f"{args.image_root}/queries_sun",      "datasets/descriptions/svox_sun.csv"),
            ("Rain",     f"{args.image_root}/queries_rain",     "datasets/descriptions/svox_rain.csv"),
            ("Snow",     f"{args.image_root}/queries_snow",     "datasets/descriptions/svox_snow.csv"),
            ("Overcast", f"{args.image_root}/queries_overcast", "datasets/descriptions/svox_overcast.csv"),
        ]
    elif args.data == 'sped':
        args.database_folder = "/home/shared/datasets/SPEDTEST/test/database"
        args.queries_folder  = "/home/shared/datasets/SPEDTEST/test/queries"
        args.image_root      = "/home/shared/datasets/SPEDTEST/test"
        args.queries_csv     = "datasets/descriptions/sped.csv"

    if 'eval_conditions' not in locals():
        eval_conditions = [(None, getattr(args, 'queries_folder', None),
                                  getattr(args, 'queries_csv', None))]

    max_results       = max(args.recall_values)
    need_local        = (getattr(args, 'reranker',       False)
                         or getattr(args, 'reranker_filip', False)
                         or getattr(args, 'filip_retrieval', False))
    stage1_cache_path = getattr(args, 'stage1_cache', None)

    db_vision_descriptors = None
    db_text_descriptors   = None
    db_all_descriptors    = None
    db_img_local_descs    = None
    db_text_local_descs   = None
    db_text_tokens_all    = None
    db_text_attentions    = None

    all_cond_results      = []
    all_retrieval_recalls = []
    all_reranking_recalls = []
    is_msls_challenge     = False
    reranking_enabled     = False

    for cond_idx, (cond_name, q_folder, q_csv) in enumerate(eval_conditions):
        args.queries_folder = q_folder
        args.queries_csv    = q_csv

        if cond_name:
            banner = (f"\n{'='*75}\n  Evaluating SVOX condition: {cond_name} "
                      f"({cond_idx+1}/{len(eval_conditions)})\n"
                      f"  Queries: {q_folder}\n  CSV: {q_csv}\n{'='*75}")
            logger.info(banner); print(banner)

        if 'msls_challenge' in getattr(args, 'image_root', ''):
            test_ds = MSLSTest(
                dataset_root=args.database_folder, image_root=args.image_root,
                csv_path=args.queries_csv, mean_std=dataset_mean_std,
                image_size=args.image_size)
            is_msls_challenge = True
        else:
            test_ds = TestDataset(
                args.database_folder, args.queries_folder, args.queries_csv,
                args.image_root, mean_std=dataset_mean_std,
                positive_dist_threshold=args.positive_dist_threshold,
                image_size=args.image_size, use_labels=args.use_labels)
        logger.info(f"Testing on {test_ds}")

        num_db    = test_ds.num_database
        num_q     = test_ds.num_queries
        total_len = len(test_ds)

        vision_descriptors = np.zeros((total_len, model.encoder_dim), dtype="float32")
        text_descriptors   = np.zeros((total_len, model.encoder_dim), dtype="float32")
        all_descriptors    = np.zeros((total_len, model.encoder_dim), dtype="float32")
        img_local_descs    = np.zeros((total_len, num_img_tokens,  img_local_dim), dtype="float32")
        text_local_descs   = np.zeros((total_len, num_text_tokens, txt_local_dim), dtype="float32")
        text_tokens_all    = np.zeros((total_len, num_text_tokens), dtype=np.int64)
        text_attentions    = np.zeros((total_len, num_text_tokens), dtype=np.int64)

        cache_suffix  = f"_{cond_name}" if cond_name else ""
        load_cache    = (Path(stage1_cache_path).with_suffix(f"{cache_suffix}.csv")
                         if stage1_cache_path else None)
        cache_loaded  = load_cache is not None and load_cache.exists()
        save_cache    = log_dir / f"stage1_cache{cache_suffix}.csv"
        skip_encoding = cache_loaded and not need_local

        if skip_encoding:
            logger.info(
                f"Stage-1 cache found — skipping gallery and query encoding.")

        with torch.inference_mode():
            if skip_encoding:
                pass
            elif db_vision_descriptors is None:
                logger.info(f"Extracting database descriptors ({num_db} images)")
                t_gallery_start = time.perf_counter()
                database_subset_ds  = Subset(test_ds, list(range(num_db)))
                database_dataloader = DataLoader(
                    dataset=database_subset_ds,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)
                for images, indices, texts in tqdm(database_dataloader,
                                                   desc="Encoding gallery"):
                    encode_batch(model, args, images, texts, indices,
                                 all_descriptors, vision_descriptors,
                                 text_descriptors, img_local_descs,
                                 text_local_descs, text_tokens_all,
                                 text_attentions)
                TIMING.record('encode_gallery_total', num_db,
                              time.perf_counter() - t_gallery_start)
                db_vision_descriptors = vision_descriptors[:num_db].copy()
                db_text_descriptors   = text_descriptors[:num_db].copy()
                db_all_descriptors    = all_descriptors[:num_db].copy()
                db_img_local_descs    = img_local_descs[:num_db].copy()
                db_text_local_descs   = text_local_descs[:num_db].copy()
                db_text_tokens_all    = text_tokens_all[:num_db].copy()
                db_text_attentions    = text_attentions[:num_db].copy()
            else:
                logger.info(f"Reusing pre-encoded gallery ({num_db} images)")
                vision_descriptors[:num_db] = db_vision_descriptors
                text_descriptors[:num_db]   = db_text_descriptors
                all_descriptors[:num_db]    = db_all_descriptors
                img_local_descs[:num_db]    = db_img_local_descs
                text_local_descs[:num_db]   = db_text_local_descs
                text_tokens_all[:num_db]    = db_text_tokens_all
                text_attentions[:num_db]    = db_text_attentions

            if not skip_encoding:
                desc_label = (f"Encoding {cond_name} queries"
                              if cond_name else "Extracting query descriptors")
                t_query_start = time.perf_counter()
                queries_subset_ds  = Subset(test_ds, list(range(num_db, num_db+num_q)))
                queries_dataloader = DataLoader(
                    dataset=queries_subset_ds,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size)
                for images, indices, texts in tqdm(queries_dataloader,
                                                   desc=desc_label):
                    encode_batch(model, args, images, texts, indices,
                                 all_descriptors, vision_descriptors,
                                 text_descriptors, img_local_descs,
                                 text_local_descs, text_tokens_all,
                                 text_attentions)
                TIMING.record('encode_queries_total', num_q,
                              time.perf_counter() - t_query_start)

        if need_local:
            logger.info(f"local buffers | img {np.abs(img_local_descs).sum():.1f}  "
                        f"text {np.abs(text_local_descs).sum():.1f}")

        if not skip_encoding:
            if args.cross_modal:
                database_descriptors = (text_descriptors[:num_db]
                                        if not args.is_image
                                        else vision_descriptors[:num_db])
                queries_descriptors  = text_descriptors[num_db:]
            else:
                database_descriptors = text_descriptors[:num_db]
                queries_descriptors  = text_descriptors[num_db:]

        if cache_loaded:
            logger.info(f"Loading stage-1 cache: {load_cache}")
            cache = load_stage1_cache(load_cache, test_ds.images_paths[:num_db])
            verify_cache_query_order(cache, test_ds)
            retrieval_predictions = cache['predictions']
            scores                = cache['scores']
            if retrieval_predictions.shape[1] < max_results:
                pad_w = max_results - retrieval_predictions.shape[1]
                retrieval_predictions = np.pad(
                    retrieval_predictions, ((0,0),(0,pad_w)), constant_values=0)
                scores = np.pad(scores, ((0,0),(0,pad_w)), constant_values=0.0)
        elif args.filip_retrieval and not skip_encoding:
            scores, retrieval_predictions = filip_full_retrieval(
                model, test_ds, vision_descriptors, text_descriptors,
                img_local_descs, text_local_descs, text_tokens_all,
                max_results=max_results,
                db_chunk=getattr(args,'filip_db_chunk',2048),
                alpha=getattr(args,'filip_alpha',0.0),
                db_on_gpu=getattr(args,'filip_db_on_gpu',True),
                device=args.device)
        elif not skip_encoding:
            scores, retrieval_predictions = get_queries_predictions(
                model.encoder_dim, database_descriptors, all_descriptors,
                queries_descriptors, max_results)
        else:
            raise RuntimeError(
                "Stage-1 cache not found but encoding was skipped.")

        save_stage1_cache(
            save_cache, retrieval_predictions, scores,
            test_ds.images_paths[num_db:], test_ds.images_paths[:num_db],
            topk=STAGE1_TOPK_SAVE)

        if getattr(args, 'filip_diag', 0) and not args.filip_retrieval:
            filip_vs_global_diagnostic(
                model, test_ds, retrieval_predictions, scores,
                vision_descriptors, text_descriptors,
                img_local_descs, text_local_descs,
                text_tokens_all=text_tokens_all,
                text_attentions=text_attentions,
                K=min(args.max_rerank, max_results),
                device=args.device)

        reranking_predictions = retrieval_predictions.copy()
        reranking_enabled     = False

        if getattr(args, 'max_sim', False):
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions = rerank_by_max_sim(
                test_ds, reranking_predictions, img_local_descs,
                text_local_descs, text_attentions,
                vision_descs=vision_descriptors if getattr(args,'rerank_alpha',0.0)>0 else None,
                text_descs  =text_descriptors   if getattr(args,'rerank_alpha',0.0)>0 else None,
                max_rerank_k=max_rerank_k,
                rerank_batch=getattr(args,'rerank_batch',64),
                alpha=getattr(args,'rerank_alpha',0.0),
                device=args.device,
                skip_cls=getattr(args,'max_sim_skip_cls',True))

        if args.reranker_filip:
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions = rerank_by_filip(
                test_ds, reranking_predictions, img_local_descs,
                text_local_descs, text_attentions,
                vision_descs=vision_descriptors if getattr(args,'rerank_alpha',0.0)>0 else None,
                text_descs  =text_descriptors   if getattr(args,'rerank_alpha',0.0)>0 else None,
                max_rerank_k=max_rerank_k,
                rerank_batch=getattr(args,'rerank_batch',64),
                alpha=getattr(args,'rerank_alpha',0.0),
                device=args.device,
                skip_cls=getattr(args,'max_sim_skip_cls',True))

        if args.reranker_mllm:
            reranking_enabled = True
            q_texts      = test_ds.descriptions[test_ds.num_database:]
            db_paths_arr = np.array(test_ds.images_paths[:test_ds.num_database])
            db_images    = db_paths_arr[reranking_predictions]
            for i in range(len(q_texts)):
                print(f"Sending query {i} to LLM")
                reranking_predictions[i] = rerank_by_mllm(
                    db_images[i], q_texts[i], reranking_predictions[i])

        if args.reranker_jev:
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions = rerank_by_jev(
                test_ds, reranking_predictions,
                max_rerank_k=max_rerank_k,
                batch_size=getattr(args,'jev_batch_size',32),
                match_weight=getattr(args,'jev_match_weight',0.6),
                device=args.device)

        if getattr(args, 'rerank_internvl', None):
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions = rerank_by_internvl(
                test_ds, reranking_predictions, scores,
                internvl_ckpt=args.rerank_internvl,
                max_rerank_k=max_rerank_k,
                rerank_batch=getattr(args,'internvl_batch',4),
                alpha=getattr(args,'rerank_alpha',0.0),
                image_size=args.image_size,
                device=args.device)

        if getattr(args,'reranker',False) and args.cross_modal==3 \
                and args.max_sim==0:
            reranking_enabled     = True
            max_rerank_k          = min(args.max_rerank, max_results)
            reranking_predictions = rerank_by_lavpr(
                model, test_ds, reranking_predictions,
                img_local_descs, text_local_descs, text_attentions,
                vision_descs=vision_descriptors, text_descs=text_descriptors,
                max_rerank_k=max_rerank_k,
                rerank_batch=getattr(args,'rerank_batch',64),
                alpha=getattr(args,'rerank_alpha',0.0),
                device=args.device)

        if not reranking_enabled and (
            bool(getattr(args,'reranker',       False))
            or bool(getattr(args,'reranker_filip',False))
            or bool(getattr(args,'reranker_mllm', False))
            or bool(getattr(args,'reranker_jev',  False))
        ):
            reranking_enabled = True

        if reranking_enabled:
            save_rerank_cache(
                log_dir / f"stage2_cache{cache_suffix}.csv",
                reranking_predictions, None,
                test_ds.images_paths[num_db:],
                test_ds.images_paths[:num_db],
                topk=STAGE1_TOPK_SAVE)

        if is_msls_challenge:
            final_predictions = (reranking_predictions
                                 if reranking_enabled else retrieval_predictions)
            test_ds.save_predictions(
                final_predictions, log_dir/"msls_challenge_predictions.txt", k=25)
        else:
            if args.use_labels:
                positives_per_query = test_ds.get_positives()

                def compute_recalls(preds_matrix):
                    rec = np.zeros(len(args.recall_values))
                    for query_index, preds in enumerate(preds_matrix):
                        for i, n in enumerate(args.recall_values):
                            if np.any(np.isin(preds[:n],
                                              positives_per_query[query_index])):
                                rec[i:] += 1; break
                    rec   = rec / test_ds.num_queries * 100
                    r_str = ", ".join([f"R@{val}: {r:.1f}"
                                      for val,r in zip(args.recall_values,rec)])
                    return rec, r_str

                retrieval_recalls, retrieval_recalls_str = compute_recalls(
                    retrieval_predictions)
                all_retrieval_recalls.append(retrieval_recalls)

                prefix = f"[SVOX {cond_name}] " if cond_name else ""
                if reranking_enabled:
                    reranking_recalls, reranking_recalls_str = compute_recalls(
                        reranking_predictions)
                    all_reranking_recalls.append(reranking_recalls)
                    logger.info(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    logger.info(f"{prefix}Reranking Recall: {reranking_recalls_str}")
                    print(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    print(f"{prefix}Reranking Recall: {reranking_recalls_str}")
                    all_cond_results.append(
                        (cond_name or "Overall",
                         retrieval_recalls_str, reranking_recalls_str))
                else:
                    logger.info(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    print(f"{prefix}Retrieval Recall: {retrieval_recalls_str}")
                    all_cond_results.append(
                        (cond_name or "Overall", retrieval_recalls_str, None))

                model_path    = (args.lora_path if args.lora_path is not None
                                 else args.model_path)
                csv_model_tag = (f"{args.model_name} [SVOX {cond_name}]"
                                 if cond_name else args.model_name)
                with open("eval_vpr_results.csv", "a") as f:
                    if reranking_enabled:
                        f.write(f"{model_path},{csv_model_tag},"
                                f"Retrieval: {retrieval_recalls_str} | "
                                f"Reranking: {reranking_recalls_str}\n")
                    else:
                        f.write(f"{model_path},{csv_model_tag},"
                                f"{retrieval_recalls_str}\n")

        if args.num_preds_to_save != 0:
            logger.info("Saving final predictions")
            final_predictions = (reranking_predictions
                                 if reranking_enabled else retrieval_predictions)
            save_dir = log_dir / cond_name if cond_name else log_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            visualizations.save_preds(
                final_predictions[:,:args.num_preds_to_save], test_ds,
                save_dir, args.save_only_wrong_preds, args.use_labels,
                test_ds.images_paths_csv, texts=test_ds.descriptions)

    if len(eval_conditions) > 1 and all_retrieval_recalls:
        mean_retrieval     = np.mean(all_retrieval_recalls, axis=0)
        mean_retrieval_str = ", ".join([f"R@{val}: {r:.1f}"
                                        for val,r in zip(args.recall_values,
                                                         mean_retrieval)])
        header = (f"\n{'='*75}\n  SVOX Evaluation Summary "
                  f"(All {len(eval_conditions)} Conditions)\n{'='*75}")
        logger.info(header); print(header)
        for c_name, r_str, rr_str in all_cond_results:
            line = (f"  {c_name:<10} | Retrieval: {r_str}"
                    + (f" | Reranking: {rr_str}" if rr_str else ""))
            logger.info(line); print(line)
        sep = f"{'-'*75}"
        logger.info(sep); print(sep)
        model_path = (args.lora_path if args.lora_path is not None
                      else args.model_path)
        if reranking_enabled and all_reranking_recalls:
            mean_reranking     = np.mean(all_reranking_recalls, axis=0)
            mean_reranking_str = ", ".join([f"R@{val}: {r:.1f}"
                                            for val,r in zip(args.recall_values,
                                                             mean_reranking)])
            mean_line = (f"  {'MEAN':<10} | Retrieval: {mean_retrieval_str} "
                         f"| Reranking: {mean_reranking_str}")
            logger.info(mean_line); print(mean_line)
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name} [SVOX Mean],"
                        f"Retrieval: {mean_retrieval_str} | "
                        f"Reranking: {mean_reranking_str}\n")
        else:
            mean_line = f"  {'MEAN':<10} | Retrieval: {mean_retrieval_str}"
            logger.info(mean_line); print(mean_line)
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name} [SVOX Mean],"
                        f"{mean_retrieval_str}\n")
        footer = f"{'='*75}\n"
        logger.info(footer); print(footer)

    # ---- Print timing summary -----------------------------------------------
    timing_str = TIMING.summary()
    logger.info(timing_str)
    print(timing_str)


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)