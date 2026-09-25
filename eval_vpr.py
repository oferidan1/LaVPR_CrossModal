import argparse
import eval_parser
from argparse import Namespace
import sys
from datetime import datetime
from pathlib import Path

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


def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _maybe(mod, name, x):
    """Apply mod.<name> if it exists, otherwise pass through.

    Keeps the eval path compatible with FILIP heads trained before the
    LayerNorms were added.
    """
    layer = getattr(mod, name, None)
    return layer(x) if layer is not None else x


def _z(x):
    return (x - x.mean()) / x.std().clamp(min=1e-6)


def encode_batch(model, args, images, texts, indices, all_descriptors,
                 vision_descriptors, text_descriptors, img_local_descs,
                 text_local_descs, text_tokens_all, text_attentions):
    if args.bfloat16:
        images = images.bfloat16()

    if args.cross_modal <= 1:
        image_features = model.encode_text(texts)
        vision_descriptors[indices.numpy(), :] = image_features.cpu().float().numpy()
        text_features = model.encode_image(images.to(args.device))
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
    else:
        # single vector of both image and text, plus per-token features
        descriptors, text_features, img_local, text_local, text_tokens = \
            model.encode_single(images.to(args.device), texts)
        vision_descriptors[indices.numpy(), :] = descriptors.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
        img_local_descs[indices.numpy(), :] = img_local.cpu().float().numpy()
        text_local_descs[indices.numpy(), :] = text_local.cpu().float().numpy()        
        #text_tokens_all[indices.numpy(), :] = text_tokens.cpu().long().numpy()
        #text_attentions[indices.numpy(), :] = text_attention_mask.cpu().long().numpy()


def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors,
                            queries_descriptors, max_results):
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    database_descriptors = normlize_features(database_descriptors)
    queries_descriptors = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    return scores, predictions


# ---------------------------------------------------------------------------
# FILIP: full-gallery retrieval (as in the original paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def filip_full_retrieval(model, test_ds, vision_descriptors, text_descriptors,
                         img_local_descs, text_local_desc, text_tokens_all,
                         max_results=100, db_chunk=2048, alpha=0.0,
                         db_on_gpu=True, device="cuda"):
    """Token-wise MaxSim over the WHOLE gallery.

        s(t->i) = sum_i w_i * max_j (t_i . v_j)

    No faiss: MaxSim is not an inner product, so it cannot be indexed.
    The database is projected once up front; each query then streams over it
    in chunks. alpha > 0 additionally fuses the z-normalised global cosine.

    Returns (scores, predictions), matching get_queries_predictions().
    """
    head = getattr(model.single_encoder, "filip_loss", None)
    if head is None:
        raise RuntimeError("Model has no filip_loss head - cannot run FILIP retrieval.")
    head.to(device).eval()

    has_cls = getattr(model.single_encoder, "has_cls", True)
    use_idf = getattr(head, "use_idf", False)
    hdt = next(head.parameters()).dtype
    num_db, num_q = test_ds.num_database, test_ds.num_queries
    logger.info(f"FILIP full-gallery retrieval over {num_db} database images")

    # ---- project the whole database once ---------------------------------
    parts = []
    for s in tqdm(range(0, num_db, db_chunk), desc="projecting database"):
        e = min(s + db_chunk, num_db)                     # <-- clamp
        v = torch.from_numpy(img_local_descs[s:e]).to(device)
        if has_cls:
            v = v[:, 1:]
        v = head.vision_proj(_maybe(head, "v_norm", v.to(hdt)))
        v = F.normalize(v, dim=-1).half()
        parts.append(v if db_on_gpu else v.cpu())
    db_v = torch.cat(parts, dim=0)
    del parts
    torch.cuda.empty_cache()
    logger.info(f"database tokens {tuple(db_v.shape)}  "
                f"{db_v.numel() * 2 / 1e9:.2f} GB on {'gpu' if db_on_gpu else 'cpu'}")

    db_g = F.normalize(torch.from_numpy(vision_descriptors[:num_db]).to(device).float(), dim=-1)
    q_g = F.normalize(torch.from_numpy(text_descriptors[num_db:]).to(device).float(), dim=-1)

    topk = min(max_results, num_db)
    predictions = np.zeros((num_q, topk), dtype=np.int64)
    out_scores = np.zeros((num_q, topk), dtype=np.float32)

    for q in tqdm(range(num_q), desc="FILIP retrieval"):
        gi = num_db + q

        raw = torch.from_numpy(text_local_desc[gi]).to(device)
        mask = raw.abs().sum(-1) > 0
        if mask.sum() == 0:
            predictions[q] = np.arange(topk)
            continue
        raw = raw[mask].unsqueeze(0)
        t = head.text_proj(_maybe(head, "t_norm", raw.to(hdt)))
        t = F.normalize(t, dim=-1).squeeze(0).half()

        if use_idf:
            ids = torch.from_numpy(text_tokens_all[gi]).to(device)[mask]
            w = head.idf_weights[ids.clamp(0, head.idf_weights.numel() - 1)].half()
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
            g = (q_g[q].unsqueeze(0) * db_g).sum(-1)
            s_final = alpha * _z(g) + (1.0 - alpha) * _z(s_final)

        vals, idx = torch.topk(s_final, topk)
        predictions[q] = idx.cpu().numpy()
        out_scores[q] = vals.cpu().numpy()

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

    reranked_predictions = predictions.copy()

    img_local_descs_tensor = torch.from_numpy(img_local_descs).to(device)
    text_local_desc_tensor = torch.from_numpy(text_local_desc).to(device)
    vision_descriptors_tensor = torch.from_numpy(vision_descriptors).to(device)
    text_descriptors_tensor = torch.from_numpy(text_descriptors).to(device)
    text_attn_tensor = (torch.from_numpy(text_attentions).to(device) if text_attentions is not None else None)
    if text_attn_tensor is None:
        logger.warning("rerank_predictions: no attention mask supplied, "
                       "falling back to non-zero-row detection.")

    n_cand = int(max_rerank_k)

    with torch.no_grad():
        for q_idx in tqdm(range(test_ds.num_queries), desc="Reranking queries"):
            actual_q_ds_idx = test_ds.num_database + q_idx

            query_text_global = text_descriptors_tensor[actual_q_ds_idx].unsqueeze(0)
            raw_text_local = text_local_desc_tensor[actual_q_ds_idx]        # (Lt, D)

            if text_attn_tensor is not None:
                mask_row = text_attn_tensor[actual_q_ds_idx]                # (Lt,)
            else:
                mask_row = raw_text_local.any(dim=-1).long()
            if mask_row.sum() == 0:
                mask_row = torch.ones_like(mask_row)

            # Trim to the longest real position, keeping tokens and mask ALIGNED.
            # Slicing by a count (the old `true_len`) silently reorders whenever
            # the real positions are not a contiguous prefix.
            true_len = int(mask_row.nonzero()[-1].item()) + 1
            query_text_local = raw_text_local[:true_len].unsqueeze(0)       # (1, L, D)
            query_mask = mask_row[:true_len].unsqueeze(0)                   # (1, L)

            candidate_db_indices = predictions[q_idx, :n_cand]
            B = len(candidate_db_indices)
            candidate_img_local = img_local_descs_tensor[candidate_db_indices]
            candidate_img_global = vision_descriptors_tensor[candidate_db_indices]

            L, D_dim = query_text_local.shape[1], query_text_local.shape[2]
            text_local_expanded = query_text_local.expand(B, L, D_dim)
            text_global_expanded = query_text_global.expand(B, -1)
            text_mask_expanded = query_mask.expand(B, L)

            if next(rerank_model.parameters()).dtype == torch.bfloat16:
                candidate_img_local = candidate_img_local.bfloat16()
                text_local_expanded = text_local_expanded.bfloat16()
                candidate_img_global = candidate_img_global.bfloat16()
                text_global_expanded = text_global_expanded.bfloat16()
                # the mask stays integer: casting it to bf16 and then comparing
                # or multiplying inside the attention module is how masks turn
                # into 0.99609375 and stop masking

            scores = rerank_model.cross_attn_classifier(
                candidate_img_local, text_local_expanded,
                candidate_img_global, text_global_expanded)
            scores = scores.float().cpu().numpy()

            reranked_order = np.argsort(-scores)
            reranked_predictions[q_idx, :n_cand] = candidate_db_indices[reranked_order]

    return reranked_predictions


def rerank_by_filip(model, test_ds, predictions, vision_descriptors, text_descriptors,
                    img_local_descs, text_local_desc, text_tokens_all=None,
                    max_rerank_k=20, alpha=0.0, device="cuda"):
    """Rerank the stage-1 shortlist with FILIP token-wise max similarity.

    alpha fuses the stage-1 global score after z-normalising both, so the
    two scales are comparable:  alpha * z(global) + (1-alpha) * z(filip).
    alpha = 1.0 must reproduce the stage-1 ordering - use it as a sanity check.
    """
    logger.info(f"Reranking top-{max_rerank_k} candidates using FILIP MaxSim...")

    head = getattr(model.single_encoder, "filip_loss", None)
    if head is None:
        logger.warning("Model has no filip_loss head; skipping FILIP rerank.")
        return predictions
    head.to(device).eval()

    has_cls = getattr(model.single_encoder, "has_cls", True)
    use_idf = getattr(head, "use_idf", False) and text_tokens_all is not None
    if getattr(head, "use_idf", False) and text_tokens_all is None:
        logger.warning("FILIP head trained with IDF weights but no token ids passed; "
                       "falling back to uniform weights (will not match training).")

    reranked = predictions.copy()
    img_l = torch.from_numpy(img_local_descs).to(device)
    txt_l = torch.from_numpy(text_local_desc).to(device)
    dtype = next(head.parameters()).dtype

    db_g = F.normalize(torch.from_numpy(vision_descriptors).to(device).float(), dim=-1)
    q_g = F.normalize(torch.from_numpy(text_descriptors).to(device).float(), dim=-1)

    with torch.no_grad():
        for q_idx in tqdm(range(test_ds.num_queries), desc="FILIP rerank"):
            q_ds_idx = test_ds.num_database + q_idx

            raw_t = txt_l[q_ds_idx]
            mask = raw_t.abs().sum(dim=-1) > 0
            if mask.sum() == 0:
                continue
            raw_t = raw_t[mask].unsqueeze(0)

            t = head.text_proj(_maybe(head, "t_norm", raw_t.to(dtype)))
            t = F.normalize(t, dim=-1).squeeze(0)

            if use_idf:
                ids = torch.from_numpy(text_tokens_all[q_ds_idx]).to(device)
                ids = ids[mask].clamp(0, head.idf_weights.numel() - 1)
                w = head.idf_weights[ids].to(t.dtype)
            else:
                w = torch.ones(t.shape[0], device=device, dtype=t.dtype)
            w = w / w.sum().clamp(min=1e-6)

            cand = predictions[q_idx, :max_rerank_k]
            v_raw = img_l[cand]
            if has_cls:
                v_raw = v_raw[:, 1:]                     # drop CLS, as in training
            v = head.vision_proj(_maybe(head, "v_norm", v_raw.to(dtype)))
            v = F.normalize(v, dim=-1)

            sim = torch.einsum('nd,kmd->knm', t, v)
            best = sim.max(dim=-1).values
            s_filip = (best * w.unsqueeze(0)).sum(-1).float()

            if alpha > 0:
                s_global = (q_g[q_ds_idx].unsqueeze(0) * db_g[cand]).sum(-1)
                s = alpha * _z(s_global) + (1.0 - alpha) * _z(s_filip)
            else:
                s = s_filip

            order = torch.argsort(s, descending=True).cpu().numpy()
            reranked[q_idx, :max_rerank_k] = cand[order]

    return reranked


def rerank_by_mllm(image_paths, target_text, predictions,
                   max_concurrent_requests=8, debug=True):
    """Rerank candidates on viewpoint-invariant scene text & landmark matching.

    Ties keep their original stage-1 order.
    """
    if len(image_paths) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(image_paths)} image paths vs {len(predictions)} predictions."
        )

    async def encode_image_async(image_path):
        def _read_and_b64(path):
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(path, "rb") as image_file:
                b64_str = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:{mime_type};base64,{b64_str}"

        return await asyncio.to_thread(_read_and_b64, image_path)

    async def evaluate_single_image(client, semaphore, orig_rank, pred_index, path):
        async with semaphore:
            if not os.path.exists(path):
                if debug:
                    print(f"[Debug] Path missing: {path}")
                return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

            try:
                img_data_url = await encode_image_async(path)

                prompt_text = f"""
You are an expert visual place recognition and OCR assistant.
Analyze this image to evaluate if it shows the location described by the target text query.

TARGET QUERY: "{target_text}"

CRITICAL VIEWPOINT-INVARIANT GUIDELINES:
1. VIEWPOINT & ANGLE INVARIANCE: The database image may be taken from a DIFFERENT camera angle, opposite street direction, or different field-of-view than the text description. DO NOT penalize the image if elements appear on different sides (left/right/center) or in a different spatial order!
2. KEY VISUAL ANCHORS: Look for shop names, storefront signs, street signs, building facades, banners, logos, and plaques mentioned in or matching the query.
3. MATCH RULE: If the key text, shop names, or primary landmarks mentioned in the query are present anywhere in the image, it is a Strong Match.

SCORING CRITERIA (0 to 100):
- 85-100: High confidence. Key shop names, street signs, or prominent landmark text from the query are clearly present in the scene.
- 50-84: Partial confidence. Moderate text match, secondary landmarks visible, or partially occluded shop name.
- 0-49: Low confidence. Completely different location, unrelated storefront text, or no matching landmarks.

OUTPUT FORMAT:
Respond ONLY with a JSON object: {{"score": <number 0-100>, "reason": "<brief justification>"}}
"""

                response = await client.chat.completions.create(
                    model="neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text.strip()},
                                {"type": "image_url", "image_url": {"url": img_data_url}},
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_tokens=80,
                )

                raw_output = response.choices[0].message.content.strip()

                score = 0.0
                try:
                    clean_json = raw_output
                    if "```" in clean_json:
                        clean_json = re.sub(r"```[a-zA-Z]*", "", clean_json).strip()
                    data = json.loads(clean_json)
                    score = float(data.get("score", data.get("confidence_score", 0.0)))
                except Exception:
                    match = re.search(
                        r'"(?:score|confidence_score)"\s*:\s*(\d+(?:\.\d+)?)', raw_output)
                    if match:
                        score = float(match.group(1))

                if debug:
                    print(f"[Debug] Orig Rank {orig_rank:02d} | Pred {pred_index} | "
                          f"Score: {score} | File: {os.path.basename(path)}")

                return {"pred_index": pred_index, "score": score, "orig_rank": orig_rank}

            except Exception as e:
                if debug:
                    print(f"[Debug Error] Candidate {pred_index} failed: {e}")
                return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

    async def run_batch():
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed") as client:
            tasks = [
                evaluate_single_image(client, semaphore, orig_idx, pred_idx, path)
                for orig_idx, (pred_idx, path) in enumerate(zip(predictions, image_paths))
            ]
            return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        results = loop.run_until_complete(run_batch())
    else:
        results = asyncio.run(run_batch())

    # primary key: score desc; secondary: original rank asc
    sorted_results = sorted(results, key=lambda x: (-x["score"], x["orig_rank"]))
    reranked_preds = [res["pred_index"] for res in sorted_results]

    if debug:
        print("\n--- Final Reranked Candidate Order ---")
        for new_rank, res in enumerate(sorted_results):
            print(f"New Rank {new_rank+1:02d}: Candidate {res['pred_index']} "
                  f"(Orig Rank: {res['orig_rank']}) -> Score: {res['score']}")

    return reranked_preds


# ---------------------------------------------------------------------------

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_time = datetime.now()

    logger.remove()
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log",
               format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD = {'mean': [0.48145466, 0.4578275, 0.40821073],
                     'std': [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    if 'blip' in args.model_name.lower() or 'clip' in args.model_name.lower() \
            or 'eva' in args.model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD

    model = LaVPR_wrapper(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")

    is_msls_challenge = False
    # ViT-B/16 at 224: (224/16)^2 = 196 patches, +1 CLS. CLIP text is 77 tokens.
    num_img_tokens = 197
    num_text_tokens = 77

    if 'msls_challenge' in args.image_root:
        test_ds = MSLSTest(dataset_root=args.database_folder, image_root=args.image_root,
                           csv_path=args.queries_csv, mean_std=dataset_mean_std,
                           image_size=args.image_size)
        is_msls_challenge = True
    else:
        test_ds = TestDataset(
            args.database_folder,
            args.queries_folder,
            args.queries_csv,
            args.image_root,
            mean_std=dataset_mean_std,
            positive_dist_threshold=args.positive_dist_threshold,
            image_size=args.image_size,
            use_labels=args.use_labels,
        )
    logger.info(f"Testing on {test_ds}")

    max_results = max(args.recall_values)
    need_local = (getattr(args, 'reranker', False)
                  or getattr(args, 'reranker_filip', False)
                  or getattr(args, 'filip_retrieval', False))

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(dataset=database_subset_ds,
                                         num_workers=args.num_workers,
                                         batch_size=args.batch_size)

        # zeros, not empty: an unpopulated buffer should fail loudly
        vision_descriptors = np.zeros((len(test_ds), model.encoder_dim), dtype="float32")
        text_descriptors = np.zeros((len(test_ds), model.encoder_dim), dtype="float32")
        all_descriptors = np.zeros((len(test_ds), model.encoder_dim), dtype="float32")        
        img_local_descs = np.zeros((len(test_ds), num_img_tokens, model.encoder_dim), dtype="float32")
        text_local_desc = np.zeros((len(test_ds), num_text_tokens, model.encoder_dim), dtype="float32")
        text_tokens_all = np.zeros((len(test_ds), num_text_tokens), dtype=np.int64)       
        text_attentions = np.zeros((len(test_ds), num_text_tokens), dtype=np.int64)       

        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors,
                         vision_descriptors, text_descriptors, img_local_descs,
                         text_local_desc, text_tokens_all, text_attentions)

        logger.debug("Extracting queries descriptors for evaluation/testing")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database,
                                test_ds.num_database + test_ds.num_queries)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds,
                                        num_workers=args.num_workers,
                                        batch_size=args.batch_size)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors,
                         vision_descriptors, text_descriptors, img_local_descs,
                         text_local_desc, text_tokens_all, text_attentions)

    if need_local:
        logger.info(f"local buffers | img {np.abs(img_local_descs).sum():.1f}  "
                    f"text {np.abs(text_local_desc).sum():.1f}  "
                    f"(both must be non-zero)")

    if args.cross_modal:
        if args.text_only:
            database_descriptors = text_descriptors[: test_ds.num_database]
        else:
            database_descriptors = vision_descriptors[: test_ds.num_database]
        queries_descriptors = text_descriptors[test_ds.num_database:]
    else:
        database_descriptors = text_descriptors[: test_ds.num_database]
        queries_descriptors = text_descriptors[test_ds.num_database:]

    # ---- 1. Stage-1 retrieval -------------------------------------------
    if args.filip_retrieval:
        # FILIP-only: token-wise MaxSim IS the retrieval similarity
        scores, predictions = filip_full_retrieval(
            model, test_ds, vision_descriptors, text_descriptors,
            img_local_descs, text_local_desc, text_tokens_all,
            max_results=max_results,
            db_chunk=getattr(args, 'filip_db_chunk', 2048),
            alpha=getattr(args, 'filip_alpha', 0.0),
            db_on_gpu=getattr(args, 'filip_db_on_gpu', True),
            device=args.device)
    else:
        scores, predictions = get_queries_predictions(
            model.encoder_dim, database_descriptors, all_descriptors,
            queries_descriptors, max_results)

    # ---- 2. Reranking ----------------------------------------------------
    if args.reranker:
        max_rerank_k = min(args.max_rerank, max_results)
        predictions = rerank_predictions(
            model, test_ds, predictions, vision_descriptors, text_descriptors,
            img_local_descs, text_local_desc, max_rerank_k=max_rerank_k,
            device=args.device)

    if args.reranker_filip:
        max_rerank_k = min(args.max_rerank, max_results)
        predictions = rerank_by_filip(
            model, test_ds, predictions, vision_descriptors, text_descriptors,
            img_local_descs, text_local_desc, text_tokens_all=text_tokens_all,
            max_rerank_k=max_rerank_k, alpha=getattr(args, 'filip_alpha', 0.0),
            device=args.device)

    if args.reranker_mllm:
        q_texts = test_ds.descriptions[test_ds.num_database:]
        db_paths_array = np.array(test_ds.images_paths[:test_ds.num_database])
        db_images = db_paths_array[predictions]
        for i in range(len(q_texts)):
            print(f"Sending query {i} to LLM")
            predictions[i] = rerank_by_mllm(db_images[i], q_texts[i], predictions[i])

    # ---- 3. Metrics ------------------------------------------------------
    if is_msls_challenge:
        test_ds.save_predictions(predictions, log_dir / "msls_challenge_predictions.txt", k=25)
    else:
        if args.use_labels:
            positives_per_query = test_ds.get_positives()
            recalls = np.zeros(len(args.recall_values))
            for query_index, preds in enumerate(predictions):
                for i, n in enumerate(args.recall_values):
                    if np.any(np.isin(preds[:n], positives_per_query[query_index])):
                        recalls[i:] += 1
                        break

            recalls = recalls / test_ds.num_queries * 100
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}"
                                     for val, rec in zip(args.recall_values, recalls)])
            logger.info(f"{recalls_str}")

            model_path = args.lora_path if args.lora_path is not None else args.model_path
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name},{recalls_str}\n")

    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir,
            args.save_only_wrong_preds, args.use_labels,
            test_ds.images_paths_csv, texts=test_ds.descriptions)


if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)