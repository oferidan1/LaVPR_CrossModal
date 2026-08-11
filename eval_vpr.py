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


def normlize_features(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)    


def encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, img_local_descs, text_local_descs):
    if args.bfloat16:
        images = images.bfloat16()

    if args.cross_modal<=1:
        image_features = model.encode_text(texts)
        vision_descriptors[indices.numpy(), :] = image_features.cpu().float().numpy()
        text_features = model.encode_image(images.to(args.device))
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
    elif args.reranker:
        # single vector of both image and text
        descriptors, text_features, img_local, text_local = model.encode_single(images.to(args.device), texts)
        vision_descriptors[indices.numpy(), :] = descriptors.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()
        img_local_descs[indices.numpy(), :] = img_local.cpu().float().numpy()
        text_local_descs[indices.numpy(), :] = text_local.cpu().float().numpy()
    else:
        descriptors, text_features, _, _ = model.encode_single(images.to(args.device), texts)
        vision_descriptors[indices.numpy(), :] = descriptors.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features.cpu().float().numpy()        
        
            
def get_queries_predictions(encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results):
     # Use a kNN to find predictions
    #faiss_index = faiss.IndexFlatL2(encoder_dim)
    faiss_index = faiss.IndexFlatIP(encoder_dim)
    #normilize descriptors for cosine similarity
    database_descriptors = normlize_features(database_descriptors)      
    queries_descriptors = normlize_features(queries_descriptors)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logger.debug("Calculating recalls")
    scores, predictions = faiss_index.search(queries_descriptors, max_results)
    return scores, predictions

# ... (Keep your existing imports at the top)

def rerank_predictions(model, test_ds, predictions, vision_descriptors, text_descriptors, img_local_descs, text_local_desc, max_rerank_k=25, device="cuda"):
    logger.info(f"Reranking top-{max_rerank_k} candidates using Cross-Attention...")

    rerank_model = model.single_encoder
    rerank_model.to(device)
    rerank_model.eval()

    reranked_predictions = predictions.copy()

    img_local_descs_tensor = torch.from_numpy(img_local_descs).to(device)
    text_local_desc_tensor = torch.from_numpy(text_local_desc).to(device)
    vision_descriptors_tensor = torch.from_numpy(vision_descriptors).to(device)
    text_descriptors_tensor = torch.from_numpy(text_descriptors).to(device)

    with torch.no_grad():
        for q_idx in tqdm(range(test_ds.num_queries), desc="Reranking queries"):
            actual_q_ds_idx = test_ds.num_database + q_idx
            
            # Extract raw query sequence
            query_text_global = text_descriptors_tensor[actual_q_ds_idx].unsqueeze(0) # [1, D]
            raw_text_local = text_local_desc_tensor[actual_q_ds_idx] # [Lt, D]
            
            # --- FIX: Dynamically identify and remove padding tokens ---
            # Find tokens that are NOT completely zero vectors
            non_zero_mask = raw_text_local.any(dim=-1) 
            true_len = non_zero_mask.sum().item()
            
            # If the entire row is somehow zero fallback to 1 token, otherwise slice down to true length
            true_len = max(1, true_len)
            query_text_local = raw_text_local[:true_len].unsqueeze(0)  # [1, True_Lt, D]
            # -----------------------------------------------------------

            candidate_db_indices = predictions[q_idx, :max_rerank_k]
            candidate_img_local = img_local_descs_tensor[candidate_db_indices]  # [K, Li, D]
            candidate_img_global = vision_descriptors_tensor[candidate_db_indices] # [K, D]

            # Expand the trimmed text tokens perfectly
            True_Lt, D_dim = query_text_local.shape[1], query_text_local.shape[2]
            text_local_expanded = query_text_local.expand(len(candidate_db_indices), True_Lt, D_dim)
            text_global_expanded = query_text_global.expand(len(candidate_db_indices), -1)

            if next(rerank_model.parameters()).dtype == torch.bfloat16:
                candidate_img_local = candidate_img_local.bfloat16()
                text_local_expanded = text_local_expanded.bfloat16()
                candidate_img_global = candidate_img_global.bfloat16()
                text_global_expanded = text_global_expanded.bfloat16()

            # Now cross-attention only sees real semantic tokens
            scores = rerank_model.cross_attn_classifier(candidate_img_local, text_local_expanded, candidate_img_global, text_global_expanded, force_local=False)
            scores = scores.cpu().numpy()
            
            reranked_order = np.argsort(-scores)
            reranked_predictions[q_idx, :max_rerank_k] = candidate_db_indices[reranked_order]

    return reranked_predictions


# import asyncio
# import base64
# import json
# import mimetypes
# import os
# import re
# from openai import AsyncOpenAI


# def rerank_by_mllm(
#     image_paths, target_text, predictions, max_concurrent_requests=128, debug=True
# ):
#     """Reranks candidates based on Viewpoint-Invariant visual scene text & landmark matching.

#     Upgraded for Gemma-4-26B-A4B execution leveraging large batching & guided JSON decoding.
#     """
#     if len(image_paths) != len(predictions):
#         raise ValueError(
#             f"Length mismatch: {len(image_paths)} image paths vs {len(predictions)} predictions."
#         )

#     async def encode_image_async(image_path):
#         def _read_and_b64(path):
#             mime_type, _ = mimetypes.guess_type(path)
#             if not mime_type:
#                 mime_type = "image/jpeg"
#             with open(path, "rb") as image_file:
#                 b64_str = base64.b64encode(image_file.read()).decode("utf-8")
#                 return f"data:{mime_type};base64,{b64_str}"

#         return await asyncio.to_thread(_read_and_b64, image_path)

#     async def evaluate_single_image(client, semaphore, orig_rank, pred_index, path):
#         async with semaphore:
#             if not os.path.exists(path):
#                 if debug:
#                     print(f"[Debug] Path missing: {path}")
#                 return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

#             try:
#                 img_data_url = await encode_image_async(path)

#                 # Gemma-4 optimized unified user message
#                 prompt_text = f"""You are an expert visual place recognition and OCR assistant.
# Analyze this image to evaluate if it shows the location described by the target text query.

# TARGET QUERY: "{target_text}"

# CRITICAL VIEWPOINT-INVARIANT GUIDELINES:
# 1. VIEWPOINT & ANGLE INVARIANCE: The database image may be taken from a DIFFERENT camera angle, opposite street direction, or different field-of-view than the text description. DO NOT penalize the image if elements appear on different sides (left/right/center) or in a different spatial order!
# 2. KEY VISUAL ANCHORS: Look for shop names, storefront signs, street signs, building facades, banners, logos, and plaques mentioned in or matching the query.
# 3. MATCH RULE: If the key text, shop names, or primary landmarks mentioned in the query are present anywhere in the image, it is a Strong Match.

# SCORING CRITERIA (0 to 100):
# - 85–100: High confidence. Key shop names, street signs, or prominent landmark text from the query are clearly present in the scene.
# - 50–84: Partial confidence. Moderate text match, secondary landmarks visible, or partially occluded shop name.
# - 0–49: Low confidence. Completely different location, unrelated storefront text, or no matching landmarks.

# Generate your chain-of-thought analysis within reasoning tags, then output the score and concise justification following the structural schema requested."""

#                 # Enforce native schema constraints at the engine level to bypass regex parsing errors
#                 response = await client.chat.completions.create(
#                     model="nvidia/Gemma-4-26B-A4B-NVFP4",  # Match your vLLM model string
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": prompt_text.strip()},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {"url": img_data_url},
#                                 },
#                             ],
#                         }
#                     ],
#                     temperature=0.0,
#                     max_tokens=256,  # Raised slightly to accommodate reasoning loops + JSON payload
#                     response_format={
#                         "type": "json_object",
#                         "schema": {
#                             "type": "object",
#                             "properties": {
#                                 "score": {"type": "number", "minimum": 0, "maximum": 100},
#                                 "reason": {"type": "string"}
#                             },
#                             "required": ["score", "reason"]
#                         }
#                     }
#                 )

#                 raw_output = response.choices[0].message.content.strip()

#                 score = 0.0
#                 try:
#                     data = json.loads(raw_output)
#                     score = float(data.get("score", 0.0))
#                 except Exception:
#                     # Fallback structural extraction regex if manual payload bypass occurs
#                     match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw_output)
#                     if match:
#                         score = float(match.group(1))

#                 if debug:
#                     print(
#                         f"[Debug] Orig Rank {orig_rank:02d} | Pred {pred_index} | Score: {score} | File: {os.path.basename(path)}"
#                     )

#                 return {"pred_index": pred_index, "score": score, "orig_rank": orig_rank}

#             except Exception as e:
#                 if debug:
#                     print(f"[Debug Error] Candidate {pred_index} failed: {e}")
#                 return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

#     async def run_batch():
#         semaphore = asyncio.Semaphore(max_concurrent_requests)
#         async with AsyncOpenAI(
#             base_url="http://localhost:8000/v1", api_key="not-needed"
#         ) as client:
#             tasks = [
#                 evaluate_single_image(client, semaphore, orig_idx, pred_idx, path)
#                 for orig_idx, (pred_idx, path) in enumerate(zip(predictions, image_paths))
#             ]
#             return await asyncio.gather(*tasks)

#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = None

#     if loop and loop.is_running():
#         import nest_asyncio

#         nest_asyncio.apply()
#         results = loop.run_until_complete(run_batch())
#     else:
#         results = asyncio.run(run_batch())

#     # --- TIE-BREAKING SORTING LOGIC ---
#     sorted_results = sorted(
#         results, key=lambda x: (-x["score"], x["orig_rank"])
#     )

#     reranked_preds = [res["pred_index"] for res in sorted_results]

#     if debug:
#         print("\n--- Final Reranked Candidate Order ---")
#         for new_rank, res in enumerate(sorted_results):
#             print(
#                 f"New Rank {new_rank+1:02d}: Candidate {res['pred_index']} (Orig Rank: {res['orig_rank']}) -> Score: {res['score']}"
#             )

#     return reranked_preds


import asyncio
import base64
import json
import mimetypes
import os
import re
from openai import AsyncOpenAI


def rerank_by_mllm(
    image_paths, target_text, predictions, max_concurrent_requests=8, debug=True
):
    """Reranks candidates based on Viewpoint-Invariant visual scene text & landmark matching.

    Preserves the original Stage 1 retrieval rank whenever candidates have identical MLLM scores.
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
- 85–100: High confidence. Key shop names, street signs, or prominent landmark text from the query are clearly present in the scene.
- 50–84: Partial confidence. Moderate text match, secondary landmarks visible, or partially occluded shop name.
- 0–49: Low confidence. Completely different location, unrelated storefront text, or no matching landmarks.

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
                                {
                                    "type": "image_url",
                                    "image_url": {"url": img_data_url},
                                },
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_tokens=80,
                )

                raw_output = response.choices[0].message.content.strip()

                # Score extraction logic
                score = 0.0
                try:
                    clean_json = raw_output
                    if "```" in clean_json:
                        clean_json = re.sub(
                            r"```[a-zA-Z]*", "", clean_json
                        ).strip()
                    data = json.loads(clean_json)
                    score = float(data.get("score", data.get("confidence_score", 0.0)))
                except Exception:
                    match = re.search(
                        r'"(?:score|confidence_score)"\s*:\s*(\d+(?:\.\d+)?)',
                        raw_output,
                    )
                    if match:
                        score = float(match.group(1))

                if debug:
                    print(
                        f"[Debug] Orig Rank {orig_rank:02d} | Pred {pred_index} | Score: {score} | File: {os.path.basename(path)}"
                    )

                return {"pred_index": pred_index, "score": score, "orig_rank": orig_rank}

            except Exception as e:
                if debug:
                    print(f"[Debug Error] Candidate {pred_index} failed: {e}")
                return {"pred_index": pred_index, "score": 0.0, "orig_rank": orig_rank}

    async def run_batch():
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        async with AsyncOpenAI(
            base_url="http://localhost:8000/v1", api_key="not-needed"
        ) as client:
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

    # --- TIE-BREAKING SORTING LOGIC ---
    # Primary Key:   score (Descending -> Higher is better)
    # Secondary Key: orig_rank (Ascending -> Smaller original index comes first)
    sorted_results = sorted(
        results, key=lambda x: (-x["score"], x["orig_rank"])
    )

    reranked_preds = [res["pred_index"] for res in sorted_results]

    if debug:
        print("\n--- Final Reranked Candidate Order ---")
        for new_rank, res in enumerate(sorted_results):
            print(
                f"New Rank {new_rank+1:02d}: Candidate {res['pred_index']} (Orig Rank: {res['orig_rank']}) -> Score: {res['score']}"
            )

    return reranked_preds



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    start_time = datetime.now()

    logger.remove()  # Remove possibly previously existing loggers
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(f"Testing with {args.model_name}")
    logger.info(f"The outputs are being saved in {log_dir}")

    IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    BLIP_MEAN_STD = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
    SIGLIP_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}

    dataset_mean_std = IMAGENET_MEAN_STD
    if 'blip' in args.model_name.lower() or 'clip' in args.model_name.lower() or 'eva' in args.model_name.lower():
        dataset_mean_std = BLIP_MEAN_STD
    elif 'siglip' in args.model_name.lower():
        dataset_mean_std = SIGLIP_MEAN_STD

    model = LaVPR_wrapper(args)
    logger.info(f"VLM encoder dim: {model.encoder_dim}")

    is_msls_challenge = False
    # Determine local feature dimensions
    # For ViT-B/16 with 224x224 images, num_patches = (224/16)^2 = 196. With CLS token, it's 197.
    # For text, it's the max token length, e.g., 77 for CLIP.
    # These are hardcoded for now but could be made dynamic.
    num_img_tokens = 197  # (14*14) + 1 for ViT-B/16
    num_text_tokens = 77 # Max length for CLIP

    if 'msls_challenge' in args.image_root:        
        test_ds = MSLSTest(dataset_root=args.database_folder, image_root=args.image_root, csv_path=args.queries_csv, mean_std=dataset_mean_std, image_size=args.image_size)
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
    all_descriptors = None
    vision_descriptors = None
    text_descriptors = None
    
    max_results = max(args.recall_values)
    query_index = 0

    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )

        vision_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        text_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")            
        all_descriptors = np.empty((len(test_ds), model.encoder_dim), dtype="float32")
        img_local_descs = np.empty((len(test_ds), num_img_tokens, 768), dtype="float32")
        text_local_desc = np.empty((len(test_ds), num_text_tokens, model.encoder_dim), dtype="float32")
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, img_local_descs, text_local_desc)

        query_index = test_ds.num_database
        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)#1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors, img_local_descs, text_local_desc)
        

    if args.cross_modal:        
        if args.text_only:
            database_descriptors = text_descriptors[: test_ds.num_database]    
        else:
            database_descriptors = vision_descriptors[: test_ds.num_database]    
        queries_descriptors = text_descriptors[test_ds.num_database :]
    else:
        database_descriptors = text_descriptors[: test_ds.num_database]    
        queries_descriptors = text_descriptors[test_ds.num_database :]        
    
    # 1. Initial Quick Retrieval via Global Vectors
    scores, predictions = get_queries_predictions(
        model.encoder_dim, database_descriptors, all_descriptors, queries_descriptors, max_results
    )       
    
    # 2. Apply Fine-Grained Cross-Attention Reranking
    # Set how many top candidates you want to rerank (e.g., top 25 or top 50)
    if args.reranker:
        max_rerank_k = min(args.max_rerank, max_results) 
        predictions = rerank_predictions(
            model, test_ds, predictions, vision_descriptors, text_descriptors, img_local_descs, text_local_desc, max_rerank_k=max_rerank_k, device=args.device
        )
        
    if args.reranker_mllm:
        # get query texts
        q_texts = test_ds.descriptions[test_ds.num_database :]
        # get db image paths for predictions per query text
        db_paths_array = np.array(test_ds.images_paths[:test_ds.num_database    ])
        db_images = db_paths_array[predictions]
        for i in range(len(q_texts)):
            print(f"Sending query {i} to LLM")
            q = q_texts[i]
            db = db_images[i]
            predictions[i] = rerank_by_mllm(db, q, predictions[i])

    # 3. Calculate metrics using the updated reranked predictions
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
            recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
            logger.info(f"{recalls_str}")
            
            model_path = args.lora_path if args.lora_path is not None else args.model_path
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name},{recalls_str}\n")
            
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels, test_ds.images_paths_csv, texts=test_ds.descriptions
        )

        
    

if __name__ == "__main__":
    args = eval_parser.parse_arguments()
    # Ensure args.device is set (fallback to cuda if missing)
    if not hasattr(args, 'device'):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)
