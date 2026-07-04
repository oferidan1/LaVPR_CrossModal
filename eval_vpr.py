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


def encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors):
    if args.bfloat16:
        images = images.bfloat16()

    if args.cross_modal==1:
        image_features = model.encode_text(texts)
        image_features = image_features.cpu().float().numpy()
        vision_descriptors[indices.numpy(), :] = image_features     
        text_features = model.encode_image(images.to(args.device))
        text_features = text_features.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features                
    else:
        # single vector of both image and text
        descriptors, text_features = model.encode_single(images.to(args.device), texts)
        descriptors = descriptors.cpu().float().numpy()
        vision_descriptors[indices.numpy(), :] = descriptors
        text_features = text_features.cpu().float().numpy()
        text_descriptors[indices.numpy(), :] = text_features
        
            
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

def rerank_predictions(model, test_ds, queries_dataloader, predictions, max_rerank_k=25, device="cuda"):
    """
    Reranks the top-k predictions using the Cross-Attention Classifier.
    """
    logger.info(f"Reranking top-{max_rerank_k} candidates using Cross-Attention...")
    
    # Access the underlying Lightning module if wrapped, or use directly
    rerank_model = model.model if hasattr(model, 'model') else model
    rerank_model.to(device)
    rerank_model.eval()

    # We need the database dataset to pull image data for candidate predictions
    # Assuming test_ds allows access by index or we can rebuild a loader
    reranked_predictions = predictions.copy()

    # Step 1: Pre-extract or dynamically fetch local features
    # To keep memory footprint low and handle variable sizes, we loop through queries
    with torch.no_grad():
        for q_idx in tqdm(range(test_ds.num_queries), desc="Reranking queries"):
            # Get the query text sequence tokens/local features
            # test_ds stores queries after database items
            actual_q_ds_idx = test_ds.num_database + q_idx
            _, _, q_text = test_ds[actual_q_ds_idx]
            
            # Encode query text to get token-level sequences
            _, text_local, _, _, _ = rerank_model.encode_text([q_text]) # shape: [1, Lt, D]
            
            # Get top-k candidate indices from global retrieval
            candidate_db_indices = predictions[q_idx, :max_rerank_k]
            
            # Gather and encode candidate images
            cand_images = []
            for db_idx in candidate_db_indices:
                img, _, _ = test_ds[db_idx]
                cand_images.append(img)
            
            # Stack images into a batch [max_rerank_k, C, H, W]
            cand_images = torch.stack(cand_images).to(device)
            if next(rerank_model.parameters()).dtype == torch.bfloat16:
                cand_images = cand_images.bfloat16()
                
            _, img_local, _ = rerank_model.encode_image(cand_images) # shape: [K, Li, D]
            
            # Expand text tokens to match the number of candidate images
            Lt, D_dim = text_local.shape[1], text_local.shape[2]
            text_local_expanded = text_local.expand(len(candidate_db_indices), Lt, D_dim) # shape: [K, Lt, D]
            
            # Compute cross-attention scores [K]
            scores = rerank_model.cross_attn_classifier(img_local, text_local_expanded)
            scores = scores.cpu().numpy()
            
            # Sort candidate indices based on the new fine-grained cross-attention scores
            reranked_order = np.argsort(-scores) # Sort descending
            
            # Update the top-k spots in your predictions matrix
            reranked_predictions[q_idx, :max_rerank_k] = candidate_db_indices[reranked_order]

    return reranked_predictions


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
            
        for images, indices, texts in tqdm(database_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors)

        query_index = test_ds.num_database
        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size)#1)
        for images, indices, texts in tqdm(queries_dataloader):
            encode_batch(model, args, images, texts, indices, all_descriptors, vision_descriptors, text_descriptors)
        

    max_results_reranking = test_ds.num_database    
        
    vision_database_descriptors = vision_descriptors[: test_ds.num_database]    
    text_queries_descriptors = text_descriptors[test_ds.num_database :]
    
    # 1. Initial Quick Retrieval via Global Vectors
    scores, predictions = get_queries_predictions(
        model.encoder_dim, vision_database_descriptors, all_descriptors, text_queries_descriptors, max_results
    )       
    
    # 2. Apply Fine-Grained Cross-Attention Reranking
    # Set how many top candidates you want to rerank (e.g., top 25 or top 50)
    max_rerank_k = min(25, max_results) 
    predictions = rerank_predictions(
        model, test_ds, queries_dataloader, predictions, max_rerank_k=max_rerank_k, device=args.device
    )

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
            logger.info(f"Reranked Metrics -> {recalls_str}")
            
            model_path = args.lora_path if args.lora_path is not None else args.model_path
            with open("eval_vpr_results.csv", "a") as f:
                f.write(f"{model_path},{args.model_name},Reranked_{max_rerank_k}: {recalls_str}\n")
            
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
