import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch

def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?'):
        
        embed_size = r_list.shape[1]
        if faiss_gpu:
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = True
            flat_config.device = 0
            faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
        # build index
        else:
            faiss_index = faiss.IndexFlatL2(embed_size)
        
        # add references
        r_list = r_list.to(torch.float32)
        q_list = q_list.to(torch.float32)

        faiss_index.add(r_list)

        # search for queries in the index
        _, predictions = faiss_index.search(q_list, max(k_values))
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.isin(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print() # print a new line
            table = PrettyTable()
            table.field_names = ['K']+[str(k) for k in k_values]
            table.add_row(['Recall@K']+ [f'{100*v:.2f}' for v in correct_at_k])
            print(table.get_string(title=f"Performances on {dataset_name}"))
        
        return d


def get_validation_recalls_rerank(
    r_list, 
    q_list, 
    k_values, 
    gt, 
    print_results=True, 
    faiss_gpu=False, 
    dataset_name='dataset without name ?',
    # --- Reranking additions ---
    rerank_model=None,          # Your LaVPR_reranker or CrossAttnClassifier model
    r_local_list=None,          # Gathered database img_local tensors [Num_Db, Li, D]
    q_local_list=None,          # Gathered query text_local tensors [Num_Q, Lt, D]
    max_rerank_k=25             # How many top FAISS candidates to rerank
):
    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)
    
    r_list = r_list.to(torch.float32).cpu().numpy()
    q_list = q_list.to(torch.float32).cpu().numpy()

    faiss_index.add(r_list)

    # Search for queries using a safe cap (at least max(k_values) or max_rerank_k)
    search_k = max(max(k_values), max_rerank_k)
    _, predictions = faiss_index.search(q_list, search_k)
    
    # -------------------------------------------------------------------------
    # --- Fine-Grained Cross-Attention Reranking Stage ---
    # -------------------------------------------------------------------------
    if rerank_model is not None and r_local_list is not None and q_local_list is not None:
        print(f"\n--> Reranking top-{max_rerank_k} validation candidates using Cross-Attention...")
        
        # Pull internal neural classifier if it's wrapped inside PyTorch Lightning
        classifier = rerank_model.cross_attn_classifier if hasattr(rerank_model, 'cross_attn_classifier') else rerank_model
        device = next(classifier.parameters()).device
        # Determine the model's expected dtype (e.g., bfloat16 or float32)
        model_dtype = next(classifier.parameters()).dtype
        classifier.eval()
        
        reranked_predictions = predictions.copy()

        # Convert global descriptors to tensors for indexing
        r_list_tensor = torch.from_numpy(r_list).to(device=device, dtype=model_dtype)
        q_list_tensor = torch.from_numpy(q_list).to(device=device, dtype=model_dtype)
        num_queries = len(predictions)
        
        with torch.no_grad():
            for q_idx in range(num_queries):
                # 1. Get the local token window for this query [1, Lt, D]
                # Cast to the model's expected dtype to prevent mismatch
                q_text_local = q_local_list[q_idx].unsqueeze(0).to(device=device, dtype=model_dtype) 
                q_text_global = q_list_tensor[q_idx].unsqueeze(0) # [1, D]
                
                # 2. Extract indices of top global candidates to rerank
                candidate_db_indices = predictions[q_idx, :max_rerank_k]
                
                # 3. Gather and cast local patch maps for these candidates [K, Li, D]
                cand_img_locals = torch.stack([r_local_list[db_idx] for db_idx in candidate_db_indices]).to(device=device, dtype=model_dtype)
                cand_img_globals = r_list_tensor[candidate_db_indices] # [K, D]
                
                # 4. Match dimensions for cross-attention broadcast
                Lt, D_dim = q_text_local.shape[1], q_text_local.shape[2]
                q_text_local_expanded = q_text_local.expand(len(candidate_db_indices), Lt, D_dim)
                q_text_global_expanded = q_text_global.expand(len(candidate_db_indices), -1) # [K, D]
                
                # 5. Compute fine-grained scores [K]
                scores = classifier(
                    cand_img_locals, 
                    q_text_local_expanded, 
                    cand_img_globals, 
                    q_text_global_expanded
                ).cpu().numpy()
                
                # 6. Sort candidate index subset descending by attention score
                reranked_order = np.argsort(-scores)
                reranked_predictions[q_idx, :max_rerank_k] = candidate_db_indices[reranked_order]
        
        # Replace predictions with our refined reranked array for metrics extraction
        predictions = reranked_predictions
    # -------------------------------------------------------------------------

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.isin(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break
    
    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100*v:.2f}' for v in correct_at_k])
        title_suffix = f" (Reranked @{max_rerank_k})" if rerank_model is not None else ""
        print(table.get_string(title=f"Performances on {dataset_name}{title_suffix}"))
    
    return d
