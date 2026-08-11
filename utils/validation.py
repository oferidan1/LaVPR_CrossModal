import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch
import tqdm

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


import torch
import numpy as np
import faiss
from prettytable import PrettyTable


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
    q_attention_mask_list=None, # [Num_Q, Lt] attention/padding masks
    max_rerank_k=25,             # How many top FAISS candidates to rerank
    force_local=True
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
    
    # Ensure numpy float32 conversion for FAISS
    r_list_np = r_list.detach().cpu().to(torch.float32).numpy() if isinstance(r_list, torch.Tensor) else r_list.astype(np.float32)
    q_list_np = q_list.detach().cpu().to(torch.float32).numpy() if isinstance(q_list, torch.Tensor) else q_list.astype(np.float32)

    faiss_index.add(r_list_np)

    # Search for queries using a safe cap
    search_k = max(max(k_values), max_rerank_k)
    _, predictions = faiss_index.search(q_list_np, search_k)
    
    # -------------------------------------------------------------------------
    # --- Fine-Grained Cross-Attention Reranking Stage ---
    # -------------------------------------------------------------------------
    if rerank_model is not None and r_local_list is not None and q_local_list is not None:
        print(f"\n--> Reranking top-{max_rerank_k} validation candidates using Cross-Attention...")
        
        # Extract underlying classifier if wrapped in PyTorch Lightning
        classifier = rerank_model.cross_attn_classifier if hasattr(rerank_model, 'cross_attn_classifier') else rerank_model
        device = next(classifier.parameters()).device
        model_dtype = next(classifier.parameters()).dtype
        classifier.eval()
        
        reranked_predictions = predictions.copy()

        # Convert global descriptors to PyTorch tensors
        r_list_tensor = torch.from_numpy(r_list_np).to(device=device, dtype=model_dtype)
        q_list_tensor = torch.from_numpy(q_list_np).to(device=device, dtype=model_dtype)
        num_queries = len(predictions)
        
        with torch.no_grad():
            for q_idx in range(num_queries):
                # 1. Get local token window for this query [1, Lt, D]
                q_text_local = q_local_list[q_idx].unsqueeze(0).to(device=device, dtype=model_dtype) 
                q_text_global = q_list_tensor[q_idx].unsqueeze(0) # [1, D]
                
                # 2. Extract index-matched top global FAISS candidates to rerank
                candidate_db_indices = predictions[q_idx, :max_rerank_k]
                num_candidates = len(candidate_db_indices)
                
                # 3. Gather local patch maps for candidate images [K, Li, D]
                cand_img_locals = torch.stack([r_local_list[db_idx] for db_idx in candidate_db_indices]).to(device=device, dtype=model_dtype)
                cand_img_globals = r_list_tensor[candidate_db_indices] # [K, D]
                
                # 4. Expand query representations across candidate dimension [K, Lt, D]
                Lt, D_dim = q_text_local.shape[1], q_text_local.shape[2]
                q_text_local_expanded = q_text_local.expand(num_candidates, Lt, D_dim)
                q_text_global_expanded = q_text_global.expand(num_candidates, -1)
                
                # --- ATTENTION MASK FIX ---
                if q_attention_mask_list is not None and q_attention_mask_list[q_idx] is not None:
                    raw_mask = q_attention_mask_list[q_idx].unsqueeze(0).to(device=device)
                    
                    # Convert mask so True strictly indicates PADDING positions
                    if raw_mask.dtype != torch.bool:
                        q_mask = (raw_mask == 0)
                    else:
                        q_mask = raw_mask
                        
                    q_mask_expanded = q_mask.expand(num_candidates, -1)
                else:
                    q_mask_expanded = None
                
                # 5. Compute cross-attention similarity scores [K]
                scores = classifier(
                    img_local=cand_img_locals, 
                    text_local=q_text_local_expanded, 
                    img_global=cand_img_globals, 
                    text_global=q_text_global_expanded,
                    text_attention_mask=q_mask_expanded,
                    force_local=force_local
                )
                
                # Ensure 1D numpy array shape [K]
                if isinstance(scores, torch.Tensor):
                    scores = scores.squeeze().detach().cpu().numpy()
                
                # 6. Sort candidate index subset descending by cross-attention score (+1.0 is best)
                reranked_order = np.argsort(-scores)
                reranked_predictions[q_idx, :max_rerank_k] = candidate_db_indices[reranked_order]
        
        predictions = reranked_predictions
    # -------------------------------------------------------------------------

    # Compute Recall@K metrics
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




def get_validation_recalls_rerank_2(
    r_list, 
    q_list, 
    k_values=[1, 5, 10, 15, 20, 50, 100], 
    gt=None, 
    print_results=True, 
    dataset_name="", 
    faiss_gpu=False,
    rerank_model=None, 
    r_local_list=None, 
    q_local_list=None, 
    q_attention_mask_list=None, 
    force_local=True,
    max_rerank_k=25
):
    """
    Evaluates global recall metrics (Stage 1 FAISS) and reranks top-K candidates 
    using Cross-Attention Score Head logits (Stage 2 rerank_model).
    """
    if isinstance(r_list, torch.Tensor):
        r_list = r_list.detach().cpu().numpy()
    if isinstance(q_list, torch.Tensor):
        q_list = q_list.detach().cpu().numpy()

    num_queries = len(q_list)
    max_k = max(k_values)
    
    # Ensure float32 contiguous arrays for FAISS
    r_list = np.ascontiguousarray(r_list, dtype=np.float32)
    q_list = np.ascontiguousarray(q_list, dtype=np.float32)

    # 1. Global Vector Retrieval (Stage 1 Candidate Retrieval)
    if r_list.shape[1] == q_list.shape[1]:
        dim = r_list.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        if faiss_gpu and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
            
        faiss_index.add(r_list)
        _, predictions = faiss_index.search(q_list, max_k)
    else:
        # Fallback if global dimensions differ: default to standard sequential candidate range
        predictions = np.tile(np.arange(max_k), (num_queries, 1))

    # 2. Stage 2 Candidate Reranking via Score Head Logits
    if rerank_model is not None and r_local_list is not None and q_local_list is not None:
        rerank_model.eval()
        device = next(rerank_model.parameters()).device
        target_dtype = next(rerank_model.parameters()).dtype

        # Ensure local feature structures are PyTorch tensors
        if not isinstance(r_local_list, torch.Tensor):
            r_local_list = torch.from_numpy(r_local_list)
        if not isinstance(q_local_list, torch.Tensor):
            q_local_list = torch.from_numpy(q_local_list)

        with torch.no_grad():
            for q_idx in range(num_queries):
                candidate_db_indices = predictions[q_idx, :max_rerank_k]
                num_cand = len(candidate_db_indices)
                
                # Fetch query local T5 sequence tokens [Lt, text_dim]
                query_text_local = q_local_list[q_idx] 
                
                # Dynamic masking: Slice valid non-padding T5 tokens
                if q_attention_mask_list is not None and q_attention_mask_list[q_idx] is not None:
                    raw_mask = q_attention_mask_list[q_idx]
                    non_zero_mask = (raw_mask != 0) if raw_mask.dtype != torch.bool else raw_mask
                    true_len = max(1, int(non_zero_mask.sum().item()))
                    
                    query_text_local = query_text_local[:true_len]
                    q_mask = raw_mask[:true_len].unsqueeze(0).to(device=device) # [1, true_len]
                    q_mask_expanded = q_mask.expand(num_cand, -1)
                else:
                    q_mask_expanded = None

                # Fetch visual patch tokens for candidates [num_cand, N_patches, img_dim]
                cand_img_local = r_local_list[candidate_db_indices].to(device=device, dtype=target_dtype)
                
                # Expand query T5 sequence tokens across candidate instances [num_cand, true_len, text_dim]
                true_lt, text_dim = query_text_local.shape[0], query_text_local.shape[1]
                text_local_expanded = query_text_local.unsqueeze(0).expand(num_cand, true_lt, text_dim).to(device=device, dtype=target_dtype)

                # Forward pass returns scalar logits from the Score Head
                logits = rerank_model(
                    img_local=cand_img_local,
                    text_local=text_local_expanded,
                    text_attention_mask=q_mask_expanded,
                    return_latent=False,
                    force_local=force_local
                )

                # Format logits to flat 1D numpy array and re-sort top-K candidates
                if isinstance(logits, torch.Tensor):
                    logits = logits.squeeze().detach().cpu().numpy()
                logits = np.atleast_1d(logits)

                # Sort candidates in descending order of matching probability
                reranked_order = np.argsort(-logits)
                predictions[q_idx, :max_rerank_k] = candidate_db_indices[reranked_order]

    # 3. Calculate Recall Metrics
    correct_at_k = {k: 0 for k in k_values}
    for q_idx in range(num_queries):
        pos_indices = set(gt[q_idx])
        pred_indices = predictions[q_idx]
        
        for k in k_values:
            if any(pred in pos_indices for pred in pred_indices[:k]):
                correct_at_k[k] += 1

    recalls = {k: correct_at_k[k] / num_queries for k in k_values}

    if print_results:
        print(f"\n--- Validation Results [{dataset_name}] ---")
        for k in k_values:
            print(f"Recall@{k:02d}: {recalls[k]:.4f}")

    return recalls