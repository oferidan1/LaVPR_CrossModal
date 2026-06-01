import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoProcessor

def precompute_dataset_idf(args):
    """
    Computes document frequencies over all Gemini captions in the training set.
    """
    token_counts = Counter()
    total_documents = 0
    
    #load clip model
    #model = AutoModel.from_pretrained(args.model_name).to("cuda")
    processor = AutoProcessor.from_pretrained(args.model_name)
    max_text_length = args.max_len
    
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} descriptions from {args.csv_file}")              
    
    for i in range(0, len(df), args.batch_size):
        batch_df = df.iloc[i:i+args.batch_size]
        batch_texts = batch_df["description"].tolist()            
     
        # If your dataloader returns raw strings, tokenize them
        text_inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_text_length)
        tokens = text_inputs.input_ids
        
        for row in tokens:
            unique_tokens = torch.unique(row).tolist()
            # Remove padding/special tokens if necessary, though IDF handles them
            token_counts.update(unique_tokens)
            total_documents += 1
    
    # Access the vocab size directly from the tokenizer component
    vocab_size = processor.tokenizer.vocab_size
    # Compute the final IDF tensor
    idf_tensor = torch.zeros(vocab_size, dtype=torch.float32)
    for token_id in range(vocab_size):
        df = token_counts.get(token_id, 0)
        # Match SuperCLIP Eq: log(|D| / (1 + df))
        idf_tensor[token_id] = np.log(total_documents / (1.0 + df))
        
    torch.save(idf_tensor, args.out_file)
    print("Dataset Token IDF cached successfully!")
    
    
import os    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--csv_file", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")
    # parser.add_argument("--out_file", type=str, default="pitts30k_val_800_queries_objects.json")
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")
    parser.add_argument("--out_file", type=str, default="gsv_cities_clip_b32_idf.pt")    
    parser.add_argument("--max_len", type=int, default="77", help="max number of tokens in text")
    parser.add_argument("--batch_size", type=int, default="1024", help="batch size for processing descriptions")    
    parser.add_argument("--load_aggregated", type=int, default="1", help="load aggregated")    
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="type of model to apply")        
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use (e.g. '0' or '0,1' for multiple GPUs)")
    
    args = parser.parse_args()           

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    precompute_dataset_idf(args)    