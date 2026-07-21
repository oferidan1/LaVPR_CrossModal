import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor
import argparse
import os

def precompute_dual_idfs(args):
    """
    Computes both image-level and location-level document frequencies 
    and saves them as separate, clean static tensors.
    """
    # 1. Initialize the tokenizer processor
    print(f"Initializing tokenizer: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    vocab_size = processor.tokenizer.vocab_size
    pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    # 2. Load description dataset (Expecting description and location identifier columns)
    df_data = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df_data)} entries from {args.csv_file}")
    
    total_images = len(df_data)
    total_locations = df_data[args.location_col].nunique()
    print(f"Dataset Stats: {total_images} images across {total_locations} unique locations.")
    
    # 3. Initialize separate frequency counters
    # Image-level counts: how many images contain token t
    image_token_counts = torch.zeros(vocab_size, dtype=torch.int64)
    
    # Location-level tracking: map token_id -> set of unique location_ids it appeared in
    token_to_locations = {i: set() for i in range(vocab_size)}
    
    # 4. Process dataset in chunks
    for i in range(0, len(df_data), args.batch_size):
        batch_df = df_data.iloc[i:i+args.batch_size]
        batch_texts = batch_df["description"].astype(str).tolist()
        batch_locs = batch_df[args.location_col].tolist()
        
        # Tokenize without padding to avoid dummy token pollution
        text_inputs = processor(
            text=batch_texts, 
            padding=False, 
            truncation=True, 
            max_length=args.max_len,
            return_tensors=None  # Explicitly prevent tensor conversion
        )
        tokens_list = text_inputs.input_ids
        
        # Map tokens to image counters and location sets
        for row, loc_id in zip(tokens_list, batch_locs):
            unique_tokens = set(row)
            unique_tokens.discard(pad_token_id)
            
            if unique_tokens:
                unique_indices = list(unique_tokens)
                
                # Increment image-level frequency
                image_token_counts[unique_indices] += 1
                
                # Append current location identity to the token's spatial history
                for t_id in unique_indices:
                    token_to_locations[t_id].add(loc_id)
                    
        if (i + args.batch_size) % (args.batch_size * 5) == 0 or (i + args.batch_size) >= len(df_data):
            processed_count = min(i + args.batch_size, len(df_data))
            print(f"  -> Processed {processed_count}/{len(df_data)} items...")

    # 5. Extract total location unique counts into a single tensor
    location_token_counts = torch.zeros(vocab_size, dtype=torch.float32)
    for t_id in range(vocab_size):
        location_token_counts[t_id] = len(token_to_locations[t_id])
        
    # =====================================================================
    # 6. COMPUTE LOG INDEPENDENT IDFs
    # =====================================================================
    print("\nVectorizing separate IDF representations...")
    df_image = image_token_counts.float()
    df_location = location_token_counts
    
    # Standard SuperCLIP Image IDF: log(|D_images| / (1 + df_image))
    idf_image = torch.log(total_images / (1.0 + df_image))
    
    # Geo-Spatial Location IDF: log(|N_locations| / (1 + df_location))
    idf_location = torch.log(total_locations / (1.0 + df_location))
    
    # Clean up structural tokens and clamp numerical roundings
    for tensor in [idf_image, idf_location]:
        tensor[pad_token_id] = 0.0
        tensor.clamp_(min=0.0)
        
    # 7. Save outputs independently
    torch.save(idf_image, args.out_img_file)
    torch.save(idf_location, args.out_loc_file)
    
    print(f"Success! Image IDF cached to: {args.out_img_file}")
    print(f"Success! Location IDF cached to: {args.out_loc_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_places.csv")
    parser.add_argument("--location_col", type=str, default="place_id", help="Column containing unique location ids/coordinates")
    
    # Output file names
    parser.add_argument("--out_img_file", type=str, default="gsv_cities_image_idf.pt")    
    parser.add_argument("--out_loc_file", type=str, default="gsv_cities_location_idf.pt")        
    
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for text processing")    
    #parser.add_argument("--max_len", type=int, default=77, help="Max text context length")
    #parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16", help="Target CLIP backbone")        
    parser.add_argument("--max_len", type=int, default=64, help="Max text context length")
    parser.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224", help="Target CLIP backbone")        
    parser.add_argument("--gpu", type=str, default="0", help="Target GPU execution device")
    
    args = parser.parse_args()           

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    precompute_dual_idfs(args)