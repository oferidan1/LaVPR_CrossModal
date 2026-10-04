import os
import argparse
import torch
import pandas as pd
from transformers import AutoProcessor

def precompute_dataset_idf(args):
    """
    Computes image-level document frequencies over all captions
    and saves a static IDF tensor for SuperCLIP token classification loss.
    """
    print(f"Initializing tokenizer: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    vocab_size = processor.tokenizer.vocab_size

    # Collect special tokens to exclude from IDF mass
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    bos_id = getattr(processor.tokenizer, "bos_token_id", 49406) or 49406
    eos_id = getattr(processor.tokenizer, "eos_token_id", 49407) or 49407
    special_ids = {pad_id, bos_id, eos_id}

    df_data = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df_data)} entries from {args.csv_file}")

    total_images = len(df_data)
    image_token_counts = torch.zeros(vocab_size, dtype=torch.int64)

    # Process captions in mini-batches
    for i in range(0, total_images, args.batch_size):
        batch_df = df_data.iloc[i : i + args.batch_size]
        batch_texts = batch_df["description"].astype(str).tolist()

        # Tokenize without padding to avoid padding token distortion
        text_inputs = processor(
            text=batch_texts,
            padding=False,
            truncation=True,
            max_length=args.max_len,
            return_tensors=None,
        )
        tokens_list = text_inputs['input_ids']

        for row in tokens_list:
            # Drop special tokens (BOS, EOS, PAD)
            unique_tokens = set(row) - special_ids
            if unique_tokens:
                image_token_counts[list(unique_tokens)] += 1

        if (i + args.batch_size) % (args.batch_size * 5) == 0 or (i + args.batch_size) >= total_images:
            processed = min(i + args.batch_size, total_images)
            print(f"  -> Processed {processed}/{total_images} items...")

    print("\nComputing log-IDF tensor...")
    df_image = image_token_counts.float()

    # Standard SuperCLIP formulation: log(|D| / (1 + df))
    idf_tensor = torch.log(total_images / (1.0 + df_image))

    # Zero out special token weights and clamp non-negative
    for s_id in special_ids:
        if s_id < vocab_size:
            idf_tensor[s_id] = 0.0
    idf_tensor.clamp_(min=0.0)

    torch.save(idf_tensor, args.out_file)
    print(f"Success! Image-level IDF cached to: {args.out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_fixed_3.csv")
    parser.add_argument("--out_file", type=str, default="gsv_cities_token_idf.pt")
    parser.add_argument("--max_len", type=int, default=77, help="Max text context length (248 for Long-CLIP)")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for text processing")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16", help="Model / tokenizer path")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    precompute_dataset_idf(args)