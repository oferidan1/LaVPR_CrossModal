import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoProcessor


def analyze_token_lengths(args):
    print(f"Initializing tokenizer: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    bos_id = getattr(processor.tokenizer, "bos_token_id", 49406) or 49406
    eos_id = getattr(processor.tokenizer, "eos_token_id", 49407) or 49407
    special_ids = {pad_id, bos_id, eos_id}

    df_data = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df_data)} entries from {args.csv_file}")

    total_images = len(df_data)
    all_lengths = []

    for i in range(0, total_images, args.batch_size):
        batch_df = df_data.iloc[i: i + args.batch_size]
        batch_texts = batch_df["description"].astype(str).tolist()

        text_inputs = processor(
            text=batch_texts,
            padding=False,
            truncation=False,  # no truncation so we see true lengths
            return_tensors=None,
        )
        tokens_list = text_inputs['input_ids']

        for row in tokens_list:
            # exclude special tokens from length count
            content_tokens = [t for t in row if t not in special_ids]
            all_lengths.append(len(content_tokens))

        if (i + args.batch_size) % (args.batch_size * 5) == 0 or (i + args.batch_size) >= total_images:
            processed = min(i + args.batch_size, total_images)
            print(f"  -> Processed {processed}/{total_images} items...")

    lengths = np.array(all_lengths)

    print("\n=== Token Length Statistics (excluding special tokens) ===")
    print(f"  Min:    {lengths.min()}")
    print(f"  Max:    {lengths.max()}")
    print(f"  Mean:   {lengths.mean():.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Std:    {lengths.std():.1f}")
    print(f"  p25:    {np.percentile(lengths, 25):.1f}")
    print(f"  p75:    {np.percentile(lengths, 75):.1f}")
    print(f"  p90:    {np.percentile(lengths, 90):.1f}")
    print(f"  p95:    {np.percentile(lengths, 95):.1f}")
    print(f"  p99:    {np.percentile(lengths, 99):.1f}")

    # show how many would be truncated at common cutoffs
    print("\n=== Truncation impact ===")
    for cutoff in [77, 120, 150, 200, 248]:
        truncated = (lengths > cutoff).sum()
        print(f"  > {cutoff} tokens: {truncated:>7} ({100*truncated/len(lengths):.1f}%)")

    # distribution histogram in terminal
    print("\n=== Length distribution ===")
    bins = [0, 20, 40, 60, 77, 100, 120, 150, 200, 248, 999]
    for b_start, b_end in zip(bins[:-1], bins[1:]):
        count = ((lengths >= b_start) & (lengths < b_end)).sum()
        bar = '█' * int(50 * count / len(lengths))
        print(f"  {b_start:>4}-{b_end:<4}: {count:>7} ({100*count/len(lengths):5.1f}%) {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--model_name", type=str, default="creative-graphic-design/LongCLIP-B")
    parser.add_argument("--gpu", type=str, default="0")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    analyze_token_lengths(args)

