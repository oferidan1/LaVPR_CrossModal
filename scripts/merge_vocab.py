import json
import asyncio
import os
from collections import defaultdict
import httpx
import ast

# Configurations
# DATASET_PATH = "pitts30k_val_800_queries_objects_intermediate.json"
# OUTPUT_LOOKUP_PATH = "lavpr_llm_negative_lookup.json"
VOCAB_PATH = "datasets/gsv_cities_scene_graph_vocab.json"
OUTPUT_VOCAB_PATH = "vocab_clean.json"
VLLM_URL = "http://localhost:8000/v1/chat/completions"


import json
from openai import OpenAI

# Connect to your local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def process_chunk(chunk_dict):
    """
    Forces Gemma-4 to return a strict dictionary mapping.
    """
    prompt = f"""
    Analyze this dictionary fragment: {json.dumps(chunk_dict)}.
    Identify redundant words (synonyms, tenses, root variations).
    Pick the word with the lowest ID as the 'Keeper'.
    Output ONLY a JSON object: {{"removed_id": "kept_id"}}.
    Do not add any conversational text.
    """
    
    response = client.chat.completions.create(
        model="nvidia/Gemma-4-26B-A4B-NVFP4",
        messages=[
            {"role": "system", "content": "You are a JSON-only API. Output only valid JSON objects."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=32096,
        # Enforce structure
        extra_body={
            "guided_json": {
                "type": "object",
                "additionalProperties": {"type": "string"}
            }
        }
    )
    
    raw_content = response.choices[0].message.content
    if not raw_content:
        print("Warning: Empty response received from API.")
        return {}
        
    content = raw_content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
        
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}\nRaw Content: '{raw_content}'")
        return {}


def merge_dups():
    # Load full dictionary
    with open(VOCAB_PATH, 'r') as f:
        full_data = json.load(f)

    # Split into chunks of 500 (to stay safely within the 8K context limit)
    items = list(full_data.items())
    chunk_size = 1000
    results = {}

    for i in range(0, len(items), chunk_size):
        chunk = dict(items[i:i + chunk_size])
        print(f"Processing chunk {i // chunk_size}...")
        try:
            chunk_result = process_chunk(chunk)
            results.update(chunk_result)
        except Exception as e:
            print(f"Error in chunk {i // chunk_size}: {e}")

    # Save final mapping
    with open(OUTPUT_VOCAB_PATH, 'w') as f:
        json.dump(results, f, indent=4)
        
def flatten_mapping(mapping):
    """
    If 3030->3031 and 3031->3032, this updates 3030->3032.
    """
    # 1. Build a map of all kept targets
    final_map = mapping.copy()
    
    # 2. Resolve chains
    for removed, kept in mapping.items():
        current_target = kept
        # Follow the chain until it hits an ID that is not a 'removed' key
        while current_target in mapping:
            current_target = mapping[current_target]
        
        final_map[removed] = current_target
        
    return final_map

def flatten():
    #load json vocab_clean.json
    with open(OUTPUT_VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    flattened = flatten_mapping(vocab)
    #save 
    # Save final mapping
    with open('vocab_clean_flattened.json', 'w') as f:
        json.dump(flattened, f, indent=4)
        
import json
import torch
from collections import defaultdict

def update_pipeline(mapping_file, old_vocab_file, old_img_map, old_idf_file):
    # 1. Load mappings and original data
    with open(mapping_file, 'r') as f:
        mapping = json.load(f) # e.g., {"removed_id": "kept_id"}
    
    with open(old_vocab_file, 'r') as f:
        old_vocab = json.load(f)
        
    with open(old_img_map, 'r') as f:
        old_img_map_data = json.load(f)
        
    old_idf = torch.load(old_idf_file)

    # 2. Build new vocabulary
    # Identify which IDs are being removed
    removed_ids = set(int(k) for k in mapping.keys())
    
    new_vocab = {}
    old_id_to_new_id = {}
    
    # Keep only words whose index (value) is not in removed_ids
    current_idx = 1
    for word, old_idx in old_vocab.items():
        if word == "<PAD>":
            new_vocab[word] = 0
            continue
            
        if old_idx not in removed_ids:
            new_vocab[word] = current_idx
            old_id_to_new_id[old_idx] = current_idx
            current_idx += 1
        else:
            # Map removed ID to the new ID of the 'kept' word
            kept_id = int(mapping[str(old_idx)])
            old_id_to_new_id[old_idx] = new_vocab.get(
                [w for w, i in old_vocab.items() if i == kept_id][0], 
                old_id_to_new_id.get(kept_id, kept_id)
            )

    # 3. Update Image Mapping
    new_img_map = {}
    for img, indices in old_img_map_data.items():
        new_indices = sorted(list(set([old_id_to_new_id.get(i, i) for i in indices])))
        new_img_map[img] = new_indices

    # 4. Update IDF Tensor
    # Create new tensor of size len(new_vocab)
    new_idf = torch.zeros(len(new_vocab))
    for word, new_idx in new_vocab.items():
        if word == "<PAD>": continue
        # Find the old index for this word
        old_idx = old_vocab[word]
        new_idf[new_idx] = old_idf[old_idx]

    # 5. Save everything
    with open("scene_graph_vocab_updated.json", "w") as f:
        json.dump(new_vocab, f, indent=4)
    with open("image_id_to_vocab_indices_updated.json", "w") as f:
        json.dump(new_img_map, f, indent=4)
    torch.save(new_idf, "gsv_cities_image_idf_updated.pt")

    print("Pipeline updated successfully.")

update_pipeline('vocab_clean_flattened.json', 'datasets/gsv_cities_scene_graph_vocab.json', 'datasets/gsv_cities_image_id_to_vocab_indices.json', 'datasets/gsv_cities_image_idf.pt')
