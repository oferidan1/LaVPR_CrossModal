import json
import asyncio
import os
from collections import defaultdict
import httpx
import ast

# Configurations
# DATASET_PATH = "pitts30k_val_800_queries_objects_intermediate.json"
# OUTPUT_LOOKUP_PATH = "lavpr_llm_negative_lookup.json"
DATASET_PATH = "gsv_cities_descriptions_queries_objects_intermediate.json"
OUTPUT_LOOKUP_PATH = "gsv_cities_descriptions_queries_negative_lookup.json"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENCY_LIMIT = 50  # Number of parallel HTTP requests to the vLLM server


def extract_unique_pairs(json_path):
    with open(json_path, "r") as f:
        dataset = json.load(f)
        
    unique_pairs = set() # Automatically handles deduplication
    
    for item in dataset:
        sg = item.get("scene_graph") or {}
        for obj in sg.get("objects", []):
            obj_type = obj.get("label", obj.get("type", obj.get("name", "")))
            obj_type = str(obj_type).lower().strip()
            attributes = obj.get("attributes", [])
            if isinstance(attributes, dict):
                attributes = list(attributes.values())
            
            for attr in attributes:
                attr_val = attr.get("value", "") if isinstance(attr, dict) else attr
                
                # Safely parse stringified lists (e.g. "['red', 'blue']") into actual Python lists
                if isinstance(attr_val, str) and attr_val.strip().startswith("[") and attr_val.strip().endswith("]"):
                    try:
                        attr_val = ast.literal_eval(attr_val.strip())
                    except (ValueError, SyntaxError):
                        pass
                
                if not isinstance(attr_val, list):
                    attr_val = [attr_val]
                    
                for single_attr in attr_val:
                    attr_clean = str(single_attr).lower().strip()
                    if obj_type and attr_clean:
                        # Store as a immutable tuple
                        unique_pairs.add((obj_type, attr_clean))
                    
    return list(unique_pairs)


# =====================================================================
# EXECUTION ORCHESTRATION PIPELINE
# =====================================================================

async def request_negative(client, semaphore, obj_type, attr):
    """Sends a structured request to vLLM to fetch 3 semantic alternatives."""
    async with semaphore:
        # Prompt forces the LLM to act as a strict ontology mutator
        prompt = (
            f"Context: Architectural features for Visual Place Recognition (VPR).\n"
            f"Object Type: {obj_type}\n"
            f"Observed Positive Attribute: {attr}\n"
            f"Task: Provide exactly 3 realistic, mutually exclusive negative attributes for this object. "
            f"They must be architecturally sound and visually distinct, avoiding style contradictions.\n"
            f"Output JSON array format: [\"alt1\", \"alt2\", \"alt3\"]"
        )

        payload = {
            "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
            "messages": [
                {"role": "system", "content": "You are a precise VPR dataset annotation assistant. Respond ONLY with a valid JSON array of strings."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2, # Low temperature for consistent structural alternatives
            "max_tokens": 64
        }

        try:
            response = await client.post(VLLM_URL, json=payload, timeout=30.0)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                
                # Clean up any potential markdown block wrappers from the LLM response
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()
                    
                negatives = json.loads(content)
                if isinstance(negatives, list):
                    return obj_type, attr, [n.lower().strip() for n in negatives]
            
            print(f"[!] Failed to get valid response for {obj_type}:{attr}. Status: {response.status_code}")
        except Exception as e:
            print(f"[!] Exception during request for {obj_type}:{attr} -> {str(e)}")
            
        return obj_type, attr, []


async def main():
    # Step 1: Deduplicate vocabulary
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Error: Could not find your 500k file at {DATASET_PATH}")
        return

    unique_pairs = extract_unique_pairs(DATASET_PATH)
    total_pairs = len(unique_pairs)
    print(f"[✓] Deduplication complete. Reduced dataset to {total_pairs} unique object-attribute pairs.")

    # Step 2: Query vLLM asynchronously over HTTP
    print(f"[*] Launching async requests to vLLM on {VLLM_URL}...")
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    # Configure low-level limits to avoid pool starvation
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=CONCURRENCY_LIMIT)
    
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [
            request_negative(client, semaphore, obj_type, attr) 
            for obj_type, attr in unique_pairs
        ]
        
        # Gather all parallel tasks
        results = await asyncio.gather(*tasks)

    # Step 3: Build the final nested dictionary layout
    print("[*] Processing API answers into structured dictionary ontology...")
    nested_lookup = defaultdict(dict)
    success_count = 0
    
    for obj_type, attr, negatives in results:
        if negatives:
            nested_lookup[obj_type][attr] = negatives
            success_count += 1

    # Save dictionary to disk
    with open(OUTPUT_LOOKUP_PATH, "w") as f:
        json.dump(nested_lookup, f, indent=2)

    print(f"\n[✓] Done! Built lookup map for {success_count}/{total_pairs} pairs.")
    print(f"[✓] File stored successfully at: {OUTPUT_LOOKUP_PATH}")


if __name__ == "__main__":
    # Run the async orchestration loop
    asyncio.run(main())