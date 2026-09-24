import asyncio
import aiohttp
import pandas as pd
import json
import re
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# ==========================================
# 1. SYSTEM PROMPT DEFINITION
# ==========================================
SYSTEM_PROMPT = """You are an expert geospatial and scene analyst curating visual place recognition datasets.
Your task is to evaluate image descriptions for a single GPS place_id sentence-by-sentence and determine if they belong to the SAME visual place or a TOTALLY DIFFERENT viewpoint.

DEFINITIONS & RULES:
1. SAME PLACE (Cluster "A"):
   - Different viewing angles or perspectives of the same street side/building facade.
   - Partial matches where at least one anchor overlaps (e.g., Image 1 mentions Billboard X + Blocks, Image 2 mentions Billboard X + Trees, Image 3 mentions Blocks + Path).
   - Temporal/seasonal variations (same building/street across different years, weather, lighting, minor renovations, or moving objects like cars/stalls).
   - Broad vs. close-up framing of the same scene.

2. DIFFERENT PLACE (Cluster "B"):
   - A completely disjoint viewpoint where NO visual anchor matches (e.g., looking at the opposite side of the street, an entirely different building across the road, or facing 180 degrees away).

3. SPLIT CONSTRAINT:
   - Form at most 2 clusters ("A" and "B").
   - Group the majority of connected/overlapping views under "A".
   - Assign to "B" ONLY if an image has zero connection or shared anchors with "A".
   - If an image is partially connected or ambiguous, MERGE IT into "A".

FEW-SHOT EXAMPLE 1 (Partial & Temporal Matches -> Same Cluster "A"):
Input [Place ID: 1805017]:
[Image 1]: "Large billboard for RICHARDSON with red circular logo, three concrete barrier blocks on paved road."
[Image 2]: "Dense green trees, large sign displaying RICHARDSON, paved path curving uphill, low metal railing."
[Image 3]: "Three gray concrete blocks, an unpaved path leading uphill, blue and white DECATHLON sign on two posts."
[Image 4]: "Large billboard for RICHARDSON, three square concrete barriers, blue and white DECATHLON sign on the right."

Output:
{
  "reasoning": "Images 1, 2, and 4 share the RICHARDSON billboard. Images 1, 3, and 4 share the three concrete blocks. Image 3 and 4 share the DECATHLON sign. They are all partial views/angles of the same street section.",
  "sub_places": {
    "A": [1, 2, 3, 4]
  }
}

FEW-SHOT EXAMPLE 2 (True Disjoint Viewpoint -> Split "A" and "B"):
Input [Place ID: 1042]:
[Image 1]: "Two-story red brick commercial building with green awnings, glass storefront, and concrete sidewalk."
[Image 2]: "Red brick storefront with green canvas awning, adjacent street lamp, and sidewalk."
[Image 3]: "Modern glass curtain-wall office tower with revolving doors and polished granite entrance across a multi-lane boulevard."

Output:
{
  "reasoning": "Images 1 and 2 share the red brick building and green awnings. Image 3 depicts a modern glass skyscraper facing the opposite side with no shared structural anchors.",
  "sub_places": {
    "A": [1, 2],
    "B": [3]
  }
}

OUTPUT FORMAT:
Return ONLY a valid JSON object containing:
1. "reasoning": A brief explanation of anchor overlaps.
2. "sub_places": Mapping of cluster keys ("A" or "B") to lists of 1-based image indices."""

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def format_prompt(place_id, group_df):
    prompt_lines = [f"Input [Place ID: {place_id}]:"]
    for i, (_, row) in enumerate(group_df.iterrows(), 1):
        desc = str(row.get('description', '')).strip()
        prompt_lines.append(f'[Image {i}]: "{desc}"')
    
    user_content = "\n".join(prompt_lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{user_content}\n\nOutput:"}
    ]

def parse_json_response(raw_text, place_id, group_indices, num_images):
    results = {}
    try:
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            sub_dict = data.get("sub_places") or data.get("sub_place_ids")

            if isinstance(sub_dict, dict):
                for sub_key, img_indices in sub_dict.items():
                    suffix = str(sub_key).split("_")[-1].upper()
                    if suffix not in ["A", "B"]:
                        suffix = "A"
                    clean_sub_id = f"{place_id}_{suffix}"

                    if isinstance(img_indices, list):
                        for item in img_indices:
                            try:
                                local_idx = int(re.search(r"\d+", str(item)).group(0))
                                if 1 <= local_idx <= num_images:
                                    global_idx = group_indices[local_idx - 1]
                                    results[global_idx] = clean_sub_id
                            except (AttributeError, ValueError, IndexError):
                                continue
    except Exception:
        pass

    for global_idx in group_indices:
        if global_idx not in results:
            results[global_idx] = f"{place_id}_A"

    return results

async def send_vllm_request(session, url, model_name, messages, semaphore):
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 512
    }
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        return res_json["choices"][0]["message"]["content"]
                    else:
                        await asyncio.sleep(1.0 * (2 ** attempt))
            except Exception:
                await asyncio.sleep(1.0 * (2 ** attempt))
        return ""

# ==========================================
# 3. ASYNC TASK PROCESSOR
# ==========================================
async def process_place_group(session, url, model_name, place_id, group_df, group_indices, num_images, semaphore):
    messages = format_prompt(place_id, group_df)
    raw_text = await send_vllm_request(session, url, model_name, messages, semaphore)
    return parse_json_response(raw_text, place_id, group_indices, num_images)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
async def run_async_pipeline(labels_file, output_file, vllm_url, model_name, concurrency_limit, save_interval):
    print(f"Loading data from: {labels_file}")
    df = pd.read_csv(labels_file)

    if Path(output_file).exists():
        print(f"Resuming from existing output file: {output_file}")
        existing_df = pd.read_csv(output_file)
        if 'sub_place_id' in existing_df.columns:
            df['sub_place_id'] = existing_df['sub_place_id']

    if 'sub_place_id' not in df.columns:
        df['sub_place_id'] = None

    grouped = list(df.groupby('place_id', sort=False))
    print(f"Total places: {len(grouped)}")

    pending_tasks = []
    immediate_assignments = 0

    for place_id, group_df in grouped:
        group_indices = group_df.index.tolist()
        num_images = len(group_indices)

        if df.loc[group_indices, 'sub_place_id'].notna().all():
            continue

        if num_images <= 1:
            df.loc[group_indices[0], 'sub_place_id'] = f"{place_id}_A"
            immediate_assignments += 1
            continue

        pending_tasks.append((place_id, group_df, group_indices, num_images))

    print(f"Single-image places assigned immediately: {immediate_assignments}")
    print(f"Place groups to query via vLLM: {len(pending_tasks)}")

    semaphore = asyncio.Semaphore(concurrency_limit)
    images_processed_since_save = immediate_assignments

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=concurrency_limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create async task batch
        tasks = [
            process_place_group(session, vllm_url, model_name, p_id, g_df, g_indices, n_imgs, semaphore)
            for p_id, g_df, g_indices, n_imgs in pending_tasks
        ]

        pbar = tqdm(total=len(tasks), desc="Processing place groups")
        
        for future in asyncio.as_completed(tasks):
            result_map = await future
            
            for global_idx, sub_id in result_map.items():
                df.loc[global_idx, 'sub_place_id'] = sub_id
            
            images_processed_since_save += len(result_map)
            pbar.update(1)

            if images_processed_since_save >= save_interval:
                df.to_csv(output_file, index=False)
                tqdm.write(f"\n[Checkpoint] Saved progress to {output_file} ({images_processed_since_save} images accumulated)")
                images_processed_since_save = 0

        pbar.close()

    df['sub_place_id'] = df['sub_place_id'].fillna(df['place_id'].astype(str) + "_A")
    df.to_csv(output_file, index=False)
    print(f"\nCompleted successfully! File saved to: {output_file}")

def main():
    labels_file = "datasets/descriptions/gsv_cities_predictions_with_place_id.csv"
    output_file = "datasets/descriptions/gsv_cities_predictions_with_sub_place_id.csv"

    VLLM_HTTP_URL = "http://localhost:8000/v1/chat/completions"
    MODEL_NAME = "nvidia/Gemma-4-26B-A4B-NVFP4"
    CONCURRENCY_LIMIT = 100
    SAVE_INTERVAL = 10000

    asyncio.run(
        run_async_pipeline(
            labels_file=labels_file,
            output_file=output_file,
            vllm_url=VLLM_HTTP_URL,
            model_name=MODEL_NAME,
            concurrency_limit=CONCURRENCY_LIMIT,
            save_interval=SAVE_INTERVAL
        )
    )

if __name__ == "__main__":
    main()