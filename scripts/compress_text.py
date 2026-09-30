import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI

# ==========================================
# 1. USER CSV READER
# ==========================================
def read_csv_file(labels_file):
    """User-provided CSV reader."""
    df = pd.read_csv(
        labels_file,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        skipinitialspace=True,
    )
    image_path = df["image_path"].values
    description = df["description"].values
    place_id = df["place_id"].values
    return df, image_path, description


# ==========================================
# 2. SYSTEM PROMPT DEFINITION
# ==========================================
SYSTEM_PROMPT = """### TASK DESCRIPTION
You are an expert dataset curator for text-to-image place recognition.
You will be given a set of numbered image descriptions representing different viewpoints of the SAME place.
Rewrite EACH description individually into a dense, high-entropy summary.

### STRICT RULES:
1. OUTPUT EXACTLY ONE REWRITTEN LINE PER INPUT IMAGE. Do NOT merge them into one long paragraph or block of text.
2. FRONT-LOAD DISCRIMINATORS (Words 1–15): Put all OCR text, distinct sign names, and unique landmarks in the first 10-15 words.
3. WORD COUNT: 35–50 words total per image (if original is shorter than 35 words, do NOT add new details).
4. REMOVE TEMPORAL ELEMENTS: Strip people, cars, weather, sky, shadows.
5. CONCISE MODIFIERS: Use compound nouns (e.g., "left asphalt shoulder").

### FEW-SHOT EXAMPLE

--- EXAMPLE INPUT ---
[Image 1] Images/PRS/PRS_0005010_2014: Paved sidewalk with square tiles, vertical pole, light-colored panel fence with horizontal lines, warning sign on the left part of the fence (triangle with exclamation mark, circular symbols), horizontal metal railings below the fence panels, single tree with a dark trunk in front of the fence, dirt ground with sparse vegetation, elevated green foliage and trees behind the fence, partial view of light-colored buildings in the background.
[Image 2] Images/PRS/PRS_0005010_2015: Pavement, a tree with green foliage, a light-colored building partially visible through trees, a long light-colored horizontal construction barrier with red and white posters featuring "FANNY J" and "STONY" and other multi-colored posters, another tree trunk with green foliage, more multi-colored posters and warning signs on the barrier, dense green foliage, a tall light-colored high-rise residential building.

--- EXAMPLE OUTPUT ---
[Image 1] "Warning sign (triangle with exclamation mark, circular symbols) on left fence panel. Light-colored panel fence with horizontal lines, horizontal metal railings below, square-tiled paved sidewalk, vertical pole, dark-trunk tree, elevated foliage, background light-colored buildings."
[Image 2] "Red and white posters featuring "FANNY J" and "STONY", multi-colored posters and warning signs on light-colored horizontal construction barrier. Tall light-colored high-rise residential building, green foliage trees, pavement."

### OUTPUT FORMAT:
Return exactly 1 line per input image using "[Image X] \"<rewritten text>\"". No extra text or explanations."""


def construct_group_prompt(group_df):
    """Formats images with explicit numbered tags for a place_id."""
    input_lines = []
    for idx, (_, row) in enumerate(group_df.iterrows(), 1):
        input_lines.append(f"[Image {idx}] {row['image_path']}: {row['description']}")
    return "\n".join(input_lines)


def parse_numbered_output(generated_text):
    """Extracts rewritten text per image index [Image X]."""
    results = {}
    pattern = r"\[Image\s*(\d+)\]\s*\"?([^\n\"]+)\"?"
    matches = re.findall(pattern, generated_text)
    
    for idx_str, text in matches:
        results[int(idx_str)] = text.strip()
        
    return results


# ==========================================
# 3. ASYNC WORKER PIPELINE
# ==========================================
async def process_place_group(client, model_name, place_id, group, semaphore):
    """Sends prompt asynchronously to vLLM server with concurrency control."""
    async with semaphore:
        user_prompt = construct_group_prompt(group)
        num_images = len(group)
        combined_content = f"{SYSTEM_PROMPT}\n\n### INPUT IMAGES ({num_images} total):\n{user_prompt}"

        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": combined_content}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=1024
            )
            gen_text = response.choices[0].message.content.strip()
            parsed_dict = parse_numbered_output(gen_text)
            
            # Map results to global row index
            results = {}
            group_indices = group.index.tolist()
            for img_idx, global_row_idx in enumerate(group_indices, 1):
                if img_idx in parsed_dict and len(parsed_dict[img_idx].split()) >= 3:
                    results[global_row_idx] = parsed_dict[img_idx]
                else:
                    results[global_row_idx] = df.at[global_row_idx, "description"]
            return results

        except Exception as e:
            print(f"Error processing place_id {place_id}: {e}")
            return {global_row_idx: df.at[global_row_idx, "description"] for global_row_idx in group.index}


async def main():
    labels_file = "datasets/descriptions/amstertime_descriptions_with_place_id.csv"
    output_file = "datasets/descriptions/amstertime_descriptions_with_place_id_rewritten.csv"
    
    VLLM_BASE_URL = "http://localhost:8000/v1"
    MODEL_NAME = "nvidia/Gemma-4-26B-A4B-NVFP4"
    CONCURRENCY_LIMIT = 64  # Sends 64 requests concurrently to saturate GPU
    
    # Use AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY"
    )
    
    global df
    print(f"Loading CSV: {labels_file}")
    df, image_paths, original_descriptions = read_csv_file(labels_file)
    df = df.sort_values(by="place_id").reset_index(drop=True)
    df["rewritten_description"] = None

    grouped = df.groupby("place_id", sort=False)
    total_places = len(grouped)
    print(f"Total place_id groups to process: {total_places}")

    # Limit maximum concurrent requests sent to vLLM server
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = [
        process_place_group(client, MODEL_NAME, place_id, group, semaphore)
        for place_id, group in grouped
    ]

    print(f"Dispatching {total_places} async requests with concurrency={CONCURRENCY_LIMIT}...")
    
    # Run all tasks concurrently
    results_list = await asyncio.gather(*tasks)

    # Merge results back into DataFrame
    print("Writing results back to DataFrame...")
    for group_results in results_list:
        for global_row_idx, text in group_results.items():
            df.at[global_row_idx, "rewritten_description"] = text

    # Final fallback cleanup
    df["rewritten_description"] = df["rewritten_description"].fillna(df["description"])

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nFinished processing! Results saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())