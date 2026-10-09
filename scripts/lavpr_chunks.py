"""
LaVPR Description Chunker  –  Batched vLLM Edition
====================================================
Splits each scene description into 3–6 semantically coherent chunks
using an LLM (Gemma via vLLM), following spatial-layer and object-category
grouping principles for place recognition retrieval.

Each input row produces one output row with a new `chunks` column
containing a JSON array of chunk strings.

Usage:
    python chunk_lavpr_descriptions.py \
        --input   datasets/descriptions/lavpr_descriptions.csv \
        --output  datasets/descriptions/lavpr_descriptions_chunked.csv \
        --batch-size  16          \
        --concurrency 8           \
        --base-url    http://localhost:8000/v1 \
        --model       google/gemma-3-4b-it
"""

import re
import json
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT       = "datasets/descriptions/pitts30k_test_descriptions.csv"
DEFAULT_OUTPUT      = "datasets/descriptions/pitts30k_test_chunked.csv"
DEFAULT_BATCH_SIZE  = 64     # descriptions packed into one LLM request
DEFAULT_CONCURRENCY = 32     # parallel batch requests in flight
DEFAULT_BASE_URL    = "http://localhost:8000/v1"
DEFAULT_MODEL       = "nvidia/Gemma-4-26B-A4B-NVFP4"

# ──────────────────────────────────────────────────────────────────────────────
# 1. CSV I/O
# ──────────────────────────────────────────────────────────────────────────────
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        skipinitialspace=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. SYSTEM PROMPT
# NOTE: Gemma 3 has no system role — this is prepended to the first user turn.
# For batch requests we use the OpenAI-compatible /v1/chat/completions endpoint
# which vLLM exposes; system role IS supported there even for Gemma.
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a precise visual scene segmenter for place recognition research.

Your task is to split each visual scene description into semantically coherent
chunks. Each chunk must be independently queryable — a person searching for
that element could retrieve it with a short natural language query.

CHUNKING RULES:
1. Spatial layers : group elements by foreground / mid-ground / background
2. Structural unity: keep components of the same object together
   (e.g. fence + railings + panels = one chunk)
3. Semantic distinctiveness: signage, landmarks, and unique features get
   their own chunk — they are high-value retrieval targets
4. Co-location: elements that appear together and would be queried together
   stay in the same chunk
5. Vegetation + ground: group natural ground-cover elements that share the
   same spatial layer

OUTPUT FORMAT:
- Return EXACTLY one line per input image, in the same order.
- Each line: [Image X] <json_array>
  where <json_array> is a JSON array of chunk strings.
  Use as many chunks as needed; use 1 only when the description is a single indivisible phrase.
- Preserve the original descriptive language — do not paraphrase.
- No labels (e.g. "Foreground:"), no markdown fences, no explanations.

FEW-SHOT EXAMPLES

--- EXAMPLE 1 ---

INPUT:
[Image 1] Paved sidewalk with square tiles, vertical pole, light-colored \
panel fence with horizontal lines, warning sign on the left part of the \
fence (triangle with exclamation mark, circular symbols), horizontal metal \
railings below the fence panels, single tree with a dark trunk in front of \
the fence, dirt ground with sparse vegetation, elevated green foliage and \
trees behind the fence, partial view of light-colored buildings in the \
background.

OUTPUT:
[Image 1] ["Paved sidewalk with square tiles, vertical pole", "Light-colored panel fence with horizontal lines, horizontal metal railings below the fence panels", "Warning sign on the left part of the fence (triangle with exclamation mark, circular symbols)", "Single tree with a dark trunk in front of the fence, dirt ground with sparse vegetation", "Elevated green foliage and trees behind the fence, partial view of light-colored buildings in the background"]

--- EXAMPLE 2 ---

INPUT:
[Image 2] Asphalt road surface with painted lane markings and pedestrian \
crossing stripes, traffic light pole on the corner with red signal \
illuminated, stop sign adjacent to the traffic light, row of parked cars \
along the right curb, residential mailboxes near the sidewalk, two-storey \
brick buildings lining both sides of the road, overcast sky visible above \
the rooflines.

OUTPUT:
[Image 2] ["Asphalt road surface with painted lane markings and pedestrian crossing stripes", "Traffic light pole on the corner with red signal illuminated, stop sign adjacent to the traffic light", "Row of parked cars along the right curb, residential mailboxes near the sidewalk", "Two-storey brick buildings lining both sides of the road, overcast sky visible above the rooflines"]\
"""


# ──────────────────────────────────────────────────────────────────────────────
# 3. RESPONSE PARSER
# ──────────────────────────────────────────────────────────────────────────────
def parse_batch_response(raw: str, batch_indices: list[int]) -> dict[int, list[str]]:
    """
    Parse a multi-item LLM response.

    Expected format (one line per item):
        [Image 1] ["chunk a", "chunk b", ...]
        [Image 2] ["chunk a", "chunk b", ...]

    Returns {original_df_index: [chunk, ...]}.
    """
    results: dict[int, list[str]] = {}
    pos_to_idx = {pos + 1: df_idx for pos, df_idx in enumerate(batch_indices)}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r'\[Image\s+(\d+)\]\s+(\[.*)', line)
        if not m:
            continue

        pos     = int(m.group(1))
        json_str = m.group(2).strip()

        # Strip stray markdown fences Gemma occasionally emits
        if json_str.startswith("```"):
            json_str = re.sub(r"```[a-z]*", "", json_str).replace("```", "").strip()

        df_idx = pos_to_idx.get(pos)
        if df_idx is None:
            continue

        try:
            chunks = json.loads(json_str)
            if (
                isinstance(chunks, list)
                and all(isinstance(c, str) for c in chunks)
                and len(chunks) >= 1
            ):
                results[df_idx] = chunks
            else:
                print(
                    f"  [WARN] df_idx={df_idx}: parsed JSON but bad shape "
                    f"(len={len(chunks) if isinstance(chunks, list) else '?'}). Skipping."
                )
        except json.JSONDecodeError as e:
            print(f"  [WARN] df_idx={df_idx}: JSON parse error – {e}. Line: {line[:120]}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. NAIVE FALLBACK CHUNKER  (no LLM — comma-split into ~equal groups)
# Used when LLM fails after all retries for a specific row.
# ──────────────────────────────────────────────────────────────────────────────
def fallback_chunk(description: str, target: int = 4) -> list[str]:
    """
    Split description on commas into `target` roughly equal chunks.
    Guarantees 3–6 chunks; marked with a leading '*' so you can find them.
    """
    parts = [p.strip() for p in description.split(",") if p.strip()]
    if len(parts) <= target:
        # Already few enough — each part becomes its own chunk (up to 6)
        chunks = parts[:6] if len(parts) >= 3 else [description]
    else:
        # Group into `target` bins
        size   = max(1, len(parts) // target)
        chunks = []
        for i in range(0, len(parts), size):
            chunks.append(", ".join(parts[i : i + size]))
        chunks = chunks  # no hard cap

    # Ensure at least 1
    if not chunks:
        chunks = [description]

    return ["* " + c if i == 0 else c for i, c in enumerate(chunks)]  # mark fallback


# ──────────────────────────────────────────────────────────────────────────────
# 5. BATCH WORKER
# ──────────────────────────────────────────────────────────────────────────────
async def chunk_batch(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    batch: list[tuple[int, str]],   # [(df_idx, description), ...]
    retries: int = 2,
) -> dict[int, list[str]]:
    """
    Send one batch request to vLLM.
    Returns {df_idx: [chunk, ...]}.
    Rows that fail parsing fall back to naive comma-chunking.
    """
    # Build numbered user message
    lines = [
        f"[Image {pos}] {desc}"
        for pos, (_, desc) in enumerate(batch, start=1)
    ]
    user_message = "\n\n".join(lines)

    batch_indices = [df_idx for df_idx, _ in batch]
    originals     = {df_idx: desc for df_idx, desc in batch}

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        # vLLM's OpenAI-compat layer accepts system role for Gemma
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    # ~100 tokens output per item (JSON array of 3-6 short strings)
                    max_tokens=min(150 * len(batch), 32768),
                    temperature=0.0,          # deterministic — critical for chunk cache
                    extra_body={
                        "repetition_penalty": 1.1,   # prevent JSON loop artifacts
                    },
                )
                raw    = response.choices[0].message.content.strip()
                parsed = parse_batch_response(raw, batch_indices)

                # Fill any missing items with fallback chunker
                for df_idx in batch_indices:
                    if df_idx not in parsed:
                        print(
                            f"  [WARN] df_idx={df_idx}: missing from response "
                            f"(attempt {attempt+1}); applying fallback chunker."
                        )
                        parsed[df_idx] = fallback_chunk(originals[df_idx])

                return parsed

            except Exception as exc:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(
                        f"  [ERROR] batch df_idx={batch_indices[0]}…{batch_indices[-1]}: "
                        f"{exc}. Retrying in {wait}s …"
                    )
                    await asyncio.sleep(wait)
                else:
                    print(
                        f"  [ERROR] batch df_idx={batch_indices[0]}…{batch_indices[-1]}: "
                        f"{exc}. Giving up — applying fallback chunker to all rows."
                    )
                    return {df_idx: fallback_chunk(desc) for df_idx, desc in batch}


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────
async def main(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"Loading CSV: {args.input}")
    df = read_csv(args.input)

    # Validate expected column
    if "description" not in df.columns:
        raise ValueError(
            f"CSV must have a 'description' column. Found: {list(df.columns)}"
        )

    total        = len(df)
    descriptions = df["description"].tolist()

    print(
        f"Total rows   : {total}\n"
        f"Batch size   : {args.batch_size}\n"
        f"Concurrency  : {args.concurrency}\n"
        f"Model        : {args.model}"
    )

    # ── Build flat list of (df_idx, description) ─────────────────────────────
    items = list(enumerate(descriptions))

    # ── Split into batches ────────────────────────────────────────────────────
    batches = [
        items[i : i + args.batch_size]
        for i in range(0, len(items), args.batch_size)
    ]
    print(f"Num batches  : {len(batches)}")

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        chunk_batch(
            client=client,
            semaphore=semaphore,
            model=args.model,
            batch=batch,
        )
        for batch in batches
    ]

    # ── Run with progress bar ─────────────────────────────────────────────────
    all_results: list[dict[int, list[str]]] = await atqdm.gather(
        *tasks, desc="Chunking batches"
    )

    # ── Write chunks back to dataframe ────────────────────────────────────────
    df["chunks"] = None  # new column: JSON array stored as string

    for result_dict in all_results:
        for df_idx, chunks in result_dict.items():
            df.at[df_idx, "chunks"] = json.dumps(chunks, ensure_ascii=False)

    # Safety net — should never trigger, but just in case
    missing = df["chunks"].isna().sum()
    if missing:
        print(f"  [WARN] {missing} rows still null after processing — applying fallback.")
        for df_idx in df[df["chunks"].isna()].index:
            df.at[df_idx, "chunks"] = json.dumps(
                fallback_chunk(df.at[df_idx, "description"]), ensure_ascii=False
            )

    # ── Stats ─────────────────────────────────────────────────────────────────
    chunk_counts = df["chunks"].apply(lambda x: len(json.loads(x)))
    fallback_rows = df["chunks"].apply(
        lambda x: any(c.startswith("* ") for c in json.loads(x))
    ).sum()

    print(
        f"\nChunk count stats:"
        f"\n  Mean : {chunk_counts.mean():.2f}"
        f"\n  Min  : {chunk_counts.min()}"
        f"\n  Max  : {chunk_counts.max()}"
        f"\n  Fallback (comma-split) rows: {fallback_rows}"
    )

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved → {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split LaVPR scene descriptions into semantic chunks (batched vLLM)."
    )
    parser.add_argument("--input",       default=DEFAULT_INPUT,
                        help="Input CSV with an 'image_path' and 'description' column")
    parser.add_argument("--output",      default=DEFAULT_OUTPUT,
                        help="Output CSV path (adds a 'chunks' column)")
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE,
                        help="Descriptions per LLM request (default: 16)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max parallel batch requests (default: 8)")
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL,
                        help="vLLM server base URL (default: http://localhost:8000/v1)")
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Model name as passed to vLLM --model")
    args = parser.parse_args()

    asyncio.run(main(args))