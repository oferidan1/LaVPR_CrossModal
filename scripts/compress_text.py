"""
LaVPR Description Compressor  –  Batched vLLM Edition
=======================================================
Packs multiple descriptions into each LLM request (true batch prompting),
so vLLM can process them in one forward pass instead of one request per row.

Only descriptions that exceed --max-tokens are sent to the model.
Short descriptions are passed through unchanged.

Usage:
    python compress_lavpr_descriptions.py \
        --input   datasets/descriptions/lavpr_descriptions.csv \
        --output  datasets/descriptions/lavpr_descriptions_compressed.csv \
        --max-tokens  64          \
        --batch-size  32          \
        --concurrency 8           \
        --base-url    http://localhost:8000/v1 \
        --model       google/gemma-3-27b-it
"""

import re
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT       = "datasets/descriptions/gsv_cities_descriptions.csv"
DEFAULT_OUTPUT      = "datasets/descriptions/gsv_cities_comprssed.csv"
DEFAULT_MAX_TOKENS  = 74      # retrieval backbone budget (CLIP=77, SigLIP=64)
DEFAULT_BATCH_SIZE  = 32      # descriptions packed into one LLM request
DEFAULT_CONCURRENCY = 8       # parallel batch requests in flight
DEFAULT_BASE_URL    = "http://localhost:8000/v1"
DEFAULT_MODEL       = "nvidia/Gemma-4-26B-A4B-NVFP4"

# Tokenizer used ONLY for token counting – must match the retrieval backbone
COUNT_TOKENIZER = "openai/clip-vit-base-patch16"

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
# 2. SYSTEM PROMPT  (shown once per request, shared across all items in batch)
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
### TASK
You are a dataset curator for text-to-image place recognition.
Each input is a scene description that reads left-to-right through a place.
Compress EACH description to at most {max_tokens} CLIP/SigLIP tokens.

### STRICT RULES (follow in this priority order)

RULE 1 — PRESERVE ALL READABLE TEXT VERBATIM
Any text readable in the scene MUST appear in the output exactly as written:
  - Sign text:      "376 EAST Monroeville", "279 SOUTH", "EXIT 1C", "No Left Turn"
  - Building names: "YMCA", "JUNIOR ACHIEVEMENT", "2 PPG PLACE"
  - Street signs:   "Grant St", "Fort Pitt Blvd", "ONE WAY"
  - Store signs:    "FOR LEASE", "MAINTENANCE", "Do Not Enter"
This text is the single most discriminative feature for place recognition.
NEVER paraphrase, abbreviate, or drop it to save tokens — drop generic
descriptions instead.

RULE 2 — KEEP UNIQUE STRUCTURAL FEATURES
Retain features that uniquely identify the place:
  - Distinctive bridge types: "yellow arched truss bridge", "Gothic castellated towers"
  - Landmark buildings: color + material + window pattern
  - Unusual structures: "white circular pavilion", "enclosed pedestrian bridge"

RULE 3 — DROP GENERIC FILLER FIRST
These phrases carry almost no discriminative value — remove them first:
  - "partial view of", "in the background", "in the foreground"
  - "in the distance", "visible behind", "partially visible"
  - Repeated material descriptions (e.g. "brick" said 3 times → say once)
  - Generic sky/ground unless color is unusual

RULE 4 — PRESERVE LEFT-TO-RIGHT ORDER
Keep the spatial reading order of the original (left → right → background).

RULE 5 — YOU MUST COMPRESS
Never return the input unchanged. Always produce a shorter output.
Use short comma-separated noun phrases, same style as the input.

### FEW-SHOT EXAMPLES

--- EXAMPLE 1: sign text must survive ---

INPUT:
[Image 1] img/PIT/A_001: Treed hillside with buildings, concrete roadway \
with raised curb, overhead sign structure with a green "376 EAST Monroeville" \
sign (left arrow), blue "Fort Pitt Blvd" sign, multiple traffic lights \
(one with a "No Left Turn" sign), central concrete island with white barriers \
and yellow ground markings, dark stone/brick structure, bridge with concrete \
barriers, overhead sign structure with a green "279 SOUTH Ft Pitt Bridge \
Airport" sign (right arrow), additional traffic lights (one with a \
"No Right Turn" sign), concrete roadway curving right.

OUTPUT:
[Image 1] "Treed hillside, concrete roadway, overhead signs '376 EAST \
Monroeville' and 'Fort Pitt Blvd', traffic lights with 'No Left Turn', \
concrete island, dark stone structure, overhead sign '279 SOUTH Ft Pitt \
Bridge Airport', concrete roadway curving right."

WHY: All six sign texts kept verbatim. Generic phrases ("raised curb",
"white barriers and yellow ground markings", "concrete roadway curving right")
trimmed or shortened. Left-to-right order preserved.

--- EXAMPLE 2: building name must survive ---

INPUT:
[Image 2] img/PIT/B_020: Dark green multi-story building with steep dark \
shingle roof, ground-level storefronts, brown brick YMCA building with \
vertical "YMCA" sign and banner, asphalt road with faded markings, traffic \
lights with "no left turn" sign, tall building with "JUNIOR ACHIEVEMENT" sign, \
red brick building with arched windows and storefront awnings, distant city \
skyline with tall buildings and hills.

OUTPUT:
[Image 2] "Dark green steep-roofed building, storefronts, brown brick YMCA \
building with 'YMCA' sign, traffic lights with 'no left turn', 'JUNIOR \
ACHIEVEMENT' building, red brick building with arched windows, distant skyline."

WHY: "YMCA", "no left turn", "JUNIOR ACHIEVEMENT" kept verbatim.
Generic phrases ("asphalt road with faded markings", "storefront awnings",
"tall buildings and hills") shortened or dropped.

--- EXAMPLE 3: no sign text — compress generic descriptions ---

INPUT:
[Image 3] img/PIT/C_005: Paved sidewalk with square tiles, vertical pole, \
light-colored panel fence with horizontal lines, warning sign on the left \
part of the fence (triangle with exclamation mark, circular symbols), \
horizontal metal railings below the fence panels, single tree with a dark \
trunk in front of the fence, dirt ground with sparse vegetation, elevated \
green foliage and trees behind the fence, partial view of light-colored \
buildings in the background.

OUTPUT:
[Image 3] "Paved tiled sidewalk, vertical pole, panel fence with warning sign \
and railings, dark-trunked tree, dirt ground, green foliage, light buildings."

WHY: No named text. Compressed by dropping "partial view of", "in the
background", redundant fence/railings detail, "sparse vegetation".

### OUTPUT FORMAT
- Return EXACTLY one line per input image, in the same order.
- Each line: [Image X] "<compressed text>"
- No extra text, no blank lines, no explanations.\
"""


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOKEN COUNTER
# ──────────────────────────────────────────────────────────────────────────────
def build_token_counter(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    def count(text: str) -> int:
        return len(tok.encode(text, add_special_tokens=False))
    return count


# ──────────────────────────────────────────────────────────────────────────────
# 4. RESPONSE PARSER
# ──────────────────────────────────────────────────────────────────────────────
def parse_batch_response(raw: str, batch_indices: list[int]) -> dict[int, str]:
    """
    Parse a multi-item LLM response.

    Expected format (one line per item):
        [Image 1] "compressed text"
        [Image 2] "compressed text"
        ...

    Returns {original_df_index: compressed_text}.
    """
    results: dict[int, str] = {}

    # Build a map from 1-based batch position → df index
    pos_to_idx = {pos + 1: df_idx for pos, df_idx in enumerate(batch_indices)}

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        # Try quoted form
        m = re.match(r'\[Image\s+(\d+)\]\s+"([^"]+)"', line)
        if not m:
            # Unquoted fallback
            m = re.match(r'\[Image\s+(\d+)\]\s+(.+)', line)
        if not m:
            continue

        pos   = int(m.group(1))
        text  = m.group(2).strip().strip('"')
        df_idx = pos_to_idx.get(pos)
        if df_idx is not None:
            results[df_idx] = text

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 5. HARD TRUNCATION FALLBACK
# ──────────────────────────────────────────────────────────────────────────────
def hard_truncate(text: str, max_tokens: int, token_counter) -> str:
    """
    Last-resort truncation: drop trailing comma-separated phrases one by one
    until the text fits within max_tokens.  Preserves left-to-right order.
    """
    # Split on ", " boundaries (the natural phrase delimiter in LaVPR descriptions)
    parts = [p.strip() for p in text.split(",")]
    while len(parts) > 1 and token_counter(", ".join(parts)) > max_tokens:
        parts.pop()
    result = ", ".join(parts)
    # Ensure it ends cleanly (no trailing comma or period duplication)
    result = result.rstrip(" ,") + "."
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 6. BATCH WORKER
# ──────────────────────────────────────────────────────────────────────────────
async def compress_batch(
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    batch: list[tuple[int, str, str]],   # [(df_idx, image_path, description), ...]
    max_tokens: int,
    token_counter,
    retries: int = 2,
) -> dict[int, str]:
    """
    Send one batch request to vLLM.  Returns {df_idx: compressed_text}.
    Items that fail parsing fall back to their original description.
    """
    # Build the user message: numbered list of descriptions
    lines = []
    for pos, (_, img_path, desc) in enumerate(batch, start=1):
        lines.append(f"[Image {pos}] {img_path}: {desc}")
    user_message = "\n".join(lines)

    batch_indices = [df_idx for df_idx, _, _ in batch]
    originals     = {df_idx: desc for df_idx, _, desc in batch}

    async with semaphore:
        for attempt in range(retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    # Give the model enough room: each output ≤ ~80 words × batch_size
                    max_tokens=min(200 * len(batch), 4096),
                    temperature=0.2,
                )
                raw = response.choices[0].message.content.strip()
                parsed = parse_batch_response(raw, batch_indices)

                # Fill in any items the parser missed
                for df_idx in batch_indices:
                    if df_idx not in parsed:
                        # Parse failure: model produced no output for this item.
                        # hard_truncate is the last resort to guarantee some output.
                        print(
                            f"  [WARN] df_idx={df_idx}: not found in response "
                            f"(attempt {attempt+1}); applying hard truncation of original."
                        )
                        parsed[df_idx] = hard_truncate(originals[df_idx], max_tokens, token_counter)
                    else:
                        compressed = parsed[df_idx]
                        c = token_counter(compressed)
                        if c > max_tokens:
                            # LLM did a valid semantic compression but is slightly
                            # over budget. Keep it — truncating would destroy
                            # discriminative content to save a handful of tokens.
                            print(
                                f"  [INFO] df_idx={df_idx}: {c} tokens "
                                f"(budget={max_tokens}) — keeping LLM output as-is."
                            )

                return parsed

            except Exception as exc:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"  [ERROR] batch starting at df_idx={batch_indices[0]}: "
                          f"{exc}. Retrying in {wait}s …")
                    await asyncio.sleep(wait)
                else:
                    print(f"  [ERROR] batch starting at df_idx={batch_indices[0]}: "
                          f"{exc}. Giving up – keeping originals.")
                    return originals


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────
async def main(args):
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_tokens=args.max_tokens)

    print(f"Loading token counter: {COUNT_TOKENIZER}")
    token_counter = build_token_counter(COUNT_TOKENIZER)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")

    print(f"Loading CSV: {args.input}")
    df = read_csv(args.input)
    # Descriptions ≤ max_tokens are never sent to the LLM and stay unchanged.
    # Only rows in to_compress (below) will have short_description overwritten.
    df["short_description"] = df["description"].copy()

    image_paths  = df["image_path"].values
    descriptions = df["description"].values
    total        = len(df)

    # ── Identify rows that need compression ──────────────────────────────────
    to_compress = [
        (i, image_paths[i], descriptions[i])
        for i in range(total)
        if token_counter(descriptions[i]) > args.max_tokens
    ]
    skipped = total - len(to_compress)
    print(
        f"Total rows       : {total}\n"
        f"Already ≤{args.max_tokens} tokens : {skipped}  (skipped)\n"
        f"Need compression : {len(to_compress)}"
    )

    if not to_compress:
        print("Nothing to compress.")
        df.to_csv(args.output, index=False, encoding="utf-8")
        return

    # ── Split into batches ────────────────────────────────────────────────────
    batches = [
        to_compress[i : i + args.batch_size]
        for i in range(0, len(to_compress), args.batch_size)
    ]
    print(
        f"Batch size       : {args.batch_size}\n"
        f"Num batches      : {len(batches)}\n"
        f"Concurrency      : {args.concurrency}"
    )

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        compress_batch(
            client=client,
            semaphore=semaphore,
            model=args.model,
            system_prompt=system_prompt,
            batch=batch,
            max_tokens=args.max_tokens,
            token_counter=token_counter,
        )
        for batch in batches
    ]

    # Gather with progress bar
    all_results: list[dict[int, str]] = await atqdm.gather(
        *tasks, desc="Compressing batches"
    )

    # ── Write results back ────────────────────────────────────────────────────
    for result_dict in all_results:
        for df_idx, text in result_dict.items():
            df.at[df_idx, "short_description"] = text

    # Safety net
    df["short_description"] = df["short_description"].fillna(df["description"])

    # ── Stats ─────────────────────────────────────────────────────────────────
    orig_counts  = [token_counter(d) for d in df["description"]]
    short_counts = [token_counter(d) for d in df["short_description"]]
    over_budget  = sum(1 for c in short_counts if c > args.max_tokens)

    print(
        f"\nToken stats (CLIP tokenizer, budget={args.max_tokens}):"
        f"\n  Original   – mean: {sum(orig_counts)/total:.1f}   max: {max(orig_counts)}"
        f"\n  Compressed – mean: {sum(short_counts)/total:.1f}   max: {max(short_counts)}"
        f"\n  Still over budget: {over_budget}"
    )

    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSaved → {args.output}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress LaVPR descriptions to ≤N tokens (batched vLLM)."
    )
    parser.add_argument("--input",       default=DEFAULT_INPUT)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tokens",  type=int, default=DEFAULT_MAX_TOKENS,
                        help="Token budget (CLIP=77, SigLIP=64)")
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE,
                        help="Descriptions per LLM request (default: 32)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max parallel batch requests (default: 8)")
    parser.add_argument("--base-url",    default=DEFAULT_BASE_URL)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    args = parser.parse_args()

    asyncio.run(main(args))