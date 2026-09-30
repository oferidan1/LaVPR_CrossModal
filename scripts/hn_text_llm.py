import ast
import asyncio
import json
import os
import posixpath
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

# =====================================================================
# CONFIGURATIONS
# =====================================================================
# SG_JSON = "pitts30k_val_800_queries_objects_intermediate.json"
# CSV_PATH = "datasets/descriptions/pitts30k_val_800_queries.csv"
# OUTPUT_DATASET_PATH = "pitts30k_val_with_hard_negatives.csv"
SG_JSON = "datasets/descriptions/gsv_cities_descriptions_sg.json"
CSV_PATH = "datasets/descriptions/gsv_cities_pos_rule_based.csv"
OUTPUT_DATASET_PATH = "gsv_cities_pos_hn.csv"
VLLM_URL = "http://localhost:8000/v1/chat/completions"
CONCURRENCY_LIMIT = 50

# Target range for multi-attribute swaps
MIN_ATTR_SWAPS = 3
MAX_ATTR_SWAPS = 5

COLOR_CLUSTERS = [
    {"white", "beige", "light grey", "grey", "cream", "off-white"},
    {"black", "dark grey", "charcoal", "dark"},
    {"orange-brown", "brown", "brick", "terracotta", "red brick"},
    {"blue", "navy", "cyan"},
    {"red", "crimson", "scarlet"},
    {"green", "olive", "emerald"},
    {"yellow", "gold", "amber"},
]

# High-contrast fallback distractors when in-scene swap pairs are exhausted
HIGH_CONTRAST_DISTRACTORS = {
    "color": [
        "bright red",
        "vivid blue",
        "neon yellow",
        "bright green",
        "dark purple",
        "orange",
    ],
    "material": [
        "polished glass",
        "corrugated metal",
        "timber wood",
        "rough concrete",
        "granite stone",
    ],
    "shape": ["curved", "rectangular", "circular", "triangular", "hexagonal"],
    "features": ["ribbed paneling", "horizontal slatted", "smooth flat"],
}

# =====================================================================
# HELPER FUNCTIONS & PATH NORMALIZATION
# =====================================================================


def normalize_path(path_str: str) -> str:
    """Normalizes file paths for consistent dictionary matching."""
    if not path_str or pd.isna(path_str):
        return ""
    path_str = str(path_str).replace("\\", "/").strip()
    return posixpath.normpath(path_str)


def read_csv_file(labels_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        labels_file,
        engine="python",
        encoding="utf-8",
        on_bad_lines="skip",
        quotechar='"',
        skipinitialspace=True,
    )
    return df


def safe_extract_json(text_content: str) -> Optional[Any]:
    """Safely extracts JSON object or array from LLM output."""
    text_content = text_content.strip()

    if text_content.startswith("```"):
        text_content = re.sub(r"^```(?:json)?", "", text_content)
        text_content = re.sub(r"```$", "", text_content).strip()

    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        pass

    dict_match = re.search(r"\{.*\}", text_content, re.DOTALL)
    if dict_match:
        try:
            return json.loads(dict_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# =====================================================================
# PART 1: CONTROLLED 3-5 ATTRIBUTE SWAPPING
# =====================================================================


def is_too_similar(v1: str, v2: str) -> bool:
    """Checks if two attribute values belong to the same color/style cluster."""
    v1_s, v2_s = v1.lower().strip(), v2.lower().strip()
    if v1_s == v2_s:
        return True
    for group in COLOR_CLUSTERS:
        if v1_s in group and v2_s in group:
            return True
    return False


def extract_all_attributes_from_sg(
    scene_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extracts scalar attribute instances from Scene Graph objects."""
    objects = scene_graph.get("objects", [])
    attr_list = []

    for obj_idx, obj in enumerate(objects):
        obj_label = str(
            obj.get("label", obj.get("type", "object"))
        ).lower().strip()
        attributes = obj.get("attributes", [])

        if isinstance(attributes, dict):
            raw_attrs = [
                {"key": k, "value": v} for k, v in attributes.items()
            ]
        elif isinstance(attributes, list):
            raw_attrs = attributes
        else:
            raw_attrs = [{"key": "attribute", "value": str(attributes)}]

        for attr in raw_attrs:
            if isinstance(attr, dict):
                k = str(attr.get("key", "attribute")).lower().strip()
                v = attr.get("value", "")
            else:
                k = "attribute"
                v = attr

            if isinstance(v, str):
                v_str = v.strip()
                if (v_str.startswith("[") and v_str.endswith("]")) or (
                    v_str.startswith("{") and v_str.endswith("}")
                ):
                    try:
                        v = ast.literal_eval(v_str)
                    except (ValueError, SyntaxError):
                        pass

            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    sub_v_list = (
                        sub_v if isinstance(sub_v, list) else [str(sub_v)]
                    )
                    for item_v in sub_v_list:
                        item_str = str(item_v).strip()
                        if item_str and k != "text":
                            attr_list.append(
                                {
                                    "obj_idx": obj_idx,
                                    "label": obj_label,
                                    "key": str(sub_k),
                                    "val": item_str,
                                }
                            )
            else:
                v_list = v if isinstance(v, list) else [str(v)]
                for item_v in v_list:
                    item_str = str(item_v).strip()
                    if item_str and k != "text":
                        attr_list.append(
                            {
                                "obj_idx": obj_idx,
                                "label": obj_label,
                                "key": k,
                                "val": item_str,
                            }
                        )

    return attr_list


def replace_3_to_5_attributes_rule_based(
    text: str,
    scene_graph: Dict[str, Any],
    min_swaps: int = MIN_ATTR_SWAPS,
    max_swaps: int = MAX_ATTR_SWAPS,
) -> str:
    """Swaps between 3 and 5 attributes in the sentence using in-scene migration first, falling back to distractor injection."""
    attr_list = extract_all_attributes_from_sg(scene_graph)
    if not attr_list:
        return text

    found_attrs = []
    for item in attr_list:
        val = item["val"]
        pattern = re.compile(r"\b" + re.escape(val) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            found_attrs.append(item)

    if not found_attrs:
        return text

    target_num_swaps = random.randint(
        min_swaps, min(max_swaps, max(min_swaps, len(found_attrs)))
    )

    # 1. Collect candidate in-scene swap pairs
    candidate_pairs = []
    for i in range(len(found_attrs)):
        for j in range(i + 1, len(found_attrs)):
            a1, a2 = found_attrs[i], found_attrs[j]
            if (
                a1["obj_idx"] != a2["obj_idx"]
                and a1["key"] == a2["key"]
                and not is_too_similar(a1["val"], a2["val"])
            ):
                candidate_pairs.append((a1["val"], a2["val"]))

    spans_to_replace = []
    used_values = set()

    # Apply in-scene swaps first
    if candidate_pairs:
        random.shuffle(candidate_pairs)
        for val1, val2 in candidate_pairs:
            if len(spans_to_replace) // 2 >= target_num_swaps:
                break
            if val1 in used_values or val2 in used_values:
                continue

            p1 = re.compile(r"\b" + re.escape(val1) + r"\b", re.IGNORECASE)
            p2 = re.compile(r"\b" + re.escape(val2) + r"\b", re.IGNORECASE)
            m1, m2 = p1.search(text), p2.search(text)

            if m1 and m2:
                s1, s2 = m1.span(), m2.span()
                if s1[1] <= s2[0] or s2[1] <= s1[0]:
                    spans_to_replace.append((s1, val2))
                    spans_to_replace.append((s2, val1))
                    used_values.add(val1)
                    used_values.add(val2)

    # 2. External Injection Fallback if more swaps are needed
    current_swaps = len(spans_to_replace) // 2
    if current_swaps < target_num_swaps:
        random.shuffle(found_attrs)
        for item in found_attrs:
            if current_swaps >= target_num_swaps:
                break
            val1 = item["val"]
            key1 = item["key"]

            if val1 in used_values:
                continue

            distractors = HIGH_CONTRAST_DISTRACTORS.get(
                key1, HIGH_CONTRAST_DISTRACTORS["color"]
            )
            valid_distractors = [
                d for d in distractors if not is_too_similar(val1, d)
            ]
            if valid_distractors:
                new_val = random.choice(valid_distractors)
                p1 = re.compile(r"\b" + re.escape(val1) + r"\b", re.IGNORECASE)
                m1 = p1.search(text)
                if m1:
                    spans_to_replace.append((m1.span(), new_val))
                    used_values.add(val1)
                    current_swaps += 1

    if not spans_to_replace:
        return text

    # Sort spans in reverse order to preserve string character indices during replacement
    spans_to_replace = sorted(spans_to_replace, key=lambda x: x[0][0], reverse=True)

    mutated_text = text
    for (start, end), new_val in spans_to_replace:
        mutated_text = mutated_text[:start] + new_val + mutated_text[end:]

    return mutated_text


# =====================================================================
# PART 2: ALWAYS MUTATE SIGN TEXT (LLM + FALLBACK)
# =====================================================================


def extract_sign_text(
    text: str, scene_graph: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """Extracts sign/OCR text from double quotes or from the Scene Graph text attributes."""
    quote_match = re.search(r'""(.*?)""', text) or re.search(r'"([^"]+)"', text)
    if quote_match:
        return quote_match.group(1), "quote"

    for obj in scene_graph.get("objects", []):
        for attr in obj.get("attributes", []):
            if isinstance(attr, dict) and attr.get("key") == "text":
                return str(attr.get("value")), "sg"

    return None, "none"


async def request_llm_sign_replacement(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, original_sign: str
) -> Tuple[str, str]:
    async with semaphore:
        prompt = (
            f'Original Sign/Street Text: "{original_sign}"\n'
            f"Task: Generate 1 alternative sign text that replaces street names, postcodes, business names, or numbers. "
            f"Keep the structural style realistic for city signage, but change the specific identifiers completely.\n"
            f'Output strictly JSON format: {{"new_sign": "ALTERNATIVE_TEXT"}}'
        )

        payload = {
            "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise dataset mutation assistant. Respond"
                        " ONLY with a valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 48,
        }

        try:
            response = await client.post(VLLM_URL, json=payload, timeout=5.0)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                parsed = safe_extract_json(content)
                if isinstance(parsed, dict) and "new_sign" in parsed:
                    new_sign = str(parsed["new_sign"]).strip()
                    if new_sign and new_sign != original_sign:
                        return original_sign, new_sign

        except Exception:
            pass

        # Guaranteed rule-based fallback if vLLM API is unreachable
        fallback_sign = (
            original_sign.replace("CHECKS CASHED", "PAYDAY LOANS")
            .replace("GREYCOAT STREET", "SILVERWOOD ROAD")
            .replace("SW1", "E1")
            .replace("221B", "104A")
        )
        if fallback_sign == original_sign:
            fallback_sign = "OXFORD STREET W1"

        return original_sign, fallback_sign


# =====================================================================
# MAIN PIPELINE
# =====================================================================


async def main():
    if not os.path.exists(CSV_PATH):
        print(f"[!] Error: Could not find CSV file at '{CSV_PATH}'")
        return
    if not os.path.exists(SG_JSON):
        print(f"[!] Error: Could not find SG JSON file at '{SG_JSON}'")
        return

    # Load CSV data
    df = read_csv_file(CSV_PATH)
    print(f"[✓] Loaded {len(df)} records from CSV: {CSV_PATH}")

    # Ensure 'flip' column exists even if missing from original input CSV
    if "flip" not in df.columns:
        df["flip"] = ""

    # Load Scene Graph JSON and build lookup dictionary indexed by normalized image path
    with open(SG_JSON, "r", encoding="utf-8") as f:
        sg_raw = json.load(f)

    sg_lookup = {}
    if isinstance(sg_raw, list):
        for item in sg_raw:
            img_key = normalize_path(
                item.get("image_path") or item.get("image_id") or ""
            )
            sg_body = item.get("scene_graph") or item
            if img_key:
                sg_lookup[img_key] = sg_body
    elif isinstance(sg_raw, dict):
        for img_key, sg_body in sg_raw.items():
            norm_key = normalize_path(img_key)
            sg_lookup[norm_key] = (
                sg_body.get("scene_graph")
                if isinstance(sg_body, dict) and "scene_graph" in sg_body
                else sg_body
            )

    print(
        f"[✓] Built Scene Graph lookup map for {len(sg_lookup)} image entries."
    )

    print("\n==================================================")
    print("      GENERATING 3-5 SWAP HARD NEGATIVES          ")
    print("==================================================\n")

    # Step 1: Extract unique sign texts from the paired data
    unique_signs = set()
    dataset_records = []

    for idx, row in df.iterrows():
        raw_img_path = str(row["image_path"])
        norm_img_path = normalize_path(raw_img_path)
        desc = str(row["description"]) if "description" in row else ""

        # Retrieve matching Scene Graph
        sg = sg_lookup.get(norm_img_path, {})

        sign_text, _ = extract_sign_text(desc, sg)
        if sign_text:
            unique_signs.add(sign_text)

        dataset_records.append(
            {
                "row_idx": idx,
                "image_path": raw_img_path,
                "normalized_path": norm_img_path,
                "text": desc,
                "scene_graph": sg,
                "row_data": row.to_dict(),
            }
        )

    print(f"[*] Found {len(unique_signs)} unique sign/OCR texts across dataset.")

    # Step 2: Fetch LLM / Fallback sign replacements
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    limits = httpx.Limits(
        max_keepalive_connections=20, max_connections=CONCURRENCY_LIMIT
    )

    sign_replacement_map = {}
    if unique_signs:
        print("[*] Generating sign text replacements...")
        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [
                request_llm_sign_replacement(client, semaphore, sign)
                for sign in unique_signs
            ]
            results = await asyncio.gather(*tasks)

        for orig_sign, new_sign in results:
            sign_replacement_map[orig_sign] = new_sign

    # Step 3: Process dataset records and create hard negatives
    hn_list = []

    for item in dataset_records:
        orig_text = item["text"]
        sg = item["scene_graph"]

        # 1. Perform 3-5 Controlled Attribute Swaps
        hard_neg_text = replace_3_to_5_attributes_rule_based(
            orig_text, sg, min_swaps=MIN_ATTR_SWAPS, max_swaps=MAX_ATTR_SWAPS
        )

        # 2. Mutate Sign / OCR text if present
        orig_sign, _ = extract_sign_text(orig_text, sg)
        if orig_sign and orig_sign in sign_replacement_map:
            new_sign = sign_replacement_map[orig_sign]
            hard_neg_text = hard_neg_text.replace(orig_sign, new_sign)

        hn_list.append(hard_neg_text)

    # Assign generated hard negative texts to new 'hn' column
    df["hn"] = hn_list

    # Force exact column ordering strictly to: image_path, description, flip, hn
    output_columns = ["image_path", "description", "flip", "hn"]

    df[output_columns].to_csv(
        OUTPUT_DATASET_PATH, index=False, encoding="utf-8"
    )

    print(
        f"\n[✓] Done! Saved {len(df)} processed records to"
        f" '{OUTPUT_DATASET_PATH}'"
    )
    print(f"[✓] CSV Header: {','.join(output_columns)}")

    # Display first sample result as verification
    if len(df) > 0:
        sample = df.iloc[0]
        print("\n--- SAMPLE #1 RECORD ---")
        print(f"IMAGE PATH: {sample.get('image_path')}")
        print(f"DESCRIPTION:\n{sample.get('description')}\n")
        print(f"FLIP:\n{sample.get('flip')}\n")
        print(f"HN:\n{sample.get('hn')}\n")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())