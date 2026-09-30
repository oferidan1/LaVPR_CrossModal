import json
import random
import re
from typing import Any, Dict, List, Optional

# Generic street/location distractors for any city
GENERIC_LOCATION_WORDS = {
    "STREET": ["ROAD", "AVENUE", "BOULEVARD", "LANE", "WAY", "PLACE"],
    "ROAD": ["STREET", "AVENUE", "DRIVE", "WAY"],
    "AVENUE": ["STREET", "ROAD", "BOULEVARD"],
    "CITY": ["TOWN", "BOROUGH", "DISTRICT"],
    "NORTH": ["SOUTH", "EAST", "WEST"],
    "SOUTH": ["NORTH", "EAST", "WEST"],
    "EAST": ["WEST", "NORTH", "SOUTH"],
    "WEST": ["EAST", "NORTH", "SOUTH"],
}

COLOR_SIMILARITY_GROUPS = [
    {"white", "beige", "light grey", "grey", "cream", "off-white"},
    {"black", "dark grey", "charcoal", "dark"},
    {"orange-brown", "brown", "brick", "terracotta", "red brick"},
    {"blue", "navy", "cyan"},
    {"red", "crimson", "scarlet"},
]

HIGH_CONTRAST_FALLBACKS = {
    "color": ["bright red", "vivid blue", "neon yellow", "bright green"],
    "material": ["polished glass", "corrugated metal", "wood timber"],
}


def is_too_similar(v1: str, v2: str) -> bool:
    """Checks if two attribute values are identical or belong to the same color cluster."""
    v1_s, v2_s = v1.lower().strip(), v2.lower().strip()
    if v1_s == v2_s:
        return True
    for group in COLOR_SIMILARITY_GROUPS:
        if v1_s in group and v2_s in group:
            return True
    return False


def mutate_any_sign_text(original_ocr: str) -> str:
    """General mutator: Alters ANY sign/OCR text while preserving casing and structure.

    Works on street names, shop signs, postcodes, or directional signs.
    """
    words = original_ocr.split()
    mutated_words = []

    for word in words:
        clean_word = re.sub(r"[^\w]", "", word).upper()

        # Strategy A: Substitute known generic street/location terms
        if clean_word in GENERIC_LOCATION_WORDS:
            replacement = random.choice(GENERIC_LOCATION_WORDS[clean_word])
            # Preserve original casing
            if word.isupper():
                replacement = replacement.upper()
            elif word.istitle():
                replacement = replacement.title()
            mutated_words.append(
                word.replace(clean_word, replacement.upper())
            )

        # Strategy B: Mutate Digits (e.g., SW1 -> SW4, 2024 -> 8024)
        elif re.search(r"\d", word):
            mutated = ""
            for char in word:
                if char.isdigit():
                    # Pick a different digit
                    new_digit = str((int(char) + random.randint(1, 8)) % 10)
                    mutated += new_digit
                else:
                    mutated += char
            mutated_words.append(mutated)

        # Strategy C: Character Swap for General Words (len > 3)
        elif len(clean_word) > 3 and random.random() < 0.4:
            chars = list(word)
            # Find candidate letter indices to swap inside word
            letter_indices = [
                i for i, c in enumerate(chars) if c.isalpha() and c.isupper()
            ]
            if len(letter_indices) >= 2:
                idx1, idx2 = letter_indices[0], letter_indices[1]
                chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
            mutated_words.append("".join(chars))

        else:
            mutated_words.append(word)

    result = " ".join(mutated_words)

    # Hard Fallback: If no word was mutated, flip 1-2 random capital letters
    if result == original_ocr and len(original_ocr) > 2:
        chars = list(original_ocr)
        caps_indices = [i for i, c in enumerate(chars) if c.isupper()]
        if caps_indices:
            idx = random.choice(caps_indices)
            # Shift character by 1 in alphabet (A -> B, Z -> A)
            chars[idx] = (
                chr((ord(chars[idx]) - 65 + 1) % 26 + 65)
                if chars[idx].isupper()
                else "X"
            )
            result = "".join(chars)

    return result


def perturb_street_sign_text(text: str, scene_graph: Dict[str, Any]) -> str:
    """Locates any sign or text attribute in the SG / Caption and mutates it generally."""
    mutated_text = text
    objects = scene_graph.get("objects", [])

    # 1. Extract exact sign text from Scene Graph if present
    sg_ocr_text = None
    for obj in objects:
        for attr in obj.get("attributes", []):
            if attr.get("key") == "text" and attr.get("value"):
                sg_ocr_text = str(attr.get("value")).strip()
                break

    # 2. Mutate quoted text (e.g. ""ANY SIGN TEXT"")
    quote_match = re.search(r'""(.*?)""', mutated_text)
    if quote_match:
        target_str = quote_match.group(1)
        new_str = mutate_any_sign_text(target_str)
        return mutated_text.replace(target_str, new_str)

    # 3. If SG contains OCR text, search and mutate directly in caption
    if sg_ocr_text:
        pattern = re.compile(re.escape(sg_ocr_text), re.IGNORECASE)
        match = pattern.search(mutated_text)
        if match:
            target_str = match.group(0)
            new_str = mutate_any_sign_text(target_str)
            return mutated_text.replace(target_str, new_str)

    return mutated_text


def generate_attribute_migration_negative(
    text: str,
    scene_graph: Dict[str, Any],
    target_keys: List[str] = ["color", "material", "shape"],
) -> str:
    """Inputs original full text and SG -> Outputs a hard negative text string

    with migrated attributes AND generally perturbed sign text.
    """
    # Step 1: General Sign / OCR Perturbation
    mutated_text = perturb_street_sign_text(text, scene_graph)

    # Step 2: Extract swappable attributes from SG
    objects = scene_graph.get("objects", [])
    attr_list = []
    for obj_idx, obj in enumerate(objects):
        for attr in obj.get("attributes", []):
            k = attr.get("key")
            v = str(attr.get("value", "")).strip()
            v_clean = (
                v.replace("[", "").replace("]", "").replace("'", "").strip()
            )

            if k in target_keys and v_clean:
                attr_list.append(
                    {
                        "obj_idx": obj_idx,
                        "label": obj.get("label"),
                        "key": k,
                        "val": v_clean,
                    }
                )

    # Step 3: Find valid candidate pairs for attribute migration
    high_contrast_pairs = []
    for i in range(len(attr_list)):
        for j in range(i + 1, len(attr_list)):
            a1, a2 = attr_list[i], attr_list[j]
            if a1["obj_idx"] != a2["obj_idx"] and a1["key"] == a2["key"]:
                if not is_too_similar(a1["val"], a2["val"]):
                    high_contrast_pairs.append((a1, a2))

    # Path A: Internal Attribute Swap
    if high_contrast_pairs:
        random.shuffle(high_contrast_pairs)
        for a1, a2 in high_contrast_pairs:
            v1, v2 = a1["val"], a2["val"]

            p1 = re.compile(r"\b" + re.escape(v1) + r"\b", re.IGNORECASE)
            p2 = re.compile(r"\b" + re.escape(v2) + r"\b", re.IGNORECASE)

            m1 = p1.search(mutated_text)
            m2 = p2.search(mutated_text)

            if m1 and m2:
                s1, s2 = m1.span(), m2.span()
                if s1[1] <= s2[0] or s2[1] <= s1[0]:
                    spans = sorted(
                        [(s1, v2), (s2, v1)],
                        key=lambda x: x[0][0],
                        reverse=True,
                    )
                    temp_text = mutated_text
                    for (start, end), new_val in spans:
                        temp_text = (
                            temp_text[:start] + new_val + temp_text[end:]
                        )
                    if temp_text != text:
                        return temp_text

    # Path B: External Attribute Injection Fallback
    random.shuffle(attr_list)
    for a in attr_list:
        k, v1 = a["key"], a["val"]
        if k in HIGH_CONTRAST_FALLBACKS:
            distractor = random.choice(HIGH_CONTRAST_FALLBACKS[k])
            p1 = re.compile(r"\b" + re.escape(v1) + r"\b", re.IGNORECASE)
            m1 = p1.search(mutated_text)
            if m1:
                start, end = m1.span()
                temp_text = mutated_text[:start] + distractor + mutated_text[end:]
                if temp_text != text:
                    return temp_text

    return mutated_text


def main():
    # Sample test with arbitrary/generic sign text
    sample_dataset = [
        {
            "image_id": "Images/Sample_01.jpg",
            "text": (
                'Black pole with round black sign, grey sidewalk, dark brick'
                " wall, orange-brown brick building with rectangular"
                " indentations and glass panel roof structure, dark grey"
                " asphalt road, curved glass facade, light grey building wall"
                ' with horizontal panels, beige brick facade, white street sign'
                ' ""BAKER STREET 221B, CITY OF LONDON"", white-framed windows,'
                " dark trim at the base of the light grey building."
            ),
            "scene_graph": {
                "objects": [
                    {
                        "id": "pole_1",
                        "label": "pole",
                        "attributes": [{"key": "color", "value": "black"}],
                    },
                    {
                        "id": "street_sign_1",
                        "label": "street sign",
                        "attributes": [
                            {"key": "color", "value": "white"},
                            {
                                "key": "text",
                                "value": "BAKER STREET 221B, CITY OF LONDON",
                            },
                        ],
                    },
                    {
                        "id": "building_1",
                        "label": "building",
                        "attributes": [
                            {"key": "color", "value": "orange-brown"}
                        ],
                    },
                ]
            },
        }
    ]

    print("==================================================")
    print("      LaVPR HARD NEGATIVE TEXT GENERATOR          ")
    print("==================================================\n")

    for idx, item in enumerate(sample_dataset, start=1):
        original_text = item["text"]
        sg = item["scene_graph"]

        hard_neg_text = generate_attribute_migration_negative(
            original_text, sg
        )

        print(f"--- SAMPLE #{idx} ---")
        print(f"ORIGINAL:\n{original_text}\n")
        print(f"HARD NEGATIVE:\n{hard_neg_text}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()