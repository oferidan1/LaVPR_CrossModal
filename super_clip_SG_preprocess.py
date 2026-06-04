# import json
# import os
# import ast
# import argparse
# import torch
# from collections import defaultdict
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.preprocessing import normalize

# def parse_attribute_value(val):
#     """
#     Safely unpacks attribute values, handling stringified lists
#     or regular string formats into clean, lowercase tokens.
#     """
#     val_str = str(val).strip()
#     if val_str.startswith('[') and val_str.endswith(']'):
#         try:
#             parsed = ast.literal_eval(val_str)
#             if isinstance(parsed, list):
#                 return [str(item).strip().lower().replace(" ", "_") for item in parsed]
#         except Exception:
#             pass
#     return [val_str.lower().replace(" ", "_")]

# def build_semantic_lookup(strings_set, model, distance_threshold):
#     """
#     Encodes strings into dense vectors, runs agglomerative hierarchical clustering,
#     and returns a mapping dict from raw string to a canonical string representative.
#     """
#     str_list = sorted(list(strings_set))
#     if not str_list:
#         return {}
        
#     # Agglomerative clustering requires at least 2 samples
#     if len(str_list) == 1:
#         raw_str = str_list[0]
#         canonical_name = raw_str.lower().strip().replace(" ", "_")
#         return {raw_str: canonical_name}
        
#     # Generate sentence embeddings and normalize to unit hypersphere
#     embeddings = model.encode(str_list, show_progress_bar=False, convert_to_numpy=True)
#     embeddings = normalize(embeddings)

#     # Use hierarchical clustering with a strict maximum semantic distance threshold
#     clustering = AgglomerativeClustering(
#         n_clusters=None,
#         metric="euclidean",
#         linkage="average",
#         distance_threshold=distance_threshold
#     )
#     labels = clustering.fit_predict(embeddings)

#     cluster_to_strs = defaultdict(list)
#     for string, c_id in zip(str_list, labels):
#         cluster_to_strs[c_id].append(string)

#     mapping = {}
#     for c_id, strs in cluster_to_strs.items():
#         # Select the shortest string length as the clean canonical representative
#         canonical_name = min(strs, key=len).lower().strip().replace(" ", "_")
#         for raw_str in strs:
#             mapping[raw_str] = canonical_name
#     return mapping

# def generate_canonical_maps(dataset, obj_threshold, attr_threshold):
#     """
#     Extracts raw pools of objects and attributes globally from the dataset
#     and constructs their decoupled canonical cluster maps.
#     """
#     raw_objects = set()
#     key_to_raw_attrs = defaultdict(set)
    
#     print("Extracting raw textual variations for global clustering...")
#     for entry in dataset:
#         sg = entry.get("scene_graph") or {}
#         if not isinstance(sg, dict):
#             sg = {}
#         for obj in (sg.get("objects") or []):
#             if not isinstance(obj, dict):
#                 continue
#             obj_label = obj.get("label")
#             if obj_label:
#                 raw_objects.add(obj_label.strip().lower())
                
#             for attr in obj.get("attributes", []):
#                 # Type guard check to prevent crashes on unnormalized dataset strings (e.g., in pitts30k)
#                 if not isinstance(attr, dict):
#                     continue
#                 key = attr.get("key", "").strip().lower()
#                 val = attr.get("value", "")
                
#                 # Skip reading signs text layout directly into clustering to avoid unique string clutter
#                 if key and val and key != "text":
#                     unpacked_vals = parse_attribute_value(val)
#                     for v in unpacked_vals:
#                         if v:
#                             key_to_raw_attrs[key].add(v)

#     print("Initializing Sentence-Transformer model (all-MiniLM-L6-v2)...")
#     model = SentenceTransformer("all-MiniLM-L6-v2") 

#     # Cluster Objects
#     print(f" -> Discovering canonical concepts across {len(raw_objects)} unique objects...")
#     object_canonical_map = build_semantic_lookup(raw_objects, model, obj_threshold)

#     # Cluster Attributes per namespace
#     attr_canonical_map = {}
#     for key, unique_vals in key_to_raw_attrs.items():
#         print(f" -> Discovering canonical clusters inside attribute namespace '{key}' ({len(unique_vals)} unique values)...")
#         namespace_map = build_semantic_lookup(unique_vals, model, attr_threshold)
#         for raw_val, canonical_val in namespace_map.items():
#             attr_canonical_map[f"{key}_{raw_val}"] = canonical_val

#     return object_canonical_map, attr_canonical_map

# def flatten_with_double_clustering(entry, object_map, attr_map):
#     """
#     Decomposes an image's scene graph nodes into decoupled pairwise 
#     canonical attributes, objects, and relation triplets.
#     """
#     tokens = set()
#     sg = entry.get("scene_graph") or {}
#     if not isinstance(sg, dict):
#         sg = {}
#     objects_list = sg.get("objects") or []
#     relationships_list = sg.get("relationships") or []
    
#     # 1. Map instance IDs to their global Canonical Labels
#     id_to_canonical_label = {}
#     for obj in objects_list:
#         if not isinstance(obj, dict):
#             continue
#         obj_id = obj.get("id")
#         raw_label = str(obj.get("label", "")).strip().lower()
        
#         canonical_obj = object_map.get(raw_label, raw_label.replace(" ", "_"))
#         if obj_id and canonical_obj:
#             id_to_canonical_label[obj_id] = canonical_obj

#     # 2. Flatten Objects and create independent attribute-object pairs
#     for obj in objects_list:
#         if not isinstance(obj, dict):
#             continue
#         obj_label = id_to_canonical_label.get(obj.get("id"))
#         if not obj_label:
#             continue
            
#         # Always maintain the base baseline object category 
#         tokens.add(obj_label)
            
#         for attr in obj.get("attributes", []):
#             if not isinstance(attr, dict):
#                 continue
#             attr_key = str(attr.get("key", "")).strip().lower().replace(" ", "_")
            
#             # Special case handle sign texts without breaking vocabulary footprint
#             if attr_key == "text":
#                 tokens.add(f"text_has_text_{obj_label}")
#                 continue
                
#             unpacked_values = parse_attribute_value(attr.get("value", ""))
#             for val in unpacked_values:
#                 if val:
#                     lookup_key = f"{attr_key}_{val}"
#                     clean_attr = attr_map.get(lookup_key, val.replace(" ", "_"))
                    
#                     # Form pairwise target token entry
#                     pairwise_token = f"{attr_key}_{clean_attr}_{obj_label}"
#                     tokens.add(pairwise_token)

#     # 3. Extract Triplets based on canonical targets
#     for rel in relationships_list:
#         if not isinstance(rel, dict):
#             continue
#         sub_id = rel.get("subject_id")
#         pred = str(rel.get("predicate", "")).strip().lower().replace(" ", "_")
#         obj_id = rel.get("object_id")
        
#         sub_label = id_to_canonical_label.get(sub_id)
#         obj_label = id_to_canonical_label.get(obj_id)
        
#         if sub_label and pred and obj_label:
#             flattened_relation = f"{sub_label}_{pred}_{obj_label}"
#             tokens.add(flattened_relation)
            
#     return tokens

# def main():
#     parser = argparse.ArgumentParser(description="Double-Sided Semantic Scene Graph Token Preprocessor")
#     #parser.add_argument("--json_file", type=str, default="pitts30k_val_800_queries_objects_intermediate.json", help="Path to input scene graph json file")
#     parser.add_argument("--json_file", type=str, default="datasets/descriptions/gsv_cities_descriptions_sg.json", help="Path to input scene graph json file")
#     parser.add_argument("--out_vocab_file", type=str, default="scene_graph_vocab.json", help="Output JSON file for categorical vocab registry")
#     parser.add_argument("--out_img_idf", type=str, default="gsv_cities_image_idf.pt", help="Output filename for static PyTorch Image IDF tensor")
#     parser.add_argument("--obj_threshold", type=float, default=0.45, help="Max semantic distance cutoff for merging object variations")
#     parser.add_argument("--attr_threshold", type=float, default=0.50, help="Max semantic distance cutoff for merging attribute variations")
    
#     args = parser.parse_args()
    
#     if not os.path.exists(args.json_file):
#         raise FileNotFoundError(f"Input JSON scene graph file not found at: {args.json_file}")
        
#     with open(args.json_file, "r") as f:
#         dataset = json.load(f)
    
#     total_images = len(dataset)
#     print(f"Successfully loaded {total_images} scene graph rows.")

#     # PHASE 1: GENERATE DECOUPLED SEMANTIC CLUSTER MAPS
#     object_map, attr_map = generate_canonical_maps(dataset, args.obj_threshold, args.attr_threshold)

#     # PHASE 2: FLATTEN SCENE GRAPHS AND CONSTRUCT GLOBAL VOCABULARY
#     print("\nFlattening elements into unique unified concepts...")
#     all_image_token_sets = []
#     global_canonical_concepts = set()
    
#     for entry in dataset:
#         entry_tokens = flatten_with_double_clustering(entry, object_map, attr_map)
#         all_image_token_sets.append(entry_tokens)
#         global_canonical_concepts.update(entry_tokens)
        
#     # ─── NEW: COMPUTE GLOBAL FREQUENCIES & APPLY CUTOFF ──────────────────
#     print("Calculating token frequencies to eliminate long-tail noise...")
#     token_global_frequencies = defaultdict(int)
#     for entry_tokens in all_image_token_sets:
#         for token in entry_tokens:
#             token_global_frequencies[token] += 1

#     # Define your minimum appearance filter constraint (adjust as needed)
#     MIN_FREQ = 5 
    
#     filtered_concepts = [
#         t for t in global_canonical_concepts 
#         if token_global_frequencies[t] >= MIN_FREQ
#     ]
    
#     print(f" -> Cutoff applied (MIN_FREQ={MIN_FREQ}).")
#     print(f" -> Compressed raw unique pool from {len(global_canonical_concepts)} down to {len(filtered_concepts)} high-yield concepts.")
#     # ─────────────────────────────────────────────────────────────────────

#     # Sort vocabulary to guarantee repeatable index mapping
#     # CRITICAL: We now sort 'filtered_concepts' instead of 'global_canonical_concepts'
#     sorted_concepts = sorted(filtered_concepts)
#     concept_vocab = {"<PAD>": 0}
#     for idx, concept in enumerate(sorted_concepts):
#         concept_vocab[concept] = idx + 1
        
#     vocab_size = len(concept_vocab)
#     print(f"Generated clean multi-label target vocabulary. Size (M) = {vocab_size} classes (including <PAD>)")
    
#     with open(args.out_vocab_file, "w") as f:
#         json.dump(concept_vocab, f, indent=4)
#     print(f" -> Vocabulary configuration registry cached to: {args.out_vocab_file}")

#     # PHASE 3: CALCULATE IMAGE DOCUMENT FREQUENCY MATRIX
#     print("\nAccumulating global dataset document frequencies...")
#     image_counts = torch.zeros(vocab_size, dtype=torch.float32)
    
#     for entry_tokens in all_image_token_sets:
#         # Map tokens present inside current image graph into explicit categorical integer slots
#         active_indices = list(set([concept_vocab[t] for t in entry_tokens if t in concept_vocab]))
#         if active_indices:
#             image_counts[active_indices] += 1

#     # Compute static SuperCLIP Image Log IDF representations
#     print("Vectorizing final log document weights...")
#     idf_image = torch.log(torch.tensor(total_images) / (1.0 + image_counts))
    
#     # Strictly zero out padding target weight and clamp numerical rounding safe floors
#     idf_image[0] = 0.0
#     idf_image.clamp_(min=0.0)
    
#     torch.save(idf_image, args.out_img_idf)
#     print(f" -> Image-level IDF weights tensor array saved to: {args.out_img_idf}")
#     print("\nPreprocessing workflow completed successfully!")

# if __name__ == "__main__":
#     main()

import json
import os
import ast
import argparse
import torch
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

def parse_attribute_value(val):
    """
    Safely unpacks attribute values, handling stringified lists
    or regular string formats into clean, lowercase tokens.
    """
    val_str = str(val).strip()
    if val_str.startswith('[') and val_str.endswith(']'):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(item).strip().lower().replace(" ", "_") for item in parsed]
        except Exception:
            pass
    return [val_str.lower().replace(" ", "_")]

def build_semantic_lookup(strings_set, model, distance_threshold):
    """
    Encodes strings into dense vectors, runs agglomerative hierarchical clustering,
    and returns a mapping dict from raw string to a canonical string representative.
    """
    str_list = sorted(list(strings_set))
    if not str_list:
        return {}
        
    # Agglomerative clustering requires at least 2 samples
    if len(str_list) == 1:
        raw_str = str_list[0]
        canonical_name = raw_str.lower().strip().replace(" ", "_")
        return {raw_str: canonical_name}
        
    # Generate sentence embeddings and normalize to unit hypersphere
    embeddings = model.encode(str_list, show_progress_bar=False, convert_to_numpy=True)
    embeddings = normalize(embeddings)

    # Use hierarchical clustering with a strict maximum semantic distance threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="euclidean",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(embeddings)

    cluster_to_strs = defaultdict(list)
    for string, c_id in zip(str_list, labels):
        cluster_to_strs[c_id].append(string)

    mapping = {}
    for c_id, strs in cluster_to_strs.items():
        # Select the shortest string length as the clean canonical representative
        canonical_name = min(strs, key=len).lower().strip().replace(" ", "_")
        for raw_str in strs:
            mapping[raw_str] = canonical_name
    return mapping

def generate_canonical_maps(dataset, obj_threshold, attr_threshold):
    """
    Extracts raw pools of objects and attributes globally from the dataset
    and constructs their decoupled canonical cluster maps.
    """
    raw_objects = set()
    key_to_raw_attrs = defaultdict(set)
    
    print("Extracting raw textual variations for global clustering...")
    for entry in dataset:
        sg = entry.get("scene_graph") or {}
        if not isinstance(sg, dict):
            sg = {}
        for obj in (sg.get("objects") or []):
            if not isinstance(obj, dict):
                continue
            obj_label = obj.get("label")
            if obj_label:
                raw_objects.add(obj_label.strip().lower())
                
            for attr in obj.get("attributes", []):
                # Type guard check to prevent crashes on unnormalized dataset strings (e.g., in pitts30k)
                if not isinstance(attr, dict):
                    continue
                key = attr.get("key", "").strip().lower()
                val = attr.get("value", "")
                
                # Skip reading signs text layout directly into clustering to avoid unique string clutter
                if key and val and key != "text":
                    unpacked_vals = parse_attribute_value(val)
                    for v in unpacked_vals:
                        if v:
                            key_to_raw_attrs[key].add(v)

    print("Initializing Sentence-Transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2") 

    # Cluster Objects
    print(f" -> Discovering canonical concepts across {len(raw_objects)} unique objects...")
    object_canonical_map = build_semantic_lookup(raw_objects, model, obj_threshold)

    # Cluster Attributes per namespace
    attr_canonical_map = {}
    for key, unique_vals in key_to_raw_attrs.items():
        print(f" -> Discovering canonical clusters inside attribute namespace '{key}' ({len(unique_vals)} unique values)...")
        namespace_map = build_semantic_lookup(unique_vals, model, attr_threshold)
        for raw_val, canonical_val in namespace_map.items():
            attr_canonical_map[f"{key}_{raw_val}"] = canonical_val

    return object_canonical_map, attr_canonical_map

def flatten_with_double_clustering(entry, object_map, attr_map):
    """
    Decomposes an image's scene graph nodes into decoupled pairwise 
    canonical attributes, objects, and relation triplets.
    """
    tokens = set()
    sg = entry.get("scene_graph") or {}
    if not isinstance(sg, dict):
        sg = {}
    objects_list = sg.get("objects") or []
    relationships_list = sg.get("relationships") or []
    
    # 1. Map instance IDs to their global Canonical Labels
    id_to_canonical_label = {}
    for obj in objects_list:
        if not isinstance(obj, dict):
            continue
        obj_id = obj.get("id")
        raw_label = str(obj.get("label", "")).strip().lower()
        
        canonical_obj = object_map.get(raw_label, raw_label.replace(" ", "_"))
        if obj_id and canonical_obj:
            id_to_canonical_label[obj_id] = canonical_obj

    # 2. Flatten Objects and create independent attribute-object pairs
    for obj in objects_list:
        if not isinstance(obj, dict):
            continue
        obj_label = id_to_canonical_label.get(obj.get("id"))
        if not obj_label:
            continue
            
        # Always maintain the base baseline object category 
        tokens.add(obj_label)
            
        for attr in obj.get("attributes", []):
            if not isinstance(attr, dict):
                continue
            attr_key = str(attr.get("key", "")).strip().lower().replace(" ", "_")
            
            # Special case handle sign texts without breaking vocabulary footprint
            if attr_key == "text":
                tokens.add(f"text_has_text_{obj_label}")
                continue
                
            unpacked_values = parse_attribute_value(attr.get("value", ""))
            for val in unpacked_values:
                if val:
                    lookup_key = f"{attr_key}_{val}"
                    clean_attr = attr_map.get(lookup_key, val.replace(" ", "_"))
                    
                    # Form pairwise target token entry
                    pairwise_token = f"{attr_key}_{clean_attr}_{obj_label}"
                    tokens.add(pairwise_token)

    # 3. Extract Triplets based on canonical targets
    for rel in relationships_list:
        if not isinstance(rel, dict):
            continue
        sub_id = rel.get("subject_id")
        pred = str(rel.get("predicate", "")).strip().lower().replace(" ", "_")
        obj_id = rel.get("object_id")
        
        sub_label = id_to_canonical_label.get(sub_id)
        obj_label = id_to_canonical_label.get(obj_id)
        
        if sub_label and pred and obj_label:
            flattened_relation = f"{sub_label}_{pred}_{obj_label}"
            tokens.add(flattened_relation)
            
    return tokens

def main():
    parser = argparse.ArgumentParser(description="Double-Sided Semantic Scene Graph Token Preprocessor")
    parser.add_argument("--json_file", type=str, default="datasets/descriptions/gsv_cities_descriptions_sg.json", help="Path to input scene graph json file")
    #parser.add_argument("--json_file", type=str, default="pitts30k_val_800_queries_objects_intermediate.json", help="Path to input scene graph json file")
    parser.add_argument("--out_vocab_file", type=str, default="gsv_cities_scene_graph_vocab.json", help="Output JSON file for categorical vocab registry")
    parser.add_argument("--out_img_idf", type=str, default="gsv_cities_image_idf.pt", help="Output filename for static PyTorch Image IDF tensor")    
    parser.add_argument("--out_img_map", type=str, default="gsv_cities_image_id_to_vocab_indices.json", help="Output JSON filename for precomputed cache lookup mapping")
    parser.add_argument("--obj_threshold", type=float, default=0.45, help="Max semantic distance cutoff for merging object variations")
    parser.add_argument("--attr_threshold", type=float, default=0.50, help="Max semantic distance cutoff for merging attribute variations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"Input JSON scene graph file not found at: {args.json_file}")
        
    with open(args.json_file, "r") as f:
        dataset = json.load(f)
    
    total_images = len(dataset)
    print(f"Successfully loaded {total_images} scene graph rows.")

    # PHASE 1: GENERATE DECOUPLED SEMANTIC CLUSTER MAPS
    object_map, attr_map = generate_canonical_maps(dataset, args.obj_threshold, args.attr_threshold)

    # PHASE 2: FLATTEN SCENE GRAPHS AND CONSTRUCT GLOBAL VOCABULARY
    print("\nFlattening elements into unique unified concepts...")
    all_image_token_sets = []
    global_canonical_concepts = set()
    
    for entry in dataset:
        entry_tokens = flatten_with_double_clustering(entry, object_map, attr_map)
        all_image_token_sets.append(entry_tokens)
        global_canonical_concepts.update(entry_tokens)
        
    # ─── COMPUTE GLOBAL FREQUENCIES & APPLY CUTOFF ──────────────────
    print("Calculating token frequencies to eliminate long-tail noise...")
    token_global_frequencies = defaultdict(int)
    for entry_tokens in all_image_token_sets:
        for token in entry_tokens:
            token_global_frequencies[token] += 1

    # Define your minimum appearance filter constraint (adjust as needed)
    MIN_FREQ = 5 
    
    filtered_concepts = [
        t for t in global_canonical_concepts 
        if token_global_frequencies[t] >= MIN_FREQ
    ]
    
    print(f" -> Cutoff applied (MIN_FREQ={MIN_FREQ}).")
    print(f" -> Compressed raw unique pool from {len(global_canonical_concepts)} down to {len(filtered_concepts)} high-yield concepts.")

    # Sort vocabulary to guarantee repeatable index mapping
    sorted_concepts = sorted(filtered_concepts)
    concept_vocab = {"<PAD>": 0}
    for idx, concept in enumerate(sorted_concepts):
        concept_vocab[concept] = idx + 1
        
    vocab_size = len(concept_vocab)
    print(f"Generated clean multi-label target vocabulary. Size (M) = {vocab_size} classes (including <PAD>)")
    
    with open(args.out_vocab_file, "w") as f:
        json.dump(concept_vocab, f, indent=4)
    print(f" -> Vocabulary configuration registry cached to: {args.out_vocab_file}")

    # =====================================================================
    # PHASE 3: CALCULATE IMAGE DOCUMENT FREQUENCY MATRIX & IMAGE INDEX MAP
    # =====================================================================
    print("\nAccumulating global dataset document frequencies and mapping cache registries...")
    image_counts = torch.zeros(vocab_size, dtype=torch.float32)
    
    # ─── NEW: MAP HOOK STORAGE FOR IMAGE FILE NAMES ──────────────────────
    image_id_to_vocab_indices = {}
    
    # We zip dataset and all_image_token_sets together to maintain image_id orientation
    for entry, entry_tokens in zip(dataset, all_image_token_sets):
        image_id = entry.get("image_id")
        
        # Translate tokens present inside current row to indices, dropping filtered items safely
        active_indices = list(set([concept_vocab[t] for t in entry_tokens if t in concept_vocab]))
        if active_indices:
            image_counts[active_indices] += 1
            
        # If the entry has a valid identifier string, commit its array target layout
        if image_id:
            image_id_to_vocab_indices[image_id] = sorted(active_indices)
            
    # Serialize image mapping database cache layout to disk
    with open(args.out_img_map, "w") as f:
        json.dump(image_id_to_vocab_indices, f, indent=4)
    print(f" -> Image ID to indices configuration translation map cached to: {args.out_img_map}")
    # ─────────────────────────────────────────────────────────────────────

    # Compute static SuperCLIP Image Log IDF representations
    print("Vectorizing final log document weights...")
    idf_image = torch.log(torch.tensor(total_images) / (1.0 + image_counts))
    
    # Strictly zero out padding target weight and clamp numerical rounding safe floors
    idf_image[0] = 0.0
    idf_image.clamp_(min=0.0)
    
    torch.save(idf_image, args.out_img_idf)
    print(f" -> Image-level IDF weights tensor array saved to: {args.out_img_idf}")
    print("\nPreprocessing workflow completed successfully!")

if __name__ == "__main__":
    main()