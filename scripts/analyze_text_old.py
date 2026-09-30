import time
from matplotlib import colors
import pandas as pd
import os
import re
import numpy as np
from numpy import nan
import json
import itertools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from pydantic import BaseModel, Field
from typing import List, Set

class ObjectAttribute(BaseModel):
    object_name: str
    attributes: List[str]
    query_ids: Set[int] = Field(default_factory=set)

class ObjectRelation(BaseModel):
    subject: str
    relation: str
    object: str
    query_ids: Set[int] = Field(default_factory=set)

class ExtractionResult(BaseModel):
    attributes: List[ObjectAttribute]
    relations: List[ObjectRelation]

class UnifiedEntity(BaseModel):
    unified_name: str
    original_entities: List[str]
    query_ids: Set[int] = Field(default_factory=set)
    frequency: int = 0
    num_places: int = 0
    max_distance: float = 0.0

class UnifiedRelation(BaseModel):
    unified_name: str
    original_relations: List[str]
    query_ids: Set[int] = Field(default_factory=set)
    frequency: int = 0
    num_places: int = 0
    max_distance: float = 0.0
    
def analyze_batch(descriptions, query_ids, model, tokenizer, max_len, is_vllm=0, model_id=None, vllm_url="http://localhost:8000/v1/chat/completions"):
    messages_batch = []
    for desc in descriptions:
        messages = [
            {"role": "system", "content": "You are a precise data extraction AI. Extract object attributes and relations from the scene description. Output ONLY a valid JSON object with 'attributes' (list of {\"object_name\": string, \"attributes\": list of strings}) and 'relations' (list of {\"subject\": string, \"relation\": string, \"object\": string})."},
            {"role": "user", "content": "Description: A large red brick building is located next to a wide canal."},
            {"role": "assistant", "content": "{\"attributes\": [{\"object_name\": \"building\", \"attributes\": [\"large\", \"red\", \"brick\"]}, {\"object_name\": \"canal\", \"attributes\": [\"wide\"]}], \"relations\": [{\"subject\": \"building\", \"relation\": \"next to\", \"object\": \"canal\"}]}"},
            {"role": "user", "content": "Description: A tall tree casts a shadow on the parked car."},
            {"role": "assistant", "content": "{\"attributes\": [{\"object_name\": \"tree\", \"attributes\": [\"tall\"]}, {\"object_name\": \"car\", \"attributes\": [\"parked\"]}, {\"object_name\": \"shadow\", \"attributes\": []}], \"relations\": [{\"subject\": \"tree\", \"relation\": \"casts\", \"object\": \"shadow\"}, {\"subject\": \"shadow\", \"relation\": \"on\", \"object\": \"car\"}]}"},
            {"role": "user", "content": f"Description: {desc}"}
        ]
        messages_batch.append(messages)

    decoded_outputs = []
    if is_vllm:
        import requests
        import concurrent.futures
        
        def fetch_vllm(msgs):
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_id,
                "messages": msgs,
                "max_tokens": int(max_len),
                "temperature": 0.1,
            }
            try:
                response = requests.post(vllm_url, headers=headers, json=data)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error querying vLLM via HTTP: {e}")
                return ""
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, max(1, len(messages_batch)))) as executor:
            decoded_outputs = list(executor.map(fetch_vllm, messages_batch))
    else:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=int(max_len), temperature=0.1, do_sample=False)
            
        input_length = inputs.input_ids.shape[1]
        decoded_outputs = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
    
    batch_attributes = []
    batch_relations = []
    
    for i, output in enumerate(decoded_outputs):
        json_str = output.strip()
        try:
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                
            # Basic cleanup for common JSON formatting issues (like trailing commas)
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            data = json.loads(json_str)
            
            for attr in data.get('attributes', []):
                batch_attributes.append(ObjectAttribute(object_name=attr['object_name'], attributes=attr.get('attributes', []), query_ids={query_ids[i]}))
            for rel in data.get('relations', []):
                batch_relations.append(ObjectRelation(subject=rel['subject'], relation=rel['relation'], object=rel['object'], query_ids={query_ids[i]}))
        except Exception as e:
            print(f"Failed to parse JSON for query {query_ids[i]}: {e}\nRaw output:\n{json_str}\n")
            
    return batch_attributes, batch_relations


def unify_entities_optimized(entities, embed_model, model, tokenizer, max_len, similarity_threshold=0.7):
    if not entities:
        return []

    # 1. Deduplicate and Normalize
    unique_entities = list(set(e.strip() for e in entities))
    
    # 2. Semantic Pre-Clustering (The "Algo" Optimization)
    # We group entities that are already 70%+ similar mathematically
    embeddings = embed_model.encode(unique_entities)
    
    # AgglomerativeClustering finds groups without needing to know 'K' beforehand
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=1 - similarity_threshold, 
        metric='cosine', 
        linkage='average'
    ).fit(embeddings)
    
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        clusters.setdefault(int(label), []).append(unique_entities[idx])
        
    """
    clusters: A list of pre-clustered entities (from Stage 1 embeddings).
    Example: [['car', 'automobile'], ['tree', 'pine'], ...]
    """
    if not clusters: return []

    # 1. Prepare a compact prompt
    # We map clusters to IDs so the LLM doesn't have to repeat the entities
    cluster_map = {i: c for i, c in enumerate(clusters.values())}
    
    # We'll process clusters in batches to fill the GPU memory
    batch_size = 8 
    final_results = {}

    system_prompt = (
        "You are a linguistic refiner. I will provide numbered clusters of synonyms. "
        "For each ID, provide ONLY the best unified name. "
        "Format: JSON object {ID: 'name'}. No prose."
    )

    for i in range(0, len(clusters), batch_size):
        batch_slice = {k: cluster_map[k] for k in range(i, min(i + batch_size, len(clusters)))}
        
        prompt = f"Clusters: {json.dumps(batch_slice)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Use efficient chat template and move to device
        inputs = tokenizer.apply_chat_template(
            [messages], 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        with torch.no_grad():
            # KEY OPTIMIZATION: Set a strict max_new_tokens. 
            # Since output is just {ID: Name}, it should be very short.
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150, 
                temperature=0.1,
                do_sample=False, # Faster and more consistent
                use_cache=True   # Essential for speed
            )

        # Decode only the NEW tokens
        decoded = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        try:
            # Clean and parse the minimal JSON
            batch_result = json.loads(decoded[decoded.find('{'):decoded.rfind('}')+1])
            for cluster_id, unified_name in batch_result.items():
                final_results[int(cluster_id)] = {
                    "unified_name": unified_name,
                    "original_entities": cluster_map[int(cluster_id)]
                }
        except Exception:
            # Fallback for failed JSON parsing
            for k, v in batch_slice.items():
                final_results[k] = {"unified_name": v[0], "original_entities": v}

    return list(final_results.values())

def unify_entities_llm(entities: List[str], model, tokenizer, max_len, is_vllm=0, model_id=None, vllm_url="http://localhost:8000/v1/chat/completions"):
    unique_entities = list(set(entities))
    if not unique_entities:
        return []
        
    system_prompt = (
        "You are an AI that unifies synonyms. Given a list of object/relation names, group synonyms together. "
        "Output ONLY a valid JSON list of objects, each containing 'unified_name' and 'original_entities' (a list of exact strings from the input)."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Entities: [\"car\", \"automobile\", \"auto\", \"building\", \"structure\", \"tree\"]"},
        {"role": "assistant", "content": "[{\"unified_name\": \"car\", \"original_entities\": [\"car\", \"automobile\", \"auto\"]}, {\"unified_name\": \"building\", \"original_entities\": [\"building\", \"structure\"]}, {\"unified_name\": \"tree\", \"original_entities\": [\"tree\"]}]"},
        {"role": "user", "content": f"Entities: {json.dumps(unique_entities)}"}
    ]
    
    if is_vllm:
        import requests
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_id,
            "messages": messages,
            "max_tokens": int(max_len) * 4,
            "temperature": 0.1,
        }
        try:
            response = requests.post(vllm_url, headers=headers, json=data)
            response.raise_for_status()
            json_str = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"Error querying vLLM via HTTP: {e}")
            return [{"unified_name": e_str, "original_entities": [e_str]} for e_str in unique_entities]
    else:
        inputs = tokenizer.apply_chat_template(
            [messages], 
            add_generation_prompt=True, 
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=int(max_len) * 4, temperature=0.1)
            
        input_length = inputs.input_ids.shape[1]
        decoded = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
        json_str = decoded.strip()
    
    try:
        json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
        json_str = re.sub(r',\s*\}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to unify entities: {e}\nRaw output:\n{json_str}\n")
        return [{"unified_name": e_str, "original_entities": [e_str]} for e_str in unique_entities]

def extract_coords(image_path):
    try:
        if pd.isna(image_path):
            return None, None
        parts = str(image_path).split("@")
        if len(parts) >= 3:
            return float(parts[1]), float(parts[2])
    except:
        pass
    return None, None

def calc_distances(query_ids, coords_dict):
    valid_coords = [coords_dict[qid] for qid in query_ids if qid in coords_dict and coords_dict[qid][0] is not None]
    unique_coords = list(set(valid_coords))
    num_places = len(unique_coords)
    
    if num_places < 2:
        return num_places, 0.0
    
    max_dist = 0.0
    for c1, c2 in itertools.combinations(unique_coords, 2):
        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        if dist > max_dist:
            max_dist = dist
            
    return num_places, max_dist
            
def analyze_texts_from_csv(args):
    df = pd.read_csv(args.csv_file)
    print(f"Loaded {len(df)} descriptions from {args.csv_file}")
    
    coords_dict = {}
    if 'image_path' in df.columns:
        for i, row in df.iterrows():
            coords_dict[i] = extract_coords(row['image_path'])
            
    if not args.is_vllm:
        print(f"Loading model {args.model_id}...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=quantization_config, device_map="auto", attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)    
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    else:
        print(f"Using vLLM via HTTP for {args.model_id}...")
        model, tokenizer = None, None
    
    aggregated_attributes = {}
    aggregated_relations = {}    
    
    if args.load_aggregated:
        aggregated_json = args.out_file.replace('.csv', '_aggregated.json')
        with open(aggregated_json, 'r') as f:
            aggregated_results = json.load(f)
        
        for attr in aggregated_results["attributes"]:
            aggregated_attributes[attr["object_name"]] = ObjectAttribute(**attr)
        for rel in aggregated_results["relations"]:
            aggregated_relations[f"{rel['subject']} | {rel['relation']} | {rel['object']}"] = ObjectRelation(**rel)
            
        print(f"Loaded aggregated results from {aggregated_json}")
        
    else:            
        for i in range(0, len(df), args.batch_size):
            batch_df = df.iloc[i:i+args.batch_size]
            batch_descriptions = batch_df["description"].tolist()
            batch_ids = batch_df.index.tolist()
            
            attrs, rels = analyze_batch(batch_descriptions, batch_ids, model, tokenizer, args.max_len, is_vllm=args.is_vllm, model_id=args.model_id)
            
            for a in attrs:
                if a.object_name not in aggregated_attributes:
                    aggregated_attributes[a.object_name] = ObjectAttribute(object_name=a.object_name, attributes=[], query_ids=set())
                aggregated_attributes[a.object_name].attributes.extend([attr for attr in a.attributes if attr not in aggregated_attributes[a.object_name].attributes])
                aggregated_attributes[a.object_name].query_ids.update(a.query_ids)
                
            for r in rels:
                rel_key = f"{r.subject} | {r.relation} | {r.object}"
                if rel_key not in aggregated_relations:
                    aggregated_relations[rel_key] = ObjectRelation(subject=r.subject, relation=r.relation, object=r.object, query_ids=set())
                aggregated_relations[rel_key].query_ids.update(r.query_ids)
                
            print(f"Processed batch {i//args.batch_size + 1}/{(len(df) + args.batch_size - 1) // args.batch_size}")

            if (i // args.batch_size + 1) % 2 == 0:
                #break
                print(f"Saving intermediate results after batch {i//args.batch_size + 1}...")
                intermediate_results = {
                    "attributes": [{**a.dict(), "query_ids": list(a.query_ids)} for a in aggregated_attributes.values()],
                    "relations": [{**r.dict(), "query_ids": list(r.query_ids)} for r in aggregated_relations.values()]
                }
                intermediate_json = args.out_file.replace('.csv', '_intermediate.json')
                with open(intermediate_json, 'w') as f:
                    json.dump(intermediate_results, f, indent=4)
                print(f"Saved intermediate results to {intermediate_json}")
            
        #save aggregated_attributes and aggregated_relations to json file   
        aggregated_results = {
            "attributes": [{**a.dict(), "query_ids": list(a.query_ids)} for a in aggregated_attributes.values()],
            "relations": [{**r.dict(), "query_ids": list(r.query_ids)} for r in aggregated_relations.values()]
        }
        aggregated_json = args.out_file.replace('.csv', '_aggregated.json')
        with open(aggregated_json, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Saved aggregated results to {aggregated_json}")
    
    
    # Load a small, fast embedding model once
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Unifying object attributes...")
    unique_objects = list(aggregated_attributes.keys())
    unified_objects_data = unify_entities_llm(unique_objects, model, tokenizer, args.max_len, is_vllm=args.is_vllm, model_id=args.model_id)
    #unified_objects_data = unify_entities_optimized(unique_objects, embed_model, model, tokenizer, args.max_len)
    
    unified_attributes = []
    for group in unified_objects_data:
        unified_name = group.get('unified_name', '')
        originals = group.get('original_entities', [])
        qids = set()
        for orig in originals:
            if orig in aggregated_attributes:
                qids.update(aggregated_attributes[orig].query_ids)
        num_places, max_dist = calc_distances(qids, coords_dict)
        unified_attributes.append(UnifiedEntity(
            unified_name=unified_name, 
            original_entities=originals, 
            query_ids=qids,
            frequency=len(qids),
            num_places=num_places,
            max_distance=max_dist
        ))
        
    print("Unifying relations...")
    unique_relations_keys = list(aggregated_relations.keys())
    unified_relations_data = unify_entities_llm(unique_relations_keys, model, tokenizer, args.max_len, is_vllm=args.is_vllm, model_id=args.model_id)
    #unified_relations_data = unify_entities_optimized(unique_relations_keys, embed_model, model, tokenizer, args.max_len)
    
    unified_relations = []
    for group in unified_relations_data:
        unified_rel = group.get('unified_name', '')
        originals = group.get('original_entities', [])
        qids = set()
        for orig in originals:
            if orig in aggregated_relations:
                qids.update(aggregated_relations[orig].query_ids)
        num_places, max_dist = calc_distances(qids, coords_dict)
        unified_relations.append(UnifiedRelation(
            unified_name=unified_rel, 
            original_relations=originals, 
            query_ids=qids,
            frequency=len(qids),
            num_places=num_places,
            max_distance=max_dist
        ))
    
    unified_results = {
        "unified_attributes": [u.dict() for u in unified_attributes],
        "unified_relations": [u.dict() for u in unified_relations]
    }
    
    for attr in unified_results["unified_attributes"]:
        attr["query_ids"] = list(attr["query_ids"])
    for rel in unified_results["unified_relations"]:
        rel["query_ids"] = list(rel["query_ids"])
        
    out_json = args.out_file.replace('.csv', '.json')
    with open(out_json, 'w') as f:
        json.dump(unified_results, f, indent=4)
        
    print(f"Saved unified entities to {out_json}")
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--csv_file", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")
    # parser.add_argument("--out_file", type=str, default="pitts30k_val_800_queries_objects.csv")
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")
    parser.add_argument("--out_file", type=str, default="gsv_cities_descriptions_queries_objects.csv")    
    parser.add_argument("--max_len", type=str, default="1024", help="max number of words in the output description")
    parser.add_argument("--batch_size", type=int, default="100", help="batch size for processing descriptions")    
    parser.add_argument("--load_aggregated", type=int, default="0", help="load aggregated")    
    parser.add_argument("--is_vllm", type=int, default="1", help="is vllm")    
    #parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="type of model to apply")
    #parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-32B", help="type of model to apply")    
    parser.add_argument("--model_id", type=str, default="qwen-fast", help="type of model to apply")        
    parser.add_argument("--gpu", type=str, default="4", help="GPU to use (e.g. '0' or '0,1' for multiple GPUs)")
    
    args = parser.parse_args()           

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    analyze_texts_from_csv(args)

        
       
