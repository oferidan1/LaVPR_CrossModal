import json
import json_repair
from collections import defaultdict
import requests
from pydantic import BaseModel, Field
import pandas as pd
import itertools
import math


# -------------------------------------------------------------------------
# Define Output Structure and Generate JSON Schema
# -------------------------------------------------------------------------
class LabelMapEntry(BaseModel):
    raw_label: str = Field(description="The exact label from the dataset.")
    canonical_entity: str = Field(description="The generic clustered parent name.")

class SemanticClusteringSchema(BaseModel):
    mappings: list[LabelMapEntry]

def cluster_entities(aggregated_results, model_name="nvidia/Gemma-4-26B-A4B-NVFP4", max_tokens=2048, vllm_url="http://localhost:8000/v1/chat/completions"):
    """
    Clusters fine-grained object labels from aggregated scene graphs into root canonical entities using an LLM.
    """

    # -------------------------------------------------------------------------
    # Step 1: Extract Unique Labels Locally
    # -------------------------------------------------------------------------
    label_occurrences = defaultdict(list)
    unique_labels = set()

    if isinstance(aggregated_results, dict) and "attributes" in aggregated_results:
        # Handle legacy aggregated dictionary format (analyze_text_old.py format)
        for attr in aggregated_results.get("attributes", []):
            lbl = attr.get("object_name")
            if lbl:
                unique_labels.add(lbl)
                for qid in attr.get("query_ids", []):
                    label_occurrences[lbl].append({
                        "image_id": qid,
                        "label": lbl
                    })
    elif isinstance(aggregated_results, list):
        # Handle new list of SceneGraph dictionaries format
        for item in aggregated_results:
            if not isinstance(item, dict):
                continue
            img_id = item.get("image_id", "unknown")
            scene_graph = item.get("scene_graph")
            objects = scene_graph.get("objects", []) if isinstance(scene_graph, dict) else []
            
            for obj in objects:
                lbl = obj.get("label")
                if lbl:
                    unique_labels.add(lbl)
                    label_occurrences[lbl].append({
                        "image_id": img_id,
                        "label": lbl
                    })

    # vLLM accepts raw JSON schema for guided decoding
    vllm_json_schema = SemanticClusteringSchema.model_json_schema()

    # -------------------------------------------------------------------------
    # Step 2: Send POST request to vLLM Server
    # -------------------------------------------------------------------------
    prompt = f"""
Group these fine-grained object labels under unified generic root entities (e.g., merge 'skyscraper' and 'high-rise building' into 'building'; merge 'trees' and 'tree' into 'tree'; merge 'lamppost' and 'utility pole' into 'pole').

    Labels:
    {list(unique_labels)}
    """

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You map raw object labels to clean parent categories. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "guided_json": vllm_json_schema,  # <-- Enforces structural JSON constraint inside vLLM
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "clustering",
                "schema": vllm_json_schema
            }
        }
    }

    print("Sending request to vLLM...")
    response = requests.post(vllm_url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()

    # -------------------------------------------------------------------------
    # Step 3: Parse Guided Output & Re-Reaggregate Data
    # -------------------------------------------------------------------------
    response_data = response.json()
    # vLLM populates the verified structured output string into the standard content field
    raw_content = response_data["choices"][0]["message"]["content"]

    # Clean up the response in case the model added markdown formatting
    cleaned_content = raw_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    cleaned_content = cleaned_content.strip()

    try:
        parsed_mappings = json.loads(cleaned_content).get("mappings", [])
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON natively. Model output was likely cut off due to max_tokens limit.")
        try:
            repaired = json_repair.loads(cleaned_content)
            parsed_mappings = repaired.get("mappings", []) if isinstance(repaired, dict) else []
            print(f"Successfully repaired JSON. Salvaged {len(parsed_mappings)} mappings.")
        except Exception:
            print(f"Failed to repair JSON. Raw output from model:\n{raw_content}")
            raise e

    # Map: raw_label -> canonical_entity
    label_to_entity_dict = {
        entry["raw_label"]: entry["canonical_entity"] 
        for entry in parsed_mappings 
        if isinstance(entry, dict) and "raw_label" in entry and "canonical_entity" in entry
    }

    # Group raw dataset items using the canonical clusters
    final_grouped_data = defaultdict(list)
    for raw_lbl, instances in label_occurrences.items():
        canonical_entity = label_to_entity_dict.get(raw_lbl, raw_lbl)
        for instance in instances:
            if instance not in final_grouped_data[canonical_entity]:
                final_grouped_data[canonical_entity].append(instance)

    # Format into final expected layout with distance calculations
    output_results = []
    for entity, instances in final_grouped_data.items():
        unique_images = {inst["image_id"] for inst in instances}
        frequency = len(unique_images)
        
        coords = []
        for img_id in unique_images:
            lat, lon = extract_coords(img_id)
            if lat is not None and lon is not None:
                coords.append((lat, lon))
        
        unique_coords = list(set(coords))
        num_places = len(unique_coords)
        
        max_dist = 0.0
        if num_places >= 2:
            for c1, c2 in itertools.combinations(unique_coords, 2):
                dist = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist > max_dist:
                    max_dist = dist
                    
        output_results.append({
            "entity": entity,
            "frequency": frequency,
            "num_places": num_places,
            "max_distance": max_dist,
            "images": instances
        })
    
    return output_results

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


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Test Execution
    # -------------------------------------------------------------------------
    test_results = [
        {
            "image_id": "database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@35@@@@@@pitch2_yaw12@.jpg",
            "scene_graph": {
                "objects": [
                    {"id": "skyscraper_1", "label": "skyscraper"},
                    {"id": "trees_1", "label": "trees"},
                    {"id": "building_ornate_1", "label": "building"},
                    {"id": "building_brick_1", "label": "building"},
                    {"id": "building_base_1", "label": "base section"}
                ]
            }
        },
        {
            "image_id": "database/@0595230.45@4477142.26@17@T@040.44056@-079.99503@000116@34@@@@@@pitch2_yaw11@.jpg",
            "scene_graph": {
                "objects": [
                    {"id": "obj_1", "label": "trees"},
                    {"id": "obj_2", "label": "skyscraper"},
                    {"id": "obj_3", "label": "ornate building"},
                    {"id": "obj_4", "label": "building"}
                ]
            }
        },
        {
            "image_id": "database/@0583230.45@45475142.26@17@T@040.44056@-079.99503@000116@32@@@@@@pitch2_yaw9@.jpg",
            "scene_graph": {
                "objects": [
                    {"id": "building_1", "label": "building"},
                    {"id": "pole_1", "label": "traffic light pole"},
                    {"id": "tree_1", "label": "tree"}
                ]
            }
        }
    ]

    VLLM_SERVER_URL = "http://localhost:8000/v1/chat/completions" # Adjust host/port if needed
    MODEL_NAME = "nvidia/Gemma-4-26B-A4B-NVFP4"                   # Replace with your active vLLM model

    final_output = cluster_entities(test_results, vllm_url=VLLM_SERVER_URL, model_name=MODEL_NAME)
    
    print("\nFinal Formatted Output:")
    print(json.dumps(final_output, indent=2))