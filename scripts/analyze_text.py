import time
from matplotlib import colors
import pandas as pd
import os
import re
import numpy as np
from numpy import nan
import json
import json_repair
import itertools

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

from pydantic import BaseModel, Field
from typing import List, Set

import json
from pydantic import BaseModel, Field, AliasChoices, field_validator
from typing import List, Union, Dict, Any
from vllm import LLM, SamplingParams

from join_entities import cluster_entities

# 1. Define the Schema
class Attribute(BaseModel):
    key: str = Field(description="The attribute type, e.g., color, material, state")
    value: str = Field(description="The value of the attribute")

class SceneObject(BaseModel):
    id: Union[int, str]
    label: str = Field(validation_alias=AliasChoices('label', 'type', 'name'), description="The name of the object")
    attributes: Union[List[Union[Attribute, str, Dict[str, Any]]], Dict[str, Any]] = Field(default_factory=list)

    @field_validator('attributes', mode='before')
    @classmethod
    def parse_attributes(cls, v):
        if isinstance(v, dict):
            return [{"key": k, "value": str(val)} for k, val in v.items()]
        return v

class Relationship(BaseModel):
    subject_id: Union[int, str, None] = Field(default=None, validation_alias=AliasChoices('subject_id', 'subject'))
    predicate: str = Field(default="", validation_alias=AliasChoices('predicate', 'relation', 'type'), description="The spatial or functional relation")
    object_id: Union[int, str, None] = Field(default=None, validation_alias=AliasChoices('object_id', 'object', 'target'))

    @field_validator('subject_id', 'predicate', 'object_id', mode='before')
    @classmethod
    def parse_ids(cls, v):
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return v

class SceneGraph(BaseModel):
    objects: List[SceneObject] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

    @field_validator('objects', 'relationships', mode='before')
    @classmethod
    def filter_lists(cls, v):
        if isinstance(v, list):
            return [item for item in v if isinstance(item, dict)]
        return v


def extract_scene_graph(args, texts: Union[str, List[str]], vllm_url="http://localhost:8000/v1/chat/completions", llm=None):
    is_single = isinstance(texts, str)
    text_list = [texts] if is_single else texts

    # Convert Pydantic model to JSON Schema for vLLM
    json_schema = SceneGraph.model_json_schema()
    raw_outputs = []

    if args.is_vllm:
        import requests
        import concurrent.futures
        
        def fetch(text):
            headers = {"Content-Type": "application/json"}
            messages = [
                {"role": "system", "content": "You are a precise data extraction AI. Output ONLY valid JSON matching the schema."},
                {"role": "user", "content": f"Analyze the following scene description and extract a structured scene graph including all objects, their specific attributes, and the relationships between them.\n\nDescription: {text}"}
            ]
            data = {
                "model": args.model_id,
                "messages": messages,
                "max_tokens": int(args.max_len),
                "temperature": 0.0,
                "guided_json": json_schema,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "scene_graph",
                        "schema": json_schema
                    }
                }
            }
            for attempt in range(3):
                try:
                    response = requests.post(vllm_url, headers=headers, json=data)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Error querying vLLM via HTTP (attempt {attempt + 1}/3): {e}")
                    time.sleep(2)
            return None
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(text_list))) as executor:
            raw_outputs = list(executor.map(fetch, text_list))
    else:
        if llm is None:
            llm = LLM(model=args.model_id)

        try:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding = GuidedDecodingParams(json=json_schema, backend="outlines")
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1024, guided_decoding=guided_decoding)
        except (ImportError, TypeError):
            # Fallback if the vLLM version is missing GuidedDecodingParams
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)

        prompts = []
        for text in text_list:
            prompt = (
                "<bos><start_of_turn>user\n"
                "Analyze the following scene description and extract a structured scene graph "
                "including all objects, their specific attributes, and the relationships between them.\n\n"
                "You must output ONLY valid JSON. Use integer IDs for objects (e.g., 1, 2, 3).\n"
                "The JSON must strictly match this schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n\n"
                f"Description: {text}\n"
                "<end_of_turn>\n<start_of_turn>model\n"
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        raw_outputs = [out.outputs[0].text for out in outputs]
            
    final_graphs = []
    for raw_json in raw_outputs:
        if not raw_json:
            final_graphs.append(None)
            continue
            
        result_json = raw_json.strip()
        if result_json.startswith("```"):
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', result_json, re.DOTALL | re.IGNORECASE)
            if match:
                result_json = match.group(1).strip()
            else:
                # Fallback to strip opening tags if generation was cut off
                result_json = re.sub(r'^```(?:json)?\s*', '', result_json, flags=re.IGNORECASE)
        
        try:
            try:
                graph = SceneGraph.model_validate_json(result_json)
            except Exception as e:
                parsed_dict = json_repair.loads(result_json)
                if isinstance(parsed_dict, dict):
                    objs = []
                    rels = []
                    
                    def search_tree(node):
                        if isinstance(node, dict):
                            if "relationships" in node and isinstance(node["relationships"], list):
                                rels.extend([r for r in node["relationships"] if isinstance(r, dict)])
                            if "objects" in node and isinstance(node["objects"], list):
                                objs.extend([o for o in node["objects"] if isinstance(o, dict)])
                            
                            if ("label" in node or "object" in node) and "attributes" in node:
                                objs.append(node)
                                
                            for k, v in node.items():
                                if k not in ["relationships", "objects"]:
                                    search_tree(v)
                        elif isinstance(node, list):
                            for item in node:
                                search_tree(item)
                                
                    search_tree(parsed_dict)
                    
                    for i, o in enumerate(objs):
                        if "id" not in o:
                            o["id"] = o.get("label", o.get("object", o.get("type", f"obj_{i}")))
                        if "label" not in o:
                            o["label"] = o.get("object", o.get("type", "unknown"))
                            
                    graph = SceneGraph.model_validate({"objects": objs, "relationships": rels})
                else:
                    graph = SceneGraph.model_validate(parsed_dict)
            final_graphs.append(graph)
        except Exception as e:
            print(f"Failed to parse JSON: {e}\nRaw output:\n{result_json}")
            final_graphs.append(None)
            
    return final_graphs[0] if is_single else final_graphs

            
def analyze_texts_from_csv(args):  
            
    llm = None
    if not args.is_vllm:
        print(f"Loading local vLLM model {args.model_id}...")
        llm = LLM(model=args.model_id)
    else:
        print(f"Using vLLM via HTTP for {args.model_id}...")
    
    aggregated_json = args.out_file.replace('.json', '_intermediate.json')
    aggregated_results = []
    
    if args.load_aggregated and os.path.exists(aggregated_json):
        with open(aggregated_json, 'r') as f:
            aggregated_results = json.load(f)        
        print(f"Loaded aggregated results from {aggregated_json}")
        
    if  args.analyze:       

        df = pd.read_csv(args.csv_file)
        print(f"Loaded {len(df)} descriptions from {args.csv_file}")            
        
        # remove already analyzed files according to aggregated_results from df 
        # remove all rows from df where image_id == image_path
        if "image_path" in df.columns:
            df = df[~df["image_path"].isin([r["image_id"] for r in aggregated_results])]
        
        for i in range(0, len(df), args.batch_size):
            batch_df = df.iloc[i:i+args.batch_size]
            batch_descriptions = batch_df["description"].tolist()            
            batch_ids = batch_df["image_path"].tolist() if "image_path" in batch_df.columns else batch_df.index.tolist()
            
            graphs = extract_scene_graph(args, batch_descriptions, llm=llm)                        
            
            for img_id, graph in zip(batch_ids, graphs):
                aggregated_results.append({
                    "image_id": img_id,
                    "scene_graph": graph.model_dump() if graph else None
                })
            
            print(f"Processed batch {i//args.batch_size + 1}/{(len(df) + args.batch_size - 1) // args.batch_size}")
            
            #save results
            if (i // args.batch_size + 1) % 2 == 0:
                #break
                print(f"Saving intermediate results after batch {i//args.batch_size + 1}...")                
                intermediate_json = args.out_file.replace('.csv', '_intermediate.json')
                with open(intermediate_json, 'w') as f:
                    json.dump(aggregated_results, f, indent=4)
                print(f"Saved intermediate results to {intermediate_json}")
            
        # Save final results
        with open(aggregated_json, 'w') as f:
            json.dump(aggregated_results, f, indent=4)
        print(f"Saved aggregated results to {aggregated_json}")
        
        # go over aggregated_results and find all images with similar object labels using LLM
        # for example, on input:
        # image_id: database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@33@@@@@@pitch2_yaw10@.jpg
        # "label": "building"
        # "image_id": "database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@32@@@@@@pitch2_yaw9@.jpg",
        # "label": "high-rise building",
        # "image_id": "database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@30@@@@@@pitch2_yaw7@.jpg",
        # "label": "lamppost",
        # output should be : 
        # "entity": "building", "images": [
            # {"image_id": "database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@33@@@@@@pitch2_yaw10@.jpg", "label":"building"},
            # {"image_id": "database/@0585230.45@4477142.26@17@T@040.44056@-079.99503@000116@32@@@@@@pitch2_yaw9@.jpg", "label": "high-rise building"}
            # ]
        
    clustered_results = cluster_entities(aggregated_results, model_name=args.model_id, max_tokens=args.max_len)
    # dump clustered_results to json
    with open(args.out_file, 'w') as f:
        json.dump(clustered_results, f, indent=4)        
    
    print(f"Clustered results: {clustered_results}")            
            

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--csv_file", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")
    # parser.add_argument("--out_file", type=str, default="pitts30k_val_800_queries_objects.json")
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/gsv_cities_descriptions.csv")
    parser.add_argument("--out_file", type=str, default="gsv_cities_descriptions_queries_objects.json")    
    parser.add_argument("--max_len", type=str, default="3096", help="max number of words in the output description")
    parser.add_argument("--batch_size", type=int, default="1024", help="batch size for processing descriptions")    
    parser.add_argument("--load_aggregated", type=int, default="1", help="load aggregated")    
    parser.add_argument("--analyze", type=int, default="1", help="continue analyyzing files")    
    parser.add_argument("--is_vllm", type=int, default="1", help="is vllm")    
    #parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="type of model to apply")
    #parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-32B", help="type of model to apply")    
    parser.add_argument("--model_id", type=str, default="nvidia/Gemma-4-26B-A4B-NVFP4", help="type of model to apply")        
    parser.add_argument("--gpu", type=str, default="4", help="GPU to use (e.g. '0' or '0,1' for multiple GPUs)")
    
    args = parser.parse_args()           

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 

    analyze_texts_from_csv(args)    

    
    
    
    


        
       
