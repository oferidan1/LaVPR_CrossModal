from matplotlib import colors
import pandas as pd
import os
import re
import numpy as np
from numpy import nan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

def augment_texts_from_csv_phi4(csv_file):    
    
    model_id = "microsoft/Phi-4-mini-instruct"

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # Optional: speeds up generation on supported GPUs
    )
    
    results = []
    # parse csv file
    df = pd.read_csv(csv_file)
    # go line by line and read columns
    for index, row in df.iterrows():
        image_path = row['image_path']
        description = row['description']    
        to_filter = row['manualy filter']
        if to_filter is nan:
            results.append([image_path, description])
        else:
            # call llm to remove to_filter text from description
            # 3. Format the Prompt
            messages = [
                {"role": "system", "content": "You are a precise editor. Your task is to rewrite paragraphs to remove specific topics while keeping the rest of the text natural and grammatically correct."},
                {"role": "user", "content": f"Rewrite the following paragraph to remove all mention of '{to_filter}':\n\n{description}"}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 4. Generate the Response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2000, 
                do_sample=False                
            )
            
            # Slice the output: [0, len(inputs.input_ids[0]):]
            # This ignores the first 'n' tokens (the prompt)
            generated_tokens = outputs[0][len(inputs.input_ids[0]):]

            # Decode only the new tokens
            new_description = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            results.append([image_path, new_description, description])
    
    # save results to updated 
    #csv_file_name = os.path.basename(csv_file)
    csv_file_name = 'amstertime_objects_cleaned.csv'
    df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    df2.to_csv(csv_file_name, index=False)
    
    
def remove_topic_v2(model, tokenizer, messages):
   
    # 3. Execution
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False # Greedy decoding for maximum consistency
    )

    #my output is batch     
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def update_descriptions_batch(model, tokenizer, messages_batch):
    """
    messages_batch: List of lists (e.g., [[{"role": "user", "content": "..."}], [...]])
    """
    # 1. CRITICAL: Set padding side to LEFT for generation
    tokenizer.padding_side = "left"
    # 1. Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply chat template to the batch
    # padding=True and return_dict=True are essential for batching
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        padding=True,
        return_dict=True
    ).to(model.device)

    # 3. Execution
    outputs = model.generate(
        **inputs, # Unpack input_ids and attention_mask
        max_new_tokens=256,
        temperature=0.1,
        do_sample=False
    )

    # 4. Extract only the NEW tokens for the entire batch
    # We slice from the length of the input sequence
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs[:, input_length:]

    # 5. Decode all responses in the batch
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)


def build_context(new_paragraph, max_len):
     # 1. Define the 1-shot example data
    example_input = (
        "A multi-story brick building with a tiled gabled roof, dormer window, and "
        "rectangular grid windows, featuring the text 'DE COST GAET VOOR DE BAET UYT.' "
        "and 'HANDELSINRICHTINGEN' on its facade; a boat docked in a canal; a dark brick "
        "bridge structure with a railing; a tall, ornate streetlight; a light-colored "
        "building with classical architectural elements; a second tall, ornate streetlight; "
        "a large, dark brick building with a prominent, narrow, pointed-roof tower and tall windows."
    )
    example_output = (
        #"brick building with 'DE COST GAET VOOR DE BAET UYT.' and 'HANDELSINRICHTINGEN' on facade, dormer, and grid windows. Canal boat docked alongside. Adjacent dark brick bridge with railing. Ornate streetlight stands between gabled structure and light classical building. Second streetlight positioned before a large, dark brick building."
        "brick building with 'DE COST GAET VOOR DE BAET UYT.' and 'HANDELSINRICHTINGEN' signs, canal, brick bridge , two ornate streetlights."
    )

    # 2. Build the message list with the 1-shot example
    messages = [        
        {"role": "system", 
        "content": f"You are an expert in Geospatial localization and Computer Vision. Your task is to compress long scene descriptions into highly discriminative Spatial Signatures for a text-to-image retrieval system. Condense this scene into a maximum of {max_len} words spatial signature, preserving unique landmarks, distinctive signs texts, and precise object-to-object positioning while removing all non-visual narrative fluff. make sure you have maximum of {max_len} words."
        },
        
        # THE 1-SHOT EXAMPLE
        {"role": "user", "content": example_input},
        {"role": "assistant", "content": example_output},
        
        # THE ACTUAL REQUEST
        {"role": "user", "content": new_paragraph}
    ]
    return messages

def augment_text_flip_description(description):
    # split the description into sentences, and flip the order of the sentences
    sentences = description.split(',')
    flipped_description = ','.join(sentences[::-1])
    return flipped_description

# The Complete Master List for VPR Augmentation
MASTER_COLORS = {
    "Greyscale": ["black", "charcoal", "onyx", "slate", "gray", "grey", "silver", 
                "metallic", "lead", "ash", "pewter", "stone", "smoke", "white", 
                "ivory", "cream", "off-white", "pearl", "alabaster"],

    "Reds": ["red", "crimson", "scarlet", "ruby", "maroon", "burgundy", "brick", 
            "terracotta", "rust", "sienna", "clay", "rose", "pink", "magenta", "coral"],

    "Blues": ["blue", "navy", "indigo", "cobalt", "azure", "sky-blue", "cyan", 
              "teal", "turquoise", "aquamarine", "sapphire", "steel-blue"],

    "Greens": ["green", "emerald", "forest-green", "olive", "moss", "sage", 
              "lime", "pine", "jade", "mint", "seaweed", "khaki"],

    "Yellows": ["yellow", "gold", "golden", "amber", "lemon", "saffron", "mustard", 
                "orange", "tangerine", "apricot", "peach", "bronze", "copper"],

    "Purples": ["purple", "violet", "lavender", "plum", "amethyst", "orchid", "lilac", "mauve"],

    "Browns": ["brown", "chocolate", "espresso", "tan", "beige", "sand", "ochre", 
                "tawny", "russet", "mahogany", "walnut", "cedar"]
}

# Flatten the MASTER_COLORS dictionary to get a list of all colors, sorted by length descending
ALL_COLORS = sorted([color for color_list in MASTER_COLORS.values() for color in color_list], key=len, reverse=True)

# Pre-compile the regex pattern to match whole words and avoid sequential overwriting bugs
COLOR_PATTERN = re.compile(r'\b(' + '|'.join(ALL_COLORS) + r')\b', re.IGNORECASE)

def augment_text_change_colors_description(description):
    def get_new_color(match):
        original_color = match.group(0)
        lower_color = original_color.lower()
        
        # Find which group the original color belongs to
        color_group = next((group for group in MASTER_COLORS.values() if lower_color in group), None)
        
        if not color_group:
            return original_color
            
        new_color = np.random.choice(color_group)
        while new_color == lower_color:
            new_color = np.random.choice(color_group)
            
        if original_color.isupper():
            return new_color.upper()
        elif original_color.istitle():
            return new_color.capitalize()
        return new_color

    return COLOR_PATTERN.sub(get_new_color, description)



def generate_positive_viewpoint_description(base_text):    
    view_change_list = [
        "top-down", "side-view", "ground-up", "low-angle", "distant", "close-up", "Interior-Outside", "Outside-Interior", "Wide-angle", "Narrow-angle", "Aerial", "Street-level"
    ]
    viewpoint = np.random.choice(view_change_list)
  
    prompt = f"""
    ### Task: Standalone Scene Reconstruction
    Create a description of this location from a {viewpoint} perspective.
    
    ### Constraints:
    - Do NOT mention that the view has changed (No "now", "appears", "reveals").
    - Do NOT mention the original text.
    - Describe the objects as they physically appear from the {viewpoint}.
    
    ### Source Data:
    "{base_text}"
    
    ### Output:
    (Provide only the final paragraph)
    """
    return prompt

def load_model(model_id):
    # 1. Define FP8 Quantization Config
    # Note: Ensure you have bitsandbytes installed
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True, # Standard 8-bit
        llm_int8_enable_fp32_cpu_offload=True # Helpful for 70B models if VRAM is tight
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 2. Load model with Flash Attention and Quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, # Compute dtype should stay bfloat16
        device_map="auto",
        attn_implementation="flash_attention_2" # Enforce Flash Attention 2
    )
    
    return model, tokenizer
    

def augment_texts_from_csv(args):
    results = []  
     # parse csv file
    df = pd.read_csv(args.csv_file)
    i = 0    
    # open csv_file_name and make short list of all files not in csv_file
    if os.path.exists(args.out_file):
        print(f"Output file {args.out_file} already exists. Loading existing descriptions to avoid duplicates.")
        df2 = pd.read_csv(args.out_file)
        files_in_csv = df2['image_path'].tolist()
        description_in_csv = df2['description'].tolist()
        original_description_in_csv = df2['original_description'].tolist()
        results = list(zip(files_in_csv, description_in_csv, original_description_in_csv))        
        df = df[~df['image_path'].isin(files_in_csv)]
        
    if args.augment_type == "llm":
        print("Using LLM for augmentation")
        model, tokenizer = load_model(args.model_id)
    else:        
        print("Using rule based augmentation")
    
    batch_llm_items = []
    batch_images = []
    orig_descriptions = []    
    
    # Defining the system role
    system_role = (
        "You are a spatial reasoning engine. Your sole purpose is to "
        "generate synthetic variations of scene descriptions for a "
        "Visual Place Recognition (VPR) dataset. Follow the user's "
        "transformation rules strictly and do not provide conversational filler."
    )
    
    flip_descriptions = []
    change_colors_descriptions = []
    
    # go line by line and read columns
    for index, row in df.iterrows():
        image_path = row['image_path']
        description = row['description']    

        if args.augment_type == "llm":
            prompt = generate_positive_viewpoint_description(description)
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": f".'Description:\n\n{prompt}"}
            ]
            batch_llm_items.append(messages)
        if args.augment_type == "rule":
            flip_descriptions.append(augment_text_flip_description(description))
            change_colors_descriptions.append(augment_text_change_colors_description(description))
        
        batch_images.append(image_path)
        orig_descriptions.append(description)
        
        if args.augment_type == "llm" and len(batch_llm_items) >= args.batch_size:
            new_descriptions = update_descriptions_batch(model, tokenizer, batch_llm_items)
            for img, new_desc, orig_desc in zip(batch_images, new_descriptions, orig_descriptions):
                results.append([img, orig_desc, new_desc.strip()])        
            
            batch_llm_items, batch_images, orig_descriptions = [], [], []

            df2 = pd.DataFrame(results, columns=['image_path', 'description', 'view change'])
            df2.to_csv(args.out_file, index=False)
            
        elif len(flip_descriptions) >= args.batch_size:
            for img, orig_desc, flip_desc, color_desc in zip(batch_images, orig_descriptions, flip_descriptions, change_colors_descriptions):
                results.append([img, orig_desc, flip_desc.strip(), color_desc.strip()])        
            
            batch_images, orig_descriptions, flip_descriptions, change_colors_descriptions = [], [], [], []             
            
            df2 = pd.DataFrame(results, columns=['image_path', 'description', 'flip', 'change_color'])
            df2.to_csv(args.out_file, index=False)
    
            
        
    # if len(batch_llm_items)>0:
    #     new_descriptions = update_descriptions_batch(model, tokenizer, batch_llm_items)
    #     for img, new_desc, orig_desc in zip(batch_images, new_descriptions, batch_descriptions):
    #         results.append([img, new_desc.strip(), orig_desc])
            
    # save results to updated 
    # df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    # df2.to_csv(csv_file_out, index=False)

    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_file", type=str, default="datasets/descriptions/pitts30k_val_800_queries.csv")
    parser.add_argument("--out_file", type=str, default="pitts30k_val_800_queries_augmented.csv")
    parser.add_argument("--max_len", type=str, default="256", help="max number of words in the output description")
    parser.add_argument("--batch_size", type=int, default="100", help="batch size for processing descriptions")
    parser.add_argument("--augment_type", type=str, default="rule", help="llm, rule")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="type of model to apply")
    
    args = parser.parse_args()           
 
    #model_id = "meta-llama/Llama-3.3-70B-Instruct"
    #model_id = "microsoft/Phi-4-mini-instruct"

    augment_texts_from_csv(args)

        
       
