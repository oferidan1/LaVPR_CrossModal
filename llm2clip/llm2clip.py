import os
import torch

from transformers import AutoModel, AutoConfig, AutoTokenizer
from .llm2vec import LLM2Vec
from PIL import Image

from transformers import CLIPImageProcessor

class LLM2VecWrapper(LLM2Vec):
    def prepare_for_tokenization(self, text):
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + text.strip()
            + "<|eot_id|>"
        )
        return  text


def load_llm2clip_eva(model_name='EVA02-CLIP-B-16'):
    from .eva_clip import create_model_and_transforms
    model, _, preprocess_val  = create_model_and_transforms(model_name, force_custom_clip=True)
    ckpt = torch.load('llm2clip/LLM2CLIP-EVA02-B-16/LLM2CLIP-EVA02-B-16.pt')    
    model.load_state_dict(ckpt)
    model = model.cuda().eval()

    # Disable xformers memory efficient attention (xattn) since the current xformers installation lacks CUDA support
    for m in model.modules():
        if hasattr(m, 'xattn'):
            m.xattn = False

    llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
    config = AutoConfig.from_pretrained(
        llm_model_name, trust_remote_code=True
    )
    llm_model = AutoModel.from_pretrained(
        llm_model_name, 
        dtype=torch.bfloat16, 
        config=config, 
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
    l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)
    return model, l2v, preprocess_val 

def load_llm2clip(model_name='llm2clip/LLM2CLIP-Openai-B-16'):
    model_name_or_path = model_name # or /path/to/local/LLM2CLIP-Openai-B-16

    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    model = AutoModel.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2").to('cuda').eval()

    llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
    #llm_model_name  = 'microsoft/LLM2CLIP-Llama-3.2-1B-Instruct-CC-Finetuned'
    config = AutoConfig.from_pretrained(
        llm_model_name, trust_remote_code=True, attn_implementation="flash_attention_2"
    )
    llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    #llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
    if '1B' in llm_model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        l2v = LLM2VecWrapper(llm_model, tokenizer, pooling_mode="mean", max_length=512, skip_instruction=True)
    else:
        l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

    # Pass the llm hidden size to the model
    model.config.text_config.hidden_size = llm_model.config.hidden_size
    
    return model, l2v, processor  

def encode_image_eva(image_path, model, processor):
    image = processor(Image.open(image_path)).cuda().unsqueeze(dim=0)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)        
    return image_features

def encode_text_eva(captions, model, l2v):
    text_features = l2v.encode(captions, convert_to_tensor=True).to('cuda')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        text_features = model.encode_text(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)        
    return text_features

def encode_image_clip(image_path, model, processor):
    image = Image.open(image_path)
    input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')
    with torch.no_grad(), torch.amp.autocast('cuda'):         
        image_features = model.get_image_features(input_pixels)
        image_features /= image_features.norm(dim=-1, keepdim=True)    
    return image_features

def encode_text_clip(captions, model, l2v):
    text_features = l2v.encode(captions, convert_to_tensor=True).to('cuda')    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        text_features = model.get_text_features(text_features)        
        text_features /= text_features.norm(dim=-1, keepdim=True)        
    return text_features

def main():
    
    #model_name='EVA02-CLIP-B-16'
    model_name='microsoft/LLM2CLIP-Openai-B-16'
    image_path = "llm2clip/LLM2CLIP-Openai-B-16/CLIP.png"
    captions = ["a diagram", "a dog", "a cat"]
    if 'EVA' in model_name:
        model, l2v, processor = load_llm2clip_eva(model_name)
        image_features = encode_image_eva(image_path, model, processor)
        text_features = encode_text_eva(captions, model, l2v)        
    else:
        model, l2v, processor = load_llm2clip(model_name)    
        image_features = encode_image_clip(image_path, model, processor)
        text_features = encode_text_clip(captions, model, l2v, True)  

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)


if __name__ == "__main__":
    main()
