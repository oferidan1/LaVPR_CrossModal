from xml.parsers.expat import model
import torch
import numpy as np
#add parent directory to path
import os
import sys
from pathlib import Path
import peft
from model.LaVPR import LaVPR
from transformers import AutoTokenizer, AutoModel
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel
from transformers import AutoModel, AutoProcessor
import open_clip



class LaVPR_wrapper():
    def __init__(self, args):
        self.model_name = args.model_name
        self.device = args.device
        self.embeds_dim = args.embeds_dim
        self.encoder_dim = args.embeds_dim
       
        if args.cross_modal==1:
            self.max_text_length = 77
            if 'blip' in self.model_name:
                self.vpr_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(self.model_name)
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            elif 'llm2clip' in self.model_name:
                from llm2clip.llm2clip import load_llm2clip
                self.vpr_encoder, self.vlm_encoder, self.processor = load_llm2clip()            
            elif 'clip' in self.model_name or 'siglip' in self.model_name:
                self.vpr_encoder = AutoModel.from_pretrained(self.model_name)
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            elif 'siglip' in self.model_name:
                 self.max_text_length = 64
            elif 'eva' in self.model_name.lower():
                self.vpr_encoder, _, self.processor = open_clip.create_model_and_transforms(self.model_name, pretrained='merged2b_s8b_b131k')#'EVA02-B-16'
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)                
        else:            
            self.single_encoder = LaVPR(   
                #---- Encoder
                model_name=args.model_name.lower(),
                train_vlm=args.train_vlm,
                embeds_dim=args.embeds_dim,           
                lora_all_linear=args.lora_all_linear,
                lora_target_modules=args.lora_target_modules,
                lora_r=args.lora_r,                  
                agg_type=args.agg_type,            
            )

            if args.lora_path is not None:
                print("loading lora from:", args.lora_path)
                self.single_encoder.vlm_encoder = peft.PeftModel.from_pretrained(self.single_encoder.vlm_encoder, args.lora_path, is_trainable=False)            
            else:            
                model_state_dict = torch.load(args.model_path)['state_dict']
                renamed_state_dict = {}
                for old_key, value in model_state_dict.items():
                    new_key = old_key.replace('text_encoder', 'vlm_encoder')
                    renamed_state_dict[new_key] = value
                self.single_encoder.load_state_dict(renamed_state_dict, strict=False)
            
            self.single_encoder = self.single_encoder.to(args.device)
            self.single_encoder.eval()            
            
            self.encoder_dim = self.embeds_dim
            if args.agg_type>0 and args.agg_type<4:
                self.encoder_dim = 1280 #8448           
            
            
        
    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # Sum of the attention mask
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
        # Mean Pooling
        return sum_embeddings / sum_mask
            
    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vpr_encoder(images)
            text_features = self.encode_text(texts)       
        return image_features, text_features
    
    def encode_single(self, images, texts):
        with torch.no_grad():
            features, text_features, _ , _ , _, _, _  = self.single_encoder(images, texts)
        return features, text_features
    
    def encode_image(self, images):
        if 'blip' in self.model_name:
            with torch.no_grad():
                image_features = self.vpr_encoder.encode_image(images)[:,0]
        elif 'llm2clip' in self.model_name:
            #image_features = self.vpr_encoder.encode_image(images)        
            image_features = self.vpr_encoder.get_image_features(images.to(self.vpr_encoder.dtype)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)   
        elif 'clip' in self.model_name or 'siglip' in self.model_name:
            with torch.no_grad():               
                image_features = self.vpr_encoder.get_image_features(pixel_values=images)
            image_features = image_features.pooler_output
        elif 'eva' in self.model_name.lower():
            with torch.no_grad():                 
                image_features = self.vpr_encoder.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)                
        else:
            with torch.no_grad():
                image_features = self.vpr_encoder(images)            
        return image_features
    
    def encode_text(self, texts):
        if 'blip' in self.model_name:            
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(self.device)
            with torch.no_grad():     
                text_features = self.vpr_encoder.encode_text(text_inputs)[:,0]
        elif 'llm2clip' in self.model_name:            
            text_features = self.vlm_encoder.encode(texts, convert_to_tensor=True).to(self.device)
            #text_features = self.vpr_encoder.encode_text(text_features)
            text_features = self.vpr_encoder.get_text_features(text_features.to(self.vpr_encoder.dtype)).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name or 'siglip' in self.model_name:            
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length).input_ids.to(self.device)
            with torch.no_grad():     
                text_features = self.vpr_encoder.get_text_features(input_ids=text_inputs)
            text_features = text_features.pooler_output  
        elif 'eva' in self.model_name.lower():
            text_tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.vpr_encoder.encode_text(text_tokens)    
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        elif 'bge' in self.text_model_name:                    
            text_tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():      
                model_output = self.vlm_encoder(**text_tokens)                        
                text_features = model_output[0][:, 0]            
            text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)          
        else:
            text_features = self.vlm_encoder.encode(texts, convert_to_tensor=True)
        return text_features


        
