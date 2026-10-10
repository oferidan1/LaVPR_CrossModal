import torch
import numpy as np
import os
import sys
from pathlib import Path
import peft
from model.LaVPR import LaVPR
from model.LaVPR_cross import LaVPR_cross
from model.LaVPR_reranker import LaVPR_reranker
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel
import open_clip


class LaVPR_wrapper():
    def __init__(self, args):
        self.model_name  = args.model_name.lower()
        self.device      = args.device
        self.embeds_dim  = args.embeds_dim
        self.encoder_dim = args.embeds_dim
        self.reranker    = args.reranker
        self.cross_modal = args.cross_modal

        if args.cross_modal == 3:
            # ---- LaVPR_cross: load from checkpoint, put in eval mode
            self.vlm_encoder = LaVPR_cross.load_from_checkpoint(
                args.model_path,
                map_location=args.device,
                strict=True,
            )
            self.vlm_encoder = self.vlm_encoder.eval().to(args.device)

        elif args.cross_modal <= 1:
            self.max_text_length = 77
            if 'blip' in self.model_name:
                self.vpr_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(self.model_name)
                self.processor   = BlipProcessor.from_pretrained(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            elif 'llm2clip' in self.model_name:
                from llm2clip.llm2clip import load_llm2clip
                self.vpr_encoder, self.vlm_encoder, self.processor = load_llm2clip()
            elif 'clip' in self.model_name or 'siglip' in self.model_name:
                self.vpr_encoder = AutoModel.from_pretrained(self.model_name)
                self.processor   = AutoProcessor.from_pretrained(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            elif 'siglip' in self.model_name:
                self.max_text_length = 64
            elif 'eva' in self.model_name.lower():
                self.vpr_encoder, _, self.processor = open_clip.create_model_and_transforms(
                    self.model_name, pretrained='merged2b_s8b_b131k')
                self.tokenizer   = open_clip.get_tokenizer(self.model_name)
                self.vpr_encoder = self.vpr_encoder.eval().to(args.device)
            elif 'bge' in self.model_name or 'all-minilm' in self.model_name:
                self.tokenizer   = AutoTokenizer.from_pretrained(self.model_name)
                self.vlm_encoder = AutoModel.from_pretrained(
                    self.model_name, attn_implementation="sdpa").to(args.device)

        else:
            if args.reranker:
                self.single_encoder = LaVPR_reranker(
                    model_name=args.model_name.lower(),
                    train_vlm=args.train_vlm,
                    embeds_dim=args.embeds_dim,
                )
            else:
                self.single_encoder = LaVPR(
                    model_name=args.model_name,
                    train_vlm=args.train_vlm,
                    embeds_dim=args.embeds_dim,
                    lora_all_linear=args.lora_all_linear,
                    lora_target_modules=args.lora_target_modules,
                    lora_r=args.lora_r,
                    filip=args.reranker_filip or args.filip_retrieval,
                    is_image=args.is_image,
                    is_llp=args.is_llp,
                )

            if args.lora_path is not None:
                print("loading lora from:", args.lora_path)
                self.single_encoder.vlm_encoder = peft.PeftModel.from_pretrained(
                    self.single_encoder.vlm_encoder, args.lora_path, is_trainable=False)
            else:
                model_state_dict = torch.load(args.model_path)['state_dict']
                renamed_state_dict = {
                    k.replace('text_encoder', 'vlm_encoder'): v
                    for k, v in model_state_dict.items()
                }
                self.single_encoder.load_state_dict(renamed_state_dict, strict=False)

            self.single_encoder = self.single_encoder.to(args.device).eval()
            self.encoder_dim = self.embeds_dim
            if 0 < args.agg_type < 4:
                self.encoder_dim = 1280

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings    = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings      = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask            = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    # Encode image
    # ------------------------------------------------------------------

    def encode_image(self, images):
        if hasattr(self, 'vlm_encoder') and isinstance(self.vlm_encoder, LaVPR_cross):
            with torch.no_grad():
                img_embeds, _ = self.vlm_encoder.encode_image(images)
            return img_embeds

        if 'blip' in self.model_name:
            with torch.no_grad():
                return self.vpr_encoder.encode_image(images)[:, 0]
        elif 'llm2clip' in self.model_name:
            image_features = self.vpr_encoder.get_image_features(
                images.to(self.vpr_encoder.dtype)).float()
            return image_features / image_features.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name or 'siglip' in self.model_name:
            with torch.no_grad():
                vision_outputs = self.vpr_encoder.vision_model(
                    pixel_values=images, output_hidden_states=True)
                image_features = self.vpr_encoder.visual_projection(
                    vision_outputs.pooler_output)
            return image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        elif 'eva' in self.model_name.lower():
            with torch.no_grad():
                image_features = self.vpr_encoder.encode_image(images)
            return image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                return self.vpr_encoder(images)

    # ------------------------------------------------------------------
    # Encode text
    # ------------------------------------------------------------------

    def encode_text(self, texts):
        if hasattr(self, 'vlm_encoder') and isinstance(self.vlm_encoder, LaVPR_cross):
            with torch.no_grad():
                text_embeds, _, _, _ = self.vlm_encoder.encode_text(texts)
            return text_embeds

        if 'blip' in self.model_name:
            text_inputs = self.processor(
                text=texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512).input_ids.to(self.device)
            with torch.no_grad():
                return self.vpr_encoder.encode_text(text_inputs)[:, 0]
        elif 'llm2clip' in self.model_name:
            text_features = self.vlm_encoder.encode(texts, convert_to_tensor=True).to(self.device)
            text_features = self.vpr_encoder.get_text_features(
                text_features.to(self.vpr_encoder.dtype)).float()
            return text_features / text_features.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name or 'siglip' in self.model_name:
            text_inputs  = self.processor(text=texts, return_tensors="pt", padding=True,
                                          truncation=True, max_length=self.max_text_length)
            text_tokens  = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                text_features = self.vpr_encoder.text_model(
                    input_ids=text_tokens, attention_mask=attention_mask,
                    output_hidden_states=True)
            text_features = self.vpr_encoder.text_projection(text_features.pooler_output)
            return text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        elif 'eva' in self.model_name.lower():
            text_tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.vpr_encoder.encode_text(text_tokens)
            return text_features / text_features.norm(dim=-1, keepdim=True)
        elif 'bge' in self.model_name or 'all-minilm' in self.model_name:
            text_tokens = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output  = self.vlm_encoder(**text_tokens)
                text_features = model_output[0][:, 0]
            return torch.nn.functional.normalize(text_features, p=2, dim=1)
        else:
            return self.vlm_encoder.encode(texts, convert_to_tensor=True)

    # ------------------------------------------------------------------
    # Encode dual  (non-cross-modal path)
    # ------------------------------------------------------------------

    def encode_dual(self, images, texts):
        with torch.no_grad():
            image_features = self.vpr_encoder(images)
            text_features  = self.encode_text(texts)
        return image_features, text_features

    # ------------------------------------------------------------------
    # Encode single  (unimodal LaVPR path, or LaVPR_cross when cross_modal==3)
    # ------------------------------------------------------------------

    def encode_single(self, images, texts):
        if self.cross_modal == 3:
            with torch.no_grad():
                (img_embeds, text_embeds, _,
                 img_local, text_local, attention_mask, text_tokens) = self.vlm_encoder(images, texts)
            return img_embeds, text_embeds, img_local, text_local, text_tokens, attention_mask

        img_local, text_local, text_tokens, attention_mask = None, None, None, None
        with torch.no_grad():
            if self.reranker:
                score_matrix, img_embeds, text_embeds, img_local, text_local = \
                    self.single_encoder(images, texts, return_embeddings=True)
            else:
                img_embeds, text_embeds, _, _, _, _, img_local, text_local, \
                    text_tokens, attention_mask, _ = self.single_encoder(images, texts)
        return img_embeds, text_embeds, img_local, text_local, text_tokens, attention_mask