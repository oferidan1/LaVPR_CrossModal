import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import os
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipModel, AutoProcessor
import open_clip
import utils
import numpy as np
from model.tokens_classify_loss import TokensClassificationLoss


class SmoothListwiseRankMarginLoss(nn.Module):
    """
    Smooth, logsumexp-based listwise margin loss.
    Penalizes all negatives that fall within margin distance of any positive,
    preventing single-outlier gradient starvation while maintaining multi-positive bounds.
    """
    def __init__(self, margin=0.2, temperature=10.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, score_matrix, pos_counts):
        B, total_eval = score_matrix.shape
        losses = []
        
        for i in range(B):
            n_pos = pos_counts[i].item()
            pos_scores = score_matrix[i, :n_pos]  # [n_pos]
            neg_scores = score_matrix[i, n_pos:]  # [n_neg]
            
            # Differentiable Soft-Min for Positives and Soft-Max for Negatives
            soft_min_pos = -torch.logsumexp(-self.temperature * pos_scores, dim=0) / self.temperature
            soft_max_neg = torch.logsumexp(self.temperature * neg_scores, dim=0) / self.temperature
            
            # Smooth margin violation check
            violation = self.margin - (soft_min_pos - soft_max_neg)
            row_loss = F.softplus(self.temperature * violation) / self.temperature
            losses.append(row_loss)
            
        return torch.stack(losses).mean()


class DecoupledCrossAttnClassifier(nn.Module):
    """
    Cross-Encoder head that processes detached backbone representations.
    Uses dedicated projection layers so local alignment logic never
    distorts the global retrieval space.
    """
    def __init__(self, embeds_dim, text_dim=512, img_dim=768, num_heads=8):
        super().__init__()
        # Dedicated local projections independent of global retrieval heads
        self.local_img_proj = nn.Linear(img_dim, embeds_dim)
        self.local_text_proj = nn.Linear(text_dim, embeds_dim)
        
        self.ln_text = nn.LayerNorm(embeds_dim)
        self.ln_img = nn.LayerNorm(embeds_dim)
        self.ln_post = nn.LayerNorm(embeds_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embeds_dim, num_heads=num_heads, batch_first=True
        )
        
        # Learnable residual gate initialized to 0.1 for gentle startup
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        self.local_scale = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, img_local, text_local, img_global=None, text_global=None, text_attention_mask=None, force_local=False, return_both=False):
        t_features = self.ln_text(self.local_text_proj(text_local))  
        i_features = self.ln_img(self.local_img_proj(img_local))    
        
        fused, _ = self.cross_attn(
            query=t_features, 
            key=i_features, 
            value=i_features
        )
        
        # Gated residual connection
        attn_logits = self.ln_post(t_features + self.alpha * fused)   
        
        if text_attention_mask is not None:
            if text_attention_mask.dtype != torch.bool:
                text_attention_mask = (text_attention_mask == 0)
                
            mask_expanded = text_attention_mask.unsqueeze(-1).expand_as(attn_logits)
            attn_logits = attn_logits.masked_fill(mask_expanded, -1e9)
        
        # Mean sequence pooling
        pooled = torch.mean(attn_logits, dim=1) # [B, embeds_dim]
        latent = F.normalize(pooled, p=2, dim=-1)
        
        # Cosine similarity score relative to normalized text query anchor
        text_query_ref = F.normalize(t_features.mean(dim=1), p=2, dim=-1)
        local_score = torch.sum(latent * text_query_ref, dim=-1) # [B]
        
        if img_global is not None and text_global is not None and not force_local:            
            global_sim = F.cosine_similarity(text_global, img_global, dim=-1)                
            final_score = global_sim + self.local_scale * local_score
        else:
            final_score = local_score

        if return_both:
            return latent, final_score
            
        return final_score


class LaVPR_reranker(pl.LightningModule):
    def __init__(self,   
                 lr=0.0001, 
                 optimizer='adamw',
                 weight_decay=1e-2,
                 momentum=0.9,
                 warmpup_steps=500,
                 milestones=[4, 6],
                 lr_mult=0.5,
                 epochs=10,
                 faiss_gpu=False,
                 model_name='Salesforce/blip-itm-base-coco',
                 embeds_dim=256,
                 freeze_vlm=True,
                 train_vlm=False,
                 pos_loss=0,
                 neg_loss=0,
                 num_mined_negatives=8,
                 max_img_tokens=197,
                 loss_name='MultiSimilarityLoss', 
                 miner_name='MultiSimilarityMiner', 
                 miner_margin=0.2,
                 tokens_idf_loss=0.0,
                 tokens_idf_file=None,
                 idf_grad_scale=0.05,
                 detach=1,
                 ):
        super().__init__()       
        
        self.model_name = model_name
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.epochs = epochs
        self.faiss_gpu = faiss_gpu
        self.num_mined_negatives = num_mined_negatives
        
        # Dual loss functions
        self.loss_fn = utils.get_loss(loss_name)
        self.miner_name = miner_name
        self.miner_margin = miner_margin         
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.listwise_loss_fn = SmoothListwiseRankMarginLoss(margin=0.2, temperature=10.0)
        
        self.save_hyperparameters()
        self.batch_acc = [] 
        self.embeds_dim = embeds_dim        
        self.train_vlm = train_vlm
        self.pos_loss = pos_loss
        self.neg_loss = neg_loss  
        self.tokens_idf_loss = tokens_idf_loss
        self.tokens_idf_file = tokens_idf_file        
        vocab_size = 49408    
        self.detach = detach
        
        if self.tokens_idf_loss:
            self.tokens_classification_loss = TokensClassificationLoss(vision_dim=768, vocab_size=vocab_size, idf_path=self.tokens_idf_file, grad_scale=idf_grad_scale)
        
        self.img_dim = 768
        self.text_dim = 512
        self.cross_attn_classifier = DecoupledCrossAttnClassifier(
            embeds_dim=embeds_dim, text_dim=self.text_dim, img_dim=self.img_dim
        )
        self.apply(self._init_weights)
        
        if 'blip' in model_name:
            self.vlm_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(model_name)
            self.processor = BlipProcessor.from_pretrained(model_name)
        elif 'llm2clip' in model_name:
            from llm2clip.llm2clip import load_llm2clip
            self.vlm_encoder, self.llm_encoder, self.processor = load_llm2clip()
            self.max_text_length = 512
        elif 'clip' in model_name or 'siglip' in model_name:
            self.max_text_length = 77
            if 'siglip' in model_name:
                self.max_text_length = 64
            self.vlm_encoder = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        elif 'eva' in model_name:
            self.vlm_encoder, _, self.processor = open_clip.create_model_and_transforms(model_name.upper(), pretrained='merged2b_s8b_b131k')
            self.tokenizer = open_clip.get_tokenizer(model_name)                        
                                
        if freeze_vlm:
            for param in self.vlm_encoder.parameters():
                param.requires_grad = False                               
            self.vlm_encoder.eval()

        self.queue_size = 4140 
        self.register_buffer("image_global_queue", torch.zeros(self.queue_size, embeds_dim))
        self.register_buffer("image_local_queue", torch.zeros(self.queue_size, max_img_tokens, self.img_dim))
        self.register_buffer("label_queue", torch.ones(self.queue_size, dtype=torch.long) * -1)
        self.queue_ptr = 0
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_embeds, img_local, labels):
        batch_size = img_embeds.shape[0]
        if self.queue_ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - self.queue_ptr
            
        if batch_size <= 0:
            self.queue_ptr = 0
            batch_size = img_embeds.shape[0]

        self.image_global_queue[self.queue_ptr:self.queue_ptr + batch_size] = img_embeds[:batch_size]
        self.image_local_queue[self.queue_ptr:self.queue_ptr + batch_size] = img_local[:batch_size]
        self.label_queue[self.queue_ptr:self.queue_ptr + batch_size] = labels[:batch_size]
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size

    def encode_image(self, img):
        img_embeds, img_local, img_all_layers, img_local_unproj = None, None, None, None
        if 'blip' in self.model_name:
            img_local = self.vlm_encoder.encode_image(img)            
            img_embeds = img_local[:,0]
        elif 'llm2clip' in self.model_name:
            img_output = self.vlm_encoder.vision_model(pixel_values=img.to(self.vlm_encoder.dtype), output_hidden_states=True)
            img_local = self.vlm_encoder.visual_projection(img_output.last_hidden_state)
            img_embeds = self.vlm_encoder.visual_projection(img_output.pooler_output)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:            
            vision_outputs = self.vlm_encoder.vision_model(pixel_values=img, output_hidden_states=True)
            img_local_unproj = vision_outputs.last_hidden_state
            img_local = self.vlm_encoder.visual_projection(img_local_unproj)
            pooled_output = vision_outputs.pooler_output
            img_embeds = self.vlm_encoder.visual_projection(pooled_output)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        elif 'siglip' in self.model_name:
            img_output = self.vlm_encoder.get_image_features(pixel_values=img)
            img_local = img_output.last_hidden_state            
            img_embeds = img_output.pooler_output
        elif 'eva' in self.model_name:            
            img_local_unproj = self.vlm_encoder.visual.trunk.forward_features(img)
            if isinstance(img_local_unproj, dict):
                img_local_unproj = img_local_unproj['x']
            img_local = self.vlm_encoder.visual.trunk.head(img_local_unproj)
            img_embeds = img_local[:, 0]
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        
        return img_embeds, img_local_unproj, img_all_layers, img_local_unproj
    
    def encode_text(self, text):
        text_embeds, attention_mask, text_local, text_all_layers = None, None, None, None

        if 'blip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokens = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_local = self.vlm_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)    
            text_embeds = text_local[:, 0]        
        elif 'llm2clip' in self.model_name:
            text_tokens = self.llm_encoder.encode(text, convert_to_tensor=True).to(self.device)
            text_embeds = self.vlm_encoder.get_text_features(text_tokens.to(self.vlm_encoder.dtype)).float()
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:        
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.device)
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_outputs = self.vlm_encoder.text_model(input_ids=text_tokens, attention_mask=attention_mask, output_hidden_states=True)                        
            text_local = self.vlm_encoder.text_projection(text_outputs.last_hidden_state)  
            pooled_text = text_outputs.pooler_output                                     
            text_embeds = self.vlm_encoder.text_projection(pooled_text)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        elif 'siglip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.device)
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_output = self.vlm_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            text_local = text_output.last_hidden_state
            text_embeds = text_output.pooler_output                
        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.device)            
            attention_mask = (text_tokens == 0)
            
            x = self.vlm_encoder.text.token_embedding(text_tokens)
            x = x + self.vlm_encoder.text.positional_embedding
            _, intermediates = self.vlm_encoder.text.transformer.forward_intermediates(
                x=x,                 
                attn_mask=self.vlm_encoder.text.attn_mask,
                indices=[-1]
            )
            text_local = intermediates[-1] 
            text_local = self.vlm_encoder.text.ln_final(text_local)
            
            eot_indices = text_tokens.argmax(dim=-1)
            eot_features = text_local[torch.arange(text_local.shape[0]), eot_indices]
            
            if hasattr(self.vlm_encoder.text, 'text_projection') and self.vlm_encoder.text.text_projection is not None:
                text_embeds = eot_features @ self.vlm_encoder.text.text_projection
            #    text_local = text_local @ self.vlm_encoder.text.text_projection
            else:
                text_embeds = eot_features
                
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)   
        
        return text_embeds, text_local, attention_mask, text_tokens, text_all_layers

    def forward(self, img, text, flip_desc=None, labels=None, return_embeddings=False):
        img_embeds, img_local, img_all_layers, img_local_unproj = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens, text_all_layers = self.encode_text(text)             
                
        B, Lt, D_text = text_local.shape
        Li, D_img = img_local.shape[1], img_local.shape[2]

        if labels is None or not self.training:
            text_pairs = text_local[:, None].expand(B, B, Lt, D_text).reshape(B * B, Lt, D_text)
            img_pairs = img_local[None].expand(B, B, Li, D_img).reshape(B * B, Li, D_img)            
            text_global_pairs = text_embeds[:, None].expand(B, B, text_embeds.shape[-1]).reshape(B * B, text_embeds.shape[-1])
            img_global_pairs = img_embeds[None].expand(B, B, img_embeds.shape[-1]).reshape(B * B, img_embeds.shape[-1])
            
            attention_mask_pairs = attention_mask[None].expand(B, B, Lt).reshape(B * B, Lt) if attention_mask is not None else None
            
            chunk_size = 2048
            total_pairs = text_pairs.size(0)
            all_flat_scores = []
            
            for chunk_start in range(0, total_pairs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pairs)
                chunk_scores = self.cross_attn_classifier(
                    img_local=img_pairs[chunk_start:chunk_end],
                    text_local=text_pairs[chunk_start:chunk_end],
                    img_global=img_global_pairs[chunk_start:chunk_end],
                    text_global=text_global_pairs[chunk_start:chunk_end],
                    text_attention_mask=attention_mask_pairs[chunk_start:chunk_end] if attention_mask_pairs is not None else None,
                    force_local=False
                )
                all_flat_scores.append(chunk_scores)
            
            scores = torch.cat(all_flat_scores, dim=0)
            score_matrix = scores.view(B, B)
            
            if return_embeddings:
                return score_matrix, img_embeds, text_embeds, img_local, text_local, attention_mask
            return score_matrix
        
        text_flip_embeds = None
        tidf_loss = 0
        if self.pos_loss and flip_desc is not None:
            text_flip_embeds, text_flip_local, attention_mask_flip, text_flip_tokens, text_flip_all_layers = self.encode_text(flip_desc)                
               
        if self.tokens_idf_loss:                         
            img_embeds_pooled = img_local_unproj[:, 1:].mean(dim=1) if img_local_unproj is not None else img_local.mean(dim=1)
            tidf_loss = self.tokens_idf_loss * self.tokens_classification_loss(vision_embeddings=img_embeds_pooled, batch_text_ids=text_tokens)

        with torch.no_grad():
            queue_is_ready = (self.label_queue[0] != -1)
            active_image_global = self.image_global_queue if queue_is_ready else img_embeds
            active_image_local = self.image_local_queue if queue_is_ready else img_local
            active_labels = self.label_queue if queue_is_ready else labels
            
            global_sim = torch.matmul(text_embeds, active_image_global.T)
            same_class_mask = (labels.unsqueeze(1) == active_labels.unsqueeze(0))
            neg_mask = ~same_class_mask
            
            if not queue_is_ready:
                self_mask = torch.eye(B, device=self.device, dtype=torch.bool)
                neg_mask = neg_mask & (~self_mask)
                
            pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
            num_neg = self.num_mined_negatives
            mining_pool_size = min(64, active_image_global.shape[0])
            final_hard_neg_indices = []
            
            for i in range(B):
                row_neg_scores = global_sim[i].clone()
                row_neg_scores[~neg_mask[i]] = -1e9
                _, candidate_pool_indices = torch.topk(row_neg_scores, k=mining_pool_size)
                
                cand_img_local = active_image_local[candidate_pool_indices]
                anchor_text_local = text_local[i:i+1].expand(mining_pool_size, -1, -1)
                anchor_attention_mask = attention_mask[i:i+1].expand(mining_pool_size, -1) if attention_mask is not None else None
                
                t_features = self.cross_attn_classifier.ln_text(self.cross_attn_classifier.local_text_proj(anchor_text_local))
                i_features = self.cross_attn_classifier.ln_img(self.cross_attn_classifier.local_img_proj(cand_img_local))

                fused, _ = self.cross_attn_classifier.cross_attn(
                    query=t_features, 
                    key=i_features, 
                    value=i_features
                )
                attn_logits = self.cross_attn_classifier.ln_post(t_features + self.cross_attn_classifier.alpha * fused)

                if anchor_attention_mask is not None:
                    if anchor_attention_mask.dtype != torch.bool:
                        anchor_attention_mask = (anchor_attention_mask == 0)
                    mask_expanded = anchor_attention_mask.unsqueeze(-1).expand_as(attn_logits)
                    attn_logits = attn_logits.masked_fill(mask_expanded, -1e9)

                pooled = torch.mean(attn_logits, dim=1)
                latent = F.normalize(pooled, p=2, dim=-1)
                
                text_query_ref = F.normalize(t_features.mean(dim=1), p=2, dim=-1)
                local_screening_scores = torch.sum(latent * text_query_ref, dim=-1)
                
                _, top_hard_meta_indices = torch.topk(local_screening_scores, k=num_neg)
                actual_hard_indices = candidate_pool_indices[top_hard_meta_indices]
                final_hard_neg_indices.append(actual_hard_indices)

        paired_text, paired_img, paired_text_global, paired_img_global, paired_attn_masks = [], [], [], [], []
        pos_counts_list = []

        for i in range(B):
            anchor_text = text_local[i:i+1]
            anchor_text_global = text_embeds[i:i+1]
            anchor_mask = attention_mask[i:i+1] if attention_mask is not None else None
            
            current_pos_mask = pos_mask[i].clone()
            if not queue_is_ready:
                current_pos_mask[i] = False
                
            pos_imgs = img_local[current_pos_mask]
            pos_imgs_global = img_embeds[current_pos_mask]
            
            if pos_imgs.shape[0] == 0:
                pos_imgs = img_local[i:i+1]
                pos_imgs_global = img_embeds[i:i+1]

            neg_imgs = active_image_local[final_hard_neg_indices[i]]
            neg_imgs_global = active_image_global[final_hard_neg_indices[i]]
            
            num_actual_pos = pos_imgs.shape[0]
            pos_counts_list.append(num_actual_pos)
            total_eval = num_actual_pos + num_neg
            
            paired_text.append(anchor_text.expand(total_eval, -1, -1))
            paired_img.append(torch.cat([pos_imgs, neg_imgs], dim=0))
            paired_text_global.append(anchor_text_global.expand(total_eval, -1))
            paired_img_global.append(torch.cat([pos_imgs_global, neg_imgs_global], dim=0))
            if anchor_mask is not None:
                paired_attn_masks.append(anchor_mask.expand(total_eval, -1))

        # --- KEY STABILITY FIX: d INPUTS TO CROSS-ATTENTION HEAD ---
        # This prevents reranking gradients from corrupting global metric space representations
        if self.detach:
            flat_text_pairs = torch.cat(paired_text, dim=0).detach()
            flat_img_pairs = torch.cat(paired_img, dim=0).detach()
            flat_text_global = torch.cat(paired_text_global, dim=0).detach()
            flat_img_global = torch.cat(paired_img_global, dim=0).detach()
        else:
            flat_text_pairs = torch.cat(paired_text, dim=0)
            flat_img_pairs = torch.cat(paired_img, dim=0)
            flat_text_global = torch.cat(paired_text_global, dim=0)
            flat_img_global = torch.cat(paired_img_global, dim=0)
        flat_attn_masks = torch.cat(paired_attn_masks, dim=0) if len(paired_attn_masks) > 0 else None
        pos_counts = torch.tensor(pos_counts_list, device=self.device, dtype=torch.long)

        latent_features, flat_scores = self.cross_attn_classifier(
            img_local=flat_img_pairs, 
            text_local=flat_text_pairs, 
            img_global=flat_img_global,
            text_global=flat_text_global,
            text_attention_mask=flat_attn_masks,
            force_local=True,
            return_both=True
        ) 
        
        latent_features = F.normalize(latent_features, p=2, dim=-1)
        score_matrix = flat_scores.view(B, -1)
        
        self._dequeue_and_enqueue(img_embeds, img_local, labels)
        
        if return_embeddings:
            return score_matrix, img_embeds, text_embeds, img_local, text_local, attention_mask
            
        return latent_features, score_matrix, pos_counts, img_embeds, text_embeds, text_flip_embeds, tidf_loss
    
    def loss_function(self, img_embeds, text_embeds, text_flip_embeds, tidf_loss, labels, score_matrix, pos_counts):        
        ref_labels = labels.clone()
        ref_embs = text_embeds        
        if self.pos_loss:
            ref_embs = torch.cat([text_embeds, text_flip_embeds], dim=0)
            ref_labels = torch.cat([ref_labels, labels], dim=0)        
            
        miner_outputs = None
        ms_loss = 0
        
        if self.miner is not None:                                                                                                              
           miner_outputs = self.miner(img_embeds, labels, ref_emb=ref_embs, ref_labels=ref_labels)                 
            #1. Primary Global Metric Loss (Maintains Fast Retrieval Performance)
           ms_loss = self.loss_fn(img_embeds, labels, indices_tuple=miner_outputs, ref_emb=ref_embs, ref_labels=ref_labels)                
        
        # 2. Smooth Reranker Loss (Optimizes Fine-Grained Precision)
        listwise_loss = self.listwise_loss_fn(score_matrix, pos_counts)
        
        # Scale margin loss dynamically
        listwise_weight = min(0.20, 0.1 + 0.03 * self.current_epoch)
        listwise_weight = 1
        total_loss = ms_loss + tidf_loss + (listwise_weight * listwise_loss)
        #total_loss = listwise_loss
        
        with torch.no_grad():
            all_pos_clean = []
            for i in range(score_matrix.shape[0]):
                n_pos = pos_counts[i].item()
                min_pos = score_matrix[i, :n_pos].min()
                max_neg = score_matrix[i, n_pos:].max()
                all_pos_clean.append((min_pos > max_neg).float())
                
            batch_acc = torch.stack(all_pos_clean).mean()
        
        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        #self.log('ms_loss', ms_loss.item(), logger=False)
        self.log('list_loss', listwise_loss.item(), logger=False)
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        places, labels, texts, flip_descs, color_change_descs, neg_attr_descs, concepts_ids = batch
        BS, N, ch, h, w = places.shape
        
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flat_texts = []
        flat_flip_descs = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])
                if self.pos_loss:
                    flat_flip_descs.append(flip_descs[j][i])

        latent_features, score_matrix, pos_counts, img_embeds, text_embeds, text_flip_embeds, tidf_loss = self(images, flat_texts, flip_desc=flat_flip_descs, labels=labels) 
        loss = self.loss_function(img_embeds, text_embeds, text_flip_embeds, tidf_loss, labels, score_matrix, pos_counts)
        
        self.log('loss', loss.item(), logger=True)        
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    def configure_optimizers(self):
        # Higher learning rate for cross-attention head to jumpstart reranking learning rate
        head_params = list(self.cross_attn_classifier.parameters())
        
        param_groups = [
            {'params': head_params, 'lr': 5e-4, 'weight_decay': self.weight_decay}
        ]
        
        if self.train_vlm:
            backbone_params = [p for p in self.vlm_encoder.parameters() if p.requires_grad]
            param_groups.append({'params': backbone_params, 'lr': self.lr, 'weight_decay': self.weight_decay})

        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(param_groups, momentum=self.momentum)
        elif self.optimizer.lower() in ['adam', 'adamw']:
            optimizer = torch.optim.AdamW(param_groups)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu, using_native_amp, using_lbfgs):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * pg.get('initial_lr', pg['lr'])
        self.trainer.strategy.optimizer_step(optimizer, optimizer_idx, optimizer_closure)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        score_matrix, img_embeds, text_embeds, img_local, text_local, attention_mask = self(places, texts, return_embeddings=True)
        
        return {
            'scores': score_matrix.detach().cpu(), 
            'img_embeds': img_embeds.detach().cpu(), 
            'text_embeds': text_embeds.detach().cpu(), 
            'img_local': img_local.detach().cpu(), 
            'text_local': text_local.detach().cpu(),
            'attention_mask': attention_mask.detach().cpu() if attention_mask is not None else None 
        }
    
    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        if len(dm.val_datasets)==1:
            val_step_outputs = [val_step_outputs]
            
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            scores, img_embeds, text_embeds, img_local, text_local, attention_masks = [], [], [], [], [], []
            for d in val_step_outputs[i]:
                for key, value in d.items():
                    if key == 'scores': scores.append(value)
                    if key == 'img_embeds': img_embeds.append(value)
                    if key == 'text_embeds': text_embeds.append(value)
                    if key == 'img_local': img_local.append(value)
                    if key == 'text_local': text_local.append(value)
                    if key == 'attention_mask': attention_masks.append(value) 
            
            scores = torch.cat(scores, dim=0)            
            feats = torch.cat(img_embeds, dim=0)
            text_feats = torch.cat(text_embeds, dim=0)
            img_local = torch.cat(img_local, dim=0)
            text_local = torch.cat(text_local, dim=0)
            
            if any(m is not None for m in attention_masks):
                attention_masks = torch.cat([m for m in attention_masks if m is not None], dim=0)
            else:
                attention_masks = None
            
            if 'pitts' in val_set_name:
                num_references = val_dataset.num_db
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(f'Please implement validation_epoch_end for {val_set_name}')

            r_list = feats[:num_references]
            q_text_list = text_feats[num_references:]
            r_list_local = img_local[:num_references]
            q_text_list_local = text_local[num_references:]
            
            q_attention_masks = attention_masks[num_references:] if attention_masks is not None else None
            
            pitts_dict = utils.get_validation_recalls_rerank(
                r_list=r_list, q_list=q_text_list, k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives, print_results=True, dataset_name=val_set_name, faiss_gpu=self.faiss_gpu,
                rerank_model=self.cross_attn_classifier, r_local_list=r_list_local, q_local_list=q_text_list_local,
                q_attention_mask_list=q_attention_masks, force_local=False
            )                                                                                                                                                
            
            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        
    def on_save_checkpoint(self, checkpoint):
        if self.train_vlm == 1:
            ckpt_cb = next((cb for cb in self.trainer.checkpoint_callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)), None)
            ckpt_dir = os.path.dirname(ckpt_cb.dirpath)
            self.vlm_encoder.save_pretrained(ckpt_dir)
            print("Saved PEFT adapter to:", ckpt_dir)