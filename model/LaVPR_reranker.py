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


class ListwiseRankMarginLoss(nn.Module):
    """
    Replaces MS Loss for rectangular score matrices [B, num_pos + num_neg].
    Optimizes relative ranking margins instead of absolute scores.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, score_matrix, num_pos):
        """
        score_matrix: [B, num_pos + num_neg] containing your blended model scores.
        num_pos: (int) The number of true positive slots grouped at the front.
        """
        # 1. Slice and isolate your positive and negative scores per row
        pos_scores = score_matrix[:, :num_pos].unsqueeze(2)  # [B, num_pos, 1]
        neg_scores = score_matrix[:, num_pos:].unsqueeze(1)  # [B, 1, num_neg]
        
        # 2. Compute pairwise distance comparisons between every positive and negative
        ranking_violations = self.margin - (pos_scores - neg_scores)
        
        # 3. Apply a ReLU constraint (hinge loss) to isolate entries violating the margin
        loss = torch.clamp(ranking_violations, min=0.0).mean()
        return loss


class CrossAttnClassifier(nn.Module):
    def __init__(self, embeds_dim, num_heads=8):
        super().__init__()
        self.img_proj = nn.Linear(embeds_dim, embeds_dim)
        self.text_proj = nn.Linear(embeds_dim, embeds_dim)
        
        self.ln_text = nn.LayerNorm(embeds_dim)
        self.ln_img = nn.LayerNorm(embeds_dim)
        self.ln_post = nn.LayerNorm(embeds_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embeds_dim, num_heads=num_heads, batch_first=True
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(embeds_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        # Learnable scaling factor to automatically balance cross-attention with baseline scores
        self.local_scale = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, img_local, text_local, img_global=None, text_global=None):
        # 1. Local Cross Attention
        t_features = self.ln_text(self.text_proj(text_local))
        i_features = self.ln_img(self.img_proj(img_local))
        
        fused, _ = self.cross_attn(t_features, i_features, i_features)
        attn_logits = self.ln_post(fused + t_features)
        
        # Max pool across the token sequence
        pooled = torch.max(attn_logits, dim=1)[0] 
        local_score = self.score_head(pooled).squeeze(-1)
        
        # 2. Combined Score Blend
        if img_global is not None and text_global is not None:            
            global_sim = F.cosine_similarity(text_global, img_global, dim=-1)                
            return global_sim + self.local_scale * local_score
            
        return local_score


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
        
        self.loss_fn = ListwiseRankMarginLoss(margin=0.3)
        
        self.save_hyperparameters()
        self.batch_acc = [] 
        self.embeds_dim = embeds_dim        
        self.train_vlm = train_vlm
        self.pos_loss = pos_loss
        self.neg_loss = neg_loss                
        
        self.cross_attn_classifier = CrossAttnClassifier(embeds_dim)
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

        # --- Global Hard Negative Memory Bank Registration ---
        # Queue capacity set to 4140 elements (Safe multiple for batch size 180)
        self.queue_size = 4140 
        self.register_buffer("image_global_queue", torch.zeros(self.queue_size, embeds_dim))
        self.register_buffer("image_local_queue", torch.zeros(self.queue_size, max_img_tokens, embeds_dim))
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
        
        # Safety fallback slice if end of data produces smaller leftover batches
        if self.queue_ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - self.queue_ptr
            
        if batch_size <= 0:
            self.queue_ptr = 0
            batch_size = img_embeds.shape[0]

        # Overwrite current target indices block
        self.image_global_queue[self.queue_ptr:self.queue_ptr + batch_size] = img_embeds[:batch_size]
        self.image_local_queue[self.queue_ptr:self.queue_ptr + batch_size] = img_local[:batch_size]
        self.label_queue[self.queue_ptr:self.queue_ptr + batch_size] = labels[:batch_size]
        
        # Advance rolling pointer tracking
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size

    def encode_image(self, img):
        img_embeds = None
        img_local = None
        img_all_layers = None
        if 'blip' in self.model_name:
            img_local = self.vlm_encoder.encode_image(img)            
            img_embeds = img_local[:,0]
        elif 'llm2clip' in self.model_name:
            img_output = self.vlm_encoder.vision_model(pixel_values=img.to(self.vlm_encoder.dtype), output_hidden_states=True)
            img_local = self.vlm_encoder.visual_projection(img_output.last_hidden_state)
            img_embeds = self.vlm_encoder.visual_projection(img_output.pooler_output)
            img_all_layers = img_output.hidden_states
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:            
            vision_outputs = self.vlm_encoder.vision_model(pixel_values=img, output_hidden_states=True)
            img_local = self.vlm_encoder.visual_projection(vision_outputs.last_hidden_state)
            img_all_layers = vision_outputs.hidden_states
            pooled_output = vision_outputs.pooler_output
            img_embeds = self.vlm_encoder.visual_projection(pooled_output)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
        elif 'siglip' in self.model_name:
            img_output = self.vlm_encoder.get_image_features(pixel_values=img)
            img_local = img_output.last_hidden_state            
            img_embeds = img_output.pooler_output
        elif 'eva' in self.model_name:            
            img_embeds = self.vlm_encoder.encode_image(img)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)            
        return img_embeds, img_local, img_all_layers
    
    def encode_text(self, text):
        text_embeds = None
        attention_mask = None
        text_local = None
        text_all_layers = None

        if 'blip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokens = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_local = self.vlm_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)    
            text_embeds= text_local[:, 0]        
        elif 'llm2clip' in self.model_name:
            text_tokens = self.llm_encoder.encode(text, convert_to_tensor=True).to(self.device)
            text_embeds = self.vlm_encoder.get_text_features(text_tokens.to(self.vlm_encoder.dtype)).float()
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:        
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_outputs = self.vlm_encoder.text_model(input_ids=text_tokens, attention_mask=attention_mask, output_hidden_states=True)                        
            text_local = self.vlm_encoder.text_projection(text_outputs.last_hidden_state)  
            text_all_layers = text_outputs.hidden_states                         
            pooled_text = text_outputs.pooler_output                                     
            text_embeds = self.vlm_encoder.text_projection(pooled_text)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        elif 'siglip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.device)                
            text_output = self.vlm_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            text_local = text_output.last_hidden_state
            text_embeds = text_output.pooler_output                
        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.device)            
            text_embeds = self.vlm_encoder.encode_text(text_tokens)    
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)        
        
        return text_embeds, text_local, attention_mask, text_tokens, text_all_layers

    def forward(self, img, text, labels=None, return_embeddings=False):
        img_embeds, img_local, img_all_layers = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens, text_all_layers = self.encode_text(text)      
                
        B, Lt, D = text_local.shape
        Li = img_local.shape[1]

        # --- Validation Phase Logic (Safely Chunked to Prevent OOM) ---
        if labels is None or not self.training:
            text_pairs = text_local[:, None].expand(B, B, Lt, D).reshape(B * B, Lt, D)
            img_pairs = img_local[None].expand(B, B, Li, D).reshape(B * B, Li, D)            
            text_global_pairs = text_embeds[:, None].expand(B, B, D).reshape(B * B, D)
            img_global_pairs = img_embeds[None].expand(B, B, D).reshape(B * B, D)
            
            chunk_size = 2048
            total_pairs = text_pairs.size(0)
            all_flat_scores = []
            
            for chunk_start in range(0, total_pairs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_pairs)
                chunk_scores = self.cross_attn_classifier(
                    img_pairs[chunk_start:chunk_end],
                    text_pairs[chunk_start:chunk_end],
                    img_global_pairs[chunk_start:chunk_end],
                    text_global_pairs[chunk_start:chunk_end]
                )
                all_flat_scores.append(chunk_scores)
            
            scores = torch.cat(all_flat_scores, dim=0)
            score_matrix = scores.view(B, B)
            
            if return_embeddings:
                return score_matrix, img_embeds, text_embeds, img_local, text_local
            return score_matrix

        # --- Training Phase Logic: Global Queue-Based Hard Mining ---
        with torch.no_grad():
            # Check if history bank queue is warm
            queue_is_ready = (self.label_queue[0] != -1)
            
            # Use active queue for negative candidates if ready, otherwise fallback to standard batch
            active_image_global = self.image_global_queue if queue_is_ready else img_embeds
            active_image_local = self.image_local_queue if queue_is_ready else img_local
            active_labels = self.label_queue if queue_is_ready else labels
            
            # Compute global cosine similarities against the entire memory pool
            global_sim = torch.matmul(text_embeds, active_image_global.T)
            
            same_class_mask = (labels.unsqueeze(1) == active_labels.unsqueeze(0))
            neg_mask = ~same_class_mask
            
            # Positives are gathered locally from the batch
            local_same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
            pos_mask = local_same_class_mask
            
            num_pos = pos_mask[0].sum().item()
            num_neg = self.num_mined_negatives
            
            # Screen the top 64 hardest global candidates using the cross-attention layers
            mining_pool_size = min(64, active_image_global.shape[0])
            final_hard_neg_indices = []
            
            for i in range(B):
                row_neg_scores = global_sim[i].clone()
                row_neg_scores[~neg_mask[i]] = -1e9
                
                _, candidate_pool_indices = torch.topk(row_neg_scores, k=mining_pool_size)
                
                cand_img_local = active_image_local[candidate_pool_indices]
                anchor_text_local = text_local[i:i+1].expand(mining_pool_size, -1, -1)
                
                # Screen using token layers without global shortcuts
                t_features = self.cross_attn_classifier.ln_text(self.cross_attn_classifier.text_proj(anchor_text_local))
                i_features = self.cross_attn_classifier.ln_img(self.cross_attn_classifier.img_proj(cand_img_local))
                
                fused, _ = self.cross_attn_classifier.cross_attn(t_features, i_features, i_features)
                attn_logits = self.cross_attn_classifier.ln_post(fused + t_features)
                pooled = torch.max(attn_logits, dim=1)[0]
                
                local_screening_scores = self.cross_attn_classifier.score_head(pooled).squeeze(-1)
                
                _, top_hard_meta_indices = torch.topk(local_screening_scores, k=num_neg)
                actual_hard_indices = candidate_pool_indices[top_hard_meta_indices]
                final_hard_neg_indices.append(actual_hard_indices)

        # Build final batches for the backward pass
        paired_text, paired_img, paired_text_global, paired_img_global = [], [], [], []
        total_eval_elements = num_pos + num_neg

        for i in range(B):
            anchor_text = text_local[i:i+1]
            anchor_text_global = text_embeds[i:i+1]
            
            pos_imgs = img_local[pos_mask[i]]
            pos_imgs_global = img_embeds[pos_mask[i]]
            
            neg_imgs = active_image_local[final_hard_neg_indices[i]]
            neg_imgs_global = active_image_global[final_hard_neg_indices[i]]
            
            paired_text.append(anchor_text.expand(total_eval_elements, -1, -1))
            paired_img.append(torch.cat([pos_imgs, neg_imgs], dim=0))
            paired_text_global.append(anchor_text_global.expand(total_eval_elements, -1))
            paired_img_global.append(torch.cat([pos_imgs_global, neg_imgs_global], dim=0))

        flat_text_pairs = torch.cat(paired_text, dim=0)
        flat_img_pairs = torch.cat(paired_img, dim=0)
        flat_text_global = torch.cat(paired_text_global, dim=0)
        flat_img_global = torch.cat(paired_img_global, dim=0)

        # Run cross-attention forward pass with gradients tracked
        flat_scores = self.cross_attn_classifier(flat_img_pairs, flat_text_pairs, flat_img_global, flat_text_global)
        score_matrix = flat_scores.view(B, total_eval_elements)
        
        # Push the current batch into the rolling memory bank
        self._dequeue_and_enqueue(img_embeds, img_local, labels)
        
        if return_embeddings:
            return score_matrix, num_pos, img_embeds, text_embeds, img_local, text_local
        return score_matrix, num_pos
        

    def loss_function(self, score_matrix, num_pos):        
        loss = self.loss_fn(score_matrix, num_pos)
        
        # Track accuracy: check if the top-scoring element falls within the positive window indices
        with torch.no_grad():
            predicted_max_indices = score_matrix.argmax(dim=1)
            batch_acc = (predicted_max_indices < num_pos).float().mean() 
        
        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        places, labels, texts, flip_descs, color_change_descs, neg_attr_descs, concepts_ids = batch
        BS, N, ch, h, w = places.shape
        
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flat_texts = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])

        # Feed forward the batch to the optimized model
        scores, num_pos = self(images, flat_texts, labels=labels) 
        loss = self.loss_function(scores, num_pos)
        
        self.log('loss', loss.item(), logger=True)        
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    # --- Rest of your original Lightning methods remain intact ---
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer.lower() in ['adam', 'adamw']:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)        
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
                       on_tpu, using_native_amp, using_lbfgs):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        self.trainer.strategy.optimizer_step(optimizer, optimizer_idx, optimizer_closure)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        score_matrix, img_embeds, text_embeds, img_local, text_local = self(places, texts, return_embeddings=True)
        return {'scores': score_matrix.detach().cpu(), 'img_embeds': img_embeds.detach().cpu(), 'text_embeds': text_embeds.detach().cpu(), 
                'img_local': img_local.detach().cpu(), 'text_local': text_local.detach().cpu()}
    
    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        if len(dm.val_datasets)==1:
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            scores, img_embeds, text_embeds, img_local, text_local= [], [], [], [], []
            for d in val_step_outputs[i]:
                for key, value in d.items():
                    if key == 'scores': scores.append(value)
                    if key == 'img_embeds': img_embeds.append(value)
                    if key == 'text_embeds': text_embeds.append(value)
                    if key == 'img_local': img_local.append(value)
                    if key == 'text_local': text_local.append(value)
            
            scores = torch.cat(scores, dim=0)            
            feats = torch.cat(img_embeds, dim=0)
            text_feats = torch.cat(text_embeds, dim=0)
            img_local = torch.cat(img_local, dim=0)
            text_local = torch.cat(text_local, dim=0)
            
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
            
            pitts_dict = utils.get_validation_recalls_rerank(
                r_list=r_list, q_list=q_text_list, k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives, print_results=True, dataset_name=val_set_name, faiss_gpu=self.faiss_gpu,
                rerank_model=self.cross_attn_classifier, r_local_list=r_list_local, q_local_list=q_text_list_local
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