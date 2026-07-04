import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
import os
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel, BlipProcessor, BlipModel, AutoProcessor
import open_clip
import utils


class LabelAwareMinedMSLoss(nn.Module):
    """
    Computes Multi-Similarity loss directly on a pre-structured 
    rectangular score matrix containing: [True Positives ..., Hard Mined Negatives ...]
    """
    def __init__(self, alpha=2.0, beta=50.0, lambda_val=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val

    def forward(self, score_matrix, num_pos):
        """
        score_matrix: [B, num_pos + num_neg] containing raw cross-attention outputs.
        num_pos: The number of true companion positive targets per row (N - 1)
        """
        B = score_matrix.size(0)
        loss = 0.0
        
        for i in range(B):
            row = score_matrix[i]
            pos_scores = row[:num_pos]
            neg_scores = row[num_pos:]
            
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue
                
            pos_term = (1.0 / self.alpha) * torch.log(1.0 + torch.sum(torch.exp(-self.alpha * (pos_scores - self.lambda_val))))
            neg_term = (1.0 / self.beta) * torch.log(1.0 + torch.sum(torch.exp(self.beta * (neg_scores - self.lambda_val))))
            
            loss += pos_term + neg_term
            
        return loss / B


class CrossAttnClassifier(nn.Module):
    def __init__(self, embeds_dim, num_heads=8):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embeds_dim,
            num_heads=num_heads,
            batch_first=True
        )
                
        self.score_head = nn.Sequential(
            nn.Linear(embeds_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, img, text, labels=None, return_embeddings=False):
        fused, _ = self.cross_attn(text, img, img)
        cls_token = fused[:, 0]
        scores = self.score_head(cls_token).squeeze(-1)
        
        if return_embeddings:
            return scores, img, text
        return scores

        

class LaVPR_reranker(pl.LightningModule):
    def __init__(self,   
                 lr=0.03, 
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 warmpup_steps=500,
                 milestones=[5, 10, 15],
                 lr_mult=0.3,
                 epochs=10,
                 faiss_gpu=False,
                 model_name='Salesforce/blip-itm-base-coco',
                 embeds_dim=256,
                 freeze_vlm=True,
                 train_vlm=False,
                 pos_loss=0,
                 neg_loss=0,
                 num_mined_negatives=8 # <-- Dynamic budget control parameter
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
        
        # Swapped to customized Multi-Similarity Loss for fixed-structure inputs
        self.loss_fn = LabelAwareMinedMSLoss(alpha=2.0, beta=50.0, lambda_val=0.5)
        
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
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
                
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
        """
        Modified forward to handle both targeted training (with labels) 
        and full-matrix extraction (during validation evaluation).
        """
        img_embeds, img_local, img_all_layers = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens, text_all_layers = self.encode_text(text)      
               
        B, Lt, D = text_local.shape
        Li = img_local.shape[1]

        # Validation phase logic: Fall back to full matching grid matrix
        if labels is None or not self.training:
            text_pairs = text_local[:, None].expand(B, B, Lt, D).reshape(B * B, Lt, D)
            img_pairs = img_local[None].expand(B, B, Li, D).reshape(B * B, Li, D)
            
            scores = self.cross_attn_classifier(img_pairs, text_pairs)            
            score_matrix = scores.view(B, B)
            
            if return_embeddings:
                return score_matrix, img_embeds, text_embeds, img_local, text_local
            
            return score_matrix

        # --- Training phase logic: Target Hard Negatives Only ---
        with torch.no_grad():
            # Cheap global cosine pre-score grid mapping to find hard negatives natively
            global_sim = torch.matmul(text_embeds, img_embeds.T)
            
            same_class_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
            identity_mask = torch.eye(B, dtype=torch.bool, device=self.device)
            
            pos_mask = same_class_mask & ~identity_mask
            neg_mask = ~same_class_mask

        paired_text = []
        paired_img = []
        
        # Pull layout dimensions
        # Each query will see all its companion variants (N-1 positives) + M hard out-of-class negatives
        num_pos = pos_mask[0].sum().item()
        num_neg = min(self.num_mined_negatives, neg_mask[0].sum().item())
        total_eval_elements = num_pos + num_neg

        for i in range(B):
            anchor_text = text_local[i:i+1] # [1, Lt, D]
            
            # Fetch true positives for query i
            pos_imgs = img_local[pos_mask[i]] # [num_pos, Li, D]
            
            # Fetch the toughest batch negatives based on global CLIP cosine alignment
            row_neg_scores = global_sim[i].clone()
            row_neg_scores[~neg_mask[i]] = -1e9
            _, hard_neg_indices = torch.topk(row_neg_scores, k=num_neg)
            neg_imgs = img_local[hard_neg_indices] # [num_neg, Li, D]
            
            # Pack and expand
            paired_text.append(anchor_text.expand(total_eval_elements, -1, -1))
            paired_img.append(torch.cat([pos_imgs, neg_imgs], dim=0))

        # Flatten into targeted evaluation inputs
        flat_text_pairs = torch.cat(paired_text, dim=0) # [B * (num_pos + num_neg), Lt, D]
        flat_img_pairs = torch.cat(paired_img, dim=0)   # [B * (num_pos + num_neg), Li, D]

        # Compute heavy cross-attention only on these targeted pairs
        flat_scores = self.cross_attn_classifier(flat_img_pairs, flat_text_pairs) # [B * (num_pos + num_neg), 1]
        
        # Reshape into a tight rectangular matrix [B, num_pos + num_neg]
        score_matrix = flat_scores.view(B, total_eval_elements)
        
        return score_matrix, num_pos, img_embeds, text_embeds, img_local, text_local
    

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
        scores, num_pos, _, _, _, _,= self(images, flat_texts, labels=labels) 
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