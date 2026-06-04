import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
import os
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel
from transformers import AutoModel, AutoProcessor
import open_clip
from model.salad import SALAD, CosineSALAD
from model.local_ot_loss import LocalOTLoss
from model.weighted_ms_loss import WeightedMultiSimilarityLossCM
from model.tokens_classify_loss import TokensClassificationLoss, HierarchicalTokensLoss, VocabClassificationLoss
from model.pooling_cm import TextGatedAttentionPooler, GeMPooling1D, AttentionGatedPatchPooler, SpatialLayoutPooler


class LaVPR(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,  
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                epochs=10,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False,
                model_name='Salesforce/blip-itm-base-coco',
                embeds_dim=256,
                is_freeze_text=True,
                is_trainable_text_encoder=False,
                cross_modal=0,
                lora_all_linear=False,
                lora_target_modules=None,
                lora_r=64,                
                agg_type=0,
                ot_loss=0.0,
                unimodal_loss=0.0,                
                pos_loss=0,
                neg_loss=0,
                latent_mixup=0.0,
                dynamic_gamma=0,
                tokens_idf_loss=0.0,
                tokens_idf_file=None,
                idf_grad_scale=0.05,
                idf_pooling = 'mean',
                vocab_path=None,
                image_idf_path=None,
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

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.faiss_gpu = faiss_gpu

        self.cross_modal = cross_modal
        self.lora_all_linear = lora_all_linear
        self.lora_target_modules = lora_target_modules
        self.lora_r = lora_r
        
        self.save_hyperparameters() # write hyperparams into a file
        
        if 'WeightedMultiSimilarityLoss' in loss_name:
            self.loss_fn = WeightedMultiSimilarityLossCM()
        else:
            self.loss_fn = utils.get_loss(loss_name)
        self.local_ot_loss = LocalOTLoss()
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
       
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        
        self.embeds_dim = embeds_dim        
        self.is_trainable_text_encoder = is_trainable_text_encoder
        self.agg_type = agg_type        
        self.ot_loss = ot_loss
        self.unimodal_loss = unimodal_loss        
        self.pos_loss = pos_loss
        self.neg_loss = neg_loss
        self.latent_mixup = latent_mixup
        self.dynamic_gamma = dynamic_gamma
        self.tokens_idf_loss = tokens_idf_loss
        self.tokens_idf_file = tokens_idf_file
        self.idf_pooling = idf_pooling
        vocab_size = 49408
        self.vocab_path = vocab_path
        self.image_idf_path = image_idf_path
        
        if self.tokens_idf_loss==1:
            self.tokens_classification_loss = TokensClassificationLoss(vision_dim=768, vocab_size=vocab_size, idf_path=self.tokens_idf_file, grad_scale=idf_grad_scale)
        elif self.tokens_idf_loss==2:
            self.tokens_classification_loss = HierarchicalTokensLoss()
        elif self.tokens_idf_loss==3:
            self.tokens_classification_loss = TokensClassificationLoss(vision_dim=768, vocab_size=vocab_size, idf_path=self.tokens_idf_file, grad_scale=idf_grad_scale)
            self.vocab_classification_loss = VocabClassificationLoss(vision_dim=768, vocab_path=vocab_path, image_idf_path=image_idf_path)
        
        if cross_modal == 4: # contrastive loss for cross modal retrieval
            self.contrastive_logit_scale = nn.Parameter(0.07*torch.ones([])) 
            self.contrastive_loss = utils.losses.contrastive_loss_cross_modal
            self.miner = None                            
        
        if agg_type == 1:
            self.agg = SALAD(num_channels=embeds_dim)
        elif agg_type == 2:
            self.agg = CosineSALAD(num_channels=embeds_dim)
        elif agg_type == 3:
            self.text_agg = CosineSALAD(num_channels=embeds_dim)
            self.img_agg = CosineSALAD(num_channels=embeds_dim)
        elif agg_type == 4:
            self.agg = TextGatedAttentionPooler(hidden_dim=embeds_dim, output_dim=embeds_dim)
            
        if idf_pooling == 'gem':
            self.idf_pooling_layer = GeMPooling1D()
        elif idf_pooling == 'attention':
            self.idf_pooling_layer = AttentionGatedPatchPooler()
        elif idf_pooling == 'spatial':
            self.idf_pooling_layer = SpatialLayoutPooler()            
                
        # init weight of linear layers but not the pretrained backbones
        self.apply(self._init_weights)
        
        # initialize the vpr encoder and text encoder        
        if 'blip' in model_name:
            self.text_encoder = BlipForImageTextRetrievalWrapper.from_pretrained(model_name)
            self.processor = BlipProcessor.from_pretrained(model_name)
        elif 'clip' in model_name or 'siglip' in model_name:
            self.max_text_length = 77
            if 'siglip' in model_name:
                self.max_text_length = 64
            self.text_encoder = AutoModel.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
        elif 'eva' in model_name:
            self.text_encoder, _, self.processor = open_clip.create_model_and_transforms(model_name.upper(), pretrained='merged2b_s8b_b131k')#'EVA02-B-16'
            self.tokenizer = open_clip.get_tokenizer(model_name)                
                        
        if is_freeze_text:
            # Freeze text encoder parameters
            for param in self.text_encoder.parameters():
                param.requires_grad = False                      
        
        # Define LoRA configuration
        # TaskType.FEATURE_EXTRACTION is appropriate for sentence embedding tasks            
        if self.is_trainable_text_encoder==1:                
            lora_targets = lora_target_modules
            if lora_all_linear:
                lora_targets = "all-linear"                    
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r*2,
                lora_dropout=0.1,
                target_modules=lora_targets,
                task_type=TaskType.SEQ_CLS,
                use_rslora=True,                    
                bias="none",
            )
            # Get the PEFT model with LoRA adapters
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        elif is_freeze_text:
            self.text_encoder.eval()        
        
        # self.register_buffer("cap_fq", torch.zeros(1, vocab_size, dtype=torch.float32))
        # self.register_buffer("num_samples", torch.zeros(1, dtype=torch.float32))

                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # For linear layers, use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # For biases, it's common to initialize them to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
                
    def encode_image(self, img):
        img_embeds = None
        img_local = None
        if 'blip' in self.model_name:
            img_local = self.text_encoder.encode_image(img)            
            img_embeds = img_local[:,0]
        elif 'clip' in self.model_name:
            img_output = self.text_encoder.get_image_features(pixel_values=img)
            img_local = img_output.last_hidden_state            
            img_embeds = img_output.pooler_output
        elif 'siglip' in self.model_name:
            img_output = self.text_encoder.get_image_features(pixel_values=img)
            #img_local = self.text_encoder.base_model.model.vision_model.head(img_output.last_hidden_state)                     
            img_local = img_output.last_hidden_state            
            img_embeds = img_output.pooler_output
        elif 'eva' in self.model_name:            
            img_embeds = self.text_encoder.encode_image(img)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)            
        return img_embeds, img_local
    
    
    def encode_text(self, text):
        text_embeds = None
        attention_mask = None
        text_local = None

        if 'blip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            text_local = self.text_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)    
            text_embeds= text_local[:, 0]        
        elif 'clip' in self.model_name:        
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            text_output = self.text_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            text_local = self.text_encoder.text_projection(text_output.last_hidden_state)
            text_embeds = text_output.pooler_output    
        elif 'siglip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            text_output = self.text_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            #text_local = self.text_encoder.base_model.model.text_model.head(text_output.last_hidden_state)
            text_local = text_output.last_hidden_state
            text_embeds = text_output.pooler_output                
        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.my_device)            
            text_embeds = self.text_encoder.encode_text(text_tokens)    
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)                   
        
        return text_embeds, text_local, attention_mask, text_tokens
    
    
    # the forward pass of the lightning model
    def forward(self, img, text, flip_desc=None, color_change_desc=None, neg_attr_desc=None, concept_ids=None):
        #encode image and text and get local features if the model has them (e.g. BLIP) for L-OT loss
        text_flip_embeds = None
        text_color_change_embeds = None
        text_neg_attr_embeds = None
        
        img_embeds, img_local = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens = self.encode_text(text)
        if self.pos_loss:
            if flip_desc is not None:
                text_flip_embeds, text_flip_local, attention_mask_flip, text_flip_tokens = self.encode_text(flip_desc)
            # if color_change_desc is not None:
            #     text_color_change_embeds, _, _ = self.encode_text(color_change_desc)
        if self.neg_loss:
            if neg_attr_desc is not None:
                text_neg_attr_embeds, text_neg_local, attention_mask_neg, text_neg_tokens = self.encode_text(neg_attr_desc)
        
        # Compute L-OT weights and loss if both modalities are present (Training)
        ot_loss = 0.0
        tidf_loss = 0.0
        w_v, w_t = None, None
        w_v_flip, w_t_flip = None, None
        w_v_neg, w_t_neg = None, None        
        
        if self.ot_loss>0 and img_local is not None and text_local is not None:
            t_mask = None
            t_mask_flip = None
            t_mask_neg = None
            if attention_mask is not None:
                t_mask = attention_mask[:, 1:]
            if attention_mask_flip is not None:
                t_mask_flip = attention_mask_flip[:, 1:]
            if attention_mask_neg is not None:
                t_mask_neg = attention_mask_neg[:, 1:]            
            
            # Calculate L-OT weights and loss
            ot_loss, w_v, w_t = self.local_ot_loss(img_local[:, 1:], text_local[:, 1:], t_mask=t_mask)
            ot_loss_flip, w_v_flip, w_t_flip = self.local_ot_loss(img_local[:, 1:], text_flip_local[:, 1:], t_mask=t_mask_flip)
            ot_loss_neg, w_v_neg, w_t_neg = self.local_ot_loss(img_local[:, 1:], text_neg_local[:, 1:], t_mask=t_mask_neg)

        if self.agg_type:
            if self.agg_type == 4: # gated attention 
                text_embeds = self.agg(text_local, attention_mask, text_embeds)
            elif self.agg_type == 3: # double salad
                img_embeds = self.img_agg(img_local, token_weights=w_v)
                text_embeds = self.text_agg(text_local, attention_mask, token_weights=w_t)               
                if flip_desc is not None: 
                    text_flip_embeds = self.text_agg(text_flip_local, attention_mask_flip, token_weights=w_t_flip)
                if neg_attr_desc is not None:
                    text_neg_attr_embeds = self.text_agg(text_neg_local, attention_mask_neg, token_weights=w_t_neg)                
            else: #salad
                img_embeds = self.agg(img_local, token_weights=w_v)
                text_embeds = self.agg(text_local, attention_mask, token_weights=w_t)
                if flip_desc is not None:
                    text_flip_embeds = self.agg(text_flip_local, attention_mask_flip, token_weights=w_t_flip)
                if neg_attr_desc is not None:
                    text_neg_attr_embeds = self.agg(text_neg_local, attention_mask_neg, token_weights=w_t_neg)    
                    
        if self.tokens_idf_loss:
            if self.idf_pooling == 'mean':
                img_embeds_pooled = img_local[:, 1:].mean(dim=1)
            else:
                img_embeds_pooled = self.idf_pooling_layer(img_local[:, 1:])            
            tidf_loss = self.tokens_classification_loss(vision_embeddings=img_embeds_pooled, batch_text_ids=text_tokens)
            if self.tokens_idf_loss==3 and concept_ids is not None:
                vocab_idf_loss = self.vocab_classification_loss(vision_embeddings=img_embeds_pooled, batch_concept_ids=concept_ids)
                tidf_loss = tidf_loss + vocab_idf_loss
            #tidf_loss = self.tokens_classification_loss(vision_embeddings=img_local[:, 0], batch_text_ids=text_tokens)
            #tidf_loss = self.tokens_classification_loss(cap_fq=self.cap_fq,  num_samples=self.num_samples, vision_embeddings=img_embeds_pooled, batch_text_ids=text_tokens)
            

        return img_embeds, text_embeds, text_flip_embeds, text_color_change_embeds, text_neg_attr_embeds, ot_loss, tidf_loss
    
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)        
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.epochs,  # Total number of epochs to decay over
        #     eta_min=1e-6        # The minimum LR floor (prevents dropping to absolute 0)
        # )
        
        # # 1. Calculate steps
        # steps_per_epoch = self.trainer.num_training_batches        
        # # Calculate total step iterations across all epochs
        # total_steps = self.epochs * steps_per_epoch        

        # # 2. Define the Warmup Phase (from 1/10th of LR up to full LR)
        # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, 
        #     start_factor=0.1, 
        #     end_factor=1.0, 
        #     total_iters=self.warmpup_steps
        # )

        # # 3. Define the Cosine Decay Phase (runs for the remaining steps)
        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, 
        #     T_max=(total_steps - self.warmpup_steps), 
        #     eta_min=1e-6
        # )

        # # 4. Chain them sequentially
        # scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, 
        #     schedulers=[warmup_scheduler, cosine_scheduler], 
        #     milestones=[self.warmpup_steps]  # Switches to cosine exactly at step 650
        # )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        # max grad norm clipping
        # max_grad_norm = 5.0                
        # clip_grad_norm_(self.parameters(), max_norm=max_grad_norm)

        optimizer.step(closure=optimizer_closure)

            
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels, text_embeds, text_flip_embeds, text_color_change_embeds, text_neg_attr_embeds, ot_loss=0.0, tidf_loss=0.0):
        
        # we mine the pairs/triplets if there is an online mining strategy
        if self.cross_modal == 5:
            desc_all = torch.cat([descriptors, text_embeds], dim=0)
            labels_all = torch.cat([labels, labels], dim=0)
            miner_outputs = self.miner(desc_all, labels_all)     
            loss = self.loss_fn(desc_all, labels_all, indices_tuple=miner_outputs)              
            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = desc_all.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)
            
        elif self.miner is not None:                        
            ref_labels = labels.clone()
            ref_embs = text_embeds
            
            # add positive augmentations
            if self.pos_loss:
                # ref_embs = torch.cat([text_embeds, text_flip_embeds, text_color_change_embeds], dim=0)
                # ref_labels = torch.cat([ref_labels, labels, labels], dim=0)
                ref_embs = torch.cat([text_embeds, text_flip_embeds], dim=0)
                ref_labels = torch.cat([ref_labels, labels], dim=0)                                
            
            # add negative augmentations
            if self.neg_loss: 
                ref_embs = torch.cat([ref_embs, text_neg_attr_embeds], dim=0)
                ref_labels = torch.cat([ref_labels, labels + 10**8], dim=0)
            
            #mine hard negatives
            miner_outputs = self.miner(descriptors, labels, ref_emb=ref_embs, ref_labels=ref_labels)     
            
            # Compute Batch-Adaptive Base ---
            if self.dynamic_gamma:
                with torch.no_grad():                    
                    
                    # 2. Compute the full similarity matrix between query and reference batches
                    sim_matrix = torch.matmul(descriptors, ref_embs.T)
                    
                    # 3. Create a binary mask where labels match (true positive pairs)
                    # Using unsqueeze allows broadcasting: [Batch_Q, 1] == [1, Batch_R]
                    pos_mask = (labels.unsqueeze(1) == ref_labels.unsqueeze(0))
                    
                    # 4. Extract positive values and safely calculate the mean
                    pos_similarities = sim_matrix[pos_mask]
                    
                    if pos_similarities.numel() > 0:
                        mean_pos_sim = pos_similarities.mean().item()
                        # Use the raw mean as the zero-point boundary.
                        # Clamp it between 0.35 and 0.45 to prevent extreme drift.
                        self.loss_fn.base = max(0.35, min(mean_pos_sim, 0.45))
                    else:
                        # Fallback to your stable 0.4 default if a weird batch has zero positive pairs
                        self.loss_fn.base = 0.4            
            #calc loss
            loss = self.loss_fn(descriptors, labels, indices_tuple=miner_outputs, ref_emb=ref_embs, ref_labels=ref_labels)              
            
            # latent mixup?
            if self.latent_mixup>0:
                # calculate latent mixup loss:
                # Extract negative pairs from the miner outputs
                a1, p, a2, n = miner_outputs
                if len(a2) > 0:
                    BS = descriptors.shape[0]
                    
                    # Vt: anchor vector (text)
                    Vt = ref_embs[n]
                    
                    # Vi-: negative image vector from mining
                    Vi_neg = descriptors[a2]
                    
                    # Vi+: positive image vector (labels are equal)
                    Vi_pos = descriptors[n % BS]
                    
                    # Mixup ratio
                    alpha = torch.rand(len(a2), 1, device=descriptors.device)
                    
                    # V' = a*Vi+ + (1-a)*Vi-
                    # Note: We L2-normalize V_prime, otherwise Vt @ V' exactly equals score2 due to linearity of the dot product, making MSE 0.
                    V_prime = torch.nn.functional.normalize(alpha * Vi_pos + (1 - alpha) * Vi_neg, p=2, dim=-1)
                    
                    # score1 = Vt@V'
                    score1 = (Vt * V_prime).sum(dim=-1)
                    
                    # score2 = a*(Vt@Vi+) + (1-a)*(Vt@Vi-)
                    score2 = alpha.squeeze(-1) * (Vt * Vi_pos).sum(dim=-1) + (1 - alpha.squeeze(-1)) * (Vt * Vi_neg).sum(dim=-1)
                    
                    mixup_loss = torch.nn.functional.mse_loss(score1, score2)
                    loss = loss + self.latent_mixup * mixup_loss
            #uni modal loss?
            if self.unimodal_loss>0:
                # calculate unimodal loss for image modality
                # miner_outputs = self.miner(descriptors, labels)     
                # img_loss = self.loss_fn(descriptors, labels, indices_tuple=miner_outputs)
                miner_outputs = self.miner(text_embeds, ref_labels)     
                txt_loss = self.loss_fn(text_embeds, ref_labels, indices_tuple=miner_outputs)
                #loss = loss + self.unimodal_loss * img_loss + self.unimodal_loss * txt_loss
                loss = loss + self.unimodal_loss * txt_loss

            loss = loss + self.ot_loss * ot_loss + self.tokens_idf_loss * tidf_loss

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            if self.cross_modal == 4: # contrastive loss
                # contrastive loss cross modal
                logit_scale = self.contrastive_logit_scale
                loss = self.contrastive_loss(descriptors, text_embeds, logit_scale)                            
            else:
                loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels, texts, flip_descs, color_change_descs, neg_attr_descs, concepts_ids = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        concepts_ids = concepts_ids.view(BS*N, -1)
        
        flat_texts = []
        flat_flip_descs = []
        flat_color_change_descs = []
        flat_neg_attr_descs = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])
                flat_flip_descs.append(flip_descs[j][i])
                flat_color_change_descs.append(color_change_descs[j][i])
                #flat_neg_attr_descs.append(neg_attr_descs[j][i])        

        # Feed forward the batch to the model
        descriptors, text_embeds, text_flip_embeds, text_color_change_embeds, neg_attr_embeds, ot_loss, tidf_loss = self(images, flat_texts, flat_flip_descs, flat_color_change_descs, flat_neg_attr_descs, concepts_ids) 
        loss = self.loss_function(descriptors, labels, text_embeds, text_flip_embeds, text_color_change_embeds, neg_attr_embeds, ot_loss, tidf_loss) # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        
        # if batch_idx == 1:   # 0, 1 → two batches
        #     self.trainer.should_stop = True
        
        return {'loss': loss}
    
    # This is called at the end of eatch training epoch
    def training_epoch_end(self, training_step_outputs):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        # calculate descriptors
        descriptors, text_embeds, _, _, _, _, _ = self(places, texts)
        #return descriptors.detach().cpu()
        descriptors = descriptors.detach().cpu()        
        text_embeds_cpu = text_embeds.detach().cpu()        
        ret_dict = {'descriptors': descriptors, 'text_embeds': text_embeds_cpu}
        return ret_dict
    
    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            # stack all descriptors
            descriptors = []
            text_embeds = []            
            for d in val_step_outputs[i]:
                for key, value in d.items():
                    if key == 'descriptors':
                        descriptors.append(value)
                    elif key == 'text_embeds' and value is not None:
                        text_embeds.append(value)                         
            
            feats = torch.cat(descriptors, dim=0)
            text_feats = None
            if text_embeds != []:
                text_feats = torch.cat(text_embeds, dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                # num_references = val_dataset.dbStruct.numDb
                num_references = val_dataset.num_db
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset)-num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            
            if self.cross_modal:
                r_text_list = text_feats[ : num_references]
                q_text_list = text_feats[num_references : ]
                
                pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                    q_list=q_text_list,
                                                    k_values=[1, 5, 10, 15, 20, 50, 100],
                                                    gt=positives,
                                                    print_results=True,
                                                    dataset_name=val_set_name,
                                                    faiss_gpu=self.faiss_gpu
                                                )               
            
            
            else:

                pitts_dict = utils.get_validation_recalls(r_list=r_list, 
                                                    q_list=q_list,
                                                    k_values=[1, 5, 10, 15, 20, 50, 100],
                                                    gt=positives,
                                                    print_results=True,
                                                    dataset_name=val_set_name,
                                                    faiss_gpu=self.faiss_gpu
                                                )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')
        
    def on_save_checkpoint(self, checkpoint):
        if self.is_trainable_text_encoder==1:
            # Lightning gives you where THIS checkpoint is being written            
            ckpt_cb = next(
                (cb for cb in self.trainer.checkpoint_callbacks 
                if isinstance(cb, pl.callbacks.ModelCheckpoint)),
                None
            )                      

            # Directory containing the checkpoint file
            ckpt_dir = os.path.dirname(ckpt_cb.dirpath)

            self.text_encoder.save_pretrained(ckpt_dir)
            print("Saved PEFT adapter to:", ckpt_dir)
    

    
