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
from model.pooling_cm import TextGatedAttentionPooler, GeMPooling1D, AttentionGatedPatchPooler, SpatialLayoutPooler, MultiLayerAttentionTextPooler, ResidualTextPooler


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
                train_vlm=False,
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
                vocab_idf_loss=0.0,                
                vocab_path=None,
                image_idf_path=None,
                vocab_grad_scale=0.05,
                cls_adapter=0,
                cmpl=0,
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
            
        if ot_loss:
            self.local_ot_loss = LocalOTLoss()
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
       
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        
        self.embeds_dim = embeds_dim        
        self.train_vlm = train_vlm
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
        vocab_size = self.get_vocab_size(model_name)        
        self.vocab_path = vocab_path
        self.image_idf_path = image_idf_path
        self.vocab_idf_loss = vocab_idf_loss
        self.vocab_grad_scale = vocab_grad_scale
        self.cls_adapter = cls_adapter
        self.cmpl = cmpl
        
        if self.tokens_idf_loss==1:
            self.tokens_classification_loss = TokensClassificationLoss(vision_dim=768, vocab_size=vocab_size, idf_path=self.tokens_idf_file, grad_scale=idf_grad_scale, cls_adapter=cls_adapter)
        elif self.tokens_idf_loss==2:
            self.tokens_classification_loss = HierarchicalTokensLoss()
        
        if self.vocab_idf_loss:
            self.vocab_classification_loss = VocabClassificationLoss(vision_dim=768, vocab_path=vocab_path, image_idf_path=image_idf_path, grad_scale=vocab_grad_scale, cls_adapter=cls_adapter)
        
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
        elif agg_type == 5:
            self.agg = MultiLayerAttentionTextPooler(text_dim=embeds_dim, joint_dim=embeds_dim)
        elif agg_type == 6:
            self.agg = ResidualTextPooler(text_dim=embeds_dim, joint_dim=embeds_dim)
            
        if idf_pooling == 'gem':
            self.idf_pooling_layer = GeMPooling1D()
        elif idf_pooling == 'attention':
            self.idf_pooling_layer = AttentionGatedPatchPooler()
        elif idf_pooling == 'spatial':
            self.idf_pooling_layer = SpatialLayoutPooler()      
            
        if cmpl:
             # Target intermediate layers P = {p3, p6, p9, p12}
            self.target_layers = [3, 6, 9, 12]
            cmpl_dim = 768
            
            # Modules per target intermediate layer
            self.sfm_modules = nn.ModuleDict({
                f"layer_{str(l)}": SaliencyFilteringModule(cmpl_dim) for l in self.target_layers
            })

            # 1. Visual Extractors: map 768 -> 512
            self.visual_E_f_modules = nn.ModuleDict({
                f"layer_{str(l)}": FineGrainedExtractor(query_dim=embeds_dim, kv_dim=768, num_queries=16) 
                for l in self.target_layers
            })

            # 2. Textual Extractors: map 512 -> 512
            self.text_E_f_modules = nn.ModuleDict({
                f"layer_{str(l)}": FineGrainedExtractor(query_dim=embeds_dim, kv_dim=512, num_queries=16) 
                for l in self.target_layers
            })
            
            self.sdm_loss = SDMLoss()
            self.local_sdm_loss = LocalSDMLoss(temperature=0.02)
                
        # init weight of linear layers but not the pretrained backbones
        self.apply(self._init_weights)
        
        # initialize vlm encoder        
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
            self.vlm_encoder, _, self.processor = open_clip.create_model_and_transforms(model_name.upper(), pretrained='merged2b_s8b_b131k')#'EVA02-B-16'
            self.tokenizer = open_clip.get_tokenizer(model_name)                  
                        
        if is_freeze_text:
            # Freeze text encoder parameters
            for param in self.vlm_encoder.parameters():
                param.requires_grad = False                      
        
        # Define LoRA configuration
        # TaskType.FEATURE_EXTRACTION is appropriate for sentence embedding tasks            
        if self.train_vlm==1:                            
            lora_targets = lora_target_modules
            if lora_all_linear:
                lora_targets = "all-linear"  
                
            # # Programmatically build the exact string matches for your 12 ViT layers
            # lora_targets = []
            # for i in range(12):  # Assuming standard 12-layer ViT baseline
            #     lora_targets.append(f"vision_model.encoder.layers.{i}.self_attn.q_proj")
            #     lora_targets.append(f"vision_model.encoder.layers.{i}.self_attn.v_proj")                  
            
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
            if 'llm2clip' in model_name:
                self.llm_encoder = get_peft_model(self.llm_encoder, lora_config)
            else:            
                self.vlm_encoder = get_peft_model(self.vlm_encoder, lora_config)
                
        elif is_freeze_text:
            self.vlm_encoder.eval()        
        
        # self.register_buffer("cap_fq", torch.zeros(1, vocab_size, dtype=torch.float32))
        # self.register_buffer("num_samples", torch.zeros(1, dtype=torch.float32))

                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # For linear layers, use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # For biases, it's common to initialize them to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)      
                
    def get_vocab_size(self, model_name):
        vocab_size = 0
        if 'blip' in model_name:
            vocab_size = 30524           
        elif 'llm2clip' in model_name:
            vocab_size = 128256
        elif 'clip' in model_name:            
            vocab_size = 49408            
        elif 'eva' in model_name:
            vocab_size = 49408
        elif 'siglip' in model_name:
            vocab_size = 256000        
        return vocab_size
                
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
            # img_output = self.vlm_encoder.get_image_features(pixel_values=img, output_hidden_states=True)            
            # img_local = img_output.last_hidden_state            
            # img_embeds = img_output.pooler_output
            # img_all_layers = img_output.hidden_states
            
            vision_outputs = self.vlm_encoder.vision_model(
                pixel_values=img, 
                output_hidden_states=True
            )
            img_local = vision_outputs.last_hidden_state            
            img_all_layers = vision_outputs.hidden_states
            pooled_output = vision_outputs.pooler_output
            if hasattr(self.vlm_encoder, 'visual_projection'):
                img_embeds = self.vlm_encoder.visual_projection(pooled_output)
            else:
                img_embeds = pooled_output            
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            
        elif 'siglip' in self.model_name:
            #img_output = self.vlm_encoder.get_image_features(pixel_values=img)
            #img_local = self.vlm_encoder.base_model.model.vision_model.head(img_output.last_hidden_state)                     
            vision_outputs = self.vlm_encoder.vision_model(
                pixel_values=img, 
                output_hidden_states=True
            )
            img_local = vision_outputs.last_hidden_state            
            img_all_layers = vision_outputs.hidden_states
            img_embeds = self.vlm_encoder.vision_model.head(vision_outputs.last_hidden_state)
            
        elif 'eva' in self.model_name:            
            # img_embeds, img_local = self.vlm_encoder.encode_image(img)
            # img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)                        
            img_local = self.vlm_encoder.visual.trunk.forward_features(img)
            if isinstance(img_local, dict):
                img_local = img_local['x']

            # 2. Extract global image token (CLS context is at index 0)
            # Shape: [Batch, Hidden_Dim]
            #img_local = self.vlm_encoder.visual.trunk.head(img_local)
            img_embeds = img_local[:, 0, :]
            # 3. Apply the final normalization if forward_features did not include it
            # (Note: Most timm-based models apply this inside forward_features, but check if needed)
            if hasattr(self.vlm_encoder.visual.trunk, 'norm') and not isinstance(img_local, dict):
                # If the trunk's final norm layer wasn't already applied inside forward_features
                img_embeds = self.vlm_encoder.visual.trunk.norm(img_embeds)
            img_embeds = self.vlm_encoder.visual.trunk.head(img_embeds)            
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)            
                        
        return img_embeds, img_local, img_all_layers
    
    
    def encode_text(self, text):
        text_embeds = None
        attention_mask = None
        text_local = None
        text_all_layers = None

        if 'blip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            text_local = self.vlm_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)    
            text_embeds= text_local[:, 0]        
        elif 'llm2clip' in self.model_name:
            text_tokens = self.llm_encoder.encode(text, convert_to_tensor=True).to(self.device)
            text_embeds = self.vlm_encoder.get_text_features(text_tokens.to(self.vlm_encoder.dtype)).float()
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:        
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            # text_output = self.vlm_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask, output_hidden_states=True)
            # text_local = self.vlm_encoder.text_projection(text_output.last_hidden_state)
            # text_embeds = text_output.pooler_output    
            # text_all_layers = text_output.hidden_states    
            
            text_outputs = self.vlm_encoder.text_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True
            )                        
            text_local = text_outputs.last_hidden_state  # Shape: [B, seq_len, hidden_dim]
            text_all_layers = text_outputs.hidden_states # Tuple of all intermediate layer states                        
            pooled_text = text_outputs.pooler_output     # Shape: [B, hidden_dim]                        
            if hasattr(self.vlm_encoder, 'text_projection'):
                text_embeds = self.vlm_encoder.text_projection(pooled_text)
            else:
                text_embeds = pooled_text                            
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
        elif 'siglip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(self.my_device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(self.my_device)                
            
            # Call the text_model directly to get all hidden states
            text_outputs = self.vlm_encoder.text_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_local = text_outputs.last_hidden_state
            text_embeds = text_outputs.pooler_output
            text_all_layers = text_outputs.hidden_states
        elif 'eva' in self.model_name:
            # text_tokens = self.tokenizer(text).to(self.my_device)            
            # text_embeds = self.vlm_encoder.encode_text(text_tokens)    
            # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)        
            #1. Tokenize text inputs as you normally do
            text_tokens = self.tokenizer(text).to(self.my_device)            
            ## 2. Extract standard global pooled embeddings [Batch, 512] for your Miner Loss
            text_embeds = self.vlm_encoder.encode_text(text_tokens)

            # 3. FIX: Prepare inputs manually for the inner text transformer module
            x = self.vlm_encoder.text.token_embedding(text_tokens)
            x = x + self.vlm_encoder.text.positional_embedding

            # 4. Call the transformer using the correct argument name 'x'
            # indices=[-1] isolates the final sequence output layer block
            _, intermediates = self.vlm_encoder.text.transformer.forward_intermediates(
                x=x,                 # Note: The key name must be x, not text
                attn_mask=self.vlm_encoder.text.attn_mask,
                indices=[-1]
            )

            # 5. Extract and apply LayerNorm to match your text encoder's output configuration
            # OpenCLIP internal format is [Sequence, Batch, Dim], so we permute to standard layout
            text_local = intermediates[-1]#.permute(1, 0, 2)  # Shape: [Batch, 77, 512]
            text_local = self.vlm_encoder.text.ln_final(text_local)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return text_embeds, text_local, attention_mask, text_tokens, text_all_layers
    
    
    # the forward pass of the lightning model
    def forward(self, img, text, flip_desc=None, color_change_desc=None, neg_attr_desc=None, concept_ids=None, labels=None):
        #encode image and text and get local features if the model has them (e.g. BLIP) for L-OT loss
        text_flip_embeds = None
        text_color_change_embeds = None
        text_neg_attr_embeds = None
        
        img_embeds, img_local, img_all_layers = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens, text_all_layers = self.encode_text(text)
        if self.pos_loss:
            if flip_desc is not None:
                text_flip_embeds, text_flip_local, attention_mask_flip, text_flip_tokens, text_flip_all_layers = self.encode_text(flip_desc)
            # if color_change_desc is not None:
            #     text_color_change_embeds, _, _ = self.encode_text(color_change_desc)
        if self.neg_loss:
            if neg_attr_desc is not None:
                text_neg_attr_embeds, text_neg_local, attention_mask_neg, text_neg_tokens, text_neg_all_layers = self.encode_text(neg_attr_desc)
        
        # Compute L-OT weights and loss if both modalities are present (Training)
        ot_loss = 0.0
        tidf_loss = 0.0
        w_v, w_t = None, None
        w_v_flip, w_t_flip = None, None
        w_v_neg, w_t_neg = None, None        
        
        if self.cmpl:
            local_features_v = {}
            local_features_t = {}
            
            # --- 1. Hierarchical Feature Extraction & Progressive Alignment ---
            for layer in self.target_layers:
                layer_key = f"layer_{layer}"
                
                # --- Modality A: Visual ---
                v_layer_out = img_all_layers[layer]
                v_cls = v_layer_out[:, 0:1, :]    # Extract [CLS]
                v_patches = v_layer_out[:, 1:, :] # Extract raw patch tokens
                
                # Filter background visual noise via SFM
                v_filtered_patches = self.sfm_modules[layer_key](v_patches)
                
                # Recombine CLS + Saliency tokens
                v_combined = torch.cat([v_cls, v_filtered_patches], dim=1)
                
                # Anchor representation distillation via E_f
                # F_v^(l) shape: (B, num_queries, embed_dim)
                F_v_l = self.visual_E_f_modules[layer_key](v_combined)
                local_features_v[layer] = F_v_l
                
                # --- Modality B: Textual ---
                t_layer_out = text_all_layers[layer] # (B, num_tokens, embed_dim)
                
                # Direct anchor representation distillation via E_f 
                # F_t^(l) shape: (B, num_queries, embed_dim)
                F_t_l = self.text_E_f_modules[layer_key](t_layer_out)
                local_features_t[layer] = F_t_l

            # --- 2. Extracting Global vs Local representations ---
            
            # For Local Loss (L_ls): Uses the fine-grained query representations directly
            # For Global Loss (L_gs): Generated by pooling the final aligned layer (p12) outputs
            final_layer = self.target_layers[-1]
            
            # Aggregate the query tokens via mean pooling to form a singular global representation vector
            global_v_raw = local_features_v[final_layer].mean(dim=1) # (B, embed_dim)
            global_t_raw = local_features_t[final_layer].mean(dim=1) # (B, embed_dim)
            
            # Hyper-sphere L2 normalization mapping
            img_embeds = F.normalize(global_v_raw, p=2, dim=-1)
            text_embeds = F.normalize(global_t_raw, p=2, dim=-1)
            
            total_local_sdm_loss = 0.0

            for layer in self.target_layers:
                # Extract specific layer tensor pairs
                F_v_layer = local_features_v[layer]
                F_t_layer = local_features_t[layer]
                
                # Compute layer-specific Local SDM loss
                L_ls_layer = self.local_sdm_loss(F_v_layer, F_t_layer, labels=labels)
    
                # Aggregate across hierarchical depths
                total_local_sdm_loss += L_ls_layer
            
            tidf_loss = total_local_sdm_loss
        
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
            if self.agg_type == 3: # double salad
                img_embeds = self.img_agg(img_local, token_weights=w_v)
                text_embeds = self.text_agg(text_local, attention_mask, token_weights=w_t)               
                if flip_desc is not None: 
                    text_flip_embeds = self.text_agg(text_flip_local, attention_mask_flip, token_weights=w_t_flip)
                if neg_attr_desc is not None:
                    text_neg_attr_embeds = self.text_agg(text_neg_local, attention_mask_neg, token_weights=w_t_neg)                
            if self.agg_type == 4: # gated attention 
                text_embeds = self.agg(text_local, attention_mask, text_embeds)
            elif self.agg_type == 5 or self.agg_type == 6: #text pooling
                text_embeds = self.agg(text_all_layers, text_embeds, attention_mask)
            else: #salad
                img_embeds = self.agg(img_local, token_weights=w_v)
                text_embeds = self.agg(text_local, attention_mask, token_weights=w_t)
                if flip_desc is not None:
                    text_flip_embeds = self.agg(text_flip_local, attention_mask_flip, token_weights=w_t_flip)
                if neg_attr_desc is not None:
                    text_neg_attr_embeds = self.agg(text_neg_local, attention_mask_neg, token_weights=w_t_neg)    
        
        if self.idf_pooling == 'mean':
            img_embeds_pooled = img_local[:, 1:].mean(dim=1)
        else:
            img_embeds_pooled = self.idf_pooling_layer(img_local[:, 1:])                     
        
        if self.tokens_idf_loss:             
            #img_embeds_pooled = img_all_layers[9][:, 1:, :].mean(dim=1)                
            #img_embeds_pooled = img_all_layers[9][:, 0, :]                  
            tidf_loss = self.tokens_idf_loss * self.tokens_classification_loss(vision_embeddings=img_embeds_pooled, batch_text_ids=text_tokens)
            
        if self.vocab_idf_loss and concept_ids is not None:
            img_features = img_embeds_pooled
            #img_features = img_all_layers[9][:, 1:, :].mean(dim=1)                
            vocab_idf_loss = self.vocab_classification_loss(vision_embeddings=img_features, batch_concept_ids=concept_ids)
            tidf_loss = tidf_loss + self.vocab_idf_loss * vocab_idf_loss
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

        #optimizer.step(closure=optimizer_closure)
          
        # CORRECT (Safe for both single GPU and Multi-GPU DDP):
        self.trainer.strategy.optimizer_step(optimizer, optimizer_idx, optimizer_closure)

            
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

            loss = loss + self.ot_loss * ot_loss + tidf_loss
            
            if self.cmpl:
                sdm_loss = self.sdm_loss(descriptors, ref_embs)
                loss = loss + sdm_loss
            

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
                if self.pos_loss:
                    flat_flip_descs.append(flip_descs[j][i])
                if self.neg_loss:
                    flat_neg_attr_descs.append(neg_attr_descs[j][i])
                flat_color_change_descs.append(color_change_descs[j][i])
                #flat_neg_attr_descs.append(neg_attr_descs[j][i])        

        # Feed forward the batch to the model
        descriptors, text_embeds, text_flip_embeds, text_color_change_embeds, neg_attr_embeds, ot_loss, tidf_loss = self(images, flat_texts, flat_flip_descs, flat_color_change_descs, flat_neg_attr_descs, concepts_ids, labels) 
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
        if self.train_vlm==1:
            # Lightning gives you where THIS checkpoint is being written            
            ckpt_cb = next(
                (cb for cb in self.trainer.checkpoint_callbacks 
                if isinstance(cb, pl.callbacks.ModelCheckpoint)),
                None
            )                      

            # Directory containing the checkpoint file
            ckpt_dir = os.path.dirname(ckpt_cb.dirpath)

            self.vlm_encoder.save_pretrained(ckpt_dir)
            print("Saved PEFT adapter to:", ckpt_dir)
    

    
class SaliencyFilteringModule(nn.Module):
    """
    SFM: Uses attention weights to dynamically select discriminative,
    geographically stable visual patches and filters out transient noise.
    """
    def __init__(self, embed_dim=768, selection_ratio=0.7):
        super().__init__()
        self.selection_ratio = selection_ratio
        # Small network or linear layer to compute saliency/attention scores
        self.score_predictor = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens):
        # patch_tokens shape: (B, num_patches, embed_dim)
        B, N, C = patch_tokens.shape
        num_to_select = int(N * self.selection_ratio)
        
        # Predict importance scores for each token
        scores = self.score_predictor(patch_tokens).squeeze(-1) # (B, num_patches)
        
        # Select top-k highest scoring patch tokens per batch instance
        _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)
        
        # Gather selected tokens
        # Expand indices across the channel dimension
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        filtered_patches = torch.gather(patch_tokens, dim=1, index=gather_indices)
        
        return filtered_patches # (B, num_to_select, embed_dim)


class FineGrainedExtractor(nn.Module):
    def __init__(self, query_dim=512, kv_dim=768, num_queries=16):
        """
        Args:
            query_dim: The unified space dimension (e.g., 512)
            kv_dim: The incoming raw feature dimension from the backbone 
                    (768 for ViT, 512 for Text)
        """
        super().__init__()
        # Queries live in the unified projection space
        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # Self-attention handles only queries (query_dim -> query_dim)
        self.query_self_attn = nn.MultiheadAttention(query_dim, num_heads=8, batch_first=True)
        self.ln1 = nn.LayerNorm(query_dim)
        
        # Cross-attention handles mixed dimensions!
        # embed_dim = Query output dimension
        # kdim/vdim = Incoming input sequence dimension from backbone
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=8, 
            kdim=kv_dim, 
            vdim=kv_dim, 
            batch_first=True
        )
        self.ln2 = nn.LayerNorm(query_dim)
        
    def forward(self, x):
        B = x.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1) # (B, num_queries, query_dim)
        
        # 1. Query Self-Attention (Stays in query_dim space)
        q_attn, _ = self.query_self_attn(q, q, q)
        q = self.ln1(q + q_attn)
        
        # 2. Cross-Attention over token sequences
        # q: (B, num_queries, query_dim)
        # x: (B, seq_len, kv_dim) -> Can be 768 or 512!
        distilled_features, _ = self.cross_attn(q, x, x)
        
        # Output is automatically projected back into query_dim space
        f_distilled = self.ln2(q + distilled_features)
        
        return f_distilled # Shape: (B, num_queries, query_dim)
    

class SDMLoss(nn.Module):
    """
    Similarity Distribution Matching (SDM) Loss.
    Computes the bidirectional KL-divergence between the predicted cross-modal
    similarity distributions and ground-truth distributions within a mini-batch.
    """
    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, visual_embeds, text_embeds, labels=None):
        """
        Args:
            visual_embeds (torch.Tensor): Global visual embeddings, shape (B, D).
            text_embeds (torch.Tensor): Global text embeddings, shape (B, D).
            labels (torch.Tensor, optional): True label identities, shape (B,).
                                            If None, assumes a diagonal matrix (perfect pairing).
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Compute the raw cosine similarity matrix
        # (Assumes visual_embeds and text_embeds are already L2-normalized)
        sim_matrix = torch.matmul(visual_embeds, text_embeds.t())  # Shape: (B, B)

        # 2. Compute predicted distributions (Softmax over rows and columns)
        p_v2t = F.softmax(sim_matrix / self.temperature, dim=-1)   # Visual to Text
        p_t2v = F.softmax(sim_matrix.t() / self.temperature, dim=-1) # Text to Visual

        # 3. Construct Ground-Truth Distribution (q)
        if labels is None:
            # Standard setup: Item i in visual matches item i in text
            q_target = torch.eye(visual_embeds.size(0), device=visual_embeds.device)
        else:
            # Handles duplicate locations / multiple positives in the same batch
            labels = labels.view(-1, 1)
            q_target = torch.eq(labels, labels.t()).float()
            
        # Normalize ground-truth rows so each row sums up to 1 (valid probability distribution)
        q_v2t = q_target / q_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q_t2v = q_target.t() / q_target.t().sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # 4. Compute Bidirectional KL-Divergence Loss
        # Avoid log(0) errors by using a small epsilon clamp
        loss_v2t = F.kl_div(p_v2t.log().clamp(min=-100), q_v2t, reduction='batchmean')
        loss_t2v = F.kl_div(p_t2v.log().clamp(min=-100), q_t2v, reduction='batchmean')

        # Total global loss is the average of both directions
        return (loss_v2t + loss_t2v) / 2.0
    
class LocalSDMLoss(nn.Module):
    """
    Local Similarity Distribution Matching (Local SDM) Loss for hierarchical layers.
    Computes bidirectional KL-divergence over token-sequence matrices by taking
    the mean similarity across local query tokens as defined in the paper.
    """
    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, F_v_layer, F_t_layer, labels=None):
        """
        Args:
            F_v_layer (torch.Tensor): Visual query features for a layer, shape (B, num_queries, D).
            F_t_layer (torch.Tensor): Textual query features for a layer, shape (B, num_queries, D).
            labels (torch.Tensor, optional): Identity labels for checking matches inside a batch.
        """
        B, num_queries, D = F_v_layer.shape

        # 1. Ensure vectors are L2-normalized across the feature dimension (D)
        F_v_norm = F.normalize(F_v_layer, p=2, dim=-1)  # (B, num_queries, D)
        F_t_norm = F.normalize(F_t_layer, p=2, dim=-1)  # (B, num_queries, D)

        # 2. Compute token-to-token similarity matrix between ALL batch pairs
        # Reshape to combine batch and query dimensions for efficient matrix multiplication
        # F_v_flat: (B * num_queries, D)
        F_v_flat = F_v_norm.view(-1, D)
        # F_t_flat: (B * num_queries, D)
        F_t_flat = F_t_norm.view(-1, D)
        
        # Total token similarity matrix shape: (B * num_queries, B * num_queries)
        token_sim_matrix = torch.matmul(F_v_flat, F_t_flat.t())
        
        # Reshape back to separate individual batch element comparisons
        # Shape: (B, num_queries, B, num_queries)
        batch_sim_tensor = token_sim_matrix.view(B, num_queries, B, num_queries)
        
        # 3. Compute the mean similarity across all local query tokens for each pair (Eq. Page 6)
        # Permute to (B, B, num_queries, num_queries) so we can average out the token dimensions
        batch_sim_tensor = batch_sim_tensor.permute(0, 2, 1, 3)
        # Average over both the visual and textual local query token axes
        sim_matrix = batch_sim_tensor.mean(dim=[-2, -1])  # Shape: (B, B)

        # 4. Convert mean token similarities to predicted probability distributions
        p_v2t = F.softmax(sim_matrix / self.temperature, dim=-1)   # Visual to Text
        p_t2v = F.softmax(sim_matrix.t() / self.temperature, dim=-1) # Text to Visual

        # 5. Define or construct target distributions (q)
        if labels is None:
            q_target = torch.eye(B, device=F_v_layer.device)
        else:
            labels = labels.view(-1, 1)
            q_target = torch.eq(labels, labels.t()).float()

        q_v2t = q_target / q_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        q_t2v = q_target.t() / q_target.t().sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # 6. Compute Bidirectional KL-Divergence Loss for this layer
        loss_v2t = F.kl_div(p_v2t.log().clamp(min=-100), q_v2t, reduction='batchmean')
        loss_t2v = F.kl_div(p_t2v.log().clamp(min=-100), q_t2v, reduction='batchmean')

        return (loss_v2t + loss_t2v) / 2.0
    
