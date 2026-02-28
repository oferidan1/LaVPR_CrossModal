import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
from torch import nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
import os
from model.blip_model import BlipForImageTextRetrievalWrapper
from transformers import BlipProcessor, BlipModel
from transformers import AutoModel, AutoProcessor
import open_clip
from model.salad import SALAD

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
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False,
                model_name='Salesforce/blip-itm-base-coco',
                embeds_dim=256,
                is_freeze_text=True,
                is_trainable_text_encoder=True,
                cross_modal=0,
                lora_all_linear=False,
                lora_target_modules=None,
                lora_r=64,                
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

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.faiss_gpu = faiss_gpu

        self.cross_modal = cross_modal
        self.lora_all_linear = lora_all_linear
        self.lora_target_modules = lora_target_modules
        self.lora_r = lora_r
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
       
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        
        self.embeds_dim = embeds_dim        
        self.is_trainable_text_encoder = is_trainable_text_encoder
        

        if cross_modal == 4: # contrastive loss for cross modal retrieval
            self.contrastive_logit_scale = nn.Parameter(0.07*torch.ones([])) 
            self.contrastive_loss = utils.losses.contrastive_loss_cross_modal
            self.miner = None                            
        
        self.agg = SALAD(num_channels=embeds_dim)
                
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
        if self.is_trainable_text_encoder:                
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

                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # For linear layers, use Kaiming uniform initialization
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            # For biases, it's common to initialize them to zero
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)        
    
    
    # the forward pass of the lightning model
    def forward(self, img, text):
        text_embeds = None

        if 'blip' in self.model_name:
            img_embeds = self.text_encoder.encode_image(img)            
        elif 'clip' in self.model_name or 'siglip' in self.model_name:
            img_embeds = self.text_encoder.get_image_features(pixel_values=img)
        elif 'eva' in self.model_name:            
            img_embeds = self.text_encoder.encode_image(img)
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)            
            
        img_embeds = self.agg(img_embeds)
        attention_mask = None        

        if 'blip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True)
            text_tokens = text_inputs.input_ids.to(img.device)
            attention_mask = text_inputs['attention_mask'].to(img.device)                
            text_embeds = self.text_encoder.encode_text(input_ids=text_tokens, attention_mask=attention_mask)    
            #text_embeds= text_embeds[:, 0]        
        elif 'clip' in self.model_name or 'siglip' in self.model_name:
            text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_text_length)
            text_tokens = text_inputs.input_ids.to(img.device)
            attention_mask = None
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'].to(img.device)                
            text_embeds = self.text_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.device)            
            text_embeds = self.text_encoder.encode_text(text_tokens)    
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)       
            
        text_embeds = self.agg(text_embeds, attention_mask)

        return img_embeds, text_embeds, 
    
    
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
    def loss_function(self, descriptors, labels, text_embeds):
        
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:                        
            ref_labels = labels.clone()
            miner_outputs = self.miner(descriptors, labels, ref_emb=text_embeds, ref_labels=ref_labels)     
            loss = self.loss_fn(descriptors, labels, indices_tuple=miner_outputs, ref_emb=text_embeds, ref_labels=ref_labels)            

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
        places, labels, texts = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        
        # reshape places and labels
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        flat_texts = []
        for i in range(BS):
            for j in range(N):
                flat_texts.append(texts[j][i])

        # Feed forward the batch to the model
        descriptors, text_embeds = self(images, flat_texts) # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels, text_embeds) # Call the loss_function we defined above
        
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
        descriptors, text_embeds = self(places, texts)
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
        if self.is_trainable_text_encoder:
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
    

    
class CLSReweightingPooler(nn.Module):
    """
    Combines the CLS token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = hidden_states[:, 0]  # [B, H]

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   
    

def mean_pooling(token_embeddings, attention_mask):
    # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum of the attention mask
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
    # Mean Pooling
    return sum_embeddings / sum_mask
    
class MeanReweightingPooler(nn.Module):
    """
    Combines the Mean token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = mean_pooling(hidden_states, mask)  # [B, H]        

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   
    

