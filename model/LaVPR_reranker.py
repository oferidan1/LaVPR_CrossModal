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
from utils.losses import MultiSimilarityLossForPairwiseScores

class LaVPR_reranker(pl.LightningModule):
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
                faiss_gpu=False,
                model_name='Salesforce/blip-itm-base-coco',
                embeds_dim=256,
                freeze_vlm=True,
                train_vlm=False,
                pos_loss=0,
                neg_loss=0,                
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
        self.loss_fn = MultiSimilarityLossForPairwiseScores()
        
        self.save_hyperparameters() # write hyperparams into a file        
     
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
       
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                
        
        self.embeds_dim = embeds_dim        
        self.train_vlm = train_vlm
        self.pos_loss = pos_loss
        self.neg_loss = neg_loss                
        
        self.classifier = nn.Sequential(
            nn.Linear(embeds_dim * 2, embeds_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embeds_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1) # Directly outputs the matching logit
        )
                
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
                        
        if freeze_vlm:
            # Freeze text encoder parameters
            for param in self.vlm_encoder.parameters():
                param.requires_grad = False                              
            self.vlm_encoder.eval()                
        
                
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
            img_output = self.vlm_encoder.get_image_features(pixel_values=img)
            #img_local = self.vlm_encoder.base_model.model.vision_model.head(img_output.last_hidden_state)                     
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
            text_output = self.vlm_encoder.get_text_features(input_ids=text_tokens, attention_mask=attention_mask)
            #text_local = self.vlm_encoder.base_model.model.text_model.head(text_output.last_hidden_state)
            text_local = text_output.last_hidden_state
            text_embeds = text_output.pooler_output                
        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.my_device)            
            text_embeds = self.vlm_encoder.encode_text(text_tokens)    
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
                
        # Concatenate along feature dimension
        fused = torch.cat([text_embeds, img_embeds], dim=-1) # [B, D*2]
        
        # Output logit score per pair
        scores = self.classifier(fused).squeeze(-1) # [B]
        return scores
    
    
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
    def loss_function(self, score_matrix, labels):        
        
        loss = self.loss_fn(score_matrix, labels)
        
        # Calculate Batch Recall@1 Accuracy Stat
        with torch.no_grad():
            # Make a copy of the score matrix to manipulate safely
            eval_scores = score_matrix.clone()
            
            # Exclude diagonal self-identity entries by setting them to a large negative number
            # This prevents the text query from trivially matching its own source image entry
            eval_scores.diagonal().fill_(-1e9)
            
            # Get the index of the highest scoring candidate image for each text query row
            top1_indices = torch.argmax(eval_scores, dim=-1) # Shape: [B]
            
            # Map those predicted indices back to their structural location labels
            predicted_labels = labels[top1_indices] # Shape: [B]
            
            # A prediction is correct if its location label matches the row's query label
            correct_predictions = (predicted_labels == labels).float()
            batch_acc = correct_predictions.mean().item() * 100.0 # Percentage float
        
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
        scores = self(images, flat_texts, flat_flip_descs, flat_color_change_descs, flat_neg_attr_descs, concepts_ids, labels) 
        loss = self.loss_function(scores, labels)
        
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
        descriptors, text_embeds, _, _, _,  = self(places, texts)
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
    
