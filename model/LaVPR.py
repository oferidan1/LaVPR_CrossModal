import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer
import utils
import numpy as np
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
from model.tokens_classify_loss import TokensClassificationLoss, HierarchicalTokensLoss, VocabClassificationLoss, FILIPLoss
from model.pooling_cm import TextGatedAttentionPooler, GeMPooling1D, AttentionGatedPatchPooler, SpatialLayoutPooler, MultiLayerAttentionTextPooler, ResidualTextPooler


def get_local_dims(model_name):
    """(vision_hidden_dim, text_hidden_dim) of the per-token features."""
    if 'blip' in model_name:
        return 768, 768
    if 'llm2clip' in model_name:
        return 1024, 1280
    if 'clip' in model_name:
        return 768, 512
    if 'siglip' in model_name:
        return 768, 768
    if 'eva' in model_name:
        return 768, 512
    return 768, 512


class LaVPR(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.
    """

    def __init__(self,

                # ---- Train hyperparameters
                lr=0.03,
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                epochs=10,

                # ----- Loss
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
                idf_pooling='mean',
                vocab_idf_loss=0.0,
                vocab_path=None,
                image_idf_path=None,
                vocab_grad_scale=0.05,
                cls_adapter=0,
                # ----- FILIP late interaction
                filip=0.0,
                filip_queue=0,
                filip_chunk=8,
                filip_use_idf=True,
                global_token=True,       # False + filip>0 -> FILIP-only retrieval
                filip_db_chunk=256,      # gallery chunk size for val-time MaxSim
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

        self.save_hyperparameters()  # write hyperparams into a file

        if 'WeightedMultiSimilarityLoss' in loss_name:
            self.loss_fn = WeightedMultiSimilarityLossCM()
        else:
            self.loss_fn = utils.get_loss(loss_name)

        if ot_loss:
            self.local_ot_loss = LocalOTLoss()
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

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

        # SigLIP's vision tower uses attention pooling and has NO CLS token,
        # every other backbone here prepends one at index 0.
        self.has_cls = 'siglip' not in model_name

        if self.tokens_idf_loss == 1:
            self.tokens_classification_loss = TokensClassificationLoss(vision_dim=768, vocab_size=vocab_size, idf_path=self.tokens_idf_file, grad_scale=idf_grad_scale, cls_adapter=cls_adapter)
        elif self.tokens_idf_loss == 2:
            self.tokens_classification_loss = HierarchicalTokensLoss()

        if self.vocab_idf_loss:
            self.vocab_classification_loss = VocabClassificationLoss(vision_dim=768, vocab_path=vocab_path, image_idf_path=image_idf_path, grad_scale=vocab_grad_scale, cls_adapter=cls_adapter)

        # ---- FILIP late interaction
        self.filip = filip
        self.global_token = global_token
        self.filip_db_chunk = filip_db_chunk        
        # FILIP replaces the global vector entirely, as in the original paper:
        # token-wise MaxSim IS the retrieval similarity, over the full gallery.
        self.filip_retrieval = (not global_token) and filip > 0
        if filip:
            v_dim, t_dim = get_local_dims(model_name)
            self.filip_loss = FILIPLoss(
                text_dim=t_dim,
                vision_dim=v_dim,
                joint_dim=embeds_dim,
                chunk=filip_chunk,
                idf_path=tokens_idf_file,
                vocab_size=vocab_size,
                use_idf=filip_use_idf,
                queue_size=filip_queue
            )

        if cross_modal == 4:  # contrastive loss for cross modal retrieval
            self.contrastive_logit_scale = nn.Parameter(0.07 * torch.ones([]))
            self.contrastive_loss = utils.losses.contrastive_loss_cross_modal
            self.miner = None

        if idf_pooling == 'gem':
            self.idf_pooling_layer = GeMPooling1D()
        elif idf_pooling == 'attention':
            self.idf_pooling_layer = AttentionGatedPatchPooler()
        elif idf_pooling == 'spatial':
            self.idf_pooling_layer = SpatialLayoutPooler()

        # init weight of linear layers but not the pretrained backbones
        self.apply(self._init_weights)
        # the global kaiming/relu init above is wrong for the FILIP projections
        if self.filip and hasattr(self.filip_loss, 'reset_parameters'):
            self.filip_loss.reset_parameters()

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
            self.vlm_encoder, _, self.processor = open_clip.create_model_and_transforms(model_name.upper(), pretrained='merged2b_s8b_b131k')
            self.tokenizer = open_clip.get_tokenizer(model_name)

        if is_freeze_text:
            for param in self.vlm_encoder.parameters():
                param.requires_grad = False

        if self.train_vlm == 1:
            lora_targets = lora_target_modules
            if lora_all_linear:
                lora_targets = "all-linear"

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.1,
                target_modules=lora_targets,
                task_type=TaskType.SEQ_CLS,
                use_rslora=True,
                bias="none",
            )

            if 'llm2clip' in model_name:
                self.llm_encoder = get_peft_model(self.llm_encoder, lora_config)
            else:
                self.vlm_encoder = get_peft_model(self.vlm_encoder, lora_config)

        elif is_freeze_text:
            self.vlm_encoder.eval()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
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

    def get_img_patches(self, img_local):
        """Strip the CLS token when the backbone has one."""
        return img_local[:, 1:] if self.has_cls else img_local

    def filip_tokens(self, img_local, text_local, t_mask, text_tokens):
        """Project + L2-normalise per-token features into the FILIP joint space.

        Returns (v, t, w): patch tokens [B,m,D], text tokens [B,n,D] and the
        normalised per-text-token weights [B,n] (0 on padding). Used by both
        the training loss and validation retrieval so they cannot drift apart.
        """
        h = self.filip_loss
        v = self.get_img_patches(img_local)
        v = h.v_norm(v) if getattr(h, 'v_norm', None) is not None else v
        v = F.normalize(h.vision_proj(v), dim=-1)
        te = h.t_norm(text_local) if getattr(h, 't_norm', None) is not None else text_local
        t = F.normalize(h.text_proj(te), dim=-1)
        w = h._weights(text_tokens, t_mask.to(t.dtype))
        return v, t, w

    def encode_image(self, img):
        img_embeds = None
        img_local = None
        img_all_layers = None
        if 'blip' in self.model_name:
            img_local = self.vlm_encoder.encode_image(img)
            img_embeds = img_local[:, 0]
        elif 'llm2clip' in self.model_name:
            img_output = self.vlm_encoder.vision_model(pixel_values=img.to(self.vlm_encoder.dtype), output_hidden_states=True)
            img_local = self.vlm_encoder.visual_projection(img_output.last_hidden_state)
            img_embeds = self.vlm_encoder.visual_projection(img_output.pooler_output)
            img_all_layers = img_output.hidden_states
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        elif 'clip' in self.model_name:
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
            vision_outputs = self.vlm_encoder.vision_model(
                pixel_values=img,
                output_hidden_states=True
            )
            img_local = vision_outputs.last_hidden_state
            img_all_layers = vision_outputs.hidden_states
            img_embeds = self.vlm_encoder.vision_model.head(vision_outputs.last_hidden_state)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

        elif 'eva' in self.model_name:
            img_local = self.vlm_encoder.visual.trunk.forward_features(img)
            if isinstance(img_local, dict):
                img_local = img_local['x']

            img_embeds = img_local[:, 0, :]
            if hasattr(self.vlm_encoder.visual.trunk, 'norm') and not isinstance(img_local, dict):
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
            text_embeds = text_local[:, 0]
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

            text_outputs = self.vlm_encoder.text_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_local = text_outputs.last_hidden_state
            text_all_layers = text_outputs.hidden_states
            pooled_text = text_outputs.pooler_output
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

            text_outputs = self.vlm_encoder.text_model(
                input_ids=text_tokens,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            text_local = text_outputs.last_hidden_state
            text_embeds = text_outputs.pooler_output
            text_all_layers = text_outputs.hidden_states
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        elif 'eva' in self.model_name:
            text_tokens = self.tokenizer(text).to(self.my_device)
            text_embeds = self.vlm_encoder.encode_text(text_tokens)

            x = self.vlm_encoder.text.token_embedding(text_tokens)
            x = x + self.vlm_encoder.text.positional_embedding

            _, intermediates = self.vlm_encoder.text.transformer.forward_intermediates(
                x=x,
                attn_mask=self.vlm_encoder.text.attn_mask,
                indices=[-1]
            )

            text_local = intermediates[-1]
            # open_clip transformers are [Sequence, Batch, Dim]; normalise to
            # [Batch, Sequence, Dim] so downstream losses see the same layout.
            if text_local.shape[0] != text_tokens.shape[0] and text_local.shape[1] == text_tokens.shape[0]:
                text_local = text_local.permute(1, 0, 2).contiguous()
            text_local = self.vlm_encoder.text.ln_final(text_local)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds, text_local, attention_mask, text_tokens, text_all_layers

    # the forward pass of the lightning model
    def forward(self, img, text, flip_desc=None, color_change_desc=None, neg_attr_desc=None, concept_ids=None, labels=None):
        text_flip_embeds = None
        text_neg_attr_embeds = None

        text_flip_local = None
        text_neg_local = None
        attention_mask_flip = None
        attention_mask_neg = None

        img_embeds, img_local, img_all_layers = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens, text_all_layers = self.encode_text(text)
        if self.pos_loss:
            if flip_desc is not None:
                text_flip_embeds, text_flip_local, attention_mask_flip, text_flip_tokens, text_flip_all_layers = self.encode_text(flip_desc)
        if self.neg_loss:
            if neg_attr_desc is not None:
                text_neg_attr_embeds, text_neg_local, attention_mask_neg, text_neg_tokens, text_neg_all_layers = self.encode_text(neg_attr_desc)

        tidf_loss = 0.0
        filip_loss = 0.0
        t_mask_filip = None

        img_patches = self.get_img_patches(img_local) if img_local is not None else None

        if img_patches is not None:
            if self.idf_pooling == 'mean':
                img_embeds_pooled = img_patches.mean(dim=1)
            else:
                img_embeds_pooled = self.idf_pooling_layer(img_patches)
        else:
            img_embeds_pooled = None

        if self.tokens_idf_loss:
            tidf_loss = self.tokens_idf_loss * self.tokens_classification_loss(vision_embeddings=img_embeds_pooled, batch_text_ids=text_tokens)

        if self.vocab_idf_loss and concept_ids is not None:
            img_features = img_embeds_pooled
            vocab_idf_loss = self.vocab_classification_loss(vision_embeddings=img_features, batch_concept_ids=concept_ids)
            tidf_loss = tidf_loss + self.vocab_idf_loss * vocab_idf_loss

        # ---- FILIP token-wise maximum similarity
        # the mask is built whenever filip is on, so validation (labels=None)
        # can still project tokens for MaxSim retrieval
        if self.filip and text_local is not None:
            t_mask_filip = attention_mask if attention_mask is not None else (text_tokens != 0)
            if labels is not None and img_patches is not None:
                filip_loss = self.filip_loss(img_patches, text_local, t_mask_filip,
                                             text_tokens, labels)

        return (img_embeds, text_embeds, text_flip_embeds, text_neg_attr_embeds,
                tidf_loss, filip_loss, img_local, text_local, text_tokens, t_mask_filip)

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

    def optimizer_step(self, epoch, batch_idx,
                       optimizer, optimizer_idx, optimizer_closure,
                       on_tpu, using_native_amp, using_lbfgs):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        self.trainer.strategy.optimizer_step(optimizer, optimizer_idx, optimizer_closure)

    def loss_function(self, img_embeds, labels, text_embeds, text_flip_embeds, text_neg_attr_embeds,
                      tidf_loss=0.0, filip_loss=0.0):

        # ---- FILIP-only: no global vector, so no metric loss on img/text embeds
        if self.filip_retrieval:
            loss = self.filip * filip_loss + tidf_loss
            self.batch_acc.append(0.0)
            self.log('filip', float(filip_loss), prog_bar=True, logger=True)
            self.log('b_acc', 0.0, prog_bar=True, logger=True)
            return loss

        if self.cross_modal == 5:
            desc_all = torch.cat([img_embeds, text_embeds], dim=0)
            labels_all = torch.cat([labels, labels], dim=0)
            miner_outputs = self.miner(desc_all, labels_all)
            loss = self.loss_fn(desc_all, labels_all, indices_tuple=miner_outputs)
            nb_samples = desc_all.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

            loss = loss + self.filip * filip_loss

        elif self.miner is not None:
            text_labels = labels.clone()
            text_embeds_all = text_embeds

            if self.pos_loss:
                text_embeds_all = torch.cat([text_embeds, text_flip_embeds], dim=0)
                text_labels = torch.cat([text_labels, labels], dim=0)

            if self.neg_loss:
                text_embeds_all = torch.cat([text_embeds_all, text_neg_attr_embeds], dim=0)
                text_labels = torch.cat([text_labels, labels + 10**8], dim=0)

            # T2I: text anchors, image refs
            miner_outputs = self.miner(text_embeds_all, text_labels, ref_emb=img_embeds, ref_labels=labels)

            if self.dynamic_gamma:
                with torch.no_grad():
                    sim_matrix = torch.matmul(img_embeds, text_embeds_all.T)
                    pos_mask = (labels.unsqueeze(1) == text_labels.unsqueeze(0))
                    pos_similarities = sim_matrix[pos_mask]

                    if pos_similarities.numel() > 0:
                        mean_pos_sim = pos_similarities.mean().item()
                        self.loss_fn.base = max(0.35, min(mean_pos_sim, 0.45))
                    else:
                        self.loss_fn.base = 0.4

            loss = self.loss_fn(text_embeds_all, text_labels, indices_tuple=miner_outputs,
                                ref_emb=img_embeds, ref_labels=labels)

            if self.latent_mixup > 0:
                a1, p, a2, n = miner_outputs
                if len(a2) > 0:
                    BS = img_embeds.shape[0]
                    # anchors (a2) index text_embeds_all, refs (n) index img_embeds
                    Vt = text_embeds_all[a2]
                    Vi_neg = img_embeds[n]
                    Vi_pos = img_embeds[a2 % BS]
                    alpha = torch.rand(len(a2), 1, device=img_embeds.device)
                    V_prime = torch.nn.functional.normalize(alpha * Vi_pos + (1 - alpha) * Vi_neg, p=2, dim=-1)
                    score1 = (Vt * V_prime).sum(dim=-1)
                    score2 = alpha.squeeze(-1) * (Vt * Vi_pos).sum(dim=-1) + (1 - alpha.squeeze(-1)) * (Vt * Vi_neg).sum(dim=-1)
                    mixup_loss = torch.nn.functional.mse_loss(score1, score2)
                    loss = loss + self.latent_mixup * mixup_loss

            if self.unimodal_loss > 0:
                miner_outputs_txt = self.miner(text_embeds_all, text_labels)
                txt_loss = self.loss_fn(text_embeds_all, text_labels, indices_tuple=miner_outputs_txt)
                loss = loss + self.unimodal_loss * txt_loss

            loss = loss + tidf_loss + self.filip * filip_loss

            nb_samples = text_embeds_all.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            if self.cross_modal == 4:
                logit_scale = self.contrastive_logit_scale
                loss = self.contrastive_loss(img_embeds, text_embeds, logit_scale)
            else:
                loss = self.loss_fn(img_embeds, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                loss, batch_acc = loss
            loss = loss + self.filip * filip_loss

        if self.filip:
            self.log('filip', float(filip_loss), prog_bar=False, logger=True)

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels, texts, flip_descs, color_change_descs, neg_attr_descs, concepts_ids = batch

        BS, N, ch, h, w = places.shape

        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        concepts_ids = concepts_ids.view(BS * N, -1)

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

        (descriptors, text_embeds, text_flip_embeds, neg_attr_embeds,
         tidf_loss, filip_loss, img_local, text_local, text_tokens, t_mask) = self(
            images, flat_texts, flat_flip_descs, flat_color_change_descs,
            flat_neg_attr_descs, concepts_ids, labels)

        loss = self.loss_function(descriptors, labels, text_embeds, text_flip_embeds,
                                  neg_attr_embeds, tidf_loss, filip_loss)

        self.log('loss', loss.item(), logger=True)

        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        (img_embeds, text_embeds, _, _, _, _,
         img_local, text_local, text_tokens, t_mask) = self(places, texts)

        ret_dict = {'descriptors': img_embeds.detach().cpu(),
                    'text_embeds': text_embeds.detach().cpu()}

        if self.filip_retrieval:
            v, t, w = self.filip_tokens(img_local, text_local, t_mask, text_tokens)
            ret_dict['v_tok'] = v.detach().half().cpu()
            ret_dict['t_tok'] = t.detach().half().cpu()
            ret_dict['t_w'] = w.detach().half().cpu()
        return ret_dict

    def _filip_recalls(self, v_all, t_all, w_all, num_references, num_queries,
                       positives, k_values):
        """Full-gallery token-wise MaxSim retrieval, as in the FILIP paper.

        Database patch tokens stay on CPU; only a chunk moves to GPU at a time.
        """
        db_v = v_all[:num_references]
        hits = np.zeros(len(k_values))
        topk = min(max(k_values), num_references)

        for q in range(num_queries):
            gi = num_references + q
            m = w_all[gi] > 0
            if m.sum() == 0:
                continue
            t = t_all[gi][m].to(self.my_device)
            w = w_all[gi][m].to(self.my_device)

            sc = torch.empty(num_references, device=self.my_device, dtype=t.dtype)
            for s in range(0, num_references, self.filip_db_chunk):
                vv = db_v[s:s + self.filip_db_chunk].to(self.my_device)
                sim = torch.einsum('nd,bmd->bnm', t, vv)
                sc[s:s + vv.shape[0]] = (sim.max(dim=-1).values * w).sum(-1)
                del sim, vv

            top = torch.topk(sc, topk).indices.cpu().numpy()
            for ki, kk in enumerate(k_values):
                if np.any(np.isin(top[:kk], positives[q])):
                    hits[ki:] += 1
                    break

        return {k: 100.0 * h / num_queries for k, h in zip(k_values, hits)}

    def validation_epoch_end(self, val_step_outputs):
        """Descriptors come back in dataset order: references then queries,
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        k_values = [1, 5, 10, 15, 20, 50, 100]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
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
                num_references = val_dataset.num_db
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.pIdx
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            if self.filip_retrieval:
                v_all = torch.cat([d['v_tok'] for d in val_step_outputs[i]], dim=0)
                t_all = torch.cat([d['t_tok'] for d in val_step_outputs[i]], dim=0)
                w_all = torch.cat([d['t_w'] for d in val_step_outputs[i]], dim=0)

                pitts_dict = self._filip_recalls(v_all, t_all, w_all,
                                                 num_references, num_queries,
                                                 positives, k_values)
                print(f"{val_set_name} [FILIP MaxSim, full gallery]: " +
                      ", ".join(f"R@{k}: {v:.1f}" for k, v in pitts_dict.items()))
                del v_all, t_all, w_all

            else:
                r_list = feats[: num_references]
                q_list = feats[num_references:]

                if self.cross_modal:
                    q_text_list = text_feats[num_references:]
                    pitts_dict = utils.get_validation_recalls(
                        r_list=r_list, q_list=q_text_list, k_values=k_values,
                        gt=positives, print_results=True,
                        dataset_name=val_set_name, faiss_gpu=self.faiss_gpu)
                else:
                    pitts_dict = utils.get_validation_recalls(
                        r_list=r_list, q_list=q_list, k_values=k_values,
                        gt=positives, print_results=True,
                        dataset_name=val_set_name, faiss_gpu=self.faiss_gpu)
                del r_list, q_list

            del feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

    def on_save_checkpoint(self, checkpoint):
        if self.train_vlm == 1:
            ckpt_cb = next(
                (cb for cb in self.trainer.checkpoint_callbacks
                 if isinstance(cb, pl.callbacks.ModelCheckpoint)),
                None
            )
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
        self.score_predictor = nn.Linear(embed_dim, 1)

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        num_to_select = int(N * self.selection_ratio)

        scores = self.score_predictor(patch_tokens).squeeze(-1)
        _, topk_indices = torch.topk(scores, k=num_to_select, dim=-1)

        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, C)
        filtered_patches = torch.gather(patch_tokens, dim=1, index=gather_indices)

        return filtered_patches

