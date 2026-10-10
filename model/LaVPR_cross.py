import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
import utils
import numpy as np
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModel
import os
from model.weighted_ms_loss import WeightedMultiSimilarityLossCM
from model.pooling_cm import CLSReweightingPooler


# Native output dims of each backbone — no guessing at runtime
_IMAGE_ENCODER_DIM = {
    'dinov2': 768,   # dinov2_vitb14  (ViT-B, patch 14)
    'sela':   2048,  # SelaVPR++ w/ dinov2-base + GeM (ResNet-style head)
}
_TEXT_ENCODER_DIM = {
    'bge-l': 1024,  # BAAI/bge-large-en-v1.5  (BERT-large)
    'bge-b':  768,  # BAAI/bge-base-en-v1.5   (BERT-base)
}


def _text_hidden_dim(model_name: str) -> int:
    """Return the CLS hidden dim for known BGE checkpoints."""
    name = model_name.lower()
    for key, dim in _TEXT_ENCODER_DIM.items():
        if key in name:
            return dim
    raise ValueError(
        f"Unknown text model '{model_name}'. "
        f"Add its hidden dim to _TEXT_ENCODER_DIM or pass text_hidden_dim explicitly."
    )


class LaVPR_cross(pl.LightningModule):
    """Cross-modal VPR: BGE text encoder aligned with DINOv2/SelaVPR image encoder.

    Both backbones output different hidden dims (e.g. 1024 vs 768).  A learned
    linear projection on each side maps them to a shared *embeds_dim* before the
    MS loss, so similarity is computed in a common space.

    Architecture
    ------------
    image  →  image_encoder (DINOv2/SelaVPR)  →  image_proj (D_img → embeds_dim)  →  L2-norm
    text   →  text_encoder  (BGE)              →  text_proj  (D_txt → embeds_dim)  →  L2-norm
    loss   →  MultiSimilarityLoss between image and text embeddings (I2T direction)
    """

    def __init__(
        self,
        # ---- Encoder names
        text_model_name: str = 'BAAI/bge-large-en-v1.5',
        image_model_name: str = 'dinov2',   # 'dinov2' or 'sela'
        embeds_dim: int = 512,              # shared projection dim

        # ---- Optimiser
        lr: float = 2e-5,
        optimizer: str = 'adamw',
        weight_decay: float = 1e-3,
        momentum: float = 0.9,
        warmpup_steps: int = 500,
        milestones: list = None,
        lr_mult: float = 0.3,
        epochs: int = 10,

        # ---- Loss
        loss_name: str = 'MultiSimilarityLoss',
        miner_name: str = 'MultiSimilarityMiner',
        miner_margin: float = 0.1,
        faiss_gpu: bool = False,

        # ---- Encoder training flags
        freeze_image: bool = False,
        freeze_text: bool = False,
        train_text: int = 1,   # 0=frozen, 1=LoRA, 2=full fine-tune
        train_image: int = 1,  # 0=frozen, 1=LoRA, 2=full fine-tune
        lora_all_linear: bool = False,
        lora_target_modules: list = None,
        lora_r: int = 64,

        # ---- Aux losses
        unimodal_loss: float = 0.0,
        pos_loss: int = 0,

        # ---- Optional modules
        is_llp: int = 0,
    ):
        super().__init__()

        if milestones is None:
            milestones = [5, 10, 15]

        self.text_model_name = text_model_name.lower()
        self.image_model_name = image_model_name.lower()
        self.embeds_dim = embeds_dim

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

        self.train_text = train_text
        self.train_image = train_image
        self.lora_all_linear = lora_all_linear
        self.lora_target_modules = lora_target_modules
        self.lora_r = lora_r

        self.unimodal_loss = unimodal_loss
        self.pos_loss = pos_loss
        self.is_llp = is_llp

        self.save_hyperparameters()

        # ---- Loss / miner
        if 'WeightedMultiSimilarityLoss' in loss_name:
            self.loss_fn = WeightedMultiSimilarityLossCM()
        else:
            self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)

        self.batch_acc = []
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Image encoder
        img_key = self.image_model_name  # 'dinov2' or 'sela'
        if img_key not in _IMAGE_ENCODER_DIM:
            raise ValueError(f"Unknown image_model_name '{image_model_name}'. "
                             f"Choose one of: {list(_IMAGE_ENCODER_DIM)}")
        img_hidden_dim = _IMAGE_ENCODER_DIM[img_key]

        if 'dinov2' in self.image_model_name:
            self.image_encoder = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitb14')
        elif 'sela' in self.image_model_name:
            self.image_encoder = torch.hub.load(
                'Lu-Feng/SelaVPRplusplus', 'SelaVPRplusplus',
                backbone='dinov2-base', aggregation='gem',
                hashing=False, rerank=False)

        # ---- Text encoder (BGE)
        txt_hidden_dim = _text_hidden_dim(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, attn_implementation="sdpa")

        # ---- Projection heads: map each backbone to shared embeds_dim
        #      BGE-large: 1024 → embeds_dim
        #      DINOv2-B / SelaVPR: 768 → embeds_dim
        self.image_proj = nn.Linear(img_hidden_dim, embeds_dim, bias=False)
        self.text_proj  = nn.Linear(txt_hidden_dim, embeds_dim, bias=False)

        # ---- Optional LLP pooler (operates in text hidden space, before proj)
        if is_llp:
            self.llp = CLSReweightingPooler(txt_hidden_dim)

        # ---- Freeze / LoRA
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        if train_image == 1:
            if 'dinov2' in self.image_model_name:
                self.image_encoder = self._init_lora(
                    self.image_encoder, lora_target_modules, lora_all_linear, lora_r)
            else:
                # sela: LoRA unsupported — fall back to full fine-tune
                for p in self.image_encoder.parameters():
                    p.requires_grad = True
        elif train_image == 2:
            # explicit full fine-tune
            for p in self.image_encoder.parameters():
                p.requires_grad = True
        elif freeze_image:
            self.image_encoder.eval()

        if train_text == 1:
            self.text_encoder = self._init_lora(
                self.text_encoder, lora_target_modules, lora_all_linear, lora_r)
        elif train_text == 2:
            for p in self.text_encoder.parameters():
                p.requires_grad = True
        elif freeze_text:
            self.text_encoder.eval()

        # ---- Init projection heads (kaiming); leave backbone weights alone
        nn.init.kaiming_uniform_(self.image_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.text_proj.weight,  mode='fan_in', nonlinearity='relu')

    # ------------------------------------------------------------------
    # LoRA helper (static — no self needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _init_lora(encoder, lora_targets, lora_all_linear, lora_r):
        """Wrap *encoder* with a PEFT LoRA adapter and return it."""
        if lora_all_linear:
            lora_targets = "all-linear"
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.1,
            target_modules=lora_targets,
            task_type=TaskType.FEATURE_EXTRACTION,
            use_rslora=True,
            bias="none",
        )
        return get_peft_model(encoder, lora_config)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_image(self, img):
        """
        Returns
        -------
        img_embeds : (B, embeds_dim)  L2-normalised, projected
        img_local  : (B, N, D_img)   raw patch tokens, or None for SelaVPR
        """
        if 'dinov2' in self.image_model_name:
            out = self.image_encoder.forward_features(img)
            img_local = out['x_norm_patchtokens']    # (B, N, 768)
            img_raw   = out['x_norm_clstoken']       # (B, 768)
        elif 'sela' in self.image_model_name:
            # SelaVPR is not a HF model — call base_model directly to bypass
            # any PEFT wrapper that would inject transformer kwargs
            encoder = (self.image_encoder.base_model
                       if hasattr(self.image_encoder, 'base_model')
                       else self.image_encoder)
            img_raw   = encoder(img)                 # (B, 2048)
            img_local = None

        img_embeds = F.normalize(self.image_proj(img_raw), p=2, dim=1)  # (B, embeds_dim)
        return img_embeds, img_local

    def encode_text(self, text):
        """
        Returns
        -------
        text_embeds   : (B, embeds_dim)  L2-normalised, projected
        text_local    : (B, L, D_txt)   full last hidden state
        attention_mask: (B, L)
        text_tokens   : tokenizer output dict
        """
        text_tokens = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        model_output = self.text_encoder(
            **text_tokens, output_hidden_states=False, return_dict=True)

        text_local = model_output.last_hidden_state          # (B, L, D_txt)

        if self.is_llp:
            text_raw = self.llp(text_local, mask=text_tokens['attention_mask'])
        else:
            text_raw = text_local[:, 0]                      # CLS token (B, D_txt)

        text_embeds = F.normalize(self.text_proj(text_raw), p=2, dim=1)  # (B, embeds_dim)
        return text_embeds, text_local, text_tokens['attention_mask'], text_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img, text, flip_desc=None, labels=None):
        """
        Returns
        -------
        img_embeds       : (B, embeds_dim)
        text_embeds      : (B, embeds_dim)
        text_flip_embeds : (B, embeds_dim) or None
        img_local        : (B, N, D_img) or None
        text_local       : (B, L, D_txt)
        attention_mask   : (B, L)
        text_tokens      : tokenizer output dict
        """
        img_embeds,  img_local   = self.encode_image(img)
        text_embeds, text_local, attention_mask, text_tokens = self.encode_text(text)

        text_flip_embeds = None
        if self.pos_loss and flip_desc is not None:
            text_flip_embeds, _, _, _ = self.encode_text(flip_desc)

        return (img_embeds, text_embeds, text_flip_embeds,
                img_local, text_local, attention_mask, text_tokens)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(
                self.parameters(), lr=self.lr,
                weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer.lower() in ('adamw', 'adam'):
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        scheduler = lr_scheduler.MultiStepLR(
            opt, milestones=self.milestones, gamma=self.lr_mult)
        return [opt], [scheduler]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Linear LR warmup — replaces the deprecated optimizer_step override."""
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_function(self, img_embeds, labels, text_embeds, text_flip_embeds=None):
        text_labels    = labels.clone()
        text_embeds_all = text_embeds

        if self.pos_loss and text_flip_embeds is not None:
            text_embeds_all = torch.cat([text_embeds, text_flip_embeds], dim=0)
            text_labels     = torch.cat([text_labels, text_labels], dim=0)

        if self.miner is not None:
            # I2T: image anchors → text references
            miner_outputs = self.miner(
                img_embeds, labels,
                ref_emb=text_embeds_all, ref_labels=text_labels)
            loss = self.loss_fn(
                img_embeds, labels,
                indices_tuple=miner_outputs,
                ref_emb=text_embeds_all, ref_labels=text_labels)
            nb_samples = img_embeds.shape[0]

            if self.unimodal_loss:
                miner_txt = self.miner(text_embeds_all, text_labels)
                txt_loss  = self.loss_fn(text_embeds_all, text_labels,
                                         indices_tuple=miner_txt)
                loss = loss + self.unimodal_loss * txt_loss

            nb_mined  = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(img_embeds, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc),
                 prog_bar=True, logger=True)
        return loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        places, labels, texts, flip_descs, *_ = batch

        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        flat_texts = [texts[j][i] for i in range(BS) for j in range(N)]
        flat_flip  = ([flip_descs[j][i] for i in range(BS) for j in range(N)]
                      if self.pos_loss else None)

        img_embeds, text_embeds, text_flip_embeds, *_ = self(
            images, flat_texts, flat_flip, labels)

        loss = self.loss_function(img_embeds, labels, text_embeds, text_flip_embeds)
        self.log('loss', loss.item(), logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    # ------------------------------------------------------------------
    # Validation — always cross-modal: text queries vs image gallery
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        img_embeds, text_embeds, *_ = self(places, texts)
        return {
            'descriptors': img_embeds.detach().cpu(),
            'text_embeds': text_embeds.detach().cpu(),
        }

    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        k_values = [1, 5, 10, 15, 20, 50, 100]

        for val_set_name, val_dataset, step_outputs in zip(
                dm.val_set_names, dm.val_datasets, val_step_outputs):

            img_feats  = torch.cat([d['descriptors'] for d in step_outputs], dim=0)
            text_feats = torch.cat([d['text_embeds']  for d in step_outputs], dim=0)

            if 'pitts' in val_set_name:
                num_references = val_dataset.num_db
                positives      = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives      = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f'Please implement validation_epoch_end for {val_set_name}')

            # Cross-modal retrieval: text queries vs image gallery
            r_list = img_feats[: num_references]        # image DB
            q_list = text_feats[num_references:]        # text queries

            recalls = utils.get_validation_recalls(
                r_list=r_list, q_list=q_list, k_values=k_values,
                gt=positives, print_results=True,
                dataset_name=val_set_name, faiss_gpu=self.faiss_gpu)

            self.log(f'{val_set_name}/R1',  recalls[1],  prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5',  recalls[5],  prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', recalls[10], prog_bar=False, logger=True)

        print('\n')

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        ckpt_cb = next(
            (cb for cb in self.trainer.checkpoint_callbacks
             if isinstance(cb, pl.callbacks.ModelCheckpoint)), None)
        if ckpt_cb is None:
            return
        ckpt_dir = os.path.dirname(ckpt_cb.dirpath)

        if self.train_text == 1:
            save_path = os.path.join(ckpt_dir, 'text_lora')
            self.text_encoder.save_pretrained(save_path)
            print(f"Saved text LoRA adapter to: {save_path}")

        if self.train_image == 1 and 'dinov2' in self.image_model_name:
            save_path = os.path.join(ckpt_dir, 'image_lora')
            self.image_encoder.save_pretrained(save_path)
            print(f"Saved image LoRA adapter to: {save_path}")
        # train_image == 2 (full fine-tune): weights saved in the PL checkpoint itself