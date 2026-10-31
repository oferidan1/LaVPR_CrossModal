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


def filip_max_sim_scores(
    img_local: torch.Tensor,       # (B, N, D)  L2-normed patch tokens
    text_local: torch.Tensor,      # (B, L, D)  L2-normed word tokens
    attention_mask: torch.Tensor,  # (B, L)     1=real token, 0=pad
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FILIP fine-grained cross-modal similarity.

    For each image token, find the most similar text token (max-sim) and
    average across image tokens.  Repeat symmetrically for text → image.

    Returns
    -------
    i2t_scores : (B, B)   row i = image i's similarity to every text j
    t2i_scores : (B, B)   row j = text j's similarity to every image i
    """
    B, N, D = img_local.shape
    _,  L, _ = text_local.shape

    # Mask padding tokens before max-sim (set their scores to -inf)
    mask = attention_mask.bool()                                       # (B, L)

    # i2t: for each image patch token, find best-matching text token
    # sim[b_i, b_t, n, l] = img_local[b_i, n] · text_local[b_t, l]
    sim = torch.einsum('ind,jtd->ijnt', img_local, text_local)        # (B, B, N, L)

    # Mask padding columns before max
    pad_mask = ~mask                                                   # (B, L)  True=pad
    pad_mask = pad_mask.unsqueeze(0).unsqueeze(2)                      # (1, B, 1, L)
    sim = sim.masked_fill(pad_mask, float('-inf'))

    i2t_scores = sim.max(dim=-1).values.mean(dim=-1)                   # (B, B)

    # t2i: for each text token, find best-matching image patch token
    sim_t = sim.permute(1, 0, 3, 2)                                   # (B, B, L, N)
    # Re-apply text mask to rows this time (exclude pad token rows)
    row_mask = ~mask                                                   # (B, L)
    row_mask = row_mask.unsqueeze(1).unsqueeze(-1)                     # (B, 1, L, 1)
    sim_t = sim_t.masked_fill(row_mask, float('-inf'))
    # Mean only over real tokens
    real_counts = mask.float().sum(dim=1, keepdim=True)                # (B, 1)
    t2i_max   = sim_t.max(dim=-1).values                               # (B, B, L)
    t2i_max   = t2i_max.masked_fill(~mask.unsqueeze(1), 0.0)          # zero out pad rows
    t2i_scores = t2i_max.sum(dim=-1) / real_counts.T                  # (B, B)

    return i2t_scores, t2i_scores


def filip_contrastive_loss(
    i2t_scores: torch.Tensor,   # (B, B)
    t2i_scores: torch.Tensor,   # (B, B)
    labels: torch.Tensor,       # (B,) place-id labels
) -> torch.Tensor:
    """
    InfoNCE-style contrastive loss on FILIP max-sim score matrices.

    Positive pairs share the same label; negatives are all others in the batch.
    Works with multiple positives per anchor (multi-positive InfoNCE).
    """
    B = i2t_scores.shape[0]
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()   # (B, B)

    # For numerical stability, subtract the diagonal max
    def multi_pos_nce(scores, pos_mask):
        log_softmax = F.log_softmax(scores, dim=1)                    # (B, B)
        # Average positive log-probs
        pos_log = (log_softmax * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        return -pos_log.mean()

    loss_i2t = multi_pos_nce(i2t_scores, pos_mask)
    loss_t2i = multi_pos_nce(t2i_scores, pos_mask)
    return (loss_i2t + loss_t2i) * 0.5


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

    FILIP extension (filip_loss > 0)
    ---------------------------------
    image patch tokens → image_proj_local (D_img → embeds_dim) → L2-norm
    text  word  tokens → text_proj_local  (D_txt → embeds_dim) → L2-norm
    filip_loss * filip_contrastive_loss(max_sim(image_local, text_local))
    SelaVPR does not expose patch tokens, so FILIP is silently disabled for it.
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

        # ---- FILIP fine-grained local-token loss
        # Weight of the FILIP contrastive loss added to the main MS loss.
        # Set to 0.0 (default) to disable entirely.
        # Requires image_model_name='dinov2' (SelaVPR has no patch tokens).
        filip_loss: float = 0.0,

        # ---- Optional modules
        is_llp: int = 0,
    ):
        super().__init__()

        if milestones is None:
            milestones = [5, 10, 15]

        self.text_model_name  = text_model_name.lower()
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

        # FILIP: only meaningful when the image encoder exposes patch tokens
        self._filip_enabled = (filip_loss > 0.0) and ('dinov2' in self.image_model_name)
        self.filip_loss = filip_loss if self._filip_enabled else 0.0
        if filip_loss > 0.0 and not self._filip_enabled:
            import warnings
            warnings.warn(
                "filip_loss > 0 but image_model_name is not 'dinov2'; "
                "SelaVPR does not expose patch tokens — FILIP disabled."
            )

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

        # ---- Global projection heads: map each backbone to shared embeds_dim
        self.image_proj = nn.Linear(img_hidden_dim, embeds_dim, bias=False)
        self.text_proj  = nn.Linear(txt_hidden_dim, embeds_dim, bias=False)

        # ---- Local (FILIP) projection heads — only built when needed
        # Separate weights from global heads so local and global projections
        # can specialise independently.
        if self._filip_enabled:
            self.image_proj_local = nn.Linear(img_hidden_dim, embeds_dim, bias=False)
            self.text_proj_local  = nn.Linear(txt_hidden_dim, embeds_dim, bias=False)

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
                for p in self.image_encoder.parameters():
                    p.requires_grad = True
        elif train_image == 2:
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

        # ---- Init projection heads (kaiming)
        nn.init.kaiming_uniform_(self.image_proj.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.text_proj.weight,  mode='fan_in', nonlinearity='relu')
        if self._filip_enabled:
            nn.init.kaiming_uniform_(self.image_proj_local.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.text_proj_local.weight,  mode='fan_in', nonlinearity='relu')

    # ------------------------------------------------------------------
    # LoRA helper
    # ------------------------------------------------------------------

    @staticmethod
    def _init_lora(encoder, lora_targets, lora_all_linear, lora_r):
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
        img_embeds       : (B, embeds_dim)        L2-normed global embedding
        img_local_proj   : (B, N, embeds_dim)     L2-normed projected patch tokens,
                           or None when FILIP is disabled or encoder is SelaVPR
        """
        if 'dinov2' in self.image_model_name:
            out = self.image_encoder.forward_features(img)
            img_local = out['x_norm_patchtokens']    # (B, N, D_img)
            img_raw   = out['x_norm_clstoken']       # (B, D_img)
        elif 'sela' in self.image_model_name:
            encoder = (self.image_encoder.base_model
                       if hasattr(self.image_encoder, 'base_model')
                       else self.image_encoder)
            img_raw   = encoder(img)                 # (B, D_img)
            img_local = None

        img_embeds = F.normalize(self.image_proj(img_raw), p=2, dim=1)  # (B, embeds_dim)

        img_local_proj = img_local
        if self._filip_enabled and img_local is not None:
            # (B, N, embeds_dim), L2-normed along the last dim
            img_local_proj = F.normalize(self.image_proj_local(img_local), p=2, dim=-1)

        return img_embeds, img_local_proj

    def encode_text(self, text):
        """
        Returns
        -------
        text_embeds      : (B, embeds_dim)         L2-normed global embedding
        text_local_proj  : (B, L, embeds_dim)      L2-normed projected word tokens
        attention_mask   : (B, L)
        text_tokens      : tokenizer output dict
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

        text_embeds = F.normalize(self.text_proj(text_raw), p=2, dim=1)

        # Local projection for FILIP
        text_local_proj = text_local
        if self._filip_enabled:
            text_local_proj = F.normalize(self.text_proj_local(text_local), p=2, dim=-1)

        return text_embeds, text_local_proj, text_tokens['attention_mask'], text_tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, img, text, flip_desc=None, labels=None):
        """
        Returns
        -------
        img_embeds        : (B, embeds_dim)
        text_embeds       : (B, embeds_dim)
        text_flip_embeds  : (B, embeds_dim) or None
        img_local_proj    : (B, N, embeds_dim) or None
        text_local_proj   : (B, L, embeds_dim) or None
        attention_mask    : (B, L)
        text_tokens       : tokenizer output dict
        """
        img_embeds,  img_local_proj = self.encode_image(img)
        text_embeds, text_local_proj, attention_mask, text_tokens = self.encode_text(text)

        text_flip_embeds = None
        if self.pos_loss and flip_desc is not None:
            text_flip_embeds, _, _, _ = self.encode_text(flip_desc)

        return (img_embeds, text_embeds, text_flip_embeds,
                img_local_proj, text_local_proj, attention_mask, text_tokens)

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
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_function(
        self,
        img_embeds: torch.Tensor,
        labels: torch.Tensor,
        text_embeds: torch.Tensor,
        text_flip_embeds: torch.Tensor | None = None,
        img_local_proj: torch.Tensor | None = None,
        text_local_proj: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        text_labels     = labels.clone()
        text_embeds_all = text_embeds

        if self.pos_loss and text_flip_embeds is not None:
            text_embeds_all = torch.cat([text_embeds, text_flip_embeds], dim=0)
            text_labels     = torch.cat([text_labels, text_labels], dim=0)

        if self.miner is not None:
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

        # ---- FILIP local-token loss
        if (self._filip_enabled
                and img_local_proj is not None
                and text_local_proj is not None
                and attention_mask is not None):
            i2t_scores, t2i_scores = filip_max_sim_scores(
                img_local_proj, text_local_proj, attention_mask)
            f_loss = filip_contrastive_loss(i2t_scores, t2i_scores, labels)
            loss = loss + self.filip_loss * f_loss
            self.log('filip_loss', f_loss.item(), prog_bar=True, logger=True)

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

        (img_embeds, text_embeds, text_flip_embeds,
         img_local_proj, text_local_proj,
         attention_mask, _) = self(images, flat_texts, flat_flip, labels)

        loss = self.loss_function(
            img_embeds, labels, text_embeds, text_flip_embeds,
            img_local_proj, text_local_proj, attention_mask)
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

            r_list = img_feats[: num_references]
            q_list = text_feats[num_references:]

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