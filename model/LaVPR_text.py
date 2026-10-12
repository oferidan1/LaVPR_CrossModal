"""
LaVPR_text  —  Text-only VPR model (T2T training stage).

SelaVPR recipe applied to BGE text encoder:
  - Backbone fully FROZEN (no LoRA, no fine-tuning)
  - Only MultiConvTextAdapter trains (inserted between frozen layers)
  - GeM pooling over adapted token sequence (same as SelaVPR image side)
  - MultiSimilarityLoss on place descriptions

Integration with LaVPR_cross (Stage 2):
    cross = LaVPR_cross(use_multiconv=True, embeds_dim=1024, ...)
    ckpt  = LaVPR_text.load_from_checkpoint(path)
    cross.text_encoder.load_state_dict(ckpt.text_encoder.state_dict())
    cross.text_adapter.load_state_dict(ckpt.text_adapter.state_dict())
"""

import math
import os
import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import utils
from model.weighted_ms_loss import WeightedMultiSimilarityLossCM


# ---------------------------------------------------------------------------
# Dim table
# ---------------------------------------------------------------------------

_TEXT_ENCODER_DIM = {
    'bge-l': 1024,  # BAAI/bge-large-en-v1.5  (BERT-large, 24 layers)
    'bge-b':  768,  # BAAI/bge-base-en-v1.5   (BERT-base,  12 layers)
}

def _text_hidden_dim(model_name: str) -> int:
    for key, dim in _TEXT_ENCODER_DIM.items():
        if key in model_name.lower():
            return dim
    raise ValueError(f"Unknown text model '{model_name}'. Add it to _TEXT_ENCODER_DIM.")


# ---------------------------------------------------------------------------
# GeM pooling (1D token sequence)
# ---------------------------------------------------------------------------

class GeM1d(nn.Module):
    """Generalised Mean Pooling over a token sequence.

    SelaVPR uses GeM over spatial patch tokens (2D → 1D after flatten).
    Here we apply it over the L token dimension directly.
    Padding tokens are excluded via the attention mask.

    p=3 matches SelaVPR's default GeM power.
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x              : (B, L, D)
        attention_mask : (B, L)  1=real token, 0=padding
        returns        : (B, D)
        """
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        x    = x.clamp(min=self.eps)
        gem  = (x.pow(self.p) * mask).sum(1) / \
               mask.sum(1).clamp(min=self.eps)               # (B, D)
        return gem.pow(1.0 / self.p)


# ---------------------------------------------------------------------------
# SelaVPR-style 1D Conv adapter modules
# ---------------------------------------------------------------------------

class BasicConv1d(nn.Module):
    """Conv1d + BN + ReLU — 1D analogue of SelaVPR's BasicConv2d."""
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, bias=False, **kwargs)
        self.bn   = nn.BatchNorm1d(out_ch, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class MulConv1dAdapter(nn.Module):
    """1D Inception-style adapter — port of SelaVPR's MulConvAdapter.

    Branches: k=1 / k=3 / k=5 Conv1d on token sequence.
    CLS token bypasses conv branches (same as SelaVPR image CLS handling).
    """
    def __init__(self, fc_in: int, inner_dim: int,
                 ch1x1: int, ch3x3red: int, ch3x3: int,
                 ch5x5red: int, ch5x5: int,
                 skip_connect: bool = True):
        super().__init__()
        assert ch1x1 + ch3x3 + ch5x5 == inner_dim, \
            "Branch output channels must sum to inner_dim"
        self.skip_connect = skip_connect
        self.branch1 = BasicConv1d(inner_dim, ch1x1,  kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv1d(inner_dim, ch3x3red, kernel_size=1),
            BasicConv1d(ch3x3red, ch3x3,    kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv1d(inner_dim, ch5x5red, kernel_size=1),
            BasicConv1d(ch5x5red, ch5x5,    kernel_size=5, padding=2),
        )
        self.D_fc1 = nn.Linear(fc_in,     inner_dim)
        self.D_fc2 = nn.Linear(inner_dim, fc_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, L, fc_in)"""
        x0    = F.relu(self.D_fc1(x), inplace=True)         # (B, L, inner)
        cls   = x0[:, 0:1, :]                                # (B, 1, inner)
        patch = x0[:, 1:,  :].transpose(1, 2)                # (B, inner, L-1)
        out   = torch.cat([self.branch1(patch),
                           self.branch2(patch),
                           self.branch3(patch)], dim=1)      # (B, inner, L-1)
        out   = out.transpose(1, 2)                          # (B, L-1, inner)
        out   = torch.cat([cls, out], dim=1) + x0            # residual (B, L, inner)
        out   = self.D_fc2(out)                              # (B, L, fc_in)
        if self.skip_connect:
            out = out + x
        return out


class MultiConvTextAdapter(nn.Module):
    """SelaVPR-style adapter stack for a frozen BGE text encoder.

    Running accumulator across all transformer layers (same pattern as
    SelaVPR's forward_features):
        y = hidden_states[1]
        for i, adapter in enumerate(adapters):
            x = hidden_states[i+1]
            y = y + adapter(y + x)
        output = LayerNorm(y)
        → GeM pooling over valid tokens
        → Linear(D, out_dim)
    """
    def __init__(self, hidden_dim: int, out_dim: int, n_layers: int,
                 pooling: str = 'gem', gem_p: float = 3.0):
        super().__init__()
        assert pooling in ('gem', 'mean', 'cls'), f"pooling must be 'gem', 'mean' or 'cls', got '{pooling}'"
        self.pooling = pooling
        inner  = hidden_dim // 2
        ch1    = inner // 2
        ch3red = inner // 16
        ch3    = inner // 4
        ch5red = inner // 16
        ch5    = inner - ch1 - ch3

        self.adapters = nn.ModuleList([
            MulConv1dAdapter(
                fc_in=hidden_dim, inner_dim=inner,
                ch1x1=ch1, ch3x3red=ch3red, ch3x3=ch3,
                ch5x5red=ch5red, ch5x5=ch5,
                skip_connect=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.gem  = GeM1d(p=gem_p) if pooling == 'gem' else None
        self.proj = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, all_hidden_states: tuple,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        all_hidden_states : tuple of (B, L, D), index 0 = embedding layer
        attention_mask    : (B, L)
        returns           : (B, out_dim)
        """
        y = all_hidden_states[1]                             # first block output
        for i, adapter in enumerate(self.adapters):
            x = all_hidden_states[i + 1]
            y = y + adapter(y + x)

        y    = self.norm(y)                                  # (B, L, D)
        mask = attention_mask.unsqueeze(-1).float()          # (B, L, 1)
        if self.pooling == 'gem':
            pooled = self.gem(y, attention_mask)             # (B, D)
        elif self.pooling == 'cls':
            pooled = y[:, 0]                                 # (B, D) CLS token
        else:
            pooled = (y * mask).sum(1) / mask.sum(1).clamp(min=1)  # (B, D) mean
        return self.proj(pooled)                             # (B, out_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LaVPR_text(pl.LightningModule):
    """Text-only VPR encoder (T2T training stage).

    BGE backbone is fully frozen. Only MultiConvTextAdapter trains.
    Follows SelaVPR's recipe: frozen backbone + adapter + GeM pooling.
    """

    def __init__(
        self,
        # ---- Encoder
        text_model_name: str = 'BAAI/bge-large-en-v1.5',
        embeds_dim: int = 1024,         # BGE-large native dim; must match LaVPR_cross

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

        # ---- Pooling
        pooling: str = 'gem',           # 'gem' / 'mean' / 'cls'
        gem_p: float = 3.0,
    ):
        super().__init__()

        if milestones is None:
            milestones = [5, 10, 15]

        self.text_model_name = text_model_name.lower()
        self.embeds_dim      = embeds_dim
        self.lr              = lr
        self.optimizer       = optimizer
        self.weight_decay    = weight_decay
        self.momentum        = momentum
        self.warmpup_steps   = warmpup_steps
        self.milestones      = milestones
        self.lr_mult         = lr_mult
        self.epochs          = epochs

        self.save_hyperparameters()

        # ---- Loss / miner
        if 'WeightedMultiSimilarityLoss' in loss_name:
            self.loss_fn = WeightedMultiSimilarityLossCM()
        else:
            self.loss_fn = utils.get_loss(loss_name)
        self.miner     = utils.get_miner(miner_name, miner_margin)
        self.faiss_gpu = faiss_gpu
        self.batch_acc = []

        # ---- Frozen text encoder
        txt_hidden_dim    = _text_hidden_dim(text_model_name)
        self.tokenizer    = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, attn_implementation="sdpa")
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # ---- Trainable adapter (only thing that trains)
        n_layers = self.text_encoder.config.num_hidden_layers
        self.text_adapter = MultiConvTextAdapter(
            hidden_dim=txt_hidden_dim,
            out_dim=embeds_dim,
            n_layers=n_layers,
            pooling=pooling,
            gem_p=gem_p,
        )

    # ------------------------------------------------------------------
    # Encoding  (identical signature to LaVPR_cross.encode_text)
    # ------------------------------------------------------------------

    def encode_text(self, text):
        """
        Returns
        -------
        text_embeds   : (B, embeds_dim)  L2-normalised
        text_local    : (B, L, D)        last hidden state
        attention_mask: (B, L)
        text_tokens   : tokenizer output dict
        """
        text_tokens = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.text_encoder(
                **text_tokens,
                output_hidden_states=True,
                return_dict=True,
            )

        attn_mask  = text_tokens['attention_mask']
        text_local = model_output.last_hidden_state           # (B, L, D)

        # Adapter runs with grad (backbone does not)
        text_raw    = self.text_adapter(model_output.hidden_states, attn_mask)
        text_embeds = F.normalize(text_raw, p=2, dim=1)
        return text_embeds, text_local, attn_mask, text_tokens

    # ------------------------------------------------------------------
    # Optimiser  (only adapter params — backbone is frozen)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if self.optimizer.lower() == 'sgd':
            opt = torch.optim.SGD(trainable, lr=self.lr,
                                  weight_decay=self.weight_decay,
                                  momentum=self.momentum)
        elif self.optimizer.lower() in ('adamw', 'adam'):
            opt = torch.optim.AdamW(trainable, lr=self.lr,
                                    weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        scheduler = lr_scheduler.MultiStepLR(
            opt, milestones=self.milestones, gamma=self.lr_mult)
        return [opt], [scheduler]

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1.0,
                float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss_function(self, text_embeds, labels):
        if self.miner is not None:
            miner_out = self.miner(text_embeds, labels)
            loss      = self.loss_fn(text_embeds, labels, indices_tuple=miner_out)
            nb_mined  = len(set(miner_out[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / text_embeds.shape[0])
        else:
            loss      = self.loss_fn(text_embeds, labels)
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
        places, labels, texts, *_ = batch
        BS, N = places.shape[:2]
        labels     = labels.view(-1)
        flat_texts = [texts[j][i] for i in range(BS) for j in range(N)]

        text_embeds, *_ = self.encode_text(flat_texts)
        loss = self.loss_function(text_embeds, labels)
        self.log('loss', loss.item(), logger=True)
        return loss

    def training_epoch_end(self, outputs):
        self.batch_acc = []

    # ------------------------------------------------------------------
    # Validation  (T2T: text queries vs text gallery)
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _, texts = batch
        text_embeds, *_ = self.encode_text(texts)
        return {'text_embeds': text_embeds.detach().cpu()}

    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]

        k_values = [1, 5, 10, 15, 20, 50, 100]

        for val_set_name, val_dataset, step_outputs in zip(
                dm.val_set_names, dm.val_datasets, val_step_outputs):

            text_feats = torch.cat(
                [d['text_embeds'] for d in step_outputs], dim=0)

            if 'pitts' in val_set_name:
                num_references = val_dataset.num_db
                positives      = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                positives      = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f'Add validation support for {val_set_name}')

            r_list = text_feats[:num_references]
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
    # Checkpoint — only adapter weights matter; backbone is frozen/pretrained
    # ------------------------------------------------------------------

    def on_load_checkpoint(self, checkpoint):
        # Backbone weights were stripped on save (frozen, reloaded from HF at init).
        # Inject them back so PL finds no missing keys and emits no warning.
        backbone_state = {
            k: v for k, v in self.state_dict().items()
            if k.startswith('text_encoder.')
        }
        checkpoint['state_dict'].update(backbone_state)

    def on_save_checkpoint(self, checkpoint):
        # Remove frozen backbone from checkpoint to save disk space
        checkpoint['state_dict'] = {
            k: v for k, v in checkpoint['state_dict'].items()
            if not k.startswith('text_encoder.')
        }