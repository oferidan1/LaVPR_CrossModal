from typing import List, Optional, Tuple, Union
import torch

from transformers import Qwen2Model, Qwen2ForCausalLM, Qwen2PreTrainedModel, Qwen2Config
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2MLP,
)
from torch import nn
from transformers.utils import logging
from .attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from peft import PeftModel

logger = logging.get_logger(__name__)


try:
    from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES as HF_QWEN2_ATTENTION_CLASSES
except ImportError:
    HF_QWEN2_ATTENTION_CLASSES = {"eager": Qwen2Attention}
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2FlashAttention2
        HF_QWEN2_ATTENTION_CLASSES["flash_attention_2"] = Qwen2FlashAttention2
    except ImportError:
        pass
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2SdpaAttention
        HF_QWEN2_ATTENTION_CLASSES["sdpa"] = Qwen2SdpaAttention
    except ImportError:
        pass

QWEN2_ATTENTION_CLASSES = {}
for key, cls in HF_QWEN2_ATTENTION_CLASSES.items():
    class ModifiedAttention(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_causal = False
    ModifiedAttention.__name__ = f"Modified{cls.__name__}"
    QWEN2_ATTENTION_CLASSES[key] = ModifiedAttention


class ModifiedQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx
        )

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen2BiModel(Qwen2Model):
    _no_split_modules = ["ModifiedQwen2DecoderLayer"]

    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                ModifiedQwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2BiForMNTP(Qwen2ForCausalLM):
    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # getter for PEFT model
    def get_model_for_peft(self):
        return self.model

    # setter for PEFT model
    def set_model_for_peft(self, model: PeftModel):
        self.model = model

    # save the PEFT model
    def save_peft_model(self, path):
        self.model.save_pretrained(path)
