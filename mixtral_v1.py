import torch
from typing import Optional
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.cache_utils import Cache
from typing_extensions import Unpack
from transformers.utils.generic import TransformersKwargs
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralRMSNorm
from transformers.models.mixtral.modeling_mixtral import MixtralAttention
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from torch.autograd.profiler import record_function


class MixtralDecoderLayerTimed(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super(MixtralDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention(config, layer_idx)

        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self._lid = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        with record_function(f"Layer[{self._lid}].attn"):
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        with record_function(f"Layer[{self._lid}].mlp"):
            hidden_states, _ = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
