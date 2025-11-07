import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from torch.autograd.profiler import record_function
import common


class Qwen3MoeDecoderLayerTimed(Qwen3MoeDecoderLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super(Qwen3MoeDecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (
                layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # use self-defined MoE block
            self.mlp = Qwen3MoeSparseMoeBlockV2(
                layer_idx=layer_idx, config=config)
        else:
            self.mlp = Qwen3MoeMLP(
                config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self._lid = layer_idx

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, cache_position=None, **kwargs):

        lid = self._lid
        is_prefill = hidden_states is not None and hidden_states.size(1) > 1
        phase = "prefill" if is_prefill else "decode"

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        common._TREG.start(f"attn_{phase}", lid)
        with record_function(f"Layer[{lid}].attn"):
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        common._TREG.stop(f"attn_{phase}", lid)

        hidden_states = residual + hidden_states

        residual = hidden_states
        common._TREG.start(f"norm_{phase}", lid)
        hidden_states = self.post_attention_layernorm(hidden_states)
        common._TREG.stop(f"norm_{phase}", lid)

        common._TREG.start(f"mlp_{phase}", lid)
        with record_function(f"Layer[{lid}].mlp"):
            hidden_states = self.mlp(is_prefill, hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
        common._TREG.stop(f"mlp_{phase}", lid)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3MoeSparseMoeBlockV2(Qwen3MoeSparseMoeBlock):
    def __init__(self, layer_idx: int, config: Qwen3MoeConfig):
        super().__init__(config)
        self._lid = layer_idx

    def forward(self, is_prefill: bool, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """

        phase = "prefill" if is_prefill else "decode"
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gating
        # router_logits: (batch * sequence_length, n_experts)
        common._TREG.start(f"gating_{phase}", self._lid)
        common._TREG.start(f"router_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].gating"):
            router_logits = self.gate(hidden_states)
        common._TREG.stop(f"gating_{phase}", self._lid)

        # softmax and top-k
        common._TREG.start(f"softmax_{phase}", self._lid)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        common._TREG.stop(f"softmax_{phase}", self._lid)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        common._TREG.stop(f"router_{phase}", self._lid)

        common._TREG.start(f"expert_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].experts.{phase}"):
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            # one-hot expert mask: [num_experts, top_k, B*T]
            common._TREG.start(f"dispatch_{phase}", self._lid)
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            common._TREG.stop(f"dispatch_{phase}", self._lid)
            common._TREG.start(f"compute_{phase}", self._lid)
            for expert_idx in expert_hit:
                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None,
                                              top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(
                    current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(
                    0, top_x, current_hidden_states.to(hidden_states.dtype))
            common._TREG.stop(f"compute_{phase}", self._lid)
        common._TREG.stop(f"expert_{phase}", self._lid)

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
