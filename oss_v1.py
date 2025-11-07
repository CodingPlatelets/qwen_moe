import torch
from torch import nn
import torch.nn.functional as F
import transformers.models.gpt_oss.modeling_gpt_oss as gmoe
from typing import Optional, Unpack
from transformers.cache_utils import Cache
from transformers.utils.generic import TransformersKwargs
import common
from torch.autograd.profiler import record_function


class GptOssDecoderLayer(gmoe.GptOssDecoderLayer):
    def __init__(self, config: gmoe.GptOssConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = gmoe.GptOssAttention(
            config=config, layer_idx=layer_idx)
        self.mlp = GptOssMLPV1(layer_idx, config)
        self.input_layernorm = gmoe.GptOssRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = gmoe.GptOssRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self._lid = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # necessary, but kept here for BC
        position_embeddings: Optional[tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        is_prefill = hidden_states is not None and hidden_states.size(1) > 1
        phase = "prefill" if is_prefill else "decode"
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        common._TREG.start(f"attn_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].attn"):
            # Self Attention
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        common._TREG.stop(f"attn_{phase}", self._lid)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # diff with llama: router scores
        common._TREG.start(f"mlp_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].mlp"):
            hidden_states, _ = self.mlp(hidden_states)
        common._TREG.stop(f"mlp_{phase}", self._lid)
        hidden_states = residual + hidden_states
        return hidden_states


# @use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLPV1(gmoe.GptOssMLP):
    # def __init__(self, layer_idx: int, config):
    #     super(gmoe.GptOssMLP, self).__init__()
    #     self.router = GptOssTopKRouterV1(config)
    #     self.experts = GptOssExpertsV1(config)
    #     self._lid = layer_idx

    def __init__(self, layer_idx: int, config):
        super().__init__(config)
        self._lid = layer_idx

    def forward(self, hidden_states):
        is_prefill = hidden_states is not None and hidden_states.size(1) > 1
        phase = "prefill" if is_prefill else "decode"
        # router timing
        common._TREG.start(f"gating_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].router"):
            router_scores, router_indices = self.router(
                hidden_states)  # (num_experts, seq_len)
        common._TREG.stop(f"gating_{phase}", self._lid)

        # experts timing
        common._TREG.start(f"expert_{phase}", self._lid)
        routed_out = self.experts(
            hidden_states, router_indices=router_indices, routing_weights=router_scores)
        common._TREG.stop(f"expert_{phase}", self._lid)
        return routed_out, router_scores


class GptOssTopKRouterV1(gmoe.GptOssTopKRouter):
    def __init__(self, config):
        super().__init__(config)
        # self.top_k = config.num_experts_per_tok
        # self.num_experts = config.num_local_experts
        # self.hidden_dim = config.hidden_size
        # self.weight = nn.Parameter(torch.empty(
        #     self.num_experts, self.hidden_dim))
        # self.bias = nn.Parameter(torch.empty(self.num_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # (seq_len, num_experts)
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(
            router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(
            router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(
            1, router_indices, router_top_value)
        return router_scores, router_indices


class GptOssExpertsV1(gmoe.GptOssExperts):
    def __init__(self, config):
        super().__init__(config)
        # self.intermediate_size = config.intermediate_size
        # self.num_experts = config.num_local_experts
        # self.hidden_size = config.hidden_size
        # self.expert_dim = self.intermediate_size
        # self.gate_up_proj = nn.Parameter(torch.empty(
        #     self.num_experts, self.hidden_size, 2 * self.expert_dim))
        # self.gate_up_proj_bias = nn.Parameter(
        #     torch.empty(self.num_experts, 2 * self.expert_dim))
        # self.down_proj = nn.Parameter(torch.empty(
        #     (self.num_experts, self.expert_dim, self.hidden_size)))
        # self.down_proj_bias = nn.Parameter(
        #     torch.empty(self.num_experts, self.hidden_size))
        # self.alpha = 1.702
        # self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        # (num_tokens, hidden_size)
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        if hidden_states.device.type == "cpu" or self.training:
            next_states = torch.zeros_like(
                hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    router_indices, num_classes=num_experts + 1
                )  # masking is also a class
                expert_mask = expert_mask.permute(2, 1, 0)
                # we sum on the top_k and on the sequence length to get which experts
                # are hit this time around
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit[:]:
                # expert_idx only have 1 element, so we can use scale for fast indexing
                expert_idx = expert_idx[0]
                # skip masking index
                if expert_idx == num_experts:
                    continue
                with torch.no_grad():
                    _, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate_up = current_state @ self.gate_up_proj[expert_idx] + \
                    self.gate_up_proj_bias[expert_idx]
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                gated_output = (up + 1) * glu
                out = gated_output @ self.down_proj[expert_idx] + \
                    self.down_proj_bias[expert_idx]
                weighted_output = out * \
                    routing_weights[token_idx, expert_idx, None]
                next_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype))
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(num_experts, 1)
            hidden_states = hidden_states.view(
                num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj) + \
                self.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            next_states = torch.bmm(((up + 1) * glu), self.down_proj)
            next_states = next_states + self.down_proj_bias[..., None, :]
            next_states = next_states.view(
                num_experts, batch_size, -1, self.hidden_size)
            next_states = next_states * \
                routing_weights.transpose(0, 1).view(
                    num_experts, batch_size, -1)[..., None]
            next_states = next_states.sum(dim=0)
        return next_states
