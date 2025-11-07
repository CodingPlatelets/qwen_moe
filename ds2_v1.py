from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Config, DeepseekV2MLP, DeepseekV2RMSNorm, DeepseekV2Attention
from transformers.models.deepseek_v2 import modeling_deepseek_v2 as ds2
from transformers.modeling_layers import GradientCheckpointingLayer
from typing import Optional, Unpack
from transformers.cache_utils import Cache
from transformers.utils.generic import TransformersKwargs

from torch.autograd.profiler import record_function
import torch
from torch import nn
from torch.nn import functional as F
import common


class DeepseekV2DecoderLayerTimed(GradientCheckpointingLayer):
    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV2Attention(
            config=config, layer_idx=layer_idx)
        self.mlp = DeepseekV2MoE(
            config, layer_idx=layer_idx) if layer_idx >= config.first_k_dense_replace else DeepseekV2MLP(config)

        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        is_prefill = hidden_states is not None and hidden_states.size(1) > 1
        phase = "prefill" if is_prefill else "decode"
        # Self Attention
        common._TREG.start(f"attn_{phase}", self._lid)
        with record_function(f"Layer[{self._lid}].attn"):
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
        common._TREG.start(f"norm_{phase}", self._lid)
        hidden_states = self.post_attention_layernorm(hidden_states)
        common._TREG.stop(f"norm_{phase}", self._lid)
        common._TREG.start(f"mlp_{phase}", self._lid)
        hidden_states = self.mlp(hidden_states)
        common._TREG.stop(f"mlp_{phase}", self._lid)
        hidden_states = residual + hidden_states
        return hidden_states


class DeepseekV2MoEGate(nn.Module):
    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.num_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(
            (self.num_experts, self.gating_dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_dim)
        logits = F.linear(hidden_states.type(torch.float32),
                          self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)

        # select top-k experts
        # greedy method is used for DeepSeek-V2-Lite
        # group_limited_greedy for DeepSeek-V2 and DeepSeek-V2-Chat
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False)
        elif self.topk_method == "group_limited_greedy":
            group_scores = scores.view(
                # [n, num_group]
                batch_size * seq_len, self.num_group, -1).max(dim=-1).values
            group_idx = torch.topk(
                # [n, top_k_group]
                group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
            group_mask = torch.zeros_like(group_scores)  # [n, num_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, num_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(batch_size * seq_len, self.num_group, self.num_experts // self.num_group)
                .reshape(batch_size * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False)

        topk_weight = topk_weight * self.routed_scaling_factor
        # expert-level computation auxiliary loss
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                (DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size))
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV2MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV2MLP(
                config=config, intermediate_size=intermediate_size)
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts
        self._lid = layer_idx

    def moe(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weight: torch.Tensor) -> torch.Tensor:
        phase = "prefill" if hidden_states.size(1) > 1 else "decode"
        common._TREG.start(f"dispatch_{phase}", self._lid)
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        indices = topk_ids.view(-1).argsort()
        sorted_tokens = hidden_states[indices // topk_ids.shape[1]]
        common._TREG.stop(f"dispatch_{phase}", self._lid)

        # Process experts
        outputs = []
        start_idx = 0
        common._TREG.start(f"compute_{phase}", self._lid)
        for i, num_tokens in enumerate(tokens_per_expert):
            if num_tokens == 0:
                continue
            end_idx = start_idx + num_tokens
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(
            outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
        common._TREG.stop(f"compute_{phase}", self._lid)

        common._TREG.start(f"aggregate_{phase}", self._lid)
        # Reorder and combine outputs
        new_x = torch.empty_like(outs)
        new_x[indices] = outs
        hidden_states = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        common._TREG.stop(f"aggregate_{phase}", self._lid)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        phase = "prefill" if hidden_states.size(1) > 1 else "decode"
        common._TREG.start(f"router_{phase}", self._lid)
        topk_indices, topk_weights = self.gate(hidden_states)
        common._TREG.stop(f"router_{phase}", self._lid)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        common._TREG.start(f"expert_{phase}", self._lid)
        hidden_states = self.moe(
            hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        common._TREG.stop(f"expert_{phase}", self._lid)
        return hidden_states
