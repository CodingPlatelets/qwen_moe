import torch
import torch.nn.functional as F
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRMSNorm
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeDecoderLayer

from torch.autograd.profiler import record_function


# model.config.num_hidden_layers
N = 48

# 预分配事件（复用）
ATTN_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
ATTN_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
MLP_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
MLP_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
SOFTMAX_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
SOFTMAX_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
GATING_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
GATING_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
PROJGATING_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
PROJGATING_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
UP_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
UP_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
DOWN_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
DOWN_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
EXPERT_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
EXPERT_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]


ATTN_MS = [0.0]*N
MLP_MS = [0.0]*N
SOFTMAX_MS = [0.0]*N
GATING_MS = [0.0]*N
UP_MS = [0.0]*N
DOWN_MS = [0.0]*N
PROJGATING_S = [0.0]*N
PROJGATING_E = [0.0]*N
PROJGATING_MS = [0.0]*N
EXPERT_MS = [0.0]*N


def reset_timers():
    global ATTN_S, ATTN_E, MLP_S, MLP_E, SOFTMAX_S, SOFTMAX_E, GATING_S, GATING_E
    global UP_S, UP_E, DOWN_S, DOWN_E, PROJGATING_S, PROJGATING_E, EXPERT_S, EXPERT_E
    global ATTN_MS, MLP_MS, SOFTMAX_MS, GATING_MS, UP_MS, DOWN_MS, PROJGATING_MS, EXPERT_MS

    ATTN_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    ATTN_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    MLP_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    MLP_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    SOFTMAX_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    SOFTMAX_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    GATING_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    GATING_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    UP_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    UP_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    DOWN_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    DOWN_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    PROJGATING_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    PROJGATING_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    EXPERT_S = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
    EXPERT_E = [torch.cuda.Event(enable_timing=True) for _ in range(N)]

    ATTN_MS = [0.0]*N
    MLP_MS = [0.0]*N
    SOFTMAX_MS = [0.0]*N
    GATING_MS = [0.0]*N
    UP_MS = [0.0]*N
    DOWN_MS = [0.0]*N
    PROJGATING_MS = [0.0]*N
    EXPERT_MS = [0.0]*N


def show_res():
    torch.cuda.synchronize()
    # just for decoder without prefill
    for i in range(N):
        ATTN_MS[i] = ATTN_S[i].elapsed_time(ATTN_E[i])  # ms
        MLP_MS[i] = MLP_S[i].elapsed_time(MLP_E[i])
        EXPERT_MS[i] = EXPERT_S[i].elapsed_time(EXPERT_E[i])
        SOFTMAX_MS[i] = SOFTMAX_S[i].elapsed_time(SOFTMAX_E[i])
        GATING_MS[i] = GATING_S[i].elapsed_time(GATING_E[i])

    print("per-layer ms:")
    for i in range(N):
        print(
            f"L{i:02d}\tattn={ATTN_MS[i]:.3f}\tmlp={MLP_MS[i]:.3f}\texpert={EXPERT_MS[i]:.3f}\tsoftmax={SOFTMAX_MS[i]:.3f}\tgating={GATING_MS[i]:.3f}")
    print("summary:")
    print(f"attn_total={sum(ATTN_MS):.3f} ms\tmlp_total={sum(MLP_MS):.3f} ms\texpert_total={sum(EXPERT_MS):.3f} ms\tsoftmax_total={sum(SOFTMAX_MS):.3f} ms\tgating_total={sum(GATING_MS):.3f} ms")
    print(f"mlp_ratio={sum(MLP_MS)/(sum(ATTN_MS)+sum(MLP_MS))*100:.3f}%")


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
            self.mlp = Qwen3MoeSparseMoeBlockV1(
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        ATTN_S[lid].record()
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
        hidden_states = residual + hidden_states
        ATTN_E[lid].record()

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        MLP_S[lid].record()
        with record_function(f"Layer[{lid}].mlp"):
            hidden_states = self.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        MLP_E[lid].record()

        return hidden_states


class Qwen3MoeSparseMoeBlockV1(Qwen3MoeSparseMoeBlock):
    def __init__(self, layer_idx: int, config: Qwen3MoeConfig):
        super().__init__(config)
        self._lid = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        GATING_S[self._lid].record()
        with record_function(f"Layer[{self._lid}].gating"):
            router_logits = self.gate(hidden_states)
        GATING_E[self._lid].record()

        SOFTMAX_S[self._lid].record()
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        SOFTMAX_E[self._lid].record()

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        EXPERT_S[self._lid].record()
        with record_function(f"Layer[{self._lid}].experts"):
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0).nonzero()
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
        EXPERT_E[self._lid].record()

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
