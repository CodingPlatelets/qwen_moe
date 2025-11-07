import torch
from typing import List, Dict
from dataclasses import dataclass, field


class CUDATimer:
    """
    Accurate per-interval device timing using CUDA events.
    Accumulates total_ms; keeps optional per-interval history.
    """

    def __init__(self, keep_history: bool = False):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0.0
        self.keep_history = keep_history
        self.history: List[float] = []

    def start(self):
        self.start_ev.record()

    def stop(self):
        # Record end, wait until completed, then read elapsed_time
        self.end_ev.record()
        self.end_ev.synchronize()
        dt = self.start_ev.elapsed_time(self.end_ev)
        self.total_ms += dt
        if self.keep_history:
            self.history.append(dt)


@dataclass
class TimerRegistry:
    """Timer registry keyed by phase and component per layer index."""
    num_layers: int
    keep_history: bool = False
    timers: Dict[str, List[CUDATimer]] = field(default_factory=dict)

    def ensure_key(self, key: str):
        if key not in self.timers:
            self.timers[key] = [CUDATimer(self.keep_history)
                                for _ in range(self.num_layers)]

    def start(self, key: str, lid: int):
        self.ensure_key(key)
        self.timers[key][lid].start()

    def stop(self, key: str, lid: int):
        self.timers[key][lid].stop()

    def sum_ms(self, key: str) -> float:
        self.ensure_key(key)
        return sum(t.total_ms for t in self.timers[key])

    def layer_ms(self, key: str, lid: int) -> float:
        self.ensure_key(key)
        return self.timers[key][lid].total_ms


_TREG: TimerRegistry = None


def init_timer_registry(num_layers: int, keep_history: bool = True):
    global _TREG
    _TREG = TimerRegistry(num_layers=num_layers, keep_history=keep_history)


def warmup_model(model, tokenizer, prompt: str | List[str], max_new_tokens: int = 2):
    """Warm up both prefill and decode path to stabilize kernels/autotune."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(
            **inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()


def print_timers_summary():
    """Pretty print per-layer and totals for both prefill & decode phases."""
    assert _TREG is not None, "Timer registry not initialized."
    N = _TREG.num_layers

    def safe(key):  # ensure key exists
        _TREG.ensure_key(key)
        return key

    keys = [
        "attn_prefill", "mlp_prefill", "gating_prefill", "softmax_prefill", "expert_prefill", "norm_prefill", "router_prefill", "dispatch_prefill", "compute_prefill", "aggregate_prefill",
        "attn_decode", "mlp_decode", "gating_decode", "softmax_decode", "expert_decode", "norm_decode", "router_decode", "dispatch_decode", "compute_decode", "aggregate_decode",
    ]
    for k in keys:
        _TREG.ensure_key(k)

    print("=== Per-layer (ms) ===")

    prefill_keys = [safe(k) for k in keys if k.endswith("_prefill")]
    decode_keys = [safe(k) for k in keys if k.endswith("_decode")]

    def format_col_name(key: str) -> str:
        name, suffix = key.split("_", 1)
        label = "PF" if suffix == "prefill" else "DEC"
        return f"{name}({label})"

    header_prefill = "\t\t".join(format_col_name(k) for k in prefill_keys)
    header_decode = "\t\t".join(format_col_name(k) for k in decode_keys)
    print(f"layer\t{header_prefill}\t||\t{header_decode}")

    for i in range(N):
        prefill_vals = [_TREG.layer_ms(k, i) for k in prefill_keys]
        decode_vals = [_TREG.layer_ms(k, i) for k in decode_keys]
        prefill_str = "\t\t".join(f"{val:.3f}" for val in prefill_vals)
        decode_str = "\t\t".join(f"{val:.3f}" for val in decode_vals)
        print(f"L{i:02d}\t{prefill_str}\t||\t{decode_str}")

    print("\n=== Totals (ms) ===")

    def total(prefix):
        return {
            "attn":   _TREG.sum_ms(f"attn_{prefix}"),
            "mlp":    _TREG.sum_ms(f"mlp_{prefix}"),
            "gating": _TREG.sum_ms(f"gating_{prefix}"),
            "softmax": _TREG.sum_ms(f"softmax_{prefix}"),
            "expert": _TREG.sum_ms(f"expert_{prefix}"),
            "norm":   _TREG.sum_ms(f"norm_{prefix}"),
            "router": _TREG.sum_ms(f"router_{prefix}"),
            "dispatch": _TREG.sum_ms(f"dispatch_{prefix}"),
            "compute": _TREG.sum_ms(f"compute_{prefix}"),
            "aggregate": _TREG.sum_ms(f"aggregate_{prefix}"),
        }

    tp = total("prefill")
    td = total("decode")
    for name in ["attn", "mlp", "gating", "softmax", "expert", "norm", "router", "dispatch", "compute", "aggregate"]:
        print(
            f"{name:7s} prefill={tp[name]:.3f} ms\tdecode={td[name]:.3f} ms\tall={tp[name]+td[name]:.3f} ms")

    mlp_all = tp["mlp"] + td["mlp"]
    attn_all = tp["attn"] + td["attn"]
    total_am = mlp_all + attn_all + tp["norm"] + td["norm"]
    if total_am > 0:
        print(f"\nmlp_ratio_over_(attn+mlp) = {mlp_all / total_am * 100:.2f}%")

    print("\n(Notes) prefill=首段批量计算；decode=逐token阶段（利用 KV cache）。")
    print("Timing is device-side via CUDA events; each segment synchronized at stop for accuracy.")
