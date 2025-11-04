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
        "attn_prefill", "mlp_prefill", "gating_prefill", "softmax_prefill", "expert_prefill",
        "attn_decode", "mlp_decode", "gating_decode", "softmax_decode", "expert_decode"
    ]
    for k in keys:
        _TREG.ensure_key(k)

    print("=== Per-layer (ms) ===")
    print("layer\tattn(PF)\tmlp(PF)\tgating(PF)\tsoftmax(PF)\texpert(PF)\t||\tattn(DEC)\tmlp(DEC)\tgating(DEC)\tsoftmax(DEC)\texpert(DEC)")
    for i in range(N):
        ap = _TREG.layer_ms(safe("attn_prefill"), i)
        mp = _TREG.layer_ms(safe("mlp_prefill"), i)
        gp = _TREG.layer_ms(safe("gating_prefill"), i)
        sp = _TREG.layer_ms(safe("softmax_prefill"), i)
        ep = _TREG.layer_ms(safe("expert_prefill"), i)

        ad = _TREG.layer_ms(safe("attn_decode"), i)
        md = _TREG.layer_ms(safe("mlp_decode"), i)
        gd = _TREG.layer_ms(safe("gating_decode"), i)
        sd = _TREG.layer_ms(safe("softmax_decode"), i)
        ed = _TREG.layer_ms(safe("expert_decode"), i)
        print(f"L{i:02d}\t{ap:.3f}\t\t{mp:.3f}\t\t{gp:.3f}\t\t{sp:.3f}\t\t{ep:.3f}\t\t||\t{ad:.3f}\t\t{md:.3f}\t\t{gd:.3f}\t\t{sd:.3f}\t\t{ed:.3f}")

    print("\n=== Totals (ms) ===")

    def total(prefix):
        return {
            "attn":   _TREG.sum_ms(f"attn_{prefix}"),
            "mlp":    _TREG.sum_ms(f"mlp_{prefix}"),
            "gating": _TREG.sum_ms(f"gating_{prefix}"),
            "softmax": _TREG.sum_ms(f"softmax_{prefix}"),
            "expert": _TREG.sum_ms(f"expert_{prefix}"),
        }

    tp = total("prefill")
    td = total("decode")
    for name in ["attn", "mlp", "gating", "softmax", "expert"]:
        print(
            f"{name:7s} prefill={tp[name]:.3f} ms\tdecode={td[name]:.3f} ms\tall={tp[name]+td[name]:.3f} ms")

    mlp_all = tp["mlp"] + td["mlp"]
    attn_all = tp["attn"] + td["attn"]
    total_am = mlp_all + attn_all
    if total_am > 0:
        print(f"\nmlp_ratio_over_(attn+mlp) = {mlp_all / total_am * 100:.2f}%")

    print("\n(Notes) prefill=首段批量计算；decode=逐token阶段（利用 KV cache）。")
    print("Timing is device-side via CUDA events; each segment synchronized at stop for accuracy.")
