# from vllm import LLM
# llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite", enforce_eager=True)  # 先和你 engine_args 保持一致
# out = llm.generate({"prompt": "Hello"}, sampling_params={"max_tokens": 16})
# print(out[0].outputs[0].text)
from optimum_benchmark import (
    Benchmark, BenchmarkConfig, InferenceConfig,
    VLLMConfig, LlamaCppConfig, PyTorchConfig, ProcessConfig
)
from optimum_benchmark.logging_utils import setup_logging

# model_id = "Qwen/Qwen3-30B-A3B"
model_id = "deepseek-ai/DeepSeek-V2-Lite"


def build_config(backend_name: str):

    vllm_config = VLLMConfig(
        model=model_id,
        device="cuda",
        device_ids="0",
        no_weights=False,
        serving_mode="offline",
        engine_args={"enforce_eager": True},
    )
    llama_cpp_config = LlamaCppConfig(
        model="./models/deepseek_V2_Lite.gguf",
        device="cuda",
        device_ids="0",
        task="text-generation",
        # filename="deepseek_V2_Lite.gguf",  # 确保文件真实存在
    )
    torch_config = PyTorchConfig(
        model=model_id, device="cuda", device_ids="0",
        no_weights=False, torch_dtype="bfloat16"
    )

    if backend_name == "vllm":
        backend = vllm_config
    elif backend_name == "llama_cpp":
        backend = llama_cpp_config
    elif backend_name == "pytorch":
        backend = torch_config
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    setup_logging(level="INFO")

    # 如需隔离显卡进程，用 device_isolation=True；只是更严格的 spawn
    launcher_config = ProcessConfig(
        device_isolation=False, device_isolation_action="warn")

    scenario_config = InferenceConfig(
        latency=True, memory=False, energy=False, warmup_runs=2,
        input_shapes={"batch_size": 1, "sequence_length": 64},
        generate_kwargs={"max_new_tokens": 64, "min_new_tokens": 64},
    )

    benchmark_config = BenchmarkConfig(
        name=f"{backend_name}_deepseek_v2_lite",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend,
    )
    return benchmark_config


def main():
    backend_name = "llama_cpp"  # "pytorch" / "llama_cpp"
    cfg = build_config(backend_name)
    # 用 launch（会起子进程）；现在已有 main-guard，不会再报错
    report = Benchmark.launch(cfg)
    Benchmark(config=cfg, report=report).save_json(
        f"{backend_name}_{model_id[:model_id.index('/')]}_benchmark.json")


if __name__ == "__main__":
    main()
