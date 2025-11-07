from llama_cpp import Llama

model_id = "deepseek-ai/DeepSeek-V2-Lite"
llm = Llama.from_pretrained(
    repo_id=model_id,
    filename="deepseek_V2_Lite.gguf",
    local_dir="./models",
    n_gpu_layers=99,  # Uncomment to use GPU acceleration
    main_gpu=0,
    seed=1337,  # Uncomment to set a specific seed
    n_ctx=2048,  # Uncomment to increase the context window
)
output = llm(
    "Q: what is llama.cpp? A: ",  # Prompt
    max_tokens=128,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
    # Stop generating just before the model would generate a new question
    stop=["Q:", "\n"],
    echo=True  # Echo the prompt back in the output
)  # Generate a completion, can also call create_completion
print(output)
