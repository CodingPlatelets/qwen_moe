#!/bin/bash
source ../venv/bin/activate
vllm bench throughput --model "deepseek-ai/DeepSeek-V2-Lite" --input-len 64  --output-len 1024 --enforce_eager --otlp-traces-endpoint http://tracing-analysis-dc-hz.aliyuncs.com/adapt_f1cy4vvlbv@369c875776e23e8_f1cy4vvlbv@53df7ad2afe8301/api/otlp/traces --collect-detailed-traces all