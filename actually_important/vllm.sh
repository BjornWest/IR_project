vllm serve openai/gpt-oss-20b \
    --gpu-memory-utilization 0.95 \
    --host 127.0.0.1 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --max-model-len 16384