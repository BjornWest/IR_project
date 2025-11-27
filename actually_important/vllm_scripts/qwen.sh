source ~/vllm_env/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.95 \
    --host 127.0.0.1 \
    --max-num-seqs 256 \
    --enable-chunked-prefill \
    --port 8000