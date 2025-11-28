source ~/vllm_env/bin/activate
vllm serve openai/gpt-oss-20b \
    --gpu-memory-utilization 0.47 \
    --host 127.0.0.1 \
    --max-num-seqs 128 \
    --enable-chunked-prefill \
    --port 8001