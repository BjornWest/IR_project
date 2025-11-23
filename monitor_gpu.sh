#!/bin/bash

while true; do
    clear
    echo "$(date +%H:%M:%S) | GPU Monitor (Ctrl+C to exit)"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits | while IFS=, read -r idx gpu_util mem_util mem_used mem_total; do
        mem_gb=$(awk "BEGIN {printf \"%.1f/%.1f\", $mem_used/1024, $mem_total/1024}")
        printf "GPU%s: Compute=%3s%% MemBW=%3s%% VRAM=%sGB\n" "$idx" "$gpu_util" "$mem_util" "$mem_gb"
    done
    sleep 1
done

