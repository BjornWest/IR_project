#!/bin/bash
#SBATCH --job-name=qwen_generation
#SBATCH --partition=csedu
#SBATCH --qos=csedu-normal
#SBATCH --account=csedui00041
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/bjornwesterlun/IR_project/logs/generation_%j.out
#SBATCH --error=/home/bjornwesterlun/IR_project/logs/generation_%j.err

# Create logs directory if it doesn't exist
mkdir -p /home/bjornwesterlun/IR_project/logs

# Activate virtual environment (using smallenv now)
source /home/bjornwesterlun/smallenv/bin/activate

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi
echo "======================="

# Run the script
cd /home/bjornwesterlun/IR_project

# Using HuggingFace Transformers (stable with multi-GPU)
python response_generation_hf.py

echo "Job completed at $(date)"

