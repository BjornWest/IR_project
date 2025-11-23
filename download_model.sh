#!/bin/bash
#SBATCH --job-name=download_qwen
#SBATCH --partition=csedu
#SBATCH --account=csedui00041
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/bjornwesterlun/IR_project/logs/download_%j.out
#SBATCH --error=/home/bjornwesterlun/IR_project/logs/download_%j.err

# Create logs directory
mkdir -p /home/bjornwesterlun/IR_project/logs

# Activate venv
source /vol/csedu-nobackup/course/I00041_informationretrieval/users/bjorn/venv/bin/activate

# Run download
cd /home/bjornwesterlun/IR_project
python download_model.py

echo "Download job completed at $(date)"

