#!/bin/bash

#SBATCH --job-name=1xA40                                                                     
#SBATCH --partition=gpuA40x4                                                                 
#SBATCH --account=bchk-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:1                                                                         
#SBATCH --time=3:00:00                                                                       
#SBATCH --output=math_word_problems_%j.out
#SBATCH --mail-user=weinberg.ai@northeastern.edu
#SBATCH --mail-type="BEGIN,END"
/scratch/bchk/aguha/venv/bin/python3 math_word_problems.py /scratch/bchk/models/Qwen3-8B-Base
