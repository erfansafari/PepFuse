#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate peptidebert-fusion
cd .
PYTHONPATH=.:$PYTHONPATH python model/train.py
