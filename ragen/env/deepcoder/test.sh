#!/usr/bin/env bash
set -e

export HF_HOME="/projects/e32695/huggingface"
export HF_DATASETS_CACHE="/projects/e32695/huggingface/datasets"
export PYTHONPATH="/gpfs/home/bba1908/RAGEN2/test2:/gpfs/home/bba1908/RAGEN2/test2/verl"
cd /gpfs/home/bba1908/RAGEN2/test2
python -m ragen.env.deepcoder.env
