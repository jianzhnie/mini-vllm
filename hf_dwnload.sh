#!/bin/bash

# 设置国内镜像
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

# 模型下载
hf download Qwen/Qwen3-0.6B --local-dir ~/hfhub/models/Qwen/Qwen3-0.6B

# 数据集下载
hf download --repo-type dataset openai/gsm8k --local-dir ~/hfhub/datasets/openai/gsm8k
