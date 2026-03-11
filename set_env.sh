#!/bin/bash

source ~/.bashrc

# 加载 CANN 环境变量（路径需根据实际安装位置调整）
# install_path=/usr/local/Ascend
install_path=/home/jianzhnie/llmtuner/Ascend/CANN8.2.RC1
source $install_path/ascend-toolkit/set_env.sh
source $install_path/nnal/atb/set_env.sh

# conda activate vllm091
source /home/jianzhnie/llmtuner/software/miniconda3/bin/activate vllm091
