# mini-vLLM 项目指南

> 本文件为 AI 编程助手提供项目背景、开发指南和最佳实践。

## 项目概述

**mini-vLLM** 是一个从零开始构建的轻量级大语言模型推理引擎，使用 Python 编写。它提供了一个简单、高效、可扩展的推理解决方案，旨在帮助开发者快速理解大语言模型的推理原理。

### 核心特性

- **CUDA Graph 优化**: 减少 decode 阶段的调度开销，提升推理速度
- **智能批处理**: 支持动态批处理和序列长度感知调度
- **KV 缓存优化**: 块式 KV 缓存管理和前缀缓存，减少内存占用
- **多设备支持**: CUDA、NPU、XPU、MPS、MLU、MUSA
- **张量并行**: 支持多 GPU 分布式推理（1-8 张量并行度）
- **两阶段调度**: 预填充（Prefill）和解码（Decode）阶段分离调度
- **FlashAttention**: 集成高性能注意力计算库

### 项目定位

- **教育平台**: 适合学习和理解大语言模型推理原理
- **开发工具**: 提供清晰的模块化实现，便于二次开发
- **轻量级部署**: 适用于需要高效推理但不想承担复杂依赖的场景
- **研究实验台**: 作为研究新推理优化技术的理想实验平台

## 技术栈

- **编程语言**: Python 3.10 - 3.12
- **深度学习框架**: PyTorch >= 2.2.0
- **模型支持**: Transformers >= 4.37.0
- **核心依赖**:
  - `torch`: 深度学习计算
  - `transformers`: HuggingFace 模型加载
  - `flash-attn`: 高性能注意力计算（可选，CUDA）
  - `triton`: GPU 内核优化（可选，CUDA）
  - `torch-npu`: NPU 支持（可选）
  - `xxhash`: 高效哈希算法（前缀缓存）
  - `colorama`: 日志颜色输出
  - `safetensors`: 安全张量序列化

## 项目结构

```
mini-vllm/
├── minivllm/                   # 主包目录
│   ├── __init__.py            # 导出核心类和版本信息
│   ├── llm.py                 # LLM 类入口（用户接口）
│   ├── config.py              # 配置类定义（Config）
│   ├── sampling_params.py     # 采样参数定义（SamplingParams）
│   ├── engine/                # 推理引擎核心模块
│   │   ├── llm_engine.py      # 推理引擎主类（LLMEngine）
│   │   ├── model_runner.py    # 模型运行器
│   │   ├── inference_executor.py  # 推理执行器
│   │   ├── scheduler.py       # 批次调度器
│   │   ├── sequence.py        # 序列数据结构
│   │   ├── block_manager.py   # KV缓存块管理器
│   │   └── distributed_manager.py  # 分布式管理
│   ├── models/                # 模型实现
│   │   ├── qwen2.py           # Qwen2 模型实现
│   │   ├── qwen3.py           # Qwen3 模型实现
│   │   ├── opt.py             # OPT 模型实现
│   │   ├── manager.py         # 模型管理器
│   │   ├── sampler.py         # 模型采样器
│   │   └── layers/            # 神经网络层
│   │       ├── attention.py   # 注意力机制
│   │       ├── attention_backend.py  # 注意力后端抽象
│   │       ├── npu_flash_attention.py # NPU Flash Attention
│   │       ├── linear.py      # 张量并行线性层
│   │       ├── embed_head.py  # 嵌入和输出层
│   │       ├── rotary_embedding.py  # 旋转位置编码（RoPE）
│   │       ├── layernorm.py   # RMSNorm 层
│   │       └── activation.py  # 激活函数
│   ├── sampling/              # 采样模块
│   │   ├── sampler.py         # 采样器实现
│   │   ├── config.py          # 采样配置
│   │   ├── functional.py      # 采样函数
│   │   └── mirostat.py        # Mirostat 采样
│   └── utils/                 # 工具模块
│       ├── context.py         # 全局上下文管理
│       ├── device.py          # 设备检测和管理（多设备支持）
│       ├── loader.py          # 模型权重加载
│       ├── memory_pool.py     # 内存池管理
│       ├── random_utils.py    # 随机工具
│       └── logger_utils.py    # 日志工具
├── tests/                     # 测试目录（18个测试文件）
├── examples/                  # 示例代码
├── docs/                      # 文档
│   ├── multi-process.md       # 多进程设计文档
│   └── npu_flash_attention_guide.md  # NPU Flash Attention 指南
├── pyproject.toml            # 项目配置
└── examples.py               # 主示例脚本
```

## 构建和安装

### 环境要求

- Python 3.10 - 3.12
- CUDA 11.8+ / NPU / XPU / MPS / MLU / MUSA（用于加速推理）
- 至少 8GB GPU 内存（推荐）

### 安装命令

```bash
# 基础安装
pip install -e .

# 安装 CUDA 优化版本（推荐）
pip install -e ".[cuda]"

# 安装 NPU 版本
pip install -e ".[npu]"

# 安装开发依赖
pip install -e ".[dev]"
```

### 验证安装

```bash
python -c "import minivllm; print('Installation successful!')"
```

## 测试策略

### 运行测试

```bash
# 运行所有测试（默认排除慢测试）
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_config.py -v

# 运行匹配特定模式的测试
python -m pytest -k test_sampling

# 运行带覆盖率的测试
python -m pytest tests/ --cov=minivllm --cov-report=term-missing

# 包含慢测试
python -m pytest tests/ --slow

# 运行集成测试
python -m pytest -m integration

# 使用测试运行脚本
python tests/run_tests.py --coverage -v
```

### 测试分类

| 测试文件 | 说明 |
|---------|------|
| `test_config.py` | 配置类验证测试 |
| `test_sampling_params.py` | 采样参数验证测试 |
| `test_llm.py` | LLM 接口测试 |
| `test_llm_engine.py` | 推理引擎测试 |
| `test_model_runner.py` | 模型运行器测试 |
| `test_scheduler.py` | 调度器测试 |
| `test_sequence.py` | 序列管理测试 |
| `test_block_manager.py` | KV 缓存块管理器测试 |
| `test_linear.py` | 张量并行线性层测试 |
| `test_sampler.py` | 采样器测试 |
| `test_rotary_embedding.py` | 旋转位置编码测试 |
| `test_loaders.py` | 模型加载器测试 |
| `test_device.py` | 设备管理测试 |
| `test_npu_attention.py` | NPU 注意力测试 |
| `test_npu_backend.py` | NPU 后端测试 |
| `test_qwen3_npu.py` | Qwen3 NPU 测试 |
| `test_integration.py` | 集成测试 |
| `run_tests.py` | 测试运行脚本 |

## 代码风格指南

### 格式化工具

项目使用以下工具进行代码格式化：

```bash
# 使用 black 格式化代码
black minivllm/ tests/

# 使用 ruff 进行代码检查
ruff check minivllm/ tests/

# 自动修复 ruff 检测到的问题
ruff check --fix minivllm/ tests/

# 使用 mypy 进行类型检查
mypy minivllm/
```

### 风格配置

- **行长度**: 88 字符（与 Black 一致）
- **目标 Python 版本**: 3.10
- **引号风格**: 双引号
- **缩进**: 4 个空格

### Pre-commit Hooks

```bash
# 安装 pre-commit hooks
pre-commit install

# 在所有文件上运行
pre-commit run --all-files
```

## 核心架构设计

### 1. 两阶段调度

```
User Prompt → LLM.generate() → LLMEngine.add_request()
    → Sequence created → Scheduler.waiting queue
    → Prefill phase: ModelRunner.execute_model() → KV cache populated
    → Decode phase: Generate one token per step
    → Sequence.FINISHED → Return to user
```

**Prefill 阶段**: 首次处理用户输入的提示文本，生成初始的 KV 缓存
**Decode 阶段**: 基于已有的 KV 缓存，每次生成一个新 token

### 2. 块化 KV 缓存

- 固定大小的块（默认 256 个 token，必须被 256 整除）
- 前缀缓存 via xxhash 用于内存共享
- 引用计数管理块生命周期
- Copy-on-write 机制

### 3. 张量并行

- Rank 0 运行在主进程，Rank 1-N 作为工作进程
- 共享内存 IPC 进行进程间通信
- 事件同步机制
- 分布式后端: NCCL (CUDA)、HCCL (NPU)、CCL (XPU)、CNCL (MLU)

### 4. CUDA Graph 优化

- 捕获 decode 阶段的计算图
- 为不同 batch size 存储图
- 通过 `enforce_eager=True` 禁用（调试用）
- NPU 设备也支持类似的图优化

### 5. 多设备支持

```python
# 支持的设备类型
DEVICE_TYPE_CUDA = 'cuda'    # NVIDIA GPU
DEVICE_TYPE_NPU = 'npu'      # Huawei Ascend
DEVICE_TYPE_XPU = 'xpu'      # Intel GPU
DEVICE_TYPE_MPS = 'mps'      # Apple Silicon
DEVICE_TYPE_MLU = 'mlu'      # Cambricon
DEVICE_TYPE_MUSA = 'musa'    # Moore Threads
DEVICE_TYPE_CPU = 'cpu'      # CPU (fallback)
```

## 配置参数

### Config 类关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必需 | 模型路径（HuggingFace 格式） |
| `max_num_batched_tokens` | int | 16384 | 每批次最大 token 数 [1, 1000000] |
| `max_num_seqs` | int | 512 | 每批次最大序列数 [1, 10000] |
| `max_model_len` | int | 4096 | 模型最大上下文长度 [1, 1000000] |
| `device_memory_utilization` | float | 0.9 | 设备内存利用率 [0.1, 1.0] |
| `tensor_parallel_size` | int | 1 | 张量并行度 [1, 8] |
| `enforce_eager` | bool | False | 强制使用 eager 模式 |
| `eos` | int | -1 | 结束符 token ID（-1 自动检测） |
| `kvcache_block_size` | int | 256 | KV 缓存块大小（必须被 256 整除） |
| `num_kvcache_blocks` | int | -1 | KV 缓存块数量（-1 自动） |
| `trust_remote_code` | bool | False | 是否信任远程代码 |
| `dtype` | str | 'auto' | 模型权重数据类型 |
| `seed` | int | None | 随机种子 |

**注意**: `gpu_memory_utilization` 是 `device_memory_utilization` 的兼容别名。

### SamplingParams 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 采样温度 (> 1e-10) |
| `top_p` | float | 1.0 | Nucleus 采样阈值 (0-1] |
| `top_k` | int | -1 | Top-k 采样 (-1 禁用，>0 启用) |
| `min_p` | float | 0.0 | 最小相对概率 [0, 1] |
| `max_tokens` | int | 64 | 最大生成 token 数 |
| `ignore_eos` | bool | False | 是否忽略结束符 |

## 开发指南

### 添加新模型架构

1. 在 `minivllm/models/` 创建模型文件，遵循 HuggingFace 格式
2. 在 `minivllm/models/manager.py` 的 `_detect_model_type()` 中注册模型类型
3. 在 `minivllm/models/manager.py` 的 `_load_model()` 中添加模型实例化逻辑

示例：
```python
# 在 _detect_model_type() 中添加
if 'my_model' in model_type:
    return 'my_model'

# 在 _load_model() 中添加
if self.model_type == 'my_model':
    self.model = MyModelForCausalLM(hf_config)
```

### 添加新设备支持

1. 在 `minivllm/utils/device.py` 中添加设备类型常量
2. 更新 `get_current_device()`, `get_device_count()`, `empty_cache()` 等函数
3. 在 `get_distributed_backend()` 中添加分布式后端

### 调试技巧

- 设置 `enforce_eager=True` 禁用 CUDA Graph，便于调试
- 使用 `logger.setLevel(logging.DEBUG)` 查看详细日志
- 通过 `scheduler.block_manager.get_num_free_blocks()` 检查 KV 缓存使用
- 设置 `trust_remote_code=True` 支持自定义模型

### 支持的模型

| 模型 | 状态 | 说明 |
|------|------|------|
| Qwen2 | ✅ 完整支持 | 推荐使用 |
| Qwen3 | ✅ 完整支持 | 推荐使用 |
| OPT | ✅ 完整支持 | 测试用模型 |
| LLaMA | ⚠️ 兼容 | 通过 Qwen2 架构兼容 |

## 安全考虑

1. **模型加载**: 默认使用 `trust_remote_code=False` 防止执行未验证的代码
2. **输入验证**: 所有配置参数都经过范围验证
3. **内存安全**: 设备内存利用率限制防止 OOM
4. **类型安全**: 使用 mypy 进行静态类型检查

## 常用命令速查

```bash
# 快速测试
python examples.py --model /path/to/model

# 多 GPU 推理
python -c "
from minivllm import LLM, SamplingParams
llm = LLM('model_path', tensor_parallel_size=4)
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=50))
print(outputs[0]['text'])
"

# NPU 推理
export ASCEND_RT_VISIBLE_DEVICES=0
python examples.py --model /path/to/model

# 性能分析
python -m pytest tests/test_llm.py -v --profile

# 代码质量检查
black --check minivllm/ tests/
ruff check minivllm/ tests/
mypy minivllm/

# 运行特定设备测试
python -m pytest tests/test_device.py -v
python -m pytest tests/test_npu_backend.py -v
```

## 相关文档

- `README.md` - 项目简介和快速开始
- `intro.md` - 详细的项目文档和教程
- `CLAUDE.md` - Claude Code 专用的开发指南
- `docs/multi-process.md` - 多进程设计文档
- `docs/npu_flash_attention_guide.md` - NPU Flash Attention 指南

## 许可证

Apache 2.0 许可证
