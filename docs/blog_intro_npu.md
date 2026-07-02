# mini-vLLM：一个对华为昇腾 NPU 友好的轻量级 LLM 推理引擎

> 在 NVIDIA 之外，国产 NPU 正在成为大模型部署的重要选择。mini-vLLM 从零开始实现了对昇腾 NPU 的完整支持，并且代码足够简洁，能看懂每一行在做什么。

## 一、为什么关注 NPU？

过去几年，LLM 推理几乎等于 CUDA 生态。但越来越多的人开始面临这样的问题：

- **算力国产化**：政企、金融、运营商等场景要求使用国产芯片
- **供应链风险**：依赖单一生态不够安全
- **成本考虑**：NPU 在某些场景下性价比更高
- **学习需求**：想理解推理引擎原理，但 vLLM 的 CUDA-only 代码无法移植到 NPU 上实验

华为昇腾 NPU 是目前国产 AI 芯片中最成熟的路线之一，生态上有 CANN、`torch_npu`、HCCL 等完整工具链。问题是：**现有主流推理引擎对 NPU 的支持要么不够原生，要么代码太复杂，二次开发门槛高。**

mini-vLLM 尝试解决这个问题。

## 二、mini-vLLM 的 NPU 支持有多完整？

mini-vLLM 不是简单地把模型 "跑在 NPU 上"，而是围绕 NPU 的硬件特性做了一套完整的推理引擎实现。

### 2.1 自动设备检测

启动时不需要手动指定设备。只要环境里有 `torch_npu`，mini-vLLM 会自动识别 NPU：

```python
from minivllm import LLM, SamplingParams

# 无需 device="npu" 之类的参数
llm = LLM(model="Qwen/Qwen3-0.6B")
```

底层在 `minivllm/utils/device.py` 里维护了一套统一的设备抽象，自动处理 `cuda` / `npu` / `xpu` / `mps` / `mlu` / `musa` 的差异。

### 2.2 三种注意力后端可选

针对 NPU，mini-vLLM 实现了三层注意力策略：

| 后端 | 触发条件 | 特点 |
|------|----------|------|
| **PyTorch SDPA** | 默认 | 兼容性最好，CANN 对 SDPA 优化已经很不错 |
| **NPU Flash Attention** | `MINIVLLM_USE_NPU_FA=1` | 调用 `torch_npu.npu_fusion_attention`，性能更优 |
| **Unified Inference** | NPU FA 开启 + 非 GQA | 调用 `npu_fused_infer_attention_score`，自动区分 prefill/decode |

之所以把 SDPA 作为默认，是因为在部分 CANN 版本上，NPU Flash Attention 对 packed prefill 3D 输入存在兼容性问题。这种 "稳妥优先、高性能可选" 的设计，让生产环境更稳。

### 2.3 专用 NPU Flash Attention 模块

`minivllm/models/layers/npu_flash_attention.py` 是专门为昇腾写的封装层：

- 使用 **BNSD 布局**（Batch × NumHeads × Seq × Dim），这是 NPU 上最优的数据排布
- 支持 **Sparse Mode 3** 右下因果掩码，适配 GPT 类解码模型
- 支持通过 `NPU_FA2_SPARSE_MODE=2` 切换到左上对齐因果掩码
- 自动缓存 attention mask，避免重复分配

核心调用大致长这样：

```python
from minivllm.models.layers.npu_flash_attention import npu_flash_attn_func

output = npu_flash_attn_func(
    q, k, v,
    causal=True,
    input_layout="BSND",  # 或 BNSD
)
```

### 2.4 统一的 Prefill / Decode 推理接口

对于支持 `npu_fused_infer_attention_score` 的 CANN 版本，mini-vLLM 会自动走统一推理路径：

```
Query 长度 = 1?  →  npu_incre_flash_attention  (decode)
Query 长度 > 1?  →  npu_prompt_flash_attention (prefill)
```

这个接口最大的优势是：**调用方不用关心当前是 prefill 还是 decode**，引擎自动选择最优分支。

### 2.5 HCCL 张量并行

多卡 NPU 推理通过 **HCCL（Huawei Collective Communication Library）** 实现，和 NCCL 对应。

mini-vLLM 支持 1-8 卡张量并行：

```python
from minivllm import LLM, SamplingParams
from minivllm.config import Config

config = Config(
    model="Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size=4,  # 4 张 NPU
    device_memory_utilization=0.85,
)
llm = LLM(config)
```

底层的 `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear` 会自动按卡切分权重，并在需要的位置插入 `all-reduce`。

### 2.6 NPU Graph 捕获

和 CUDA Graph 类似，NPU 也支持把 decode 阶段的计算图捕获下来重复执行。mini-vLLM 在 `minivllm/utils/device.py` 里统一了 `CUDAGraph` 和 `NPUGraph` 的捕获逻辑：

```python
from minivllm.utils.device import DeviceGraphContext, get_device_graph_class

GraphClass = get_device_graph_class()  # torch.npu.NPUGraph
```

NPU 捕获时需要非默认 stream，`DeviceGraphContext` 已经自动处理了这些细节。

## 三、NPU 上的实际用法

### 3.1 基础推理

```python
import os
from minivllm import LLM, SamplingParams
from minivllm.config import Config

# 可选：开启 NPU Flash Attention
os.environ["MINIVLLM_USE_NPU_FA"] = "1"

config = Config(
    model="Qwen/Qwen3-0.6B",
    max_num_seqs=8,
    max_model_len=512,
    device_memory_utilization=0.85,
    dtype="float16",
)

llm = LLM(config)

params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=64)
outputs = llm.generate(
    ["Hello, who are you?", "What is the capital of China?"],
    params,
)

for out in outputs:
    print(out["text"])
```

### 3.2 多卡张量并行

```python
config = Config(
    model="Qwen/Qwen2-7B-Instruct",
    tensor_parallel_size=4,
    device_memory_utilization=0.8,
    dtype="float16",
)
llm = LLM(config)
```

### 3.3 快速检查 NPU 环境

项目里提供了一个环境检查脚本：

```bash
python examples/check_npu_graph.py
```

输出会告诉你：

- 当前有多少张 NPU
- 设备名称
- 哪些 `torch_npu` Flash Attention API 可用
- SDPA 前向是否能跑通

## 四、NPU 相关的代码设计亮点

### 4.1 设备抽象足够薄

`minivllm/utils/device.py` 没有搞过度封装，而是提供了一组统一的函数：

```python
get_current_device()      # 根据 LOCAL_RANK 返回 npu:0 / npu:1 ...
get_distributed_backend() # NPU 返回 "hccl"
empty_cache()             # 调用 torch.npu.empty_cache()
synchronize()             # 调用 torch.npu.synchronize()
mem_get_info()            # 获取 NPU 显存
get_device_graph_class()  # 返回 torch.npu.NPUGraph
```

这些函数对上层隐藏了 `cuda` 和 `npu` 的 API 差异，但本身非常薄，读起来不费力。

### 4.2 KV Cache 更新针对 NPU 优化

在 `attention_backend.py` 里，KV Cache 的 scatter 更新对 NPU 做了特殊处理：

- indices 转成 `int32`，NPU 上更稳定
- 使用 `index_copy_` 替代 `index_put_`，在 bbfloat16 等场景下验证过正确性
- 保证 tensor contiguous，减少 NPU 内部转换开销

### 4.3 采样层禁用 torch.compile

`minivllm/sampling/functional.py` 里检测到 NPU 时自动跳过 `torch.compile`，避免后端不支持导致的错误。

### 4.4 随机种子统一设置

`minivllm/utils/random_utils.py` 会同时设置 Python / numpy / torch / CUDA / NPU 的随机种子，保证实验可复现。

### 4.5 GQA 模型自动降级

NPU Flash Attention 目前不支持 GQA（`num_heads != num_kv_heads`）。mini-vLLM 在 `Attention.forward` 里做了安全检查：

```python
_npu_fa_safe = self.num_heads == self.num_kv_heads
```

如果是 GQA 模型，自动走 SDPA 路径，避免输出乱码。

## 五、性能与兼容性

### 环境要求

| 组件 | 推荐版本 |
|------|----------|
| 硬件 | Atlas 910B / 910B A2 / A3 |
| CANN | 8.0.RC1+ |
| PyTorch | 2.3.0+（含 torch_npu） |
| Python | 3.10–3.12 |

### 启用 NPU Flash Attention 的推荐配置

```bash
export MINIVLLM_USE_NPU_FA=1
# 如果你的模型使用标准 GPT 因果掩码（多数都是）
export NPU_FA2_SPARSE_MODE=3
```

如果因果掩码方向不对，可以尝试：

```bash
export NPU_FA2_SPARSE_MODE=2
```

### 什么时候用 SDPA，什么时候用 NPU FA？

| 场景 | 推荐 |
|------|------|
| 调试开发 / 不稳定 CANN | SDPA（默认） |
| 生产性能优化 | NPU FA（`MINIVLLM_USE_NPU_FA=1`） |
| GQA 模型（如 Llama-3-8B） | SDPA 或 unified inference |
| 长序列 / 高并发 | NPU FA + unified inference |

## 六、mini-vLLM 适合哪些 NPU 用户？

- **国产化部署**：需要在昇腾 NPU 上跑 Qwen、OPT 等模型的团队
- **推理引擎学习**：想深入理解 NPU 推理引擎实现，但不想读 vLLM 十几万行代码的人
- **二次开发**：需要在 NPU 上验证新调度策略、采样策略、KV 缓存优化的研究者
- **跨平台产品**：希望同一套代码同时支持 CUDA 和 NPU 的开发者

## 七、写在最后

NPU 生态正在快速成熟，但好的开发工具还是太少。mini-vLLM 想做的是：**提供一份清晰、可运行、可修改的 NPU 推理引擎参考实现**。

它不是 vLLM 的替代品，而是理解原理、快速验证、轻量部署的利器。如果你手上有昇腾 NPU，或者只是对国产 AI 芯片生态感兴趣，都可以试试这个项目。

项目已经开源，欢迎大家试用、提 issue、贡献代码：

**GitHub**: https://github.com/jianzhnie/mini-vllm

相关示例：

```bash
python examples/check_npu_graph.py          # 检查 NPU 环境
python examples/npu_inference_example.py    # 单卡推理
python examples/npu_tp_example.py --all     # TP=1/2/4 对比
python examples/npu_flash_attention_example.py  # NPU FA 接口示例
```

如果觉得有帮助，欢迎在 GitHub 点个 star，也欢迎在评论区交流 NPU 部署经验。

---

**许可证**：Apache 2.0

**致谢**：vLLM、nano-vllm、Transformers、PyTorch、FlashAttention、torch_npu
