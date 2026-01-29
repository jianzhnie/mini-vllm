# NPU Flash Attention 使用文档

## 概述

NPU Flash Attention 是华为昇腾 NPU 提供的高性能注意力计算接口，主要包括以下核心API：

### 训练场景
- `torch_npu.npu_fusion_attention`: 融合注意力计算，适用于训练场景

### 推理场景

- `torch_npu.npu_incre_flash_attention`: 增量Flash Attention，适用于推理阶段
- `torch_npu.npu_prompt_flash_attention`: 全量Flash Attention，适用于prefill阶段
- `torch_npu.npu_fused_infer_attention_score`: 融合推理注意力，统一接口
- `torch_npu.npu_advance_step_flashattn`: vLLM专用的step flash attention

## 1. 训练场景 - torch_npu.npu_fusion_attention

### 功能简介

`npu_fusion_attention` 是用于处理变长序列（varlen）场景的融合注意力计算接口, 在 Ascend NPU 上融合计算 Transformer 中的 Attention Score：

```python
attention_out = Softmax( (Q·Kᵀ) * scale + mask ) · V
```

### 函数原型
```python
torch_npu.npu_fusion_attention(
    query,               # Tensor
    key,                 # Tensor
    value,               # Tensor
    head_num,            # int
    input_layout,        # str
    pse=None,            # Tensor, optional
    padding_mask=None,   # Tensor, 暂不支持
    atten_mask=None,     # Tensor, optional
    scale=1.0,           # float, optional
    keep_prob=1.0,       # float, optional
    pre_tockens=2147483647,    # int, optional
    next_tockens=2147483647,   # int, optional
    inner_precise=0,     # int, optional
    prefix=None,         # Tensor, optional
    actual_seq_qlen=None,# Tensor, optional (varlen)
    actual_seq_kvlen=None,# Tensor, optional (varlen)
    sparse_mode=0,       # int, optional
    gen_mask_parallel=True, # bool, optional
    sync=False,          # bool, optional
    softmax_layout=None, # str, optional
    sink=None            # Tensor, optional
)
```

### 稀疏模式详解
- `0`: defaultMask（默认全量／带宽／上下三角等由 pre_tockens/next_tockens 控制）
- `1`: allMask（全量 Mask）
- `2/3`: leftUpCausal/rightDownCausal（上下三角压缩）
- `4`: band（带宽）
- `5/6`: prefix（非压缩/压缩）
- `7/8`: varlen 外切场景（基于 3/2）

### 使用示例

#### 基础训练场景
```python
import torch
import torch_npu
import math

# 构造输入数据 [B, S, N, D]
batch_size, seq_len, num_heads, head_dim = 2, 512, 8, 64
query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).npu()

# 构造注意力掩码
atten_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().npu()

# 调用融合注意力
attn_out, softmax_max, softmax_sum, _, seed, offset, mask_len = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num=num_heads,
    input_layout="BSNH",
    scale=1.0 / math.sqrt(head_dim),
    keep_prob=0.9,  # 启用dropout
    atten_mask=atten_mask,
    sparse_mode=0
)

print(f"输出形状: {attn_out.shape}")  # [2, 512, 8, 64]
```

#### 变长序列训练场景
```python
import torch
import torch_npu
import math

# 变长序列场景：batch中有不同长度的句子
total_tokens, num_heads, head_dim = 1000, 8, 64
query = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16).npu()
key = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16).npu()
value = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.float16).npu()

# 实际序列长度：句子1长度100，句子2长度200，句子3长度150，句子4长度550
actual_seq_qlen = [100, 300, 450, 1000]  # 累加长度
actual_seq_kvlen = [100, 300, 450, 1000]

# 调用变长序列注意力
attn_out, *_ = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num=num_heads,
    input_layout="TND",  # TotalTokens, NumHeads, HeadDim
    scale=1.0 / math.sqrt(head_dim),
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
    sparse_mode=0
)
```

## 2. torch_npu.npu_incre_flash_attention

### 功能简介

`npu_incre_flash_attention` 用于解码阶段的增量注意力计算，适用于自回归（autoregressive）推理场景：

```python
atten_out = softmax(scale_value * (query · key) + atten_mask) · value
```

### 函数原型
```python
torch_npu.npu_incre_flash_attention(
    query,
    key,
    value,
    *,
    padding_mask=None,
    pse_shift=None,
    atten_mask=None,
    actual_seq_lengths=None,
    dequant_scale1=None,
    quant_scale1=None,
    dequant_scale2=None,
    quant_scale2=None,
    quant_offset2=None,
    antiquant_scale=None,
    antiquant_offset=None,
    block_table=None,
    kv_padding_size=None,
    num_heads=None,
    scale_value=None,
    input_layout=None,
    num_key_value_heads=None,
    block_size=None,
    inner_precise=None
) → Tensor
```

### 主要参数说明

#### 必选参数
- **query**: Query 输入，形状支持 3D/4D（如 B×H×S×D 或 B×N×S×D 等）
- **key**: Key 输入，shape 与 query 保持一致的前三维
- **value**: Value 输入，shape 与 key 保持一致

#### 可选参数
- **actual_seq_lengths**: 每个 batch 的有效序列长度，一维向量，长度 = B
- **num_heads**: 注意力头数 H，默认从输入推断
- **scale_value**: 缩放系数，典型值 1/√D，默认 1.0
- **input_layout**: 输入布局，"BSH"或"BNSD"或"BSND"，默认"BSH"
- **num_key_value_heads**: K/V 头数，用于 Grouped-Query Attention 场景
- **block_table**: 二维映射表，用于 KV cache 的 block 索引映射
- **block_size**: page-attention 模式下每个 block 最大 token 数
- **inner_precise**: 精度控制，"high_precise"（高精度）或"high_performance"（高性能）

### 返回值
- **atten_out**: 与输入 query 形状一致的输出 Attention 值

### 使用示例

#### 单算子调用
```python
import torch
import torch_npu

# 假设已经构造好 query, key, value 张量
atten_out = torch_npu.npu_incre_flash_attention(
    query, key, value,
    num_heads=8,
    scale_value=1.0 / math.sqrt(head_dim),
    input_layout="BSH"
)
```

#### 图模式调用
```python
@torch.jit.script
def model(q, k, v):
    return torch_npu.npu_incre_flash_attention(
        q, k, v,
        num_heads=8,
        scale_value=1.0 / math.sqrt(head_dim),
        input_layout="BSH"
    )
```

## 3. 使用约束

### torch_npu.npu_fusion_attention
- 仅支持训练模式，不支持图模式
- Q/K/V 的数据类型和布局必须一致
- Batch 大小 B：非 varlen 场景 1 ～ 2,000,000；varlen 场景 1 ～ 2000
- 序列长度 S：1 ～ 1,000,000；varlen 场景下 B×S ≤ 1,000,000
- head_dim ≤ 768；支持多头（MHA）和分组查询注意力（GQA）

### torch_npu.npu_incre_flash_attention
- 仅支持推理（inference）场景，且需在图（Graph）模式下调用
- query/key/value 的 batch、head、seq_len、head_dim 等维度必须匹配
- head_dim（D）需16对齐
- 对于 page-attention，需同时传入 block_table 与 actual_seq_lengths

## 4. 在 mini-vllm 中的应用

在 mini-vllm 中，我们主要使用 `torch_npu.npu_incre_flash_attention` 来优化推理阶段的注意力计算，特别是在 decode 阶段处理增量 attention 计算。

### 接入要点
1. **Prefill 阶段**: 可以使用 `npu_flash_attn_varlen_func`（如果可用）
2. **Decode 阶段**: 使用 `torch_npu.npu_incre_flash_attention` 处理增量 attention
3. **GQA/MQA 支持**: 通过 `num_key_value_heads` 参数支持分组查询注意力
4. **KV Cache**: 集成 block_table 和 actual_seq_lengths 参数支持分块注意力

这些接口可以显著提升 NPU 设备上的注意力计算性能，特别适用于大语言模型的高效推理。

## 5. 新增API接口详解

### 5.1 torch_npu.npu_prompt_flash_attention

**功能**: Prefill阶段全量注意力计算，支持GQA、量化等高级特性。

**函数签名**:
```python
torch_npu.npu_prompt_flash_attention(
    query, key, value,
    *,
    pse_shift=None, padding_mask=None, atten_mask=None,
    actual_seq_lengths=None, actual_seq_lengths_kv=None,
    deq_scale1=None, quant_scale1=None, deq_scale2=None,
    quant_scale2=None, quant_offset2=None,
    num_heads=1, scale_value=1.0, pre_tokens=2147473647,
    next_tokens=0, input_layout="BSH", num_key_value_heads=0,
    sparse_mode=0
) -> Tensor
```

**使用示例**:
```python
import torch
import torch_npu
import math

# Prefill阶段: 处理多个tokens
batch_size, query_len, kv_len = 1, 64, 64
num_heads, head_dim = 8, 128

query = torch.randn(batch_size, query_len, num_heads, head_dim, dtype=torch.float16).npu()
key = torch.randn(batch_size, kv_len, num_heads, head_dim, dtype=torch.float16).npu()
value = torch.randn(batch_size, kv_len, num_heads, head_dim, dtype=torch.float16).npu()

scale = 1.0 / math.sqrt(head_dim)

# Prefill全量注意力，启用Causal掩码
prefill_out = torch_npu.npu_prompt_flash_attention(
    query, key, value,
    num_heads=num_heads,
    scale_value=scale,
    input_layout="BNSD",
    sparse_mode=3,  # rightDownCausal
    pre_tokens=65535,
    next_tokens=0
)

print(f"Prefill输出: {prefill_out.shape}")  # [1, 64, 8, 128]
```

### 5.2 torch_npu.npu_fused_infer_attention_score

**功能**: 统一推理接口，自动选择增量或全量计算模式。

**自适应逻辑**:
- Query序列长度=1 → 增量分支 (npu_incre_flash_attention)
- Query序列长度>1 → 全量分支 (npu_prompt_flash_attention)

**使用示例**:
```python
import torch
import torch_npu
import math

def attention_inference(query, key_cache, value_cache, num_heads, head_dim, kv_len):
    """统一的推理注意力接口"""
    scale = 1.0 / math.sqrt(head_dim)
    actual_seq_lengths = [kv_len]
    actual_seq_lengths_kv = [kv_len]

    return torch_npu.npu_fused_infer_attention_score(
        query, key_cache, value_cache,
        num_heads=num_heads,
        scale_value=scale,
        input_layout="BNSD",
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_mode=3,  # causal
        pre_tokens=65535,
        next_tokens=0,
        softmax_lse_flag=True  # 返回log-sum-exp
    )

# 示例调用
num_heads, head_dim = 8, 64
kv_len = 100

# 初始化KV cache
key_cache = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float16).npu()
value_cache = torch.randn(1, kv_len, num_heads, head_dim, dtype=torch.float16).npu()

# Prefill阶段
query_prefill = torch.randn(1, 64, num_heads, head_dim, dtype=torch.float16).npu()
prefill_out, prefill_lse = attention_inference(query_prefill, key_cache, value_cache, num_heads, head_dim, kv_len)

# Decode阶段 (增量)
query_decode = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float16).npu()
decode_out, decode_lse = attention_inference(query_decode, key_cache, value_cache, num_heads, head_dim, kv_len)
```

### 5.3 torch_npu.npu_advance_step_flashattn

**功能**: vLLM专用的step flash attention，管理生成状态。

**使用示例**:
```python
import torch
import torch_npu
import numpy as np

# vLLM风格的生成步骤
num_seqs = 16
num_queries = 8  # 当前要生成的序列数
block_size = 16

# 当前batch状态
input_tokens = torch.randint(0, 10000, (num_seqs,), dtype=torch.int64).npu()
input_positions = torch.randint(0, 1000, (num_seqs,), dtype=torch.int64).npu()
seq_lens = torch.randint(1, 100, (num_seqs,), dtype=torch.int64).npu()
slot_mapping = torch.randint(0, 1000, (num_seqs,), dtype=torch.int64).npu()

# 新生成的token ids
sampled_token_ids = torch.randint(0, 10000, (num_queries, 1), dtype=torch.int64).npu()

# Block映射表
max_blocks_per_seq = 64
block_tables = torch.randint(0, 1000, (num_seqs, max_blocks_per_seq), dtype=torch.int64).npu()

# 执行step更新
torch_npu.npu_advance_step_flashattn(
    input_tokens, sampled_token_ids, input_positions,
    seq_lens, slot_mapping, block_tables,
    num_seqs, num_queries, block_size
)

print(f"Step完成，更新了 {num_seqs} 个序列的状态")
```

## 6. 完整使用样例

### 6.1 mini-vLLM集成示例

```python
import torch
import torch_npu
import math
from typing import Optional, Tuple

class NPUAttention:
    """NPU Flash Attention的mini-vLLM集成类"""

    def __init__(self, num_heads: int, head_dim: int, block_size: int = 16):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.scale = 1.0 / math.sqrt(head_dim)

    def prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """Prefill阶段注意力计算"""
        return torch_npu.npu_prompt_flash_attention(
            query, key, value,
            num_heads=self.num_heads,
            scale_value=self.scale,
            input_layout="BNSD",
            sparse_mode=3,  # causal mask
            pre_tokens=65535,
            next_tokens=0
        )

    def decode_attention(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_lengths: torch.Tensor,
        block_table: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode阶段增量注意力计算"""
        kwargs = {
            "num_heads": self.num_heads,
            "scale_value": self.scale,
            "input_layout": "BNSD",
        }

        # 如果支持PageAttention，添加block参数
        if block_table is not None:
            kwargs.update({
                "block_table": block_table,
                "block_size": self.block_size,
                "actual_seq_lengths": seq_lengths
            })

        return torch_npu.npu_incre_flash_attention(
            query, key_cache, value_cache, **kwargs
        )

    def unified_inference(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """统一推理接口"""
        actual_seq_lengths = [seq_length]
        actual_seq_lengths_kv = [seq_length]

        return torch_npu.npu_fused_infer_attention_score(
            query, key_cache, value_cache,
            num_heads=self.num_heads,
            scale_value=self.scale,
            input_layout="BNSD",
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_mode=3,  # causal
            pre_tokens=65535,
            next_tokens=0,
            softmax_lse_flag=True
        )

# 使用示例
def main():
    # 初始化注意力模块
    attention = NPUAttention(num_heads=8, head_dim=128, block_size=16)

    batch_size = 1
    seq_len = 64
    max_kv_len = 1024

    # 构造输入
    query = torch.randn(batch_size, seq_len, 8, 128, dtype=torch.float16).npu()
    key_cache = torch.randn(batch_size, max_kv_len, 8, 128, dtype=torch.float16).npu()
    value_cache = torch.randn(batch_size, max_kv_len, 8, 128, dtype=torch.float16).npu()

    # Prefill阶段
    prefill_out = attention.prefill_attention(query, key_cache[:, :seq_len], value_cache[:, :seq_len])
    print(f"Prefill输出: {prefill_out.shape}")

    # Decode阶段
    decode_query = torch.randn(batch_size, 1, 8, 128, dtype=torch.float16).npu()
    seq_lengths = torch.tensor([seq_len], dtype=torch.int32).npu()

    decode_out = attention.decode_attention(decode_query, key_cache, value_cache, seq_lengths)
    print(f"Decode输出: {decode_out.shape}")

if __name__ == "__main__":
    main()
```

### 6.2 量化推理示例

```python
import torch
import torch_npu
import math

def quantized_inference_example():
    """量化推理示例"""
    batch_size, seq_len, num_heads, head_dim = 1, 1, 8, 64

    # Query使用FP16，KV使用INT8
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).npu()
    key_int8 = torch.randint(-128, 127, (1, 100, num_heads, head_dim), dtype=torch.int8).npu()
    value_int8 = torch.randint(-128, 127, (1, 100, num_heads, head_dim), dtype=torch.int8).npu()

    # 量化参数
    dequant_scale1 = torch.randn(1, dtype=torch.float32).npu()  # BMM1反量化系数
    quant_scale2 = torch.randn(1, dtype=torch.float32).npu()   # BMM2量化系数
    quant_offset2 = torch.randn(1, dtype=torch.float32).npu()  # BMM2量化偏移

    # 量化增量注意力
    atten_out_int8 = torch_npu.npu_incre_flash_attention(
        query, key_int8, value_int8,
        dequant_scale1=dequant_scale1,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
        num_heads=num_heads,
        scale_value=1.0 / math.sqrt(head_dim),
        input_layout="BNSD"
    )

    print(f"量化输出: {atten_out_int8.shape}, dtype: {atten_out_int8.dtype}")
    return atten_out_int8
```

## 7. 常见问题与排查

### 7.1 基础问题

*   **RuntimeError: input shapes mismatch**: 检查 `input_layout` 是否与实际 Tensor 维度一致。例如 `BSH` 要求输入为 `(Batch, Seq, Hidden)`。
*   **Accuracy Issue**: 检查 `scale` 参数是否正确设置。FlashAttention 默认不包含 scale，需手动传入 `1/sqrt(d)`。
*   **Unsupported data type**: 确保输入为 `float16` 或 `bfloat16`，NPU FlashAttention 通常不支持 `float32`。
*   **OOM (Out of Memory)**: 尝试减小 `batch_size` 或使用 `block_table` 优化 KV Cache 显存占用。

### 7.2 新API特有问题

*   **Sparse Mode 不匹配**: 不同API支持的sparse_mode范围不同，`npu_prompt_flash_attention` 目前只支持0-4。
*   **量化参数缺失**: 使用量化推理时，必须同时提供对应的量化参数组合。
*   **Block Table 维度错误**: PageAttention场景下，`block_table`的第二维必须足够大以容纳最长序列。

### 7.3 性能调优建议

1.  **API选择**:
    - 训练场景使用 `npu_fusion_attention`
    - 推理场景优先使用 `npu_fused_infer_attention_score` (自适应)
    - vLLM场景使用 `npu_advance_step_flashattn`

2.  **数据格式**: 尽量使用 `BNSD` 或 `TND` 格式，这些格式在 NPU 内部处理效率较高。

3.  **对齐优化**: `head_dim` 建议为 16 的倍数（如 64, 128），以充分利用 NPU 的 Cube 单元。

4.  **稀疏模式**: 明确指定 `sparse_mode`（如 Causal=3）比传入巨大的 bool mask 性能更好且更省显存。

5.  **量化优化**: 在内存受限场景下，可以考虑使用 INT8 量化推理。

## 8. 版本兼容性说明

### 8.1 API演进

*   **PyTorch 2.1**: 基础版本，支持核心的 fusion_attention 和 incre_flash_attention
*   **PyTorch 2.3+**: 新增 prompt_flash_attention 和 fused_infer_attention_score
*   **PyTorch 2.5+**: 新增 advance_step_flashattn，强化量化支持

### 8.2 接口变更

*   `npu_fusion_attention` 在不同版本中对 `atten_mask` 的支持程度可能不同，建议优先使用 `sparse_mode`。
*   Varlen 支持在新版本中更加完善，通过 `actual_seq_qlen` 完美支持变长序列，无需 Padding。
*   量化支持在新版本中大幅增强，支持更多量化组合和精度控制。

### 8.3 硬件支持

*   **Atlas A2 训练系列**: 支持全部功能，包括量化、PageAttention等高级特性
*   **Atlas 推理系列**: 主要支持推理场景，部分高级功能可能有限制
*   **Atlas A3 训练系列**: 最新硬件，支持所有最新特性，性能最优

---
*文档来源参考: [昇腾社区官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/)*
- torch_npu.npu_fusion_attention (60RC1)
- torch_npu.npu_prompt_flash_attention (700)
- torch_npu.npu_incre_flash_attention (60RC3)
- torch_npu.npu_fused_infer_attention_score (600)
- torch_npu.npu_advance_step_flashattn (700)
