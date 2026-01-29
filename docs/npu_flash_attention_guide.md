# NPU Flash Attention 使用文档

## 概述

NPU Flash Attention 是华为昇腾 NPU 提供的高性能注意力计算接口，主要包括两个核心API：

- `torch_npu.npu_fusion_attention`: 融合注意力计算，适用于训练场景
- `torch_npu.npu_incre_flash_attention`: 增量Flash Attention，适用于推理场景

## 1. torch_npu.npu_fusion_attention

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

### 主要参数说明
- **query, key, value**: 输入张量，支持 float16、bfloat16
  - 支持布局：BSNH、SBHN、BSND、BNSD，以及 TND（varlen 场景）
  - Q·Kᵀ 中 Q 的最后一维与 K 的最后一维一致，V 的最后一维可小于等于 K
- **head_num**: 注意力头数
- **input_layout**: 输入数据排布，支持以下格式：
  - `"BSH"`: Batch, Sequence, Hidden_Size
  - `"SBH"`: Sequence, Batch, Hidden_Size
  - `"BSND"`: Batch, Sequence, NumHeads, HeadDim
  - `"BNSD"`: Batch, NumHeads, Sequence, HeadDim
  - `"TND"`: TotalTokens, NumHeads, HeadDim (变长序列场景)
- **pse**: 位置编码，支持 NDNDND 布局；可用于 alibi 等压缩位置编码场景
- **atten_mask**: 掩码输入，1 表示遮蔽该位置，0 表示参与计算；支持 BNSS、B1SS、11SS、SS（varlen）格式
- **scale**: 缩放系数，默认 1.0, 通常设置为 1/sqrt(head_dim)
- **keep_prob**: Dropout 保留率 (0,1]，默认 1.0（不做 Dropout）
- **actual_seq_qlen**：Host侧（CPU端）的整型数组。在变长序列（varlen）场景下（即一个Batch中包含不同长度的句子），该参数必选。它描述了每个 Query 序列的结束位置（即长度的累加和），用于在打包的数据中区分不同的句子。例如：如果有两个长度分别为 4 和 6 的句子，该参数应为 `[4, 10]`。
- **actual_seq_kvlen**：Host侧（CPU端）的整型数组。在变长序列（varlen）场景下必选。与 `actual_seq_qlen` 类似，它描述了每个 Key/Value 序列的结束位置（即长度的累加和）。

- **sparse_mode**: 稀疏模式选择，稀疏模式，用于控制注意力掩码，当整网的atten_mask都相同且shape小于2048*2048时，建议使用defaultMask模式，来减少内存使用量
  - `0`: 默认掩码
  - `1`: Full mask
  - `2`: Top-left aligned causal mask (上三角掩码)
  - `3`: Down-right aligned causal mask (下三角掩码，用于因果掩码)

### 返回值
7 个输出，依次为：
1. **attention_out**: 注意力输出张量
2. **softmax_max**: Softmax 过程中的最大值
3. **softmax_sum**: Softmax 过程中的求和值
4. **reserved**: 保留未用
5. **philox_seed**: Dropout 随机数种子
6. **philox_offset**: Dropout 随机数偏移量
7. **mask_length**: Dropout mask 长度

### 使用示例
```python
import torch_npu

# 假设 query/key/value 均为形状 [B, S, head_num, head_dim] 的 Tensor
attn_out, max_v, sum_v, _, seed, offset, mlen = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num=8,
    input_layout="BSNH",
    scale=1.0,
    keep_prob=0.9,
    atten_mask=mask_tensor,
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
