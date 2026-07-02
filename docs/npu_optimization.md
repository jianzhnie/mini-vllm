# Mini-vLLM NPU 推理性能优化文档

## 概述

本文档记录了 mini-vllm 在华为昇腾 NPU (Ascend 910) 上的全面推理性能优化工作。优化覆盖了从底层注意力算子到上层推理执行器的完整推理链路，重点解决 NPU 上的内核启动开销、内存分配开销和数据传输瓶颈。

**环境信息:**
- 硬件: Ascend 910 (65GB HBM)
- 软件: CANN 9.0.0, PyTorch 2.12.0, torch_npu 2.12.0rc1
- 镜像: `torchtitan-npu:cann9.0.0-torch2.12.0`

---

## 优化项目详细说明

### 1. 注意力后端优化 (`attention_backend.py`)

#### 1.1 注意力掩码缓存

**问题:** `unified_inference` 每次调用都通过 `torch.triu` 创建因果注意力掩码，产生不必要的内核启动和内存分配。

**优化:** 引入 `_attn_mask_cache` 字典，按 `q_len` 缓存 2D 掩码张量。后续调用直接查找缓存，仅需一次 `.view()` 零拷贝操作。

```python
# 优化前: 每步创建
atten_mask = torch.triu(torch.ones(q_len, q_len, ...), diagonal=1)

# 优化后: 缓存复用
if q_len in self._attn_mask_cache:
    atten_mask = self._attn_mask_cache[q_len]
else:
    atten_mask = torch.triu(...)
    self._attn_mask_cache[q_len] = atten_mask
```

**收益:** 消除 prefill 阶段每层的掩码创建开销。

#### 1.2 位置网格缓存

**问题:** `prepare_npu_cache` 中每步通过 `torch.arange` 创建位置索引张量。

**优化:** 引入 `_seq_pos_cache`，按 `(max_seqlen, device)` 缓存位置网格。

**收益:** 减少 decode 阶段每步每层的 `torch.arange` 调用（在 NPU 上每次约 0.1ms kernel launch）。

#### 1.3 KV Cache 存储优化

**问题:** 原始 `store_kv_cache` 有多层冗余检查（设备类型判断、contiguity 检查、try-except 分支）。

**优化:**
- 移除冗余的 `is_contiguous()` 检查（`reshape` 后张量已经是连续的）
- 移除设备类型条件分支（这是 NPU 专属后端）
- 合并 dtype 转换为单次判断
- 直接使用 `index_copy_`（NPU 上最高效的 scatter 操作）

**收益:** 减少每步 KV cache 写入时的条件判断和冗余操作。

---

### 2. 注意力模块优化 (`attention.py`)

#### 2.1 批量 dtype 转换

**问题:** 原始代码对 q、k、v 分别检查 dtype 并逐一转换。

**优化:** 以 `q.dtype` 作为代理，一次判断后同时转换三者。

```python
# 优化后
if q.dtype != target_dtype:
    q = q.to(target_dtype)
    k = k.to(target_dtype)
    v = v.to(target_dtype)
```

#### 2.2 消除 decode 路径冗余转换

**问题:** `npu_incre_flash_attention` 路径原先显式将 q/k_cache/v_cache 转为 float16，即使 forward 入口已确保了 dtype 一致。

**优化:** 直接传递 `q.unsqueeze(2)` 和 `k_cache`/`v_cache`，省去 3 次 `.to()` 调用。

**收益:** 每个 decode step 减少 3 次潜在的 dtype 转换操作。

---

### 3. 推理执行器优化 (`inference_executor.py`)

#### 3.1 预分配 Decode 张量

**问题:** 每个 decode step 从 Python list 创建新张量 → pinned memory → H2D 传输，产生频繁的内存分配/释放。

**优化:** 初始化时预分配固定大小的设备张量，decode 阶段直接填充复用。

```python
# 初始化时
self._decode_input_ids = torch.zeros(max_seqs, dtype=torch.long, device=self.device)
self._decode_positions = torch.zeros(max_seqs, dtype=torch.long, device=self.device)
self._decode_slot_mapping = torch.zeros(max_seqs, dtype=torch.int32, device=self.device)
self._decode_context_lens = torch.zeros(max_seqs, dtype=torch.int32, device=self.device)
```

**收益:** 消除 decode 阶段每步 4 次 tensor 创建 + pinned memory 分配 + GC 压力。

#### 3.2 移除热路径计时开销

**问题:** `execute_batch` 中的 CUDA Event 计时在 NPU 上无效且引入同步开销。

**优化:** 移除 CUDA Event 相关代码，保持推理路径纯净。

#### 3.3 采样参数优化

**问题:** `_sample_tokens` 每步通过 `torch.tensor([list])` 创建采样参数张量。

**优化:** 使用 `torch.empty` + 逐元素填充，避免中间 Python list 创建。

#### 3.4 NPU 编译模式设置

新增 `npu_set_compile_mode(jit_compile=False)` 确保 eager 执行模式稳定性。

---

### 4. 页式注意力优化 (`page_attention.py`, `attention_gather.py`)

#### 4.1 位置网格缓存

**问题:** `PageAttention` 和 `BufferedPageAttention` 每次调用都创建 `torch.arange` 位置张量。

**优化:** 两个类都引入 `_seq_pos_cache`，缓存已创建的位置网格。

**收益:** 对于 8 batch decode，每步减少 ~8 次 `torch.arange` kernel launch。

---

### 5. 旋转位置编码优化 (`rotary_embedding.py`)

#### 5.1 默认启用 NPU RoPE 内核

**问题:** `npu_rotary_mul` 内核默认关闭（需 `MINIVLLM_USE_NPU_ROPE=1`），用户需手动开启。

**优化:** 默认启用，改为 opt-out（`MINIVLLM_USE_NPU_ROPE=0` 关闭）。

**收益:** 利用 NPU 硬件加速的融合旋转乘法，替代纯 PyTorch 实现。

#### 5.2 减少形状操作

**优化:** 合并 cos/sin 的 shape 检查、expand、contiguous 操作为更少的步骤。

---

### 6. RMSNorm 优化 (`layernorm.py`)

#### 6.1 原地残差加法

**问题:** `add_rms_forward` 中 `x + residual` 创建中间张量。

**优化:** 当 `x` 连续时使用 `x.add_(residual)` 原地操作，避免中间张量分配。

```python
x = x.add_(residual) if x.is_contiguous() else x + residual
```

**收益:** 每层每步减少一次张量分配（对于 28 层模型，每步减少 56 次分配）。

---

### 7. NPU Flash Attention 掩码缓存 (`npu_flash_attention.py`)

#### 7.1 Power-of-2 掩码大小

**问题:** 掩码按实际 size 分配，不同序列长度创建不同掩码。

**优化:** 掩码大小向上取整到 2 的幂次，提高跨序列长度的缓存命中率。

```python
new_size = 1 << (new_size - 1).bit_length()
```

---

## 优化效果总结

| 优化类别 | 影响范围 | 主要收益 |
|---------|---------|---------|
| 掩码/网格缓存 | 每步每层 | 减少 kernel launch |
| 预分配张量 | 每 decode step | 消除内存分配 |
| dtype 批量转换 | 每步每层 | 减少 .to() 调用 |
| 原地 RMSNorm | 每步每层×2 | 减少中间张量 |
| NPU RoPE 默认启用 | 每步每层 | 硬件加速 RoPE |
| index_copy_ 精简 | 每步每层 | 减少条件分支 |

---

## 已激活的 NPU 专用算子

| 算子 | 来源 | 用途 |
|------|------|------|
| `npu_rms_norm` | torch_npu | RMSNorm 融合计算 |
| `npu_rotary_mul` | torch_npu | RoPE 旋转乘法 |
| `npu_swiglu` | torch_npu | SwiGLU 激活函数 |
| `npu_fusion_attention` | torch_npu | Prefill 注意力 |
| `npu_incre_flash_attention` | torch_npu | Decode 增量注意力 |
| `npu_fused_infer_attention_score` | torch_npu | 统一推理注意力 |
| `scaled_dot_product_attention` | PyTorch | SDPA (GQA 回退) |
