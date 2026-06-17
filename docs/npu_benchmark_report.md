# Mini-vLLM NPU Benchmark 报告

## 测试环境

| 项目 | 配置 |
|------|------|
| **硬件** | Ascend 910 (65GB HBM) x 8 |
| **CANN** | 9.0.0 |
| **PyTorch** | 2.12.0 |
| **torch_npu** | 2.12.0rc1 |
| **triton-ascend** | 3.2.1 |
| **镜像** | `torchtitan-npu:cann9.0.0-torch2.12.0` |
| **测试配置** | max_num_seqs=8, max_model_len=512, enforce_eager=True, dtype=float16 |

---

## 1. NPU vs CPU 推理吞吐量

| 模型 | 参数量 | NPU (tok/s) | CPU (tok/s) | 加速比 |
|------|:------:|:-----------:|:-----------:|:------:|
| GPT2 | 117M | **84.0** | — | — |
| OPT-125M | 125M | **89.5** | 28.6 | **3.1x** |
| Qwen3-0.6B | 600M | **39.9** | 8.7 | **4.6x** |
| Qwen3-1.7B | 1.7B | **43.5** | 3.1 | **14.0x** |
| Qwen3-4B | 4B | **32.9** | — | — |

> 测试条件: 4 prompts, max_tokens=16, temperature=0.7, top_p=0.95

**结论:** 模型越大 CPU 越慢，NPU HBM 带宽优势越明显。

---

## 2. NPU Eager vs Flash-Attention

| 模型 | Eager (tok/s) | FA (tok/s) | 加速比 |
|------|:------------:|:----------:|:------:|
| OPT-125M | 63.5 | 78.4 | **1.1x** |
| Qwen3-0.6B | 34.4 | 39.2 | **1.1x** |
| Qwen3-1.7B | 32.3 | 37.2 | **1.2x** |

> Qwen3 使用 GQA，统一推理路径回退 SDPA；FA 仍通过掩码缓存和 NPU 融合算子获得加速。

---

## 3. Example 脚本测试结果

### 3.1 `check_npu_graph.py`

| 检测项 | 结果 |
|--------|------|
| NPU 设备数 | 8 (Ascend910_9392) |
| npu_fusion_attention | ✅ AVAILABLE |
| npu_incre_flash_attention | ✅ AVAILABLE |
| npu_prompt_flash_attention | ✅ AVAILABLE |
| npu_fused_infer_attention_score | ✅ AVAILABLE |
| SDPA 前向计算 | ✅ PASSED |

### 3.2 `npu_inference_example.py`

| 模型 | max_tokens | 吞吐量 | 状态 |
|------|:----------:|:------:|:----:|
| OPT-125M | 16 | 89.5 tok/s | ✅ |
| GPT2 | 16 | 84.0 tok/s | ✅ |
| Qwen3-0.6B | 16 | 39.9 tok/s | ✅ |
| Qwen3-4B | 16 | 32.9 tok/s | ✅ |

### 3.3 `npu_flash_attention_example.py`

| 模型 | Eager | FA | FA 加速 | 状态 |
|------|:-----:|:--:|:-------:|:----:|
| OPT-125M | 63.5 | 78.4 | 1.1x | ✅ |
| Qwen3-0.6B | 34.4 | 39.2 | 1.1x | ✅ |
| Qwen3-1.7B | 32.3 | 37.2 | 1.2x | ✅ |
| Low-level Attn (Prefill+Decode) | — | — | — | ✅ PASSED |

### 3.4 `cpu_inference_opt.py`

| 模型 | 总时间 | 吞吐量 | 状态 |
|------|:------:|:------:|:----:|
| OPT-125M | 6.69s | 28.6 tok/s | ✅ |
| Qwen3-0.6B | 31.64s | 8.7 tok/s | ✅ |
| Qwen3-1.7B | 87.62s | 3.1 tok/s | ✅ |

### 3.5 `npu_tp_example.py`

| 模型 | TP=1 | 状态 |
|------|:----:|:----:|
| OPT-125M | 63.7 tok/s | ✅ |
| GPT2 | 75.3 tok/s | ✅ |
| Qwen3-0.6B | 30.2 tok/s | ✅ |

### 3.6 `mp_event_demo.py`

| 测试项 | 状态 |
|--------|:----:|
| 多进程 Event 同步 | ✅ 正常退出 |

---

## 4. NPU 专用算子激活状态

```
INFO - NPU SwiGLU kernel available
INFO - NPU Flash Attention available
INFO - NPU RMSNorm kernel available
INFO - NPU RoPE kernel available
```

---

## 5. 测试通过率

| 类别 | 测试数 | 通过 | 失败 |
|------|:------:|:----:|:----:|
| Example 脚本 | 7 | 7 | 0 |
| 模型测试组合 | 21 | 21 | 0 |
| NPU API 检测 | 4 | 4 | 0 |
| 功能验证 | 2 | 2 | 0 |
| **总计** | **34** | **34** | **0** |

---

*报告生成日期: 2026-06-16*
*测试执行环境: torchtitan-npu:cann9.0.0-torch2.12.0, Ascend 910*
