# 调试 Qwen3 模型乱码输出的排查与修复过程

## 问题描述

使用 Qwen3-0.6B 模型进行推理时，输出为乱码（重复的无效 token），例如：

```
'icularlyicularlyicularlybilder<vectoricularly<vectoricularlyicularlyeterminatebilder...'
```

而相同的模型在 HuggingFace Transformers 下可以正常生成。

## 排查过程

### 第一步：验证模型文件完整性

对比 safetensors 权重文件的 key 与 mini-vllm 模型参数，确认：

- 所有 packed 模块（`qkv_proj`, `gate_up_proj`）的权重通过 weight\_loader 正确加载
- 无 shape 不匹配
- 无缺失权重

结论：**权重加载无误**。

### 第二步：对比 mini-vllm 与 HuggingFace 的 logits

用相同的输入对比两个框架的输出 logits：

- Embedding 权重：完全匹配
- LM head 权重：完全匹配
- 最终 logits 余弦相似度：**0.04**（极低）

结论：问题出在 transformer 层的内部计算中，而非输入/输出层。

### 第三步：逐层定位发散点

通过 hook 捕获 HuggingFace 每层输出，与 mini-vllm 逐层对比：

```
Layer 0: cos_sim=0.488888  → 第一层就已经发散
```

结论：**发散从 Layer 0 开始**。

### 第四步：逐子组件排查 Layer 0

对 Layer 0 的每个子组件分别对比：

| 子组件                   | 是否匹配         | 备注           |
| --------------------- | ------------ | ------------ |
| Embedding             | 匹配           | <br />       |
| Input LayerNorm       | 匹配           | <br />       |
| QKV Projection        | 匹配           | Q/K/V 各自完全一致 |
| Q/K Norm (RMSNorm)    | 匹配           | <br />       |
| RoPE                  | **不匹配**      | cos/sin 值不同  |
| Attention Computation | 匹配（RoPE 修复后） | <br />       |
| O Projection          | 匹配           | <br />       |
| MLP                   | 权重匹配，但输出错误   | <br />       |

### 第五步：定位 Bug #1 — RoPE theta 错误

对比 mini-vllm 与 HuggingFace 的 cos/sin 值：

```
mini-vllm cos[1,:8]: [0.5403, 0.6479, 0.7318, 0.7965, ...]
HF cos[1,:8]:        [0.5403, 0.6925, 0.7965, 0.8662, ...]
```

从 cos\_cache 反推实际使用的 inv\_freq，发现与 `base=10000` 一致，而非配置文件中的 `1000000`。

**根因**：`Qwen3Config` 不直接暴露 `rope_theta` 属性，而是存储在嵌套结构中：

```python
config.rope_theta           # AttributeError！不存在
config.rope_parameters      # {'rope_theta': 1000000, 'rope_type': 'default'}
config.rope_scaling         # {'rope_theta': 1000000, 'rope_type': 'default'}
```

代码中使用 `getattr(config, 'rope_theta', 10000)` 获取不到正确的值，回退到了 `Qwen3Attention.default_rope_theta = 10000`。

**修复**：在 `qwen_base.py` 中添加 `_resolve_rope_theta()` 函数，依次从 `config.rope_theta`、`config.rope_parameters['rope_theta']`、`config.rope_scaling['rope_theta']` 中获取值。

### 第六步：RoPE 修复后仍然发散

修复 RoPE theta 后，cos/sin 值正确，但 logits 余弦相似度仍然只有 0.035。

重新逐层排查发现 Layer 0 仍然发散（cos\_sim=0.56），尽管各子组件单独测试时都匹配。

### 第七步：发现 RMSNorm 的输入突变

对比手动逐步执行与调用完整 layer forward 的结果：

```python
# 手动执行
residual = emb
h, _ = layer.input_layernorm(emb, None)
...

# 完整调用
h, r = layer(pos, emb.clone(), None)
```

两者输出**不一致**。进一步测试发现：

```python
norm = RMSNorm(4)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
out, _ = norm(x)
print(x)  # [0.3651, 0.7303, 1.0954, 1.4606] — 输入被篡改！
```

**根因**：`RMSNorm.rms_forward` 中使用 in-place 操作：

```python
x = x.float()          # 当 x 已是 float32 时，返回同一个 tensor
x.mul_(rsqrt(...))     # in-place 修改，篡改了原始输入
```

在 `QwenDecoderLayer.forward` 中：

```python
residual = hidden_states                    # residual 和 hidden_states 指向同一 tensor
hidden_states, _ = self.input_layernorm(hidden_states)  # layernorm 篡改了 hidden_states
# 此时 residual 也被篡改！它不再是原始输入，而是 LN(原始输入)
```

这导致所有残差连接的值都是错误的，且误差逐层累积。

**影响条件**：当输入已经是 float32 时（CPU 推理、或 dtype='float32'），`x.float()` 返回原始 tensor，触发此 bug。当输入是 float16/bfloat16 时，`x.float()` 创建新 tensor，不受影响。

**修复**：将 `rms_forward` 和 `add_rms_forward` 中的 in-place 操作替换为普通操作：

```python
# Before (in-place, mutates input)
x = x.float()
x.mul_(torch.rsqrt(var + self.eps))
x = x.to(orig_dtype).mul_(self.weight)

# After (creates new tensors, input preserved)
x = x.float()
x = x * torch.rsqrt(var + self.eps)
return x.to(orig_dtype) * self.weight
```

## 验证结果

两个 bug 修复后，mini-vllm 与 HuggingFace 的 logits 余弦相似度达到 **1.000000**：

```
mini-vllm top-5: 'oulos': 0.9998, '<|im_start|>': 0.0000, ...
HF top-5:        'oulos': 0.9998, '<|im_start|>': 0.0000, ...
```

## 修改文件

| 文件                                    | 修改内容                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------- |
| `minivllm/models/qwen_base.py`        | 添加 `_resolve_rope_theta()` 和 `_resolve_rope_scaling()`，替换直接的 `getattr` 调用 |
| `minivllm/models/layers/layernorm.py` | `rms_forward` 和 `add_rms_forward` 中移除 in-place 操作                         |

## 经验总结

1. **HuggingFace Config 兼容性**：不同版本的 transformers 库对同一模型配置的属性暴露方式不同。`rope_theta` 在旧版中是直接属性，在新版 Qwen3Config 中嵌套在 `rope_parameters` 内。解析配置时需要兼容多种存储格式。
2. **In-place 操作的隐患**：`x = x.float()` 在 `x` 已经是目标 dtype 时返回原 tensor，后续 in-place 操作会意外修改原始数据。应当避免在可能存在多引用的场景中使用 in-place 操作。

**调试方法**：当模型输出异常时，采用"二分法"逐层、逐子组件对比参考实现（HuggingFace），可以高效定位问题。
