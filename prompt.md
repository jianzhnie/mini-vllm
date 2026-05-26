# mini-vllm NPU Backend 调试与优化

目标：确保 mini-vllm 在 NPU 环境下各功能模块能正确运行并输出合理结果。

## 环境信息

- 可用模型路径：
  - `/home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m`（轻量，优先用于快速验证）
  - `/home/jianzhnie/llmtuner/hfhub/models/Qwen`（主要测试模型）
  - `/home/jianzhnie/llmtuner/hfhub/models/openai-community/gpt2`（兼容性验证）
- 后端：NPU（通过 `minivllm/utils/device.py` 检测和适配）

## 任务（按优先级顺序执行）

### 1. 基础推理验证（最高优先级）

逐个运行 `minivllm/examples/` 下的示例脚本，先用 `--enforce_eager` 模式确保 NPU 后端能跑通 e2e 推理。

- **验证标准**：
  - 无报错退出
  - 输出 token 数量符合 `max_tokens` 设定
  - 输出文本语义通顺（与 CPU/CUDA 参考输出对比，允许浮点精度差异）
- **覆盖顺序**：先用 `opt-125m` 快速跑通，再用 `Qwen` 验证
- **关键检查点**：
  - 模型加载（`minivllm/models/manager.py` → `_detect_model_type()`）
  - 权重加载（`minivllm/utils/loader.py`）
  - KV Cache 分配和 block 管理
- 遇到错误时，定位报错模块并修复后再继续下一个示例

### 2. NPU Flash-Attention 算子调试

验证并优化 NPU 下的 attention 计算路径。

- **相关文件**：
  - `minivllm/models/layers/attention.py`（FlashAttention 调用入口）
  - `minivllm/models/layers/attention_backend.py`（backend 抽象）
  - `minivllm/models/layers/npu_flash_attention.py`（NPU 专用实现）
- **验证方式**：
  - 关闭 `enforce_eager`，确认 attention 走 NPU flash-attention 路径
  - 对比 eager 和 flash-attn 的输出，结果应一致
  - 如不一致，排查精度问题（dtype 转换、scale 计算）
- **性能基线**：记录 `opt-125m` 在 eager vs flash-attn 的耗时，flash-attn 应有明显加速

### 3. 张量并行（TP）验证

验证 TP=2/4/8 配置下 NPU 多卡推理的正确性。

- **相关文件**：
  - `minivllm/engine/distributed_manager.py`（多进程协调、backend 选择）
  - `minivllm/models/layers/linear.py`（Column/Row/QKV ParallelLinear）
- **验证方式**：
  - 同一 prompt 在 TP=1 和 TP=N 下的输出应一致
  - 如不一致，重点检查 all-reduce / all-gather 的位置和通信 backend
- **测试步骤**：
  1. TP=1 基线（步骤 1 已完成即可）
  2. TP=2 → 验证 → 修复 → 再验证
  3. TP=4 → 同上
  4. TP=8 → 同上
- **注意**：TP 大小不能超过可用 NPU 卡数，且需是 2 的幂

## 调试方法论

- **定位问题**：从错误堆栈自底向上追踪，找到 mini-vllm 代码中最靠近出错点的文件
- **对比验证**：同一份代码在 CPU（`--dtype float32`）下运行作为 golden reference
- **日志级别**：`import logging; logging.getLogger("minivllm").setLevel(logging.DEBUG)`
- **最小复现**：用 opt-125m + 最简单的示例脚本复现问题

## 测试的Python 环境

```bash
source /home/jianzhnie/llmtuner/llm/mini-vllm/set_env.sh
```

## 常见 NPU 适配问题

- `torch.npu` vs `torch.cuda` API 差异（`minivllm/utils/device.py` 中的统一抽象是否正确）
- 通信 backend：NPU 用 `hccl`，CUDA 用 `nccl`（`distributed_manager.py` 中是否正确分支）
- 内存分配和 device 同步（`torch.npu.synchronize()` vs `torch.cuda.synchronize()`）
- Flash-Attention 接口差异（输入参数、返回值格式）
