---
name: "code-optimizer"
description: "智能代码专家：全方位执行代码审查、性能调优、重构与错误修复，严格强制执行 mini-vllm 项目开发规范。"
---

# mini-vllm 代码优化与规范专家 (Code Optimizer)

本 Skill 是 mini-vllm 项目的官方代码治理工具，旨在通过自动化的审查与优化流程，确保代码库的高质量、一致性与高性能。它集成了静态分析、风格强制、逻辑修复与性能调优四大核心能力。

## 1. 快速开始 (Quick Start)

在对话中输入以下任一指令即可触发：
- "优化这段代码"
- "检查当前文件是否存在 Bug"
- "格式化此文件"
- "Review my code"

**最佳实践**：
在提交代码 (Commit) 前，务必调用此 Skill 对修改的文件进行最终检查，以确保通过 CI/CD 流水线的 Pre-commit 检查。

## 2. 核心规范 (Core Specifications)

本项目严格遵循以下规范，**所有生成的代码必须符合**：

### 2.1 Python 代码风格 (Style Guide)
| 规则项 | 规范要求 | 解释/工具 |
| :--- | :--- | :--- |
| **行长度** | **Max 79 字符** | 严格遵循 PEP 8，便于多窗口并排阅读。 |
| **Import 排序** | 分组并字母排序 | 使用 `isort` 标准。Application imports: `minivllm`。 |
| **字符串引号** | 双引号 `"` | 统一风格 (`double-quote-string-fixer`)。 |
| **格式化** | Yapf 风格 | 即使不习惯，也必须保持项目一致性。 |

### 2.2 Linter 配置 (Flake8)
基于 `.flake8` 配置文件，以下规则被明确忽略：
- `W503`, `W504`: 二元运算符前后的换行 (交给 Formatter 处理)。
- `E501`: 行过长 (尽量遵守 79 字符，但 url/长字符串除外)。
- `E126`: 悬挂缩进过度 (兼容性调整)。
- `F811`: 重定义未使用的名称 (Redefinition of unused name)。

### 2.3 文件与通用规范 (Pre-commit)
- **文件结尾**: 必须以 **LF** 换行符结尾，且仅有一行空行。
- **尾部空格**: 行尾不得有多余空格。
- **编码声明**: 禁止使用 `# -*- coding: utf-8 -*-` (`fix-encoding-pragma --remove`)。
- **类型注解**: 必须使用 `typing` 模块进行类型提示。
- **YAML 检查**: 必须符合 YAML 语法 (`check-yaml`)。

## 3. 功能矩阵 (Capabilities Matrix)

### 3.1 代码审查 (Code Review)
- **静态分析**: 检查未使用的变量、未定义的名称、危险的默认参数。
- **逻辑验证**: 识别可能的死循环、边界溢出、空指针异常。
- **依赖检查**: 确保引入的包在 `requirements.txt` 或环境中可用。

### 3.2 性能优化 (Performance Optimization)
- **计算加速**: 建议使用 `torch` 或 `numpy` 的向量化操作替代 Python 循环。
- **NPU 适配**: 针对 Ascend NPU 环境，优先使用 `minivllm.models.layers.npu_flash_attention` 等优化算子。
- **内存管理**: 检查 Tensor 的生命周期，避免不必要的显存占用。

### 3.3 重构与质量 (Refactoring & Quality)
- **函数拆分**: 识别圈复杂度过高的函数并建议拆分。
- **命名规范**: 修正不清晰的变量名 (如 `x`, `data`, `temp`) 为语义化名称。
- **文档增强**: 自动生成符合 Google 风格的 Docstrings。

## 4. 工作流 (Workflow)

当收到优化请求时，Agent 将执行以下标准流程：

1.  **Context Analysis**: 分析代码上下文与依赖关系。
2.  **Lint & Check**: 运行静态检查，标记风格违规与潜在 Bug。
3.  **Optimize**:
    *   应用 `isort` 逻辑排序 Imports (Group: stdlib, third-party, `minivllm`)。
    *   应用 `yapf` 逻辑格式化代码。
    *   修复检测到的 Bug。
    *   添加缺失的 Type Hints。
4.  **Verify**: 再次检查修改后的代码是否符合 79 字符限制。
5.  **Report**: 输出修改摘要与优化理由。

## 5. 示例 (Examples)

### 优化前 (Before)
```python
import sys, os
from minivllm import llm
def load_config( p ):
    f = open(p) # 忘记关闭文件
    c = f.read()
    return c
```

### 优化后 (After)
```python
import os
import sys
from typing import Any

from minivllm import llm


def load_config(path: str) -> Any:
    """Load configuration from the specified path.

    Args:
        path (str): The file path to read from.

    Returns:
        Any: The configuration content.
    """
    with open(path, "r") as f:
        return f.read()
```
*变更说明：修复了 Import 排序和格式，使用了 Context Manager 确保文件关闭，添加了类型注解和 Docstring，规范了变量命名。*

## 6. 常见问题 (FAQ)

**Q: 为什么要限制 79 字符？**
A: mini-vllm 项目遵循严格的 PEP 8 标准，79 字符限制有助于在 Code Review 工具中并排查看 Diff，同时也方便在终端编辑器中编辑。

**Q: 如何处理 `minivllm` 内部模块的引用？**
A: 推荐使用绝对引用，例如 `from minivllm.engine import LLMEngine`。仅在同一包内的紧密耦合模块间可使用相对引用。

**Q: NPU 环境下有什么特殊注意事项？**
A: 在涉及 Attention 计算时，请检查是否可以使用 `npu_flash_attention` 模块进行加速。
