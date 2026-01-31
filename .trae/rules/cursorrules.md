# mini-vllm Project Rules

## 1. Project Overview
**mini-vllm** is a lightweight, educational implementation of the vLLM inference engine, built from scratch to demonstrate core concepts of high-performance LLM serving. It supports multiple hardware backends, with specific optimizations for Ascend NPUs.

## 2. Technology Stack
- **Language**: Python 3.10+
- **Core Framework**: PyTorch >= 2.2.0
- **Key Libraries**:
  - `transformers` (Model loading & tokenization)
  - `torch-npu` (Ascend NPU support)
  - `numpy`, `tqdm`, `safetensors`
- **Build System**: `pyproject.toml` (Setuptools)

## 3. Coding Standards (Mandatory)

### 3.1 Style & Formatting
The project strictly enforces coding standards via `pre-commit` hooks.
- **Linter**: `flake8`
  - **Max Line Length**: **79 characters** (Strict PEP 8).
  - **Ignored Errors**:
    - `W503`: Line break before binary operator
    - `W504`: Line break after binary operator
    - `E501`: Line too long (Exception for URLs/imports)
    - `E126`: Continuation line over-indented for hanging indent
    - `F811`: Redefinition of unused name
- **Formatter**: `yapf` (Code style) and `isort` (Import sorting).
- **Strings**: Use double quotes (`"`) for all strings (`double-quote-string-fixer`).

### 3.2 Imports
- **Sorting**: Must be sorted by `isort`.
- **Structure**:
  1. Standard library
  2. Third-party libraries (`torch`, `transformers`, etc.)
  3. Local project imports (`minivllm`)
- **Convention**: Use absolute imports for clarity.
  ```python
  # Good
  from minivllm.engine.llm_engine import LLMEngine

  # Avoid (unless in closely related modules)
  from ..engine import LLMEngine
  ```

### 3.3 Type Hints
- All function signatures (arguments and return values) **MUST** have type annotations.
- Use `typing` module (e.g., `List`, `Dict`, `Optional`, `Any`).

## 4. Project Structure
```text
minivllm/
├── engine/           # Core inference engine & scheduler
│   ├── llm_engine.py
│   └── scheduler.py
├── models/           # Model implementations
│   ├── layers/       # Neural network layers (Attention, MLP)
│   ├── qwen3.py      # Specific model architectures
│   └── npu_*.py      # NPU-specific implementations
├── sampling/         # Sampling strategies (Greedy, Top-P, etc.)
├── utils/            # Utilities (Device, Logging, Config)
└── config.py         # Global configuration
examples/             # Usage examples (e.g., npu_flash_attention_example.py)
tests/                # Pytest suite
docs/                 # Documentation
```

## 5. Development Workflow

### 5.1 Setup
1. **Install Dependencies**:
   ```bash
   pip install -e ".[dev,npu]"  # For NPU development
   # OR
   pip install -e ".[dev,cuda]" # For CUDA development
   ```
2. **Install Hooks**:
   ```bash
   pre-commit install
   ```

### 5.2 Verification (Before Commit)
Always run the `code-optimizer` skill or manual checks:
1. **Format & Lint**:
   ```bash
   pre-commit run --all-files
   ```
2. **Run Tests**:
   ```bash
   python run_tests.py
   # OR specific tests
   pytest tests/test_llm_engine.py
   ```

### 5.3 NPU/Ascend Guidelines
- **Device Management**: Use `minivllm.utils.device` instead of raw `torch.device`.
  ```python
  from minivllm.utils.device import get_current_device
  device = get_current_device()
  ```
- **Environment**: Use `ASCEND_RT_VISIBLE_DEVICES` for device isolation.
- **Ops**: Prefer `minivllm.models.layers.npu_flash_attention` for attention on NPU.

## 6. Critical Rules Checklist
- [ ] **Pass Pre-commit**: No trailing whitespace, correct YAML, sorted imports.
- [ ] **Pass Tests**: All unit tests in `tests/` must pass.
- [ ] **Type Check**: Critical paths must have type hints.
- [ ] **No Hardcoded Devices**: Do not hardcode "cuda" or "cpu"; use the device utility.
