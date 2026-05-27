# Python Style Guide

This guide follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with standard adaptations for modern Python (3.10+).

---

## 1 Formatting

### 1.1 Line length

Maximum 88 characters. Use implicit line joining inside `()`, `[]`, `{}`. Never use `\` backslash continuation.

### 1.2 Indentation

4 spaces. No tabs.

### 1.3 Blank lines

- 2 blank lines between top-level definitions (class, function)
- 1 blank line between method definitions
- 1 blank line between import groups

### 1.4 Whitespace

- No spaces inside `()`, `[]`, `{}`
- One space after `,`, `:`, `;`
- One space on both sides of binary operators (`+`, `-`, `==`, etc.)
- No spaces around `=` in keyword arguments: `func(a=1)`
- One space around `=` in type-annotated defaults: `def func(a: int = 1)`

### 1.5 Trailing comma

Required when closing bracket is on its own line:

```python
my_list = [
    1,
    2,
    3,
]
```

### 1.6 Imports

Order (alphabetical within each group, blank line between groups):

1. `from __future__ import ...`
2. Standard library
3. Third-party
4. First-party / local

Rules:

- `import x` for packages and modules
- `from x import y` where `x` is package prefix, `y` is module name
- `from x import y as z` when `y` conflicts, is too long, or too generic
- `import y as z` only for standard abbreviations (e.g. `np`, `pd`, `torch` as `pt`)
- No relative imports
- No `from module import *`
- No multiple imports on one line: `import os, sys`

```python
from __future__ import annotations

import os
from pathlib import Path

import torch

from myproject.utils import helper
```

---

## 2 Naming

| Type | Convention | Example |
|------|------------|---------|
| Module / package | `lower_snake_case` | `my_module` |
| Class | `CapWords` | `MyClass` |
| Exception | `CapWords` + `Error` suffix | `ValueError` |
| Function / method | `lower_snake_case()` | `my_function` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_SIZE` |
| Variable / parameter | `lower_snake_case` | `my_variable` |
| Private (internal) | leading underscore | `_internal_func` |
| Type alias | `CapWords` | `TensorDict` |

Prohibited:

- Names with hyphens `-`
- Names containing type info (`id_to_name_dict`)
- `__double_leading_and_trailing_underscore__` (reserved)
- Single-char names except in very short blocks (`i`, `j`, `k`, `e`, `f`)

---

## 3 Type Annotations

### 3.1 Required

All public functions and methods must have type annotations.

### 3.2 Modern syntax

Use `from __future__ import annotations` and Python 3.10+ syntax:

- `X | Y` instead of `Union[X, Y]`
- `X | None` instead of `Optional[X]`
- Built-in generics: `list[str]`, `dict[str, int]`, `tuple[int, str]`

### 3.3 Special cases

- `self` and `cls` need not be annotated
- `__init__` return type need not be annotated
- Use `ClassVar` for class-level mutable attributes
- Use `Any` only when the type truly cannot be expressed

### 3.4 None defaults

Explicit `None` default requires explicit `None` in type:

```python
# Yes
def func(a: str | None = None) -> None:
    ...

# No
def func(a: str = None) -> None:
    ...
```

---

## 4 Docstrings

Always use `"""` triple double-quotes.

### 4.1 Module

```python
"""One-line summary of the module.

Leave one blank line, then describe the module in more detail.
"""
```

### 4.2 Function / method

```python
def my_function(arg1: int, arg2: str) -> bool:
    """One-line summary, no more than 88 chars, ending with a period.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When arg1 is negative.
    """
```

Rules:

- Summary line: one line, ends with period / question mark / exclamation
- `Args`, `Returns`, `Raises` sections for public API
- Private methods (`_`) may omit docstring or use one-line only
- Do not document the obvious (e.g. "gets x" for `get_x`)

### 4.3 Class

```python
class MyClass:
    """One-line summary.

    Attributes:
        attr1: Description of attr1.
        attr2: Description of attr2.
    """
```

---

## 5 Comments

- Comments explain **WHY**, not **WHAT**
- Use block comments (`#`) above the code they describe, not at end of line
- Delete commented-out code before committing
- Do not reference PR numbers or ticket IDs in code comments

TODO format:

```python
# TODO(username): Description of what needs to be done.
```

---

## 6 Error Handling

### 6.1 Exception types

- Catch the most specific exception possible
- Never use bare `except:` or `except Exception:` unless re-raising immediately
- Never swallow exceptions silently; at minimum log them
- Custom exceptions must inherit an existing exception class and end with `Error`

### 6.2 assert

Use `assert` only for internal invariants, never for input validation or runtime checks (assertions can be disabled with `-O`).

```python
# Yes (internal invariant)
assert index < len(data), "Index out of bounds"

# No (runtime check)
assert os.path.exists(path), "Path must exist"
```

### 6.3 Re-raising

Use `raise from` to preserve tracebacks:

```python
try:
    data = load_file(path)
except FileNotFoundError as e:
    raise RuntimeError(f"Failed to load {path}") from e
```

### 6.4 Resource cleanup

Use `with` for files, sockets, locks. Use `finally` for cleanup.

### 6.5 try/except scope

Minimize code inside `try/except` blocks:

```python
# Yes
value = compute_something()
try:
    result = value.risky_operation()
except ValueError:
    result = default

# No
try:
    value = compute_something()
    result = value.risky_operation()
except ValueError:
    result = default
```

---

## 7 Boolean Comparisons

| Prefer | Avoid |
|--------|-------|
| `if seq:` | `if len(seq) != 0:` |
| `if not seq:` | `if len(seq) == 0:` |
| `if val is None:` | `if val == None:` |
| `if val is not None:` | `if val != None:` |
| `if flag:` | `if flag == True:` |
| `if not flag:` | `if flag == False:` |

When `0` is a valid value, explicitly compare: `if x == 0:` not `if not x:`.

---

## 8 Default Arguments

Never use mutable objects as default arguments:

```python
# Yes
def foo(a, b=None):
    if b is None:
        b = []

def foo(a, b: Sequence = ()):
    ...

# No
def foo(a, b=[]):
def foo(a, b: Mapping = {}):
def foo(a, b=time.time()):  # evaluated at import time
```

---

## 9 String Formatting

- Prefer f-strings for inline formatting
- Never use `+` for string concatenation in loops
- Never accumulate strings with `+=` in loops (use `list` + `join` or `io.StringIO`)

Logging: use `%`-style placeholders for lazy evaluation:

```python
# Yes
logger.info("Loading from %s", path)

# No
logger.info(f"Loading from {path}")
```

---

## 10 Comprehensions

Allowed for simple cases. Prohibited with multiple `for` clauses:

```python
# Yes
result = [x * 2 for x in data if x > 0]

# No
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
```

Use generator expressions for `any()` / `all()` / `sum()`:

```python
# Yes
has_positive = any(x > 0 for x in data)

# No
has_positive = any([x > 0 for x in data])
```

---

## 11 Lambda

Allowed only for trivial one-liners. Use nested functions for anything longer:

```python
# Yes
sorted_items = sorted(items, key=lambda x: x[0])

# No
result = map(lambda x: complex_transform(x), items)
```

---

## 12 Classes

- Prefer `@dataclass` for data containers
- Use `@property` instead of trivial getter/setter methods
- Instance attributes initialized in `__init__`, not dynamically added later
- Avoid `staticmethod` — use module-level functions instead
- `classmethod` only for named constructors or modifying class-level state
- Inherit from `object` explicitly only in Python 2 compatibility code

---

## 13 Main Entry Point

All executable scripts must define and call a `main()` function:

```python
def main() -> None:
    ...

if __name__ == "__main__":
    main()
```

---

## 14 General Conventions

- Use `pathlib.Path` over `os.path` for file operations
- Use `logging` module, not `print()`, for diagnostic output
- Prefer `isinstance(x, type)` over `type(x) == type`
- Keep functions focused; consider splitting when exceeding ~40 lines
- Be consistent with existing code in the same file
