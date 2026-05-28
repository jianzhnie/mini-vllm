---
paths:
  - "**/*.py"
  - "**/*.pyi"
---

# Python conventions

Tooling: ruff, black (line-length 88, double quotes), isort (black profile), mypy (strict).

## Project-specific rules

- Python 3.10+ — use `X | Y`, `list[int]`, built-in generics. No `Union`, `Optional`.
- Prefer `collections.abc` over `typing` for abstract types (`Iterator`, `Sequence`).
- Imports: stdlib → third-party → first-party. No relative imports, no wildcard imports.
- f-strings for formatting; `%`-style for logging (lazy eval): `logger.info("val: %s", val)`.
- Trailing comma required when closing bracket is on its own line.
- `logger = logging.getLogger(__name__)` at module level. Never `print()`.
- Dataclass validation in `__post_init__`. Use `Enum` + `auto()` for enumerations.
- `# TODO(username): description` for future work. Delete commented-out code before committing.
- Catch most specific exception. Use `raise from` to preserve tracebacks.
- `assert` only for internal invariants, never for input validation.
- Match existing code style in the file; three similar lines > premature abstraction.
