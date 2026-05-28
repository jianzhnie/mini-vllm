---
paths:
  - "**/*.sh"
  - "**/*.bash"
---

# Shell conventions

- `#!/bin/bash` (not `/usr/bin/env bash`). `set -euo pipefail` in executable scripts.
- Sourced files must NOT set shell options — they inherit from the caller.
- `[[ ]]` for conditionals, never `[ ]` or `test`. `$(command)` for substitution.
- Double-quote all variable refs: `"$var"`, `"${array[@]}"`.
- `local` for function-scoped variables, `readonly` for constants (UPPER_SNAKE_CASE).
- Functions under 50 lines, scripts under 400 lines. 4-space indent, max 120 char lines.
- `${var:-default}` for defaults; `${1:?error}` for required args.
- Return exit status only (0/1), not computed values.

## Compatibility

- bash 4.2+ — no `declare -A`, no namerefs.
- Dependencies: bash 4+, coreutils, openssh, docker, ray, vllm.
- Run `shellcheck` on all scripts. Hook auto-runs it on every `.sh` edit.
