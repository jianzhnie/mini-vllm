---
description: Run all pre-commit checks before committing
argument-hint: none
---

Run the full pre-commit suite and report results:

```bash
pre-commit run --all-files
```

If checks pass: "All checks passed" in Chinese.
If any fail: list each failure with the file and fix hint. Suggest running the auto-fixers first (ruff, black, isort) and re-checking.
