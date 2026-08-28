```markdown
# mini-vllm Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you the core development patterns and conventions used in the `mini-vllm` TypeScript codebase. You'll learn about file organization, import/export styles, commit message formatting, and how to write and run tests. This guide is designed to help contributors maintain consistency and quality across the project.

## Coding Conventions

### File Naming
- Use **snake_case** for all file names.
  - Example: `model_utils.ts`, `api_handler.ts`

### Import Style
- Use **relative imports** for modules within the project.
  - Example:
    ```typescript
    import { processInput } from './input_utils';
    ```

### Export Style
- Use **named exports** for all exported functions, types, or constants.
  - Example:
    ```typescript
    // In tokenizer.ts
    export function tokenize(text: string): string[] { ... }
    ```

    ```typescript
    // In another file
    import { tokenize } from './tokenizer';
    ```

### Commit Messages
- Follow **Conventional Commits**.
- Use prefixes such as `docs`.
- Keep commit messages concise (average ~41 characters).
  - Example:
    ```
    docs: update README with usage instructions
    ```

## Workflows

### Documentation Updates
**Trigger:** When updating or improving documentation files.
**Command:** `/update-docs`

1. Make changes to documentation files (e.g., `README.md`, inline comments).
2. Use a conventional commit message with the `docs` prefix.
   - Example: `docs: clarify setup instructions`
3. Push your changes and open a pull request.

## Testing Patterns

- Test files follow the pattern: `*.test.*` (e.g., `tokenizer.test.ts`).
- The specific test framework is **unknown**, but tests are likely colocated with source files or in a dedicated `tests` directory.
- To write a test, create a file like `module_name.test.ts` and use the project's standard test assertions.

  Example test file:
  ```typescript
  import { tokenize } from './tokenizer';

  describe('tokenize', () => {
    it('splits text into tokens', () => {
      expect(tokenize('hello world')).toEqual(['hello', 'world']);
    });
  });
  ```

## Commands
| Command        | Purpose                                      |
|----------------|----------------------------------------------|
| /update-docs   | Start the documentation update workflow      |
```
