## Commit Message Instructions (Commitizen/Conventional Commits Style)

All commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This ensures clarity and consistency in our project history.

### Commit Message Format

```
<type>(optional scope): <short description>

[optional body]

[optional footer(s)]
```

#### Examples

```
feat: add CUDA matrix multiplication kernel
fix(l2_cache): correct L2 cache size calculation
docs: update README with build instructions
refactor(async_overlap): simplify async logic
test: add tests for pinned memory
chore: update dependencies
```

### Types

- feat:     A new feature
- fix:      A bug fix
- docs:     Documentation only changes
- style:    Changes that do not affect the meaning of the code (white-space, formatting, etc)
- build:    Changes that affect the build system or external dependencies 
- refactor: A code change that neither fixes a bug nor adds a feature
- perf:     A code change that improves performance
- test:     Adding missing tests or correcting existing tests
- chore:    Changes to the build process or auxiliary tools and libraries
- ci:       Changes to our CI configuration files and scripts

### Scope (optional)
Scope is a noun describing a section of the codebase (e.g., async_overlap, l2_cache, matrix_multiply).

### Description
Use the imperative mood, present tense, and keep it concise (max 72 characters).

### Body (optional)
Provide additional context about the commit. Use when necessary.

### Footer (optional)
For breaking changes or referencing issues (e.g., BREAKING CHANGE: or Closes #123).

---
