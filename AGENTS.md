# AI Agent Behavior & Coding Standards

## 1. Core Philosophy: Minimal & Essential
* **YAGNI (You Ain't Gonna Need It):** Write the *minimum* amount of code necessary to satisfy the requirements. Do not over-engineer or add "future-proofing" features unless explicitly asked.
* **Conciseness:** Avoid boilerplate. If a library or native function can do it, use that instead of writing custom logic.

## 2. Simplicity & Clarity
* **Readability first:** Write code that is easy for a human to understand immediately. Avoid complex one-liners or "clever" hacks.
* **Naming:** Use descriptive, verbose variable and function names (e.g., `calculateTotalPrice` instead of `calc`).

## 3. Modularity
* **Single Responsibility:** Each function or component should do exactly one thing.
* **File Structure:** Break large files into smaller, specific modules.
* **DRY (Don't Repeat Yourself):** Extract repeated logic into shared utility functions immediately.

## 4. Boundaries
* **Never** modify files in the `/dist` or `/build` directories.
* Only run tests using the `npm test` command.