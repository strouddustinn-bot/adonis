---
name: test-diagnostics-cleanup
description: Clean up recurring test-file diagnostics in this repo, especially unused imports and semicolon-separated statements, without changing test behavior.
---

# Test Diagnostics Cleanup

Use this skill when a test or test-helper file in `adonis/tests/` has small hygiene issues that do not require behavioral changes.

## What to fix

- Remove unused imports.
- Split multiple statements written on one line into separate lines.
- Keep helper fakes and fixtures behavior-compatible with existing tests.
- Preserve async signatures and public fake interfaces unless the tests require a deliberate change.

## How to work

- Make the smallest edit that resolves the diagnostic.
- Prefer readability over compact one-line statements.
- Do not refactor unrelated test code while cleaning up diagnostics.
- After editing, check the touched test file with diagnostics before finishing.

## Scope

Apply this to `tests/conftest.py` and similar files under `adonis/tests/` when the issue is the same class of cleanup problem.
