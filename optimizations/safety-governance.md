# Safety & Governance: Prometheus and plugin sandboxing

Adds CI gating, plugin sandboxing proposals, and tamper protections for Prometheus. Includes acceptance criteria and test plan.

Proposed files/changes (high level):
- Add CI checks and gating for plugins against Prometheus risk ceilings
- Add sandboxing/permission model for plugins and vault writes
- Add unit/integration tests for fuse scoring logic

