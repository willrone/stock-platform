# AGENTS.md

## Agent Execution Rules

These rules apply to all work in this repository.

### Complex remediation and project closeout

- The controller agent is an orchestrator only: split work, dispatch subagents, review results, consolidate findings, run final acceptance checks, and update authoritative documentation.
- Code changes, root-cause investigation, and test repairs must be delegated to subagents by default.
- Subagents use `glm-5.2` through `custom:tokenhub`.
- Give each subagent a narrow task, explicit file whitelist, exact test commands, and a prohibition on unrelated edits.
- If a subagent times out or fails, inspect its transcript, reduce the task scope, and dispatch another subagent. The controller must not take over implementation merely to accelerate closeout.
- The controller may edit or investigate directly only when the user explicitly authorizes it for the current task.
- Do not claim completion from subagent self-reports. The controller must inspect diffs and run the final verification chain itself.

### Verification

- Use the project Python environment: `PYTHONPATH=backend backend/.venv-py313/bin/python`.
- Do not install packages into system Python or the Hermes runtime environment for project tests.
- Completion requires real command output, not a plan or inferred result.
- Preserve unrelated user changes in the dirty worktree.
