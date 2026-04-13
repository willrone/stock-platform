# Official-Style Model Ranking Backtest Implementation Plan

> For Hermes: Use subagent-driven-development skill to implement this plan task-by-task.

Goal: Add an official-style Qlib-inspired model ranking backtest flow to stock-platform, centered on TopK + Dropout portfolio rotation instead of threshold-triggered per-stock signals.

Architecture: Keep the existing executor/reporting stack, but separate two concerns cleanly: (1) score-producing model strategies and (2) portfolio trade-mode execution. Add an extensible trade-mode registry so official-style ranking execution is a first-class backend capability rather than an if/else buried inside model_signal. Preserve the formal /backtest and /tasks flow and make the new strategy configurable with official-style names (topk, n_drop, benchmark, deal_price, costs, etc.).

Tech Stack: FastAPI, Python dataclasses, stock-platform backtest executor, Qlib-style TopkDropout semantics, pytest.

---

## Task breakdown

1. Document current architecture and pin desired API surface.
2. Add failing tests for strategy normalization and strategy-factory wiring.
3. Add failing tests for official TopK/Dropout trade-mode execution.
4. Add failing tests for formal /tasks model-driven ranking backtest flow.
5. Implement reusable trade-mode abstraction/registry.
6. Implement official-style model ranking strategy.
7. Wire strategy normalization for /backtest and /tasks.
8. Update reporting/config payloads to expose ranking-mode config clearly.
9. Run targeted regression tests.
10. Run one formal task smoke verification.

## Acceptance criteria

- New model-driven ranking strategy exists as a first-class strategy, not an ad-hoc config hack.
- Official-style config names are supported:
  - strategy_name alias -> model_topk_dropout
  - topk
  - n_drop
  - optional hold_thresh / buffer
  - benchmark
  - deal_price
  - open_cost / close_cost / min_cost (mapped into current commission/slippage model where applicable and carried through config/reporting)
- Formal /api/v1/tasks and direct /api/v1/backtest both support the new strategy.
- Executor uses an extensible trade-mode registry/interface, not a one-off branch for this strategy.
- Tests cover:
  - request normalization
  - strategy creation
  - ranking execution semantics
  - formal task flow
- Existing model_signal threshold path continues to work.
