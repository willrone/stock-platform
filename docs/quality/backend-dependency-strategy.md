# Backend Dependency Strategy

## Current state

The current GitHub Actions workflows still install `backend/requirements.txt` for backend quality, test, and security jobs.

This is not ideal because `requirements.txt` currently mixes:

- runtime dependencies
- test dependencies
- quality tooling
- security tooling
- heavy ML / Qlib dependencies

However, the workflow files cannot be changed from the current token because GitHub rejects workflow updates without the `workflow` scope. Therefore, the split requirement files below are maintained as the intended migration target while keeping the current green baseline intact.

Current green baseline:

- Commit: `34d9b0f56d4e770186ef70ba6d91b7d983babbc8`
- Latest follow-up baseline: `7108bb3` and later docs/test hygiene commits remained green at the time they were pushed.

## Files

### `backend/requirements.txt`

Current all-in-one install file used by GitHub Actions.

Important constraints:

- `numpy>=1.26.0,<2.0` is required while `pyarrow==14.0.1` remains pinned. Without `<2.0`, CI may install NumPy 2.x and fail with pyarrow ABI errors.
- `safety==2.3.5` must not be pinned here because it conflicts with `black==23.11.0` through `packaging` version constraints.
- `black==23.11.0`, `isort==5.12.0`, and `flake8==6.1.0` define the current formatting/lint baseline.

### `backend/requirements-quality.txt`

Future lightweight install target for the backend quality workflow.

Intended workflow replacement:

```bash
pip install -r requirements-quality.txt
black --check app/
isort --check-only app/
flake8 app/
mypy app/ --ignore-missing-imports || true
bandit -r app/ -f json -o bandit-report.json || true
```

Notes:

- Keeps quality tool versions aligned with the already-green CI gate.
- Includes only enough runtime/test dependencies for imports and smoke checks.
- Avoids heavy ML packages by default.
- Does not include Safety because Safety 3.x requires `pydantic>=2.6`, while the application currently pins `pydantic==2.5.0`.

### `backend/requirements-test.txt`

Future lightweight install target for the backend test workflow.

Intended workflow replacement:

```bash
pip install -r requirements-test.txt
pytest tests/ \
  --cov=app \
  --cov-report=xml \
  --cov-report=html \
  --cov-report=term \
  --junitxml=test-results.xml \
  -v
```

Notes:

- Keeps NumPy `<2.0` for pyarrow compatibility.
- Excludes heavy ML/Qlib packages by default.
- The current CI still uses `tests/conftest.py` to ignore legacy/flaky suites under `GITHUB_ACTIONS=true`; see `docs/quality/backend-ci-tail-cleanup.md`.


### `backend/requirements-security.txt`

Future isolated install target for dependency/security scanning tools.

Intended workflow replacement:

```bash
python -m venv .security-venv
.security-venv/bin/pip install -r requirements-security.txt
.security-venv/bin/safety check --json || true
```

Notes:

- Keep this separate from app/test/quality dependencies because Safety 3.x depends on newer Pydantic than the current app runtime pin.
- Bandit can run from either `requirements-quality.txt` or this security environment, but Safety should remain isolated until runtime Pydantic is upgraded.

### `backend/requirements-ml.txt`

Optional heavy ML / quant dependencies.

Install this only for workflows or environments that need:

- Qlib official workflows
- PyTorch / Transformers
- XGBoost
- Optuna/vectorbt heavy experiments

## Migration plan once workflow edits are allowed

1. Update `.github/workflows/code-quality.yml` backend install step to use `requirements-quality.txt`.
2. Update `.github/workflows/test.yml` backend install step to use `requirements-test.txt`.
3. Update security scanning to install `requirements-security.txt` in an isolated environment for Safety.
4. Keep `requirements-ml.txt` out of default CI; add a separate optional ML workflow if needed.
5. Add a small sync/doctor check that verifies:
   - quality tool versions match this document
   - CI tail cleanup ledger matches `tests/conftest.py`
5. Only after workflows are stable, simplify `requirements.txt` toward runtime-only dependencies.

## Guardrails

- Do not upgrade Black casually. A previous mismatch between latest Black and `black==23.11.0` caused GitHub formatting failures.
- Do not remove `numpy<2.0` while `pyarrow==14.0.1` remains in use.
- Do not reintroduce `safety==2.3.5` into `requirements.txt`.
- Do not place Safety 3.x in the same requirements file as `pydantic==2.5.0`; use `requirements-security.txt` separately.
- Keep `talib` optional; code should continue to provide pandas/numpy fallbacks.
