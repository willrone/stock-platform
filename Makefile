.PHONY: doctor setup setup-backend setup-frontend dev dev-backend dev-frontend stop status smoke-local smoke-backtest logs prod-build prod-up prod-down prod-status install-systemd audit-docs audit-docs-selftest manifest-build

doctor:
	./scripts/doctor.sh

setup: setup-backend setup-frontend

setup-backend:
	./scripts/setup-backend.sh

setup-frontend:
	./scripts/setup-frontend.sh

dev:
	./scripts/start-dev.sh

dev-backend:
	./scripts/dev-backend.sh

dev-frontend:
	./scripts/dev-frontend.sh

stop:
	./scripts/stop-dev.sh

status:
	./scripts/status.sh

smoke-local:
	./scripts/smoke-local.js

smoke-backtest:
	./scripts/smoke-backtest.js

logs:
	./scripts/logs.sh all

prod-build:
	./scripts/prod-build.sh

prod-up:
	./scripts/prod-up.sh

prod-down:
	./scripts/prod-down.sh

prod-status:
	./scripts/prod-status.sh

install-systemd:
	./scripts/install-systemd.sh

# ── 回测文档数字审计 ──────────────────────────────────────
# 文档里的每个数字都必须能回溯到产物；产物被改动/丢失 → 指纹校验先拦。
audit-docs:
	env -u PYTHONPATH -u PYTHONHOME python3 backend/scripts/audit_doc_numbers.py

# 审计器自检：动态注入错误 → 必须抓到（抓不到说明审计器本身有盲区）
audit-docs-selftest:
	env -u PYTHONPATH -u PYTHONHOME python3 backend/scripts/audit_doc_numbers.py --neg

# 重新生成产物指纹清单（产物更新后必须重建，否则审计器会报指纹不一致）
manifest-build:
	env -u PYTHONPATH -u PYTHONHOME python3 backend/scripts/artifact_manifest.py build
