set shell := ["bash", "-cu"]

default:
    @just --list

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

logs service="all":
    ./scripts/logs.sh {{service}}

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
