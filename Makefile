LOCAL_SMOKE_OUTPUT_ROOT ?= outputs/experiments/local-smoke
LOCAL_PARITY_RUN_DIR ?=
REMOTE_PARITY_RUN_DIR ?=
PARITY_REPORT ?= outputs/reports/local-remote-parity.json

.PHONY: sync sync-all test test-unit test-integration test-smoke local-smoke compare-local-remote lint format format-check typecheck check

sync:
	uv sync

sync-all:
	uv sync --all-groups

test:
	uv run --group development pytest

test-unit:
	uv run --group development pytest tests/unit -m unit

test-integration:
	uv run --all-groups pytest tests/integration -m integration

test-smoke:
	uv run --all-groups pytest tests/smoke -m smoke

local-smoke:
	uv run --group ml nlp-lab run --experiment hf-text-classification --config configs/experiments/local_smoke_tiny_sst2.yaml --output-root $(LOCAL_SMOKE_OUTPUT_ROOT)

compare-local-remote:
	uv run nlp-lab compare-runs --local-run-dir $(LOCAL_PARITY_RUN_DIR) --remote-run-dir $(REMOTE_PARITY_RUN_DIR) --report $(PARITY_REPORT)

lint:
	uv run --group development ruff check .

format:
	uv run --group development ruff format .

format-check:
	uv run --group development ruff format --check .

typecheck:
	uv run --group development mypy src

check: lint format-check typecheck test-unit
