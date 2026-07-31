# NLP Engineering Practical Lab

This repository is a practical workspace for building, evaluating, and running NLP experiments with
a clear path from local development to remote execution.

The project follows the study plan in `Baykar_NLP_Uygulamali_Calisma_Plani.pdf`: shared experiment
infrastructure first, then Hugging Face based NLP tasks, RAG experiments, parameter-efficient
fine-tuning, PyTorch training and evaluation flows, reliability checks, and Modal-based remote
execution.

## Aim

The goal is to make NLP engineering work reproducible and testable from the first day. The codebase
is organized around:

- reusable config models and experiment definitions
- data preparation and artifact management
- model training, inference, and evaluation
- local-first development with optional remote execution
- quality checks that stay lightweight for normal development

## Local And Remote Model

Local development is the default workflow. Unit tests, linting, formatting checks, type checking,
small data transforms, and CPU-friendly experiments should run locally through `uv`.

Remote execution is reserved for workflows that need Modal resources, longer jobs, shared Volumes,
or environment isolation. Remote code should live under `modal_apps/` and package-level remote
helpers should live under `src/nlp_lab/remote/`.

Generated artifacts should be written under `outputs/` for local runs or to a Modal Volume for
remote runs. Large datasets and generated outputs are ignored by git; keep only small samples,
configs, code, and reproducibility metadata in version control.

## Setup

Install dependencies with `uv` and Python 3.11:

```bash
uv sync
```

Install specific dependency groups as needed:

```bash
uv sync --group core
uv sync --group ml
uv sync --group remote
uv sync --group development
```

Install every dependency group:

```bash
uv sync --all-groups
```

Run commands inside the managed environment:

```bash
uv run python --version
```

Copy the environment template before adding local secrets or paths:

```bash
cp .env.example .env
```

## Quality Commands

Run the default local quality gate:

```bash
make check
```

Run individual checks:

```bash
make lint
make format-check
make typecheck
make test-unit
```

Apply Ruff formatting:

```bash
make format
```

Run broader test levels when their prerequisites are available:

```bash
make test-integration
make test-smoke
```

Run the local Hugging Face smoke experiment:

```bash
make local-smoke
```

Preview the end-to-end acceptance flow without running network or Modal work:

```bash
make acceptance-plan
```

Run the full local and Modal infrastructure acceptance flow:

```bash
make acceptance-test
```

The acceptance flow performs dependency sync, lint, format check, type check, unit tests, a local
classification smoke run, an intentional local failed run, two Modal CPU classification runs, an
optional Modal GPU run, Volume artifact export, cache timing inspection, and local/remote parity
comparison. It writes `outputs/reports/acceptance.json` and
`outputs/reports/acceptance-parity.json`. Use `uv run nlp-lab acceptance-test --skip-gpu` when GPU
quota is not available, or `uv run nlp-lab acceptance-test --skip-remote` for a local-only
acceptance pass.

Run Modal launchers from the repository root:

```bash
uv run --group remote modal run modal_apps/smoke_test.py
uv run --group remote modal run modal_apps/smoke_test.py --fail
uv run --group remote modal run modal_apps/classification.py
uv run --group remote modal run modal_apps/classification.py --gpu
```

The GPU launcher defaults to `configs/experiments/modal_smoke_tiny_sst2_gpu.yaml`, which requests
`inference.device: cuda`. Remote artifacts and caches are written to the `nlp-lab-storage` Modal
Volume mounted at `/storage`.

The initial storage layout is:

```text
/storage/
├── cache/
│   ├── huggingface/
│   ├── datasets/
│   └── transformers/
├── experiments/
├── checkpoints/
└── models/
```

Remote experiment output defaults to `/storage/experiments`; local output still defaults to
`outputs/experiments`. The artifact writer only receives the selected root path, so local and Modal
runs use the same artifact lifecycle.

Inspect the Volume from a Modal worker:

```bash
uv run --group remote modal run modal_apps/storage.py
uv run --group remote modal run modal_apps/storage.py --path /storage/cache
```

List or download artifacts with Modal's Volume CLI:

```bash
uv run --group remote modal volume ls nlp-lab-storage /experiments
uv run --group remote modal volume get nlp-lab-storage /experiments/<run-id> outputs/modal-downloads/<run-id>
```

Compare a local run with a downloaded remote run:

```bash
uv run nlp-lab compare-runs \
  --local-run-dir outputs/experiments/local-smoke/<local-run-id> \
  --remote-run-dir outputs/modal-downloads/<remote-run-id> \
  --report outputs/reports/local-remote-parity.json
```

The strict parity check should use runs created from the same resolved experiment config. The
Modal GPU smoke config intentionally differs by requesting `inference.device: cuda`, so the
acceptance command compares the local run against the second Modal CPU run and records the GPU run
separately for CUDA and device metadata verification.

For a cache-hit check, run the same classification command twice and compare `model_load_seconds`
in each run's `runtime.json`. The Modal runner overrides experiment cache paths so Hugging Face,
dataset, and model caches resolve under `/storage/cache` or `/storage/models`; the second run should
reuse those files instead of downloading the model again. Cache is intentionally not deleted after
each run; keep used models, selectively remove old checkpoints, and clean failed partial downloads
when storage grows.

## Config Layers

Experiment configs are resolved in three steps:

1. `configs/common/default.yaml`
2. `configs/experiments/<experiment>.yaml`
3. optional command-line overrides

Experiment files override common defaults by normal nested key merge. Command-line overrides are
intentionally limited to frequently changed fields only:

- `batch_size` -> `inference.batch_size`
- `max_samples` -> `dataset.max_samples`
- `model_id` -> `model.model_id`
- `dataset_split` -> `dataset.split`
- `threshold` -> `inference.threshold`
- `output_root` -> `runtime.output_root`
- `seed` -> `runtime.seed`

## Test Levels

Unit tests live in `tests/unit` and must be fast, offline, CPU-only, and avoid model downloads.

Integration tests live in `tests/integration`; they may download small Hugging Face models, run on
CPU, and verify that components work together.

Smoke tests live in `tests/smoke`; they verify Modal execution, artifact writes to a Volume, and
results returned to the local client.
