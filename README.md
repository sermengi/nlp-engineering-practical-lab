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

## Test Levels

Unit tests live in `tests/unit` and must be fast, offline, CPU-only, and avoid model downloads.

Integration tests live in `tests/integration`; they may download small Hugging Face models, run on
CPU, and verify that components work together.

Smoke tests live in `tests/smoke`; they verify Modal execution, artifact writes to a Volume, and
results returned to the local client.
