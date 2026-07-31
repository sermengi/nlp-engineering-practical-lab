import argparse
import sys
from pathlib import Path

from nlp_lab.core.config import ConfigOverrides
from nlp_lab.experiments.local import run_local_experiment
from nlp_lab.experiments.runner import (
    CONFIG_VALIDATION_EXIT_CODE,
    EXPERIMENT_FAILURE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    ExperimentRunFailedError,
    is_config_validation_error,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlp-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment locally.")
    run_parser.add_argument("--config", required=True, type=Path, help="Experiment config path.")
    run_parser.add_argument(
        "--common-config",
        default=Path("configs/common/default.yaml"),
        type=Path,
        help="Common config path.",
    )
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--max-samples", type=int, default=None)
    run_parser.add_argument("--batch-size", type=int, default=None)
    run_parser.add_argument("--output-root", type=Path, default=None)
    run_parser.add_argument(
        "--dummy-experiment",
        choices=("success", "failure"),
        default="success",
        help="Temporary local experiment implementation.",
    )
    run_parser.set_defaults(handler=run_command)
    return parser


def run_command(args: argparse.Namespace) -> int:
    try:
        overrides = build_overrides(args)
        run = run_local_experiment(
            experiment_config_path=args.config,
            common_config_path=args.common_config,
            overrides=overrides,
            dummy_experiment=args.dummy_experiment,
        )
    except ExperimentRunFailedError as exc:
        print(f"Experiment failed: {exc.original}", file=sys.stderr)
        print(f"Run artifacts: {exc.paths.run_dir}", file=sys.stderr)
        return EXPERIMENT_FAILURE_EXIT_CODE
    except Exception as exc:
        if is_config_validation_error(exc):
            print(f"Config validation failed: {exc}", file=sys.stderr)
            return CONFIG_VALIDATION_EXIT_CODE
        raise

    print(f"Run completed: {run.paths.run_dir}")
    return SUCCESS_EXIT_CODE


def build_overrides(args: argparse.Namespace) -> ConfigOverrides | None:
    raw_overrides = {
        "seed": args.seed,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "output_root": args.output_root,
    }
    selected_overrides = {key: value for key, value in raw_overrides.items() if value is not None}
    if not selected_overrides:
        return None
    return ConfigOverrides.model_validate(selected_overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
