import argparse
import sys
from pathlib import Path

from nlp_lab.acceptance import AcceptanceOptions, build_acceptance_plan, run_acceptance
from nlp_lab.core.config import ConfigOverrides
from nlp_lab.core.parity import compare_run_parity, write_parity_report
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
        "--experiment",
        choices=("dummy-success", "dummy-failure", "hf-text-classification"),
        default="dummy-success",
        help="Local experiment implementation to run.",
    )
    run_parser.add_argument(
        "--dummy-experiment",
        choices=("success", "failure"),
        default=None,
        help="Temporary local experiment implementation.",
    )
    run_parser.set_defaults(handler=run_command)

    compare_parser = subparsers.add_parser(
        "compare-runs",
        help="Compare local and remote run artifacts for parity.",
    )
    compare_parser.add_argument("--local-run-dir", required=True, type=Path)
    compare_parser.add_argument("--remote-run-dir", required=True, type=Path)
    compare_parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    compare_parser.add_argument("--confidence-tolerance", type=float, default=1e-4)
    compare_parser.add_argument("--report", type=Path, default=None)
    compare_parser.set_defaults(handler=compare_runs_command)

    acceptance_parser = subparsers.add_parser(
        "acceptance-test",
        help="Run the end-to-end local and Modal infrastructure acceptance flow.",
    )
    acceptance_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/local_smoke_tiny_sst2.yaml"),
    )
    acceptance_parser.add_argument(
        "--gpu-config",
        type=Path,
        default=Path("configs/experiments/modal_smoke_tiny_sst2_gpu.yaml"),
    )
    acceptance_parser.add_argument(
        "--local-output-root",
        type=Path,
        default=Path("outputs/experiments/acceptance/local"),
    )
    acceptance_parser.add_argument(
        "--failed-output-root",
        type=Path,
        default=Path("outputs/experiments/acceptance/failed"),
    )
    acceptance_parser.add_argument(
        "--remote-download-root",
        type=Path,
        default=Path("outputs/modal-downloads"),
    )
    acceptance_parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/reports/acceptance.json"),
    )
    acceptance_parser.add_argument(
        "--parity-report",
        type=Path,
        default=Path("outputs/reports/acceptance-parity.json"),
    )
    acceptance_parser.add_argument("--skip-clean-checks", action="store_true")
    acceptance_parser.add_argument("--skip-remote", action="store_true")
    acceptance_parser.add_argument("--skip-gpu", action="store_true")
    acceptance_parser.add_argument("--dry-run", action="store_true")
    acceptance_parser.set_defaults(handler=acceptance_test_command)
    return parser


def run_command(args: argparse.Namespace) -> int:
    try:
        overrides = build_overrides(args)
        run = run_local_experiment(
            experiment_config_path=args.config,
            common_config_path=args.common_config,
            overrides=overrides,
            experiment=args.experiment,
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


def compare_runs_command(args: argparse.Namespace) -> int:
    try:
        report = compare_run_parity(
            args.local_run_dir,
            args.remote_run_dir,
            metric_tolerance=args.metric_tolerance,
            confidence_tolerance=args.confidence_tolerance,
        )
    except (OSError, ValueError) as exc:
        print(f"Run parity comparison failed: {exc}", file=sys.stderr)
        return CONFIG_VALIDATION_EXIT_CODE

    if args.report is not None:
        write_parity_report(args.report, report)

    print(f"Parity status: {report.status}")
    for check in report.checks:
        print(f"- {check.name}: {check.status}")
    if args.report is not None:
        print(f"Parity report: {args.report}")
    return SUCCESS_EXIT_CODE if report.passed else EXPERIMENT_FAILURE_EXIT_CODE


def acceptance_test_command(args: argparse.Namespace) -> int:
    options = AcceptanceOptions(
        config=args.config,
        gpu_config=args.gpu_config,
        local_output_root=args.local_output_root,
        failed_output_root=args.failed_output_root,
        remote_download_root=args.remote_download_root,
        report_path=args.report,
        parity_report_path=args.parity_report,
        run_clean_checks=not args.skip_clean_checks,
        run_remote=not args.skip_remote,
        run_gpu=not args.skip_gpu,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Acceptance test plan:")
        for step in build_acceptance_plan(options):
            print(f"- {step.name}: {' '.join(step.command)}")
    report = run_acceptance(options)
    print(f"Acceptance status: {report.status}")
    print(f"Acceptance report: {args.report}")
    if report.parity_report_path is not None:
        print(f"Parity report: {report.parity_report_path}")
    if report.status == "SKIPPED":
        return SUCCESS_EXIT_CODE
    return SUCCESS_EXIT_CODE if report.passed else EXPERIMENT_FAILURE_EXIT_CODE


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
