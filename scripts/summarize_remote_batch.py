import argparse
import csv
import re
from pathlib import Path


TEST_PATTERN = re.compile(r"\[ITER\s+(?P<iter>\d+)\]\s+Evaluating\s+test:\s+L1\s+(?P<l1>[0-9.eE+-]+)\s+PSNR\s+(?P<psnr>[0-9.eE+-]+)")


def load_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_timing(rows, skip_first: int):
    filtered = rows[skip_first:]
    if not filtered:
        raise ValueError(f"No timing rows left after skip_first={skip_first}")

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in filtered) / len(filtered)

    return {
        "avg_tile_ratio": mean("avg_tile_ratio"),
        "iter_time_ms": mean("iter_time_ms"),
        "optimizer_ms": mean("optimizer_ms"),
        "step_total_ms": mean("iter_time_ms") + mean("optimizer_ms"),
        "train_render_ms": mean("train_render_ms"),
        "backward_ms": mean("backward_ms"),
        "mask_update_ms": mean("mask_update_ms"),
        "mask_build_ms": mean("mask_build_ms"),
        "full_render_ms": mean("full_render_ms"),
    }


def parse_test_metrics(log_path: Path):
    matches = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TEST_PATTERN.search(line)
        if match:
            matches.append(
                {
                    "iteration": int(match.group("iter")),
                    "l1": float(match.group("l1")),
                    "psnr": float(match.group("psnr")),
                }
            )
    return matches


def main():
    parser = argparse.ArgumentParser(description="Summarize a remote experiment batch from timing logs and Slurm logs.")
    parser.add_argument("--skip-first", type=int, default=100)
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "TIMING_LOG", "SLURM_LOG"),
        help="One run entry with label, timing_log.csv path, and slurm log path.",
    )
    args = parser.parse_args()

    if not args.run:
        raise ValueError("At least one --run entry is required")

    print("| Label | Test iter | Test PSNR | Test L1 | Avg tile ratio | Step total ms | Train render ms | Backward ms | Mask update ms | Full render ms |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for label, timing_log, slurm_log in args.run:
        timing_summary = summarize_timing(load_rows(Path(timing_log)), args.skip_first)
        test_metrics = parse_test_metrics(Path(slurm_log))
        if test_metrics:
            final_test = test_metrics[-1]
            test_iter = final_test["iteration"]
            test_psnr = final_test["psnr"]
            test_l1 = final_test["l1"]
        else:
            test_iter = "N/A"
            test_psnr = float("nan")
            test_l1 = float("nan")

        print(
            f"| {label} | {test_iter} | {test_psnr:.6f} | {test_l1:.6f} | "
            f"{timing_summary['avg_tile_ratio']:.4f} | {timing_summary['step_total_ms']:.3f} | "
            f"{timing_summary['train_render_ms']:.3f} | {timing_summary['backward_ms']:.3f} | "
            f"{timing_summary['mask_update_ms']:.3f} | {timing_summary['full_render_ms']:.3f} |"
        )


if __name__ == "__main__":
    main()
