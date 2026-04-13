import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


class IterationProfiler:
    def __init__(
        self,
        enabled=False,
        output_dir=None,
        start_iter=1,
        warmup_iters=2,
        active_iters=10,
    ):
        self.enabled = enabled
        self.output_dir = output_dir
        self.start_iter = max(1, start_iter)
        self.warmup_iters = max(0, warmup_iters)
        self.active_iters = active_iters

        self.csv_path = None
        self.summary_path = None
        self._current_iteration = None
        self._current_sections = defaultdict(lambda: {"cuda_ms": 0.0, "wall_ms": 0.0, "calls": 0})
        self._aggregate_sections = defaultdict(lambda: {"cuda_ms": 0.0, "wall_ms": 0.0, "calls": 0})
        self._active = False
        self._profiled_iterations = 0

        if self.enabled:
            if not self.output_dir:
                raise ValueError("Profiling output_dir must be set when profiling is enabled.")
            os.makedirs(self.output_dir, exist_ok=True)
            self.csv_path = os.path.join(self.output_dir, "train_iteration_profile.csv")
            self.summary_path = os.path.join(self.output_dir, "train_iteration_profile_summary.txt")
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            "iteration",
                            "section",
                            "calls",
                            "cuda_ms_total",
                            "cuda_ms_avg",
                            "wall_ms_total",
                            "wall_ms_avg",
                            "num_points",
                        ]
                    )

    def should_profile(self, iteration):
        if not self.enabled:
            return False
        if iteration < self.start_iter:
            return False

        relative_iter = iteration - self.start_iter
        if relative_iter < self.warmup_iters:
            return False

        if self.active_iters < 0:
            return True

        return relative_iter < self.warmup_iters + self.active_iters

    def is_active(self):
        return self._active

    def start_iteration(self, iteration):
        self._current_iteration = iteration
        self._current_sections.clear()
        self._active = self.should_profile(iteration)

    @contextmanager
    def section(self, name):
        if not self._active:
            yield
            return

        wall_start = time.perf_counter()
        cuda_enabled = torch.cuda.is_available()
        start_event = torch.cuda.Event(enable_timing=True) if cuda_enabled else None
        end_event = torch.cuda.Event(enable_timing=True) if cuda_enabled else None
        if start_event is not None:
            start_event.record()

        try:
            yield
        finally:
            cuda_ms = 0.0
            if end_event is not None:
                end_event.record()
                end_event.synchronize()
                cuda_ms = start_event.elapsed_time(end_event)

            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            section_stats = self._current_sections[name]
            section_stats["cuda_ms"] += cuda_ms
            section_stats["wall_ms"] += wall_ms
            section_stats["calls"] += 1

    def end_iteration(self, num_points):
        if not self._active:
            return

        self._profiled_iterations += 1
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for section_name, stats in self._current_sections.items():
                calls = max(1, stats["calls"])
                writer.writerow(
                    [
                        self._current_iteration,
                        section_name,
                        stats["calls"],
                        f"{stats['cuda_ms']:.6f}",
                        f"{stats['cuda_ms'] / calls:.6f}",
                        f"{stats['wall_ms']:.6f}",
                        f"{stats['wall_ms'] / calls:.6f}",
                        int(num_points),
                    ]
                )

                aggregate = self._aggregate_sections[section_name]
                aggregate["cuda_ms"] += stats["cuda_ms"]
                aggregate["wall_ms"] += stats["wall_ms"]
                aggregate["calls"] += stats["calls"]

        self._write_summary()

    def _write_summary(self):
        if not self.enabled:
            return

        lines = [
            f"profiled_iterations: {self._profiled_iterations}",
            f"csv_path: {self.csv_path}",
            "",
            "average section time per profiled iteration (cuda_ms / wall_ms / calls):",
        ]

        sorted_sections = sorted(
            self._aggregate_sections.items(),
            key=lambda item: item[1]["cuda_ms"],
            reverse=True,
        )
        for section_name, stats in sorted_sections:
            denom = max(1, self._profiled_iterations)
            lines.append(
                (
                    f"{section_name}: "
                    f"{stats['cuda_ms'] / denom:.4f} / "
                    f"{stats['wall_ms'] / denom:.4f} / "
                    f"{stats['calls'] / denom:.2f}"
                )
            )

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def close(self):
        if self.enabled:
            self._write_summary()
