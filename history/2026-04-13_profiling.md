## 2026-04-13 - Training iteration / rasterizer profiling

- Added opt-in profiling switches to `train.py`.
- Added per-iteration section profiling for Python/CUDA-visible training stages:
  - `lr_and_sh`
  - `view_to_cuda`
  - `view_render`
  - `view_loss`
  - `view_backward`
  - `batch_reduce`
  - `logging_and_report`
  - `densify`
  - `optimizer`
- Added CUDA rasterizer profiling rows for:
  - forward: `preprocess`, `scan`, `copy_rendered`, `duplicate`, `sort`, `zero_ranges`, `identify_ranges`, `render`, `copy_alpha`
  - backward: `backward_render`, `backward_preprocess`
- Profiling outputs are written under `<model_path>/profiling` by default, or `--profile_output_dir`.

### Example

```bash
conda run -n 4dgs python train.py --config <config_file> --profile_training --profile_rasterizer --profile_from_iter 1 --profile_warmup_iters 2 --profile_iters 10
```

### Output files

- `train_iteration_profile.csv`
- `train_iteration_profile_summary.txt`
- `rasterizer_profile.csv`

## Run notes

- Dataset/config used: `configs/dynerf/cook_spinach_debug.yaml`
- Profiling target window: `300-500 iter`
- Profiling command:

```bash
python train.py --config configs/dynerf/cook_spinach_debug.yaml --profile_training --profile_rasterizer --profile_from_iter 280 --profile_warmup_iters 20 --profile_iters 201 --profile_output_dir output/N3DV/cook_spinach_debug_profile_20260413/profiling
```

- Due to the local GPU being `RTX 2080 Ti (sm_75)`, custom CUDA extensions were rebuilt for `TORCH_CUDA_ARCH_LIST=7.5`.
- Rebuilt/verified:
  - `simple-knn`
  - `pointops2`
  - `diff_gaussian_rasterization`

## Measured bottlenecks (300-500 iter average)

- Python training sections, average CUDA time per iter:
  - `view_backward`: `49.48 ms`
  - `logging_and_report`: `33.70 ms`
  - `view_render`: `18.48 ms`
  - `optimizer`: `16.10 ms`
- Rasterizer internal average CUDA time per call:
  - forward total: `7.85 ms`
  - forward `render`: `4.86 ms`
  - forward `sort`: `1.44 ms`
  - backward total: `27.68 ms`
  - backward `backward_render`: `26.94 ms`

## Saved artifacts

- Profile logs: `output/N3DV/cook_spinach_debug_profile_20260413/profiling`
- Visual summaries:
  - `train_section_cuda_avg.png`
  - `train_section_wall_avg.png`
  - `rasterizer_forward_avg.png`
  - `rasterizer_backward_avg.png`

## Follow-up: disable reporting during profiling

- Updated `train.py` so `training_report(...)` and PSNR computation are skipped while the profiler is actively collecting samples.
- Added `IterationProfiler.is_active()` to gate profiling-only behavior cleanly.
- Re-ran the same debug training with profiling window `300-500 iter`:

```bash
python train.py --config configs/dynerf/cook_spinach_debug.yaml --profile_training --profile_rasterizer --profile_from_iter 280 --profile_warmup_iters 20 --profile_iters 201 --profile_output_dir output/N3DV/cook_spinach_debug_profile_20260413_nolog/profiling
```

### Updated bottlenecks without `logging_and_report`

- Python training sections, average CUDA time per iter:
  - `view_backward`: `42.47 ms`
  - `view_render`: `18.81 ms`
  - `optimizer`: `14.37 ms`
  - `view_loss`: `4.75 ms`
- Rasterizer internal average CUDA time per call:
  - forward total: `7.28 ms`
  - backward total: `26.36 ms`
  - backward `backward_render`: `25.52 ms`

### Backward residual analysis

- `view_backward - rasterizer_backward` leaves about `16.11 ms`/iter outside the rasterizer kernel timer.
- A single-iteration `torch.profiler` trace points to quaternion normalization backward as the main non-rasterizer source:
  - `LinalgVectorNormBackward0`
  - nested ops such as `aten::eq`, `aten::where`, `ClampMinBackward0`
- This maps back to `scene/gaussian_model.py`, where both `get_rotation()` and `get_rotation_r()` call `torch.nn.functional.normalize(...)` before rasterization.
- SSIM/L1 backward is comparatively small in the sampled trace.

### Additional saved artifacts

- Profile logs: `output/N3DV/cook_spinach_debug_profile_20260413_nolog/profiling`
- Backward trace table: `output/N3DV/cook_spinach_debug_profile_20260413/profiling/backward_trace/backward_cuda_table.txt`

## Follow-up: averaged backward operator profile

- Added `scripts/profile_backward_ops.py` to measure backward-only CUDA time from a saved checkpoint with `torch.profiler`.
- Run command:

```bash
python scripts/profile_backward_ops.py --config configs/dynerf/cook_spinach_debug.yaml --checkpoint output/N3DV/cook_spinach_debug/chkpnt500.pth --warmup_iters 5 --profile_iters 20 --output_dir output/N3DV/cook_spinach_debug_profile_20260413_nolog/profiling/backward_ops_avg
```

### Averaged backward-only result

- `avg_backward_ms`: `24.26 ms`
- `avg_rasterizer_backward_ms`: `18.24 ms`
- `avg_norm_backward_ms`: `0.33 ms`
- `norm_share_of_backward`: `1.37%`
- `rasterizer_share_of_backward`: `75.20%`

### Interpretation update

- The earlier single-iteration trace overestimated `normalize` as a bottleneck.
- In averaged backward-only profiling, quaternion normalization backward is small.
- The dominant backward bottleneck remains rasterizer backward.

## Follow-up: stabilize averaged backward operator profile

- Re-ran the backward-only checkpoint replay with more samples for a steadier estimate:

```bash
python scripts/profile_backward_ops.py --config configs/dynerf/cook_spinach_debug.yaml --checkpoint output/N3DV/cook_spinach_debug/chkpnt500.pth --warmup_iters 10 --profile_iters 100 --output_dir output/N3DV/cook_spinach_debug_profile_20260413_nolog/profiling/backward_ops_avg_100
```

### Stable backward-only result

- `avg_backward_ms`: `24.48 ms`
- `avg_rasterizer_backward_ms`: `18.76 ms`
- `avg_norm_backward_ms`: `0.37 ms`
- `norm_share_of_backward`: `1.52%`
- `rasterizer_share_of_backward`: `76.64%`

### Final correction

- The single-iteration backward trace should be treated as anecdotal only.
- The 100-step averaged backward replay is the final reference for operator attribution.
- `normalize` is not a major bottleneck; the main optimization target remains rasterizer backward.

## Visualization refresh

- Added `scripts/render_profile_dashboard.py` to generate a corrected, presentation-friendly dashboard from:
  - train section summary
  - rasterizer summary
  - backward operator replay summary
- Saved artifact:
  - `output/N3DV/cook_spinach_debug_profile_20260413_nolog/profiling/profiling_breakdown_dashboard_v2.png`

## Follow-up: tile-wise training pipeline and tile ratio sweep

- Added tile-wise training controls to `PipelineParams`:
  - `tile_training`
  - `tile_size`
  - `tile_ratio`
  - `tile_selection_mode`
- Added `utils/tile_utils.py` to build per-view tile selections and active-pixel masks.
- Updated `train.py` so tile training:
  - samples active tiles before rendering
  - forwards tile selections into the rasterizer
  - applies masked L1 / SSIM / opacity-mask losses on active pixels only
  - logs the effective active tile ratio during training
- Updated the diff Gaussian rasterizer so the tile-wise path:
  - launches forward/backward render kernels only for active tiles
  - remaps active launch indices back to image tile coordinates
  - recomputes `tiles_touched` against the active tile set and drops gaussians that touch no active tiles
  - clears only active tile ranges instead of the whole tile grid
- Added rasterizer profile fields:
  - `active_tiles`
  - `total_tiles`
  - `active_tile_ratio`

### Supporting implementation notes

- Fixed CLI-over-config precedence in `train.py` so sweep-time overrides such as `--model_path`, `--tile_ratio`, and profiling paths are not overwritten by YAML config values.
- Added a CUDA arch fallback in `gaussian_renderer/diff_gaussian_rasterization.py` to rebuild the extension for the actual local GPU capability when `TORCH_CUDA_ARCH_LIST` does not include it.
- Replaced the unstable CUB temp-storage query path in the rasterizer with:
  - `thrust::inclusive_scan`
  - `thrust::sort_by_key`
- This avoided invalid temp-buffer sizes observed on the local `RTX 2080 Ti (sm_75)` environment.

### Sweep command

```bash
conda run -n 4dgs python scripts/profile_tile_ratio_sweep.py --config configs/dynerf/cook_spinach_debug.yaml --output_root output/N3DV/tile_ratio_sweep_20260413
```

### Sweep setup

- Dataset/config: `configs/dynerf/cook_spinach_debug.yaml`
- Profiling window: warmup at `280-299 iter`, averaged on `300-500 iter`
- Tile size: `16x16`
- Cases:
  - baseline
  - `tile_ratio=0.75`
  - `tile_ratio=0.5`
  - `tile_ratio=0.25`
  - `tile_ratio=0.125`

### Average iteration/rasterizer results

- Baseline:
  - `iter_cuda_ms`: `69.50`
  - rasterizer forward total: `7.34 ms`
  - rasterizer backward total: `21.22 ms`
- `tile_ratio=0.75`:
  - `iter_cuda_ms`: `72.48`
  - speedup vs baseline: `0.959x`
- `tile_ratio=0.5`:
  - `iter_cuda_ms`: `65.01`
  - speedup vs baseline: `1.069x`
- `tile_ratio=0.25`:
  - `iter_cuda_ms`: `56.52`
  - speedup vs baseline: `1.230x`
- `tile_ratio=0.125`:
  - `iter_cuda_ms`: `53.83`
  - speedup vs baseline: `1.291x`

### Interpretation

- Tile-wise rendering becomes worthwhile once the active tile ratio is low enough; on this debug setup, `0.75` is slightly slower than baseline.
- The main gains come from rasterizer render/backward work shrinking with active tiles:
  - forward total: `7.34 ms -> 3.09 ms` from baseline to `0.125`
  - backward total: `21.22 ms -> 2.90 ms`
  - forward render: `3.48 ms -> 0.20 ms`
  - backward render: `20.67 ms -> 2.37 ms`
- Overall iteration speedup saturates because non-rasterizer costs remain.
- For this configuration, the practical efficiency region starts around `tile_ratio=0.25-0.5`.

### Saved artifacts

- Summary/report:
  - `output/N3DV/tile_ratio_sweep_20260413/tile_ratio_sweep_summary.csv`
  - `output/N3DV/tile_ratio_sweep_20260413/tile_ratio_sweep_report.txt`
- Visualizations:
  - `output/N3DV/tile_ratio_sweep_20260413/tile_ratio_iter_cuda_ms.png`
  - `output/N3DV/tile_ratio_sweep_20260413/tile_ratio_speedup.png`
