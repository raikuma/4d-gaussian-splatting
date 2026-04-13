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
