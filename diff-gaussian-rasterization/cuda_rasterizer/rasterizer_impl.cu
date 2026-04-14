/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace {
constexpr int kForwardProfileSize = 8;
constexpr int kBackwardProfileSize = 3;

struct CudaSectionTimer {
	bool enabled;
	cudaEvent_t start;
	cudaEvent_t stop;

	explicit CudaSectionTimer(bool enabled_) : enabled(enabled_), start(nullptr), stop(nullptr)
	{
		if (enabled)
		{
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start);
		}
	}

	~CudaSectionTimer()
	{
		if (enabled)
		{
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
		}
	}

	float stopAndElapsed()
	{
		if (!enabled)
			return 0.0f;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float elapsed = 0.0f;
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

template <typename Fn>
void runProfiledSection(bool enabled, float* slot, Fn&& fn)
{
	if (!enabled)
	{
		fn();
		if (slot != nullptr)
			*slot = 0.0f;
		return;
	}

	CudaSectionTimer timer(true);
	fn();
	if (slot != nullptr)
		*slot = timer.stopAndElapsed();
}
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	float3 orig_point={orig_points[3*idx],orig_points[3*idx+1],orig_points[3*idx+2]};
	present[idx] = in_frustum(orig_point, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

__global__ void countActiveTilesTouched(
	int P,
	const float2* points_xy,
	const int* radii,
	const int* active_tile_rank_map,
	dim3 grid,
	uint32_t* tiles_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;
	if (radii[idx] <= 0)
		return;

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	uint32_t count = 0;
	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			const int dense_tile_id = y * grid.x + x;
			if (active_tile_rank_map[dense_tile_id] >= 0)
			{
				count++;
			}
		}
	}

	tiles_touched[idx] = count;
}

__global__ void duplicateWithActiveKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
	const int* active_tile_rank_map,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] <= 0)
		return;

	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			const int dense_tile_id = y * grid.x + x;
			const int active_tile_rank = active_tile_rank_map[dense_tile_id];
			if (active_tile_rank < 0)
			{
				continue;
			}

			uint64_t key = static_cast<uint64_t>(active_tile_rank);
			key <<= 32;
			key |= *((uint32_t*)&depths[idx]);
			gaussian_keys_unsorted[off] = key;
			gaussian_values_unsorted[off] = idx;
			off++;
		}
	}
}

__global__ void fillBackgroundImage(
	int W,
	int H,
	const float* background,
	float* out_color,
	float* out_flow,
	float* out_depth,
	float* out_T,
	uint32_t* n_contrib)
{
	auto idx = cg::this_grid().thread_rank();
	const int pixels = W * H;
	if (idx >= pixels)
		return;

	out_T[idx] = 0.0f;
	n_contrib[idx] = 0;
	out_depth[idx] = 0.0f;
	out_flow[idx] = 0.0f;
	out_flow[pixels + idx] = 0.0f;
	for (int ch = 0; ch < NUM_CHANNELS; ch++)
	{
		out_color[ch * pixels + idx] = background[ch];
	}
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int D_t, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	float* out_means3D,
	const float* shs,
	const float* colors_precomp,
	const float* flows_precomp,
	const float* opacities,
	const float* ts,
	const float* scales,
	const float* scales_t,
	const float scale_modifier,
	const float* rotations,
	const float* rotations_r,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float timestamp,
	const float time_duration,
	const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
	const float tan_fovx, float tan_fovy,
	const int active_tile_count,
	const int* active_tile_ids,
	const int* active_tile_xy,
	const bool* active_tile_mask,
	const int* active_tile_rank_map,
	const bool profile,
	const bool prefiltered,
	float* out_color,
	float* out_flow,
	float* out_depth,
	float* out_T,
	int* radii,
	float* profile_out,
	bool debug)
{
	(void)active_tile_ids;
	(void)active_tile_mask;
	const bool use_active_tiles = active_tile_count >= 0;
	const int* active_tile_xy_ptr = (use_active_tiles && active_tile_count > 0) ? active_tile_xy : nullptr;
	const int* active_tile_rank_map_ptr = (use_active_tiles && active_tile_count > 0) ? active_tile_rank_map : nullptr;
	const bool enable_profile = profile && profile_out != nullptr;
	if (profile_out != nullptr)
	{
		std::fill(profile_out, profile_out + kForwardProfileSize, 0.0f);
	}
	CudaSectionTimer total_timer(enable_profile);

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[0] : nullptr, [&] {
		CHECK_CUDA(FORWARD::preprocess(
			P, D, D_t, M,
			means3D,
			out_means3D,
			ts,
			(glm::vec3*)scales,
			scales_t,
			scale_modifier,
			(glm::vec4*)rotations,
			(glm::vec4*)rotations_r,
			opacities,
			shs,
			geomState.clamped,
			cov3D_precomp,
			colors_precomp,
			viewmatrix, projmatrix,
			(glm::vec3*)cam_pos,
			timestamp,
			time_duration,
			rot_4d, gaussian_dim, force_sh_3d,
			width, height,
			focal_x, focal_y,
			tan_fovx, tan_fovy,
			radii,
			geomState.means2D,
			geomState.depths,
			geomState.cov3D,
			geomState.rgb,
			geomState.conic_opacity,
			tile_grid,
			geomState.tiles_touched,
			prefiltered
		), debug)
	});

	if (use_active_tiles && active_tile_count == 0)
	{
		runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[6] : nullptr, [&] {
			CHECK_CUDA(cudaMemset(imgState.accum_alpha, 0, width * height * sizeof(float)), debug);
			CHECK_CUDA(cudaMemset(imgState.n_contrib, 0, width * height * sizeof(uint32_t)), debug);
			const int pixels = width * height;
			fillBackgroundImage << <(pixels + 255) / 256, 256 >> > (
				width,
				height,
				background,
				out_color,
				out_flow,
				out_depth,
				out_T,
				imgState.n_contrib);
			CHECK_CUDA(, debug)
		});
		if (enable_profile)
			profile_out[7] = total_timer.stopAndElapsed();
		return 0;
	}

	if (use_active_tiles && active_tile_count > 0)
	{
		runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[1] : nullptr, [&] {
			countActiveTilesTouched << <(P + 255) / 256, 256 >> > (
				P,
				geomState.means2D,
				radii,
				active_tile_rank_map_ptr,
				tile_grid,
				geomState.tiles_touched);
			CHECK_CUDA(, debug)
		});
	}

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[2] : nullptr, [&] {
		CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)
	});

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[3] : nullptr, [&] {
		if (use_active_tiles && active_tile_count > 0)
		{
			duplicateWithActiveKeys << <(P + 255) / 256, 256 >> > (
				P,
				geomState.means2D,
				geomState.depths,
				geomState.point_offsets,
				binningState.point_list_keys_unsorted,
				binningState.point_list_unsorted,
				radii,
				active_tile_rank_map_ptr,
				tile_grid);
		}
		else
		{
			duplicateWithKeys << <(P + 255) / 256, 256 >> > (
				P,
				geomState.means2D,
				geomState.depths,
				geomState.point_offsets,
				binningState.point_list_keys_unsorted,
				binningState.point_list_unsorted,
				radii,
				tile_grid);
		}
		CHECK_CUDA(, debug)
	});

	// int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	int bit = 32;

	// Sort complete list of (duplicated) Gaussian indices by keys
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[4] : nullptr, [&] {
		CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
			binningState.list_sorting_space,
			binningState.sorting_size,
			binningState.point_list_keys_unsorted, binningState.point_list_keys,
			binningState.point_list_unsorted, binningState.point_list,
			num_rendered, 0, 32 + bit), debug)
	});

	const size_t range_count = use_active_tiles ? static_cast<size_t>(active_tile_count) : static_cast<size_t>(tile_grid.x * tile_grid.y);
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[5] : nullptr, [&] {
		CHECK_CUDA(cudaMemset(imgState.ranges, 0, range_count * sizeof(uint2)), debug);

		// Identify start and end of per-tile workloads in sorted list
		if (num_rendered > 0)
			identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
				num_rendered,
				binningState.point_list_keys,
				imgState.ranges);
		CHECK_CUDA(, debug)
	});

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* flow_ptr = flows_precomp;
	const dim3 render_grid = use_active_tiles ? dim3(active_tile_count, 1, 1) : tile_grid;
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[6] : nullptr, [&] {
		CHECK_CUDA(FORWARD::render(
			render_grid, block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			geomState.means2D,
			feature_ptr,
			flow_ptr,
			geomState.depths,
			geomState.conic_opacity,
			imgState.accum_alpha,
			imgState.n_contrib,
			background,
			out_color,
			out_flow,
			out_depth,
			active_tile_xy_ptr), debug)

		CHECK_CUDA(cudaMemcpy(out_T, imgState.accum_alpha, width * height * sizeof(float), cudaMemcpyDeviceToDevice), debug);
	});
	if (enable_profile)
		profile_out[7] = total_timer.stopAndElapsed();
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int D_t, int M, int R,
	const float* background,
	const int width, int height,
	const float* out_means3D,
	const float* shs,
	const float* colors_precomp,
	const float* flows_2d,
	const float* opacities,
	const float* ts,
	const float* scales,
	const float* scales_t,
	const float scale_modifier,
	const float* rotations,
	const float* rotations_r,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float timestamp,
    const float time_duration,
    const bool rot_4d, const int gaussian_dim, const bool force_sh_3d,
	const float tan_fovx, float tan_fovy,
	const int active_tile_count,
	const int* active_tile_ids,
	const int* active_tile_xy,
	const bool* active_tile_mask,
	const int* active_tile_rank_map,
	const bool profile,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_depths,
	const float* dL_masks,
	const float* dL_dpix_flow,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dflows,
	float* dL_dts,
	float* dL_dscale,
	float* dL_dscale_t,
	float* dL_drot,
	float* dL_drot_r,
	float* profile_out,
	bool debug)
{
	(void)active_tile_count;
	(void)active_tile_ids;
	(void)active_tile_mask;
	(void)active_tile_rank_map;
	const bool use_active_tiles = active_tile_count >= 0;
	const int* active_tile_xy_ptr = (use_active_tiles && active_tile_count > 0) ? active_tile_xy : nullptr;
	const bool enable_profile = profile && profile_out != nullptr;
	if (profile_out != nullptr)
	{
		std::fill(profile_out, profile_out + kBackwardProfileSize, 0.0f);
	}
	CudaSectionTimer total_timer(enable_profile);

	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const dim3 render_grid = use_active_tiles ? dim3(active_tile_count, 1, 1) : tile_grid;

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[0] : nullptr, [&] {
		CHECK_CUDA(BACKWARD::render(
			render_grid,
			block,
			imgState.ranges,
			binningState.point_list,
			width, height,
			background,
			geomState.means2D,
			geomState.conic_opacity,
			color_ptr,
			depth_ptr,
			flows_2d,
			imgState.accum_alpha,
			imgState.n_contrib,
			dL_dpix,
			dL_depths,
			dL_masks,
			dL_dpix_flow,
			(float3*)dL_dmean2D,
			(float4*)dL_dconic,
			dL_dopacity,
			dL_dcolor, dL_dflows,
			active_tile_xy_ptr), debug)
	});

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	runProfiledSection(enable_profile, profile_out != nullptr ? &profile_out[1] : nullptr, [&] {
		CHECK_CUDA(BACKWARD::preprocess(P, D, D_t, M,
			(float3*)out_means3D,
			radii,
			shs,
			ts,
			opacities,
			geomState.clamped,
			geomState.tiles_touched,
			(glm::vec3*)scales,
			scales_t,
			(glm::vec4*)rotations,
			(glm::vec4*)rotations_r,
			scale_modifier,
			cov3D_ptr,
			viewmatrix,
			projmatrix,
			focal_x, focal_y,
			tan_fovx, tan_fovy,
			(glm::vec3*)campos,
			timestamp,
			time_duration,
			rot_4d, gaussian_dim, force_sh_3d,
			(float3*)dL_dmean2D,
			dL_dconic,
			(glm::vec3*)dL_dmean3D,
			dL_dcolor,
			dL_dcov3D,
			dL_dsh, dL_dts,
			(glm::vec3*)dL_dscale,
			dL_dscale_t,
			(glm::vec4*)dL_drot,
			(glm::vec4*)dL_drot_r,
			dL_dopacity), debug)
	});
	if (enable_profile)
		profile_out[2] = total_timer.stopAndElapsed();
}
