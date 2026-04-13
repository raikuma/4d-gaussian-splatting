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
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

namespace {

std::string getRasterizerProfilePath() {
	const char* profile_dir = std::getenv("FOCUSGS_PROFILE_DIR");
	if (profile_dir == nullptr || profile_dir[0] == '\0') {
		return "";
	}
	return std::string(profile_dir) + "/rasterizer_profile.csv";
}

void appendRasterizerProfileRow(
	const char* phase,
	int iteration,
	int points,
	int image_width,
	int image_height,
	int active_tiles,
	int total_tiles,
	int num_rendered,
	float preprocess_ms,
	float scan_ms,
	float copy_rendered_ms,
	float duplicate_ms,
	float sort_ms,
	float zero_ranges_ms,
	float identify_ranges_ms,
	float render_ms,
	float copy_alpha_ms,
	float backward_render_ms,
	float backward_preprocess_ms,
	float total_ms)
{
	const std::string profile_path = getRasterizerProfilePath();
	if (profile_path.empty()) {
		return;
	}

	const bool write_header = !std::ifstream(profile_path).good();
	std::ofstream stream(profile_path, std::ios::app);
	if (!stream.is_open()) {
		return;
	}

	if (write_header) {
		stream
			<< "phase,iteration,points,image_width,image_height,active_tiles,total_tiles,active_tile_ratio,num_rendered,"
			<< "preprocess_ms,scan_ms,copy_rendered_ms,duplicate_ms,sort_ms,zero_ranges_ms,"
			<< "identify_ranges_ms,render_ms,copy_alpha_ms,backward_render_ms,"
			<< "backward_preprocess_ms,total_ms\n";
	}

	const float active_tile_ratio = total_tiles > 0 ? static_cast<float>(active_tiles) / static_cast<float>(total_tiles) : 1.0f;
	stream
		<< phase << ","
		<< iteration << ","
		<< points << ","
		<< image_width << ","
		<< image_height << ","
		<< active_tiles << ","
		<< total_tiles << ","
		<< active_tile_ratio << ","
		<< num_rendered << ","
		<< preprocess_ms << ","
		<< scan_ms << ","
		<< copy_rendered_ms << ","
		<< duplicate_ms << ","
		<< sort_ms << ","
		<< zero_ranges_ms << ","
		<< identify_ranges_ms << ","
		<< render_ms << ","
		<< copy_alpha_ms << ","
		<< backward_render_ms << ","
		<< backward_preprocess_ms << ","
		<< total_ms << "\n";
}

}  // namespace

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
	int* radii,
	const int* active_tile_mask,
	uint32_t* tiles_touched,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	if (radii[idx] <= 0)
	{
		tiles_touched[idx] = 0;
		return;
	}

	uint2 rect_min, rect_max;
	getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

	uint32_t active_count = 0;
	for (int y = rect_min.y; y < rect_max.y; y++)
	{
		for (int x = rect_min.x; x < rect_max.x; x++)
		{
			const uint32_t tile_id = y * grid.x + x;
			active_count += active_tile_mask[tile_id] ? 1u : 0u;
		}
	}

	tiles_touched[idx] = active_count;
	if (active_count == 0)
	{
		radii[idx] = 0;
	}
}

__global__ void clearActiveTileRanges(
	int active_tile_count,
	const int* active_tile_ids,
	uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= active_tile_count)
		return;

	const int tile_id = active_tile_ids[idx];
	ranges[tile_id] = make_uint2(0, 0);
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
	geom.scan_size = 0;
	geom.scanning_space = nullptr;
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
	binning.point_list_unsorted = binning.point_list;
	obtain(chunk, binning.point_list_keys, P, 128);
	binning.point_list_keys_unsorted = binning.point_list_keys;
	binning.sorting_size = 0;
	binning.list_sorting_space = nullptr;
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
	const int* active_tile_mask,
	const int* active_tile_ids,
	const int active_tile_count,
	const bool prefiltered,
	float* out_color,
	float* out_flow,
	float* out_depth,
	float* out_T,
	int* radii,
	bool debug,
	bool profile,
	int iteration)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);
	const bool profile_enabled = profile && !getRasterizerProfilePath().empty();
	float preprocess_ms = 0.0f;
	float scan_ms = 0.0f;
	float copy_rendered_ms = 0.0f;
	float duplicate_ms = 0.0f;
	float sort_ms = 0.0f;
	float zero_ranges_ms = 0.0f;
	float identify_ranges_ms = 0.0f;
	float render_ms = 0.0f;
	float copy_alpha_ms = 0.0f;
	CudaEventTimer total_timer(profile_enabled);

	size_t chunk_size = required<GeometryState>(P);
	if (chunk_size > (1ULL << 32))
	{
		std::ostringstream oss;
		oss << "Geometry buffer too large: " << chunk_size << " bytes for P=" << P;
		throw std::runtime_error(oss.str());
	}
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int total_tiles = tile_grid.x * tile_grid.y;
	const int render_tile_count = active_tile_ids != nullptr ? active_tile_count : total_tiles;
	dim3 render_grid = active_tile_ids != nullptr ? dim3(render_tile_count, 1, 1) : tile_grid;

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	if (img_chunk_size > (1ULL << 32))
	{
		std::ostringstream oss;
		oss << "Image buffer too large: " << img_chunk_size
			<< " bytes for width=" << width << " height=" << height;
		throw std::runtime_error(oss.str());
	}
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	{
		CudaEventTimer timer(profile_enabled);
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
		preprocess_ms = timer.stop();
	}

	if (active_tile_mask != nullptr)
	{
		countActiveTilesTouched << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			radii,
			active_tile_mask,
			geomState.tiles_touched,
			tile_grid);
		CHECK_CUDA(, debug)
	}

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	{
		CudaEventTimer timer(profile_enabled);
		if (P > 0)
		{
			thrust::inclusive_scan(
				thrust::device,
				geomState.tiles_touched,
				geomState.tiles_touched + P,
				geomState.point_offsets);
			CHECK_CUDA(, debug)
		}
		scan_ms = timer.stop();
	}

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	uint32_t num_rendered_u32 = 0;
	{
		CudaEventTimer timer(profile_enabled);
		CHECK_CUDA(cudaMemcpy(&num_rendered_u32, geomState.point_offsets + P - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost), debug);
		copy_rendered_ms = timer.stop();
	}
	const uint64_t max_rendered = static_cast<uint64_t>(P) * static_cast<uint64_t>(total_tiles);
	if (num_rendered_u32 > max_rendered)
	{
		std::ostringstream oss;
		oss << "Invalid num_rendered=" << num_rendered_u32
			<< " exceeds max_possible=" << max_rendered
			<< " (P=" << P << ", total_tiles=" << total_tiles << ")";
		throw std::runtime_error(oss.str());
	}
	const int num_rendered = static_cast<int>(num_rendered_u32);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	{
		CudaEventTimer timer(profile_enabled);
		duplicateWithKeys << <(P + 255) / 256, 256 >> > (
			P,
			geomState.means2D,
			geomState.depths,
			geomState.point_offsets,
			binningState.point_list_keys,
			binningState.point_list,
			radii,
			tile_grid);
		CHECK_CUDA(, debug)
		duplicate_ms = timer.stop();
	}

	// int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	int bit = 32;

	// Sort complete list of (duplicated) Gaussian indices by keys
	{
		CudaEventTimer timer(profile_enabled);
		if (num_rendered > 0)
		{
			thrust::sort_by_key(
				thrust::device,
				binningState.point_list_keys,
				binningState.point_list_keys + num_rendered,
				binningState.point_list);
			CHECK_CUDA(, debug)
		}
		sort_ms = timer.stop();
	}

	{
		CudaEventTimer timer(profile_enabled);
		if (active_tile_ids != nullptr)
		{
			clearActiveTileRanges << <(render_tile_count + 255) / 256, 256 >> > (
				render_tile_count,
				active_tile_ids,
				imgState.ranges);
			CHECK_CUDA(, debug)
		}
		else
		{
			CHECK_CUDA(cudaMemset(imgState.ranges, 0, total_tiles * sizeof(uint2)), debug);
		}
		zero_ranges_ms = timer.stop();
	}

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0) {
		CudaEventTimer timer(profile_enabled);
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
		CHECK_CUDA(, debug)
		identify_ranges_ms = timer.stop();
	}

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* flow_ptr = flows_precomp;
	{
		CudaEventTimer timer(profile_enabled);
		CHECK_CUDA(FORWARD::render(
			render_grid, block,
			imgState.ranges,
			active_tile_ids,
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
			out_depth), debug)
		render_ms = timer.stop();
	}

	{
		CudaEventTimer timer(profile_enabled);
		CHECK_CUDA(cudaMemcpy(out_T, imgState.accum_alpha, width * height * sizeof(float), cudaMemcpyDeviceToDevice), debug);
		copy_alpha_ms = timer.stop();
	}

	if (profile_enabled) {
		const float total_ms = total_timer.stop();
		appendRasterizerProfileRow(
			"forward",
			iteration,
			P,
			width,
			height,
			render_tile_count,
			total_tiles,
			num_rendered,
			preprocess_ms,
			scan_ms,
			copy_rendered_ms,
			duplicate_ms,
			sort_ms,
			zero_ranges_ms,
			identify_ranges_ms,
			render_ms,
			copy_alpha_ms,
			0.0f,
			0.0f,
			total_ms);
	}
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
	const int* active_tile_ids,
	const int active_tile_count,
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
	bool debug,
	bool profile,
	int iteration)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);
	const bool profile_enabled = profile && !getRasterizerProfilePath().empty();
	float backward_render_ms = 0.0f;
	float backward_preprocess_ms = 0.0f;
	CudaEventTimer total_timer(profile_enabled);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);
	const int total_tiles = tile_grid.x * tile_grid.y;
	const int render_tile_count = active_tile_ids != nullptr ? active_tile_count : total_tiles;
	const dim3 render_grid = active_tile_ids != nullptr ? dim3(render_tile_count, 1, 1) : tile_grid;

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* depth_ptr = geomState.depths;
	{
		CudaEventTimer timer(profile_enabled);
		CHECK_CUDA(BACKWARD::render(
			render_grid,
			block,
			imgState.ranges,
			active_tile_ids,
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
			dL_dcolor, dL_dflows), debug)
		backward_render_ms = timer.stop();
	}

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	{
		CudaEventTimer timer(profile_enabled);
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
		backward_preprocess_ms = timer.stop();
	}

	if (profile_enabled) {
		const float total_ms = total_timer.stop();
		appendRasterizerProfileRow(
			"backward",
			iteration,
			P,
			width,
			height,
			render_tile_count,
			total_tiles,
			R,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			0.0f,
			backward_render_ms,
			backward_preprocess_ms,
			total_ms);
	}
}
