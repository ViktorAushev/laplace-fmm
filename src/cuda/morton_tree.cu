#include "morton_tree.h"
#include "../common/simple_math.h"

#include "omp.h"
#include <iostream>
#include <iterator>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/atomic>
#include <device_launch_parameters.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <compare>
#include <thrust/extrema.h>

namespace fmm::gpu {

template <int dim>
__device__ __host__  size_t ExpandBits(double x, int tree_depth);

template <int dim, typename point>
__device__ __host__ size_t MortonCode(point val, int tree_depth, point shift, double scale);

template <typename point>
__device__ __host__ point PointFromMortonCode(size_t val, int depth, point shift, double scale);

template <typename point, typename value>
__device__ __host__ TreeCell<point, value>& TreeCell<point, value>::operator=(const particle<point, value>& particle)
{
	morton_code = particle.morton_code; return *this;
}

template <typename point, typename value>
__device__ __host__ TreeCell<point, value>& TreeCell<point, value>::operator=(const TreeCell<point, value>& cell)
{
	morton_code = cell.morton_code; return *this;
}

template <>
__device__ __host__  size_t ExpandBits<2>(double x, int tree_depth)
{
	size_t shift = (1ull << tree_depth);
	size_t val = shift * x;
	val = ((val << 16) | val) & 0xffff0000ffff;
	val = ((val << 8) | val) & 0xff00ff00ff00ff;
	val = ((val << 4) | val) & 0xf0f0f0f0f0f0f0f;
	val = ((val << 2) | val) & 0x3333333333333333;
	val = ((val << 1) | val) & 0x5555555555555555;
	return val;
}

template <>
__device__ __host__  size_t ExpandBits<3>(double x, int tree_depth)
{
	size_t bit_shift = (1ull << tree_depth);
	size_t val = bit_shift * x;
	val = ((val << 32) | val) & 0xffff00000000ffff;
    val = ((val << 16) | val) & 0xff0000ff0000ff;
    val = ((val << 8) | val) & 0xf00f00f00f00f00f;
    val = ((val << 4) | val) & 0x30c30c30c30c30c3;
    val = ((val << 2) | val) & 0x9249249249249249;
	return val;
}

template <>
__device__ __host__ size_t MortonCode<2,cuda_complex>(cuda_complex val, int tree_depth, cuda_complex shift, double scale)
{
	double x = (val.real() - shift.real()) / scale;
	double y = (val.imag() - shift.imag()) / scale;
	return (ExpandBits<2>(x, tree_depth) << 1) | ExpandBits<2>(y, tree_depth);
}

template <>
__device__ __host__  size_t MortonCode<3, point3d>(point3d val, int tree_depth, point3d shift, double scale)
{
	double x = (val[0] - shift[0]) / scale;
	double y = (val[1] - shift[1])  / scale;
	double z = (val[2] - shift[2])  / scale;
	return (ExpandBits<3>(x, tree_depth) << 2) | (ExpandBits<3>(y, tree_depth) << 1) | ExpandBits<3>(z, tree_depth);
}

template <>
__device__ __host__ cuda_complex PointFromMortonCode(size_t val, int depth, cuda_complex shift, double scale)
{
	double step = 1.0 / (1 << (depth + 1));
	double x = step, y = step;
	step *= 2;
	for (int i = 0; i < depth; ++i)
	{
		y += step * (val & 1);
		val >>= 1;
		x += step * (val & 1);
		val >>= 1;
		step *= 2;
	}
	x *= scale;
	x += shift.real();
	y *= scale;
	y += shift.imag();
	return { x, y };
}

template <>
__device__ __host__ point3d PointFromMortonCode(size_t val, int depth, point3d shift, double scale)
{
	double step = 1.0 / (1 << (depth + 1));
	double x = step, y = step, z = step;
	step *= 2;
	for (int i = 0; i < depth; ++i)
	{
		z += step * (val & 1);
		val >>= 1;
		y += step * (val & 1);
		val >>= 1;
		x += step * (val & 1);
		val >>= 1;
		step *= 2;
	}
	x *= scale;
	x += shift[0];
	y *= scale;
	y += shift[1];
	z *= scale;
	z += shift[2];
	return { x, y, z };
}

template <int dim, typename point, typename value>
__global__ void FillNeighbours(TreeCell<point, value>* top_level, size_t top_level_size, TreeCell<point, value>* bottom_level, size_t bottom_level_size,
								size_t* top_level_closenbrs, size_t* top_level_closenbrs_sizes,
								size_t* bottom_level_closenbrs, size_t* bottom_level_closenbrs_sizes,
								size_t* bottom_level_farnbrs, size_t* bottom_level_farnbrs_sizes, int level_num, double* scale)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (level_num == 1 && idx == 0)
	{
		top_level_closenbrs[0] = 0;
		top_level_closenbrs_sizes[0] = 1;
		for (int i = 0; i < bottom_level_size; ++i)
		{
			for (int j = 0; j < bottom_level_size; ++j)
			{
				if constexpr (dim == 2)
					bottom_level_closenbrs[9 * i + j] = j;
				else
					bottom_level_closenbrs[27 * i + j] = j;
			}
			bottom_level_closenbrs_sizes[i] = bottom_level_size;
			bottom_level_farnbrs_sizes[i] = 0;
		}
	}
	else
	{
		if (idx < bottom_level_size)
		{
			double dw = sqrt(dim + 1.e-5) / (1 << level_num) * (*scale);
			dw *= dw;

			auto& cell = bottom_level[idx];
			size_t& closenbrs_size = bottom_level_closenbrs_sizes[idx];
			closenbrs_size = 0;
			size_t& farnbrs_size = bottom_level_farnbrs_sizes[idx];
			farnbrs_size = 0;
			for (size_t j = 0; j < top_level_closenbrs_sizes[cell.parent]; ++j)
			{
				size_t nbr_idx;
				if constexpr (dim == 2)
					nbr_idx = top_level_closenbrs[9 * cell.parent + j];
				else
					nbr_idx = top_level_closenbrs[27 * cell.parent + j];

				const auto& [s, e] = top_level[nbr_idx].source_range;
				for (int k = s; k < e; ++k)
				{
					if (norm(bottom_level[k].center - cell.center) < dw) {
						if constexpr (dim == 2)
							bottom_level_closenbrs[9 * idx + closenbrs_size] = k;
						else
							bottom_level_closenbrs[27 * idx + closenbrs_size] = k;
						++closenbrs_size;
					}
					else {
						if constexpr (dim == 2)
							bottom_level_farnbrs[27 * idx + farnbrs_size] = k;
						else
							bottom_level_farnbrs[189 * idx + farnbrs_size] = k;
						++farnbrs_size;
					}
				}
					
			}
		}	
	}
}

template <int dim, typename point, typename value>
__global__ void for_each_morton_code(particle<point, value>* data, size_t size, size_t tree_depth, point* shift, double* scale)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		data[idx].morton_code = MortonCode<dim, point>(data[idx].center, tree_depth, *shift, *scale);
		assert(data[idx].morton_code < MyPow(MyPow(2, dim), tree_depth));
	}
}

template <typename point, typename value>
__global__ void for_each_counts(particle<point, value>* data, cuda::atomic<size_t>* counts, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		counts[1 + data[idx].morton_code]++;
}

template <typename point, typename value>
__global__ void for_each_copy(particle<point, value>* data, particle<point, value>* temp, cuda::atomic<size_t>* counts, size_t size, size_t* positions_map)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		auto& x = data[idx];
		size_t pos = counts[x.morton_code]++;
		positions_map[idx] = pos;
		assert(pos < size);
		temp[pos] = x;
	}
}

template <typename Val, typename point, int dim, typename value>
__global__ void fill_indexes(Val* bottom_level, TreeCell<point, value>* top_level, 
							 size_t bottom_level_size, size_t top_level_size, size_t tree_depth, point* shift, double* scale)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < top_level_size)
	{
		auto& cell = top_level[idx];
		cell.morton_code >>= dim;
		cell.center = PointFromMortonCode(cell.morton_code, tree_depth, *shift, *scale);

		size_t start = 0;
		size_t end = bottom_level_size;
		size_t middle = (start + end) / 2;

		for (;;)
		{
			auto cmp = (cell.morton_code <=> (bottom_level[middle].morton_code >> dim));
			if (std::is_lt(cmp)) {
				end = middle;
			}
			else {
				if (std::is_gt(cmp))
					start = middle;
				else
					break;
			}
			middle = (start + end) / 2;
		}
		start = middle;
		for (long long i = start - 1; i >= 0; --i)
		{
			if ((bottom_level[i].morton_code >> dim) == cell.morton_code)
				start = i;
			else
				break;
		}

		end = bottom_level_size;
		for (size_t j = start; j < bottom_level_size; ++j)
		{
			if ((bottom_level[j].morton_code >> dim) != cell.morton_code) {
				end = j;
				break;
			}
			else
			{
				if constexpr (std::is_same_v<Val, TreeCell<point, value>>)
					bottom_level[j].parent = idx;
			}
		}
		cell.source_range = { start, end };
	}
}

template <int dim, typename Val>
struct comp {
__device__ __host__ bool operator()(const Val& lhs, const Val& rhs) { return (lhs.morton_code >> dim) == (rhs.morton_code >> dim); }
};

struct compx {
	__device__ __host__ bool operator()(const fmm::gpu::particle2d& lhs, const fmm::gpu::particle2d& rhs) { return lhs.center.real() < rhs.center.real(); }
};

struct compy {
	__device__ __host__ bool operator()(const fmm::gpu::particle2d& lhs, const fmm::gpu::particle2d& rhs) { return lhs.center.imag() < rhs.center.imag(); }
};

template <int direction, typename value>
struct comp_d {
	__device__ __host__ bool operator()(const fmm::gpu::particle<point3d, value>& lhs, const fmm::gpu::particle<point3d, value>& rhs) { return lhs.center[direction] < rhs.center[direction]; }
};

__global__ void box_extent(fmm::gpu::particle2d* xmin, fmm::gpu::particle2d* xmax, fmm::gpu::particle2d* ymin, fmm::gpu::particle2d* ymax, cuda_complex* shift, double* dw)
{
	double dx = xmax->center.real() - xmin->center.real();
	double dy = ymax->center.imag() - ymin->center.imag();
	*dw = std::max(dx, dy);
	*shift = { xmin->center.real(), ymin->center.imag() };
	*shift -= cuda_complex{ 1.e-7, 1.e-7 }; // min particles shouldnt have coordinates equal zero
	*dw *= (1.0 + 1.e-6);// *(1.0 + 1.e-7 + std::max(std::abs(shift->real()), std::abs(shift->imag()))); // max particles shouldnt have coordinate equal 1
	printf("scale factors: shift = (%f, %f), extent = %f\n", shift->real(), shift->imag(), *dw);
}

template <typename value>
__global__ void box_extent(particle<point3d, value>* xmin, particle<point3d, value>* xmax, particle<point3d, value>* ymin, particle<point3d, value>* ymax, particle<point3d, value>* zmin, particle<point3d, value>* zmax, Vector3d* shift, double* dw)
{
	double dx = xmax->center[0] - xmin->center[0];
	double dy = ymax->center[1] - ymin->center[1];
	double dz = zmax->center[2] - zmin->center[2];
	*dw = std::max({dx, dy, dz});
	*shift = { xmin->center[0], ymin->center[1], zmin->center[2] };
	*shift -= Vector3d{ 1.e-7, 1.e-7, 1.e-7 }; // min particles shouldnt have coordinates equal zero
	*dw += 1.e-6;// max particles shouldnt have coordinate equal 1
	printf("scale factors: shift = (%f, %f, %f), extent = %f\n", (*shift)[0], (*shift)[1], (*shift)[2], *dw);
}

__global__ void fill_map(const size_t* positions_map, size_t* inv_positions_map, const size_t num_particles)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_particles) 
		inv_positions_map[positions_map[idx]] = idx;
}

template<typename point, typename value>
__global__ void fill_target_range(TreeCell<point, value>* leaves, size_t num_leaves, particle<point, value>* particles, size_t* inverse_positions_map, value Zero)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_leaves) 
	{
		auto& leaf = leaves[idx];
		auto& [s_begin, s_end] = leaf.source_range;
		auto& [t_begin, t_end] = leaf.target_range;
		t_end = s_end;
		t_begin = s_begin;
		for (size_t i = s_begin; i < s_end; ++i)
		{
			if (particles[i].q != Zero)
			{
				std::swap(particles[t_begin], particles[i]);
				std::swap(inverse_positions_map[t_begin++], inverse_positions_map[i]);
			}
		}
		s_end = t_begin;
	}
}

template <typename point, typename value>
struct cell2int {
	__device__ __host__ int operator()(const fmm::gpu::TreeCell<point, value>& cell) { 
	return cell.target_range.second - cell.target_range.first; }
};

template <int dim, typename point, typename host_point, typename value>
MortonTree<dim, point, host_point, value>::MortonTree(const std::vector<fmm::particle<host_point, value>>& particles_, size_t tree_depth_) : tree_depth(tree_depth_), num_particles(particles_.size())
{
	double T = omp_get_wtime();
	std::cout << "\n************ Start Building Tree ************" << std::endl;
	const int dim2 = MyPow(2, dim);
	double t = omp_get_wtime();
	static_assert(sizeof(fmm::particle2d) == sizeof(fmm::gpu::particle2d));
	static_assert(sizeof(fmm::particle3d) == sizeof(fmm::gpu::particle3d));
	static_assert(sizeof(fmm::particle3d3) == sizeof(fmm::gpu::particle3d3));
	const size_t counts_size = MyPow(dim2, tree_depth - 1) + 1; // max leaves size + 1
	const size_t num_blocks = (num_particles + 1024 - 1) / 1024;

	std::cout << "particles num = " << num_particles << std::endl;
	particle_t* particle_buf;
	cuda::atomic<size_t>* int_buf;

	{
		cudaMalloc(&particle_buf, sizeof(particle_t) * num_particles);
		cudaMalloc(&dev_particles, sizeof(particle_t) * num_particles);
		cudaMalloc(&dev_positions_map, sizeof(size_t) * num_particles);
		cudaMalloc(&int_buf, sizeof(cuda::atomic<size_t>) * counts_size);
		cudaMemset(int_buf, 0, sizeof(cuda::atomic<size_t>) * counts_size);
		dev_levels.resize(tree_depth);
		for (size_t i = 0; i < tree_depth; ++i)
			cudaMalloc(&dev_levels[i], sizeof(TreeCell_t) * MyPow(size_t(dim2), i));
		level_sizes.resize(tree_depth);
		dev_closeneighbours.resize(tree_depth);
		dev_closeneighbours_sizes.resize(tree_depth);
		dev_farneighbours.resize(tree_depth);
		dev_farneighbours_sizes.resize(tree_depth);
	}
	std::cout << "allocation time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	cudaMemcpy(particle_buf, particles_.data(), sizeof(particle_t) * num_particles, cudaMemcpyHostToDevice);
	std::cout << "copy particles time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	{
		cudaMalloc(&shift, sizeof(point));
		cudaMalloc(&scale, sizeof(double));
		if constexpr (dim == 2)
		{
			auto xminmax = thrust::minmax_element(thrust::device, particle_buf, particle_buf + num_particles, compx());
			auto yminmax = thrust::minmax_element(thrust::device, particle_buf, particle_buf + num_particles, compy());
			box_extent<<<1, 1>>>(xminmax.first, xminmax.second, yminmax.first, yminmax.second, shift, scale);
		}
		else
		{
			auto xminmax = thrust::minmax_element(thrust::device, particle_buf, particle_buf + num_particles, comp_d<0, value>());
			auto yminmax = thrust::minmax_element(thrust::device, particle_buf, particle_buf + num_particles, comp_d<1, value>());
			auto zminmax = thrust::minmax_element(thrust::device, particle_buf, particle_buf + num_particles, comp_d<2, value>());
			box_extent<<<1, 1>>>(xminmax.first, xminmax.second, yminmax.first, yminmax.second, zminmax.first, zminmax.second, shift, scale);
		}
		cudaDeviceSynchronize();
	}
	std::cout << "compute extent time: " << omp_get_wtime() - t << std::endl;

	for_each_morton_code<dim, point><<<num_blocks, 1024>>>(particle_buf, num_particles, tree_depth - 1, shift, scale);
	cudaDeviceSynchronize();

	t = omp_get_wtime();
	{
		// counting sort, using known number of unique numbers
		for_each_counts << <num_blocks, 1024 >> > (particle_buf, int_buf, num_particles);
		cudaDeviceSynchronize();

		auto ptr = new size_t[counts_size];
		ptr[0] = 0;

		static_assert(sizeof(size_t) == sizeof(cuda::atomic<size_t>));
		cudaMemcpy(ptr, int_buf, counts_size * sizeof(cuda::atomic<size_t>), cudaMemcpyDeviceToHost);

		for (size_t i = 1; i < counts_size; ++i)
			ptr[i] += ptr[i - 1];
		assert(ptr[counts_size - 1] == num_particles);

		cudaMemcpy(int_buf, ptr, counts_size * sizeof(cuda::atomic<size_t>), cudaMemcpyHostToDevice);
		delete[] ptr;

		for_each_copy << <num_blocks, 1024 >> > (particle_buf, dev_particles, int_buf, num_particles, dev_positions_map);
		cudaDeviceSynchronize();

		//std::cout << std::is_sorted(dev_particles, dev_particles + num_particles, [](const auto& x, const auto& y) {return x.morton_code < y.morton_code; }) << std::endl;
	}
	std::cout << "sort time: " << omp_get_wtime() - t << std::endl;
	t = omp_get_wtime();

	{
		auto& top_level = dev_levels.back();
		auto ptr = thrust::unique_copy(thrust::device, dev_particles, dev_particles + num_particles, top_level, comp<0, particle_t>());
		level_sizes.back() = ptr - top_level;

		fill_indexes<particle_t, point, 0><<<(level_sizes.back() + 1024 - 1) / 1024, 1024 >> >(dev_particles, top_level, num_particles, level_sizes.back(), tree_depth - 1, shift, scale);
		cudaDeviceSynchronize();

		std::cout << "level " << tree_depth - 1 << " size: " << level_sizes.back() << std::endl;
	}

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = dev_levels[i];
		auto& bottom_level = dev_levels[i + 1];

		auto ptr = thrust::unique_copy(thrust::device, bottom_level, bottom_level + level_sizes[i + 1], top_level, comp<dim, TreeCell_t>());
		level_sizes[i] = ptr - top_level;
		std::cout << "level " << i << " size: " << level_sizes[i] << std::endl;

		fill_indexes<TreeCell_t, point, dim><<<(level_sizes[i] + 1024 - 1) / 1024, 1024 >> >(bottom_level, top_level, level_sizes[i + 1], level_sizes[i], i, shift, scale);
		cudaDeviceSynchronize();
	}
	std::cout << "create levels time: " << omp_get_wtime() - t << std::endl;

	for (int i = 0; i < tree_depth; ++i)
	{
		if constexpr (dim == 2)
		{
			cudaMalloc(&dev_closeneighbours[i], sizeof(size_t) * level_sizes[i] * 9);
			cudaMalloc(&dev_farneighbours[i], sizeof(size_t) * level_sizes[i] * 27);
		}
		else
		{
			cudaMalloc(&dev_closeneighbours[i], sizeof(size_t) * level_sizes[i] * 27);
			cudaMalloc(&dev_farneighbours[i], sizeof(size_t) * level_sizes[i] * 189);
		}
		cudaMalloc(&dev_closeneighbours_sizes[i], sizeof(size_t) * level_sizes[i]);
		cudaMalloc(&dev_farneighbours_sizes[i], sizeof(size_t) * level_sizes[i]);
	}

	t = omp_get_wtime();
	for (int i = 1; i < tree_depth; ++i)
	{
		FillNeighbours<dim, point><<<(level_sizes[i] + 1024 - 1) / 1024, 1024>>>(dev_levels[i - 1], level_sizes[i - 1], dev_levels[i], level_sizes[i],
			dev_closeneighbours[i - 1], dev_closeneighbours_sizes[i - 1],
			dev_closeneighbours[i], dev_closeneighbours_sizes[i],
			dev_farneighbours[i], dev_farneighbours_sizes[i], i, scale);
		cudaDeviceSynchronize();
	}
	std::cout << "fill neighbours time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	{
		size_t* dev_inverse_positions_map;
		cudaMalloc(&dev_inverse_positions_map, sizeof(size_t) * num_particles);
		fill_map<<<(num_particles + 1024 - 1) / 1024, 1024>>>(dev_positions_map, dev_inverse_positions_map, num_particles);
		cudaDeviceSynchronize();
		value Zero;
		if constexpr (std::is_same_v<value, double>)
			Zero = 0;
		if constexpr (std::is_same_v<value, Vector3d>)
			Zero = { 0,0,0 };
		fill_target_range<<<(level_sizes.back() + 1024 - 1) / 1024, 1024>>>(dev_levels.back(), level_sizes.back(), dev_particles, dev_inverse_positions_map, Zero);
		cudaDeviceSynchronize();
		fill_map<<<(num_particles + 1024 - 1) / 1024, 1024>>>(dev_inverse_positions_map, dev_positions_map, num_particles);
		cudaDeviceSynchronize();
		targets_num = thrust::transform_reduce(thrust::device, dev_levels.back(), dev_levels.back() + level_sizes.back(), cell2int<point, value>(), 0, thrust::plus<int>());
		std::cout << "targets num: " << targets_num << std::endl;
		cudaFree(&dev_inverse_positions_map);
	}
	std::cout << "source/target partition time: " << omp_get_wtime() - t << std::endl;

	cudaFree(&int_buf);
	cudaFree(&particle_buf);

	std::cout << "total build time: " << omp_get_wtime() - T << std::endl;
	std::cout << "************ End Building Tree ************\n\n";
}

template <int dim, typename point, typename host_point, typename value>
MortonTree<dim, point, host_point, value>::~MortonTree()
{
	cudaFree(&dev_particles);
	cudaFree(&dev_positions_map);
	for (size_t i = 0; i < tree_depth; ++i)
		cudaFree(&dev_levels[i]);
	for (int i = 0; i < tree_depth; ++i)
	{
		cudaFree(&dev_closeneighbours[i]);
		cudaFree(&dev_farneighbours[i]);
		cudaFree(&dev_closeneighbours_sizes[i]);
		cudaFree(&dev_farneighbours_sizes[i]);
	}
	cudaFree(&shift);
	cudaFree(&scale);
}

template class MortonTree<2, cuda_complex, fmm::point2d, double>;
template class MortonTree<3, Vector3d, Vector3d, double>;
template class MortonTree<3, Vector3d, Vector3d, Vector3d>;

}
