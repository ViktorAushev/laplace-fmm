#include "multipole.h"
#include "../common/simple_math.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include "omp.h"
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda/atomic>
#include <thrust/fill.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <cmath>

namespace fmm::gpu {

using namespace std::complex_literals;

const int MAX_MULTIPOLE_NUM = fmm::detail::_2d_MAX_MULTIPOLE_NUM;

template<int n>
struct cuda_binom
{
public:
	std::array<double, n * n> cft = fmm::detail::binom.cft;
	int dim = n;
	constexpr double operator()(int p, int q) const
	{
		return cft[p * dim + q];
	}
};

__constant__ cuda_binom<2 * MAX_MULTIPOLE_NUM> binom;

__global__ void multipole(TreeCell2d* leaves, particle2d* particles, cuda_complex* outer, int N)
{
	size_t leaf_idx = blockIdx.x;
	const auto [s, e] = leaves[leaf_idx].source_range;
	const int particle_num = e - s;
	const int particles_per_thread = (particle_num + blockDim.x - 1) / blockDim.x;

	const cuda_complex center = leaves[leaf_idx].center;
	cuda_complex local_outer[MAX_MULTIPOLE_NUM];
	for (int mult_num = 0; mult_num < N; ++mult_num)
	{
		local_outer[mult_num] = 0;
	}

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			auto local_particle = particles[particle_idx];
			for (int mult_num = 1; mult_num < N; ++mult_num)
			{
				local_outer[mult_num] -= local_particle.q * MyPow(local_particle.center - center, mult_num);
			}
			local_outer[0] += local_particle.q;
		}
	}

	assert(blockDim.x <= 32);
	__shared__ cuda_complex shared_outer[32];
	for (int i = 0; i < N; ++i)
	{
		shared_outer[threadIdx.x] = local_outer[i];
		__syncthreads();
		for (int s = blockDim.x / 2; s >= 1; s >>= 1)
		{
			if (threadIdx.x < s)
				shared_outer[threadIdx.x] += shared_outer[s + threadIdx.x];
			__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			outer[leaf_idx * N + i] = shared_outer[0] / double(std::max(1,i));
		}
	}
}

__global__ void M2M(TreeCell2d* top_level, size_t top_level_size, TreeCell2d* bottom_level, size_t bottom_level_size, cuda_complex* top_outer, cuda_complex* bottom_outer, int N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < top_level_size)
	{
		const auto& cell = top_level[idx];
		const auto [s, e] = cell.source_range;
		const cuda_complex center = cell.center;

		cuda_complex local_outer[MAX_MULTIPOLE_NUM];
		cuda_complex work[MAX_MULTIPOLE_NUM];
		cuda_complex other_outer[MAX_MULTIPOLE_NUM];
		work[0] = 1.0;
		for (int j = 0; j < N; ++j)
		{
			 local_outer[j] = 0;
		}

		for (size_t k = s; k < e; ++k)
		{
			const auto& child_cell = bottom_level[k];
			const cuda_complex z0 = child_cell.center - center;
			for (int j = 0; j < N; ++j)
			{
				other_outer[j] = bottom_outer[k * N + j];
			}
			for (int i = 1; i < N; ++i)
				work[i] = work[i - 1] * z0;

			for (int j = 1; j < N; ++j)
			{
				for (int i = 1; i <= j; ++i)
				{
					local_outer[j] += other_outer[i] * work[j - i] * binom(j - 1, i - 1);
				}
				local_outer[j] -= other_outer[0] * work[j] / double(j);
			}
			local_outer[0] += other_outer[0];
		}	

		auto outer = top_outer + idx * N;
		for (int j = 0; j < N; ++j)
		{
			outer[j] = local_outer[j];
		}
	}
}
 
__global__ void M2L(TreeCell2d* level, size_t level_size, cuda_complex* level_inner, cuda_complex* level_outer, int N, size_t* farneighbours, size_t* farneighbours_sizes)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < level_size)
	{
		const TreeCell2d& cell = level[idx];
		const cuda_complex center = cell.center;
		cuda_complex work[MAX_MULTIPOLE_NUM];
		cuda_complex other_outer[MAX_MULTIPOLE_NUM];
		cuda_complex inner[MAX_MULTIPOLE_NUM];

		for (int j = 0; j < N; ++j)
		{
			inner[j] = level_inner[idx * N + j];
		}

		for (size_t k = 0; k < farneighbours_sizes[idx]; ++k)
		{
			size_t nbr_idx = farneighbours[idx * 27 + k];
			const TreeCell2d& nbr_cell = level[nbr_idx];
			const cuda_complex z0 = nbr_cell.center - center;
			for (int j = 0; j < N; ++j)
			{
				other_outer[j] = level_outer[nbr_idx * N + j];
			}

			work[0] = other_outer[0];
			inner[0] += other_outer[0] * cuda::std::conj(log(-z0));
			
			for (int i = 1; i < N; ++i)
			{
				work[i] = ni(i) * other_outer[i] / MyPow(z0, i);
				inner[0] += work[i];
			}

			for (int i = 1; i < N; ++i)
			{
				cuda_complex tmp{ 0.0, 0.0 };
				for (int j = 1; j < N; ++j)
				{
					tmp += work[j] * double(binom(i + j - 1, j - 1));
				}
				inner[i] += (- other_outer[0] / double(i) + tmp) / MyPow(z0, i);
			}
		}

		for (int j = 0; j < N; ++j)
		{
			level_inner[idx * N + j] = inner[j];
		}
	}
}

__global__ void L2L(TreeCell2d* top_level, size_t top_level_size, TreeCell2d* bottom_level, size_t bottom_level_size, cuda_complex* top_inner, cuda_complex* bottom_inner, int N)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < bottom_level_size)
	{
		const TreeCell2d& cell = bottom_level[idx];
		const TreeCell2d& parent_cell = top_level[cell.parent];
		const cuda_complex z0 = parent_cell.center - cell.center;
		cuda_complex b[MAX_MULTIPOLE_NUM];

		auto a = top_inner + cell.parent * N;
		for (int i = 0; i < N; ++i)
			b[i] = a[i];
		for (int j = 0; j < N; ++j)
		{
			for (int k = N - j - 1; k < N - 1; ++k)
			{
				b[k] -= z0 * b[k + 1];
			}
		}
		for (int i = 0; i < N; ++i)
			bottom_inner[idx * N + i] = b[i];
	}
}

void FastMultipole::Upward()
{
	double t1 = omp_get_wtime();

	auto& leaves = tree->dev_levels.back();
	auto& leaves_outer = outer_expansions.back();
	const auto& particles = tree->dev_particles;
	
	const size_t num_leaves = tree->level_sizes.back();
	multipole<<<num_leaves, 32>>>(leaves, particles, leaves_outer, N);
	cudaDeviceSynchronize();

	std::cout << "multipole time: " << omp_get_wtime() - t1 << std::endl;

	t1 = omp_get_wtime();

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = tree->dev_levels[i];
		auto& top_outer = outer_expansions[i];
		const auto& bottom_level = tree->dev_levels[i + 1];
		const auto& bottom_outer = outer_expansions[i + 1];

		const size_t num_blocks = (tree->level_sizes[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;

		M2M<<<num_blocks, BLOCK_SIZE>>>(top_level, tree->level_sizes[i], bottom_level, tree->level_sizes[i + 1], top_outer, bottom_outer, N);
		cudaDeviceSynchronize();
		}

	std::cout << "m2m time: " << omp_get_wtime() - t1 << std::endl;
}

void FastMultipole::Downward()
{
	for (int i = 2; i < tree_depth; ++i)
	{
		const auto& top_level = tree->dev_levels[i - 1];
		const auto& top_inner = inner_expansions[i - 1];
		auto& bottom_level = tree->dev_levels[i];
		auto& bottom_inner = inner_expansions[i];
		const auto& bottom_outer = outer_expansions[i];
		const auto& neighbours = tree->dev_farneighbours[i];
		const auto& nbr_sizes = tree->dev_farneighbours_sizes[i];
		const size_t num_blocks = (tree->level_sizes[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;

		L2L<<<num_blocks, BLOCK_SIZE>>>(top_level, tree->level_sizes[i - 1], bottom_level, tree->level_sizes[i], top_inner, bottom_inner, N);
		cudaDeviceSynchronize();
		M2L<<<num_blocks, BLOCK_SIZE>>>(bottom_level, tree->level_sizes[i], bottom_inner, bottom_outer, N, neighbours, nbr_sizes);
		cudaDeviceSynchronize();
		}
		}

//__global__ void Compute_Forces_Error(particle2d* particles, cuda_complex* dev_forces, size_t num_particles, std::pair<cuda_complex, cuda_complex>* cmp_forces)
//{
//	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//	if (idx < num_particles)
//	{
//		const cuda_complex r = particles[idx].center;
//		cuda_complex force{ 0.0, 0.0 };
//		for (size_t i = 0; i < num_particles; ++i)
//		{
//			const cuda_complex dz = r - particles[i].center;
//			double n = cuda::std::norm(dz);
//			force += particles[i].q * cuda::std::conj(dz) / std::max(n, CUDA_FORCE_EPS2);
//		}
//		cmp_forces[idx] = { force, dev_forces[idx] };
//	}
//}
//
//struct absolute_error {
//	__device__ double operator()(const std::pair<cuda_complex, cuda_complex>& rhs) { return cuda::std::abs(rhs.first - rhs.second); }
//};
//
//__device__ double sqr(double x) { return x * x; }
//
//struct l2_error {
//	__device__ double operator()(const std::pair<cuda_complex, cuda_complex>& rhs) { return sqr(cuda::std::abs(rhs.first - rhs.second) / cuda::std::abs(rhs.first)); }
//};

//void FastMultipole::ComputeForcesError()
//{
//	auto& particles = tree->dev_particles;
//	const size_t num_particles = tree->num_particles;
//	const size_t num_blocks = (num_particles + 1024 - 1) / 1024;
//
//	std::pair<cuda_complex, cuda_complex>* cmp_forces;
//	cudaMalloc(&cmp_forces, sizeof(std::pair<cuda_complex, cuda_complex>) * num_particles);
//	Compute_Forces_Error<<<num_blocks, 1024>>>(particles, dev_forces, num_particles, cmp_forces);
//	cudaDeviceSynchronize();
//	double max_error = thrust::transform_reduce(thrust::device, cmp_forces, cmp_forces + num_particles, absolute_error(), 0.0, thrust::maximum<double>());
//	std::cout << "max error = " << max_error << std::endl;
//	double l2error = thrust::transform_reduce(thrust::device, cmp_forces, cmp_forces + num_particles, l2_error(), 0.0, thrust::maximum<double>());
//	std::cout << "l2error = " << sqrt(l2error / num_particles) << std::endl;
//	cudaFree(&cmp_forces);
//}

template <bool USE_SOURCE>
__global__ void Compute_Potentials(TreeCell2d* leaves, size_t num_leaves, particle2d* particles, cuda_complex* leaves_inner, int N, size_t* close_neighbours, size_t* close_neighbours_sizes, double* dev_potentials)
{
	size_t leaf_idx = blockIdx.x;
	const auto& cell = leaves[leaf_idx];
	size_t s,e;
	if constexpr (USE_SOURCE == true)
	{
		s = cell.source_range.first;
		e = cell.source_range.second;
	}
	else
	{
		s = cell.target_range.first;
		e = cell.target_range.second;
	}
	const int particle_num = e - s;
	const int particles_per_thread = (particle_num + blockDim.x - 1) / blockDim.x;

	__shared__ cuda_complex inner[MAX_MULTIPOLE_NUM];
	if (particle_num < N)
	{
		if (threadIdx.x == 0)
			for (int i = 0; i < N; ++i)
				inner[i] = leaves_inner[leaf_idx * N + i];
	}
	else
	{
		if (threadIdx.x < N)
			inner[threadIdx.x] = leaves_inner[leaf_idx * N + threadIdx.x];
	}
	__syncthreads();

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			double potential{ 0.0 };
			cuda_complex dzpow{ 1.0, 0.0 };
			cuda_complex dz;

			const particle2d p1 = particles[particle_idx];

			for (size_t k = 0; k < close_neighbours_sizes[leaf_idx]; ++k)
			{
				size_t nbr_idx = close_neighbours[leaf_idx * 9 + k];
				const auto [ns, ne] = leaves[nbr_idx].source_range;

				for (size_t j = ns; j < ne; ++j) {
					potential += Potential2d(p1, particles[j]);
				}
			}

			dz = p1.center - cell.center;
			dzpow = 1;
			for (int k = 0; k < N; ++k)
			{
				potential += inner[k].real() * dzpow.real() - inner[k].imag() * dzpow.imag();
				dzpow *= dz;
			}
			dev_potentials[particle_idx] += potential;
		}
	}
}

template <bool USE_SOURCE>
__global__ void Compute_Forces(TreeCell2d* leaves, size_t num_leaves, particle2d* particles, cuda_complex* leaves_inner, int N, size_t* close_neighbours, size_t* close_neighbours_sizes, cuda_complex* dev_forces)
{
	size_t leaf_idx = blockIdx.x;
	const auto& cell = leaves[leaf_idx];
	size_t s, e;
	if constexpr (USE_SOURCE == true)
	{
		s = cell.source_range.first;
		e = cell.source_range.second;
	}
	else
	{
		s = cell.target_range.first;
		e = cell.target_range.second;
	}
	const int particle_num = e - s;
	const int particles_per_thread = (particle_num + blockDim.x - 1) / blockDim.x;

	__shared__ cuda_complex inner[MAX_MULTIPOLE_NUM];
	if (particle_num < N)
	{
		if (threadIdx.x == 0)
			for (int i = 0; i < N; ++i)
				inner[i] = leaves_inner[leaf_idx * N + i];
	}
	else
	{
		if (threadIdx.x < N)
			inner[threadIdx.x] = leaves_inner[leaf_idx * N + threadIdx.x];
	}
	__syncthreads();

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			cuda_complex force{ 0.0, 0.0 };
			cuda_complex dzpow{ 1.0, 0.0 };
			cuda_complex dz;

			const particle2d p1 = particles[particle_idx];

			for (size_t k = 0; k < close_neighbours_sizes[leaf_idx]; ++k)
			{
				size_t nbr_idx = close_neighbours[leaf_idx * 9 + k];
				const auto [ns, ne] = leaves[nbr_idx].source_range;

				for (size_t j = ns; j < ne; ++j) {
					force += Force2d(p1, particles[j]);
				}
			}

			dz = p1.center - cell.center;
			dzpow = 1;
			for (int k = 1; k < N; ++k)
			{
				force += double(k) * inner[k] * dzpow;
				dzpow *= dz;
			}
			dev_forces[particle_idx] += force;
		}
	}
}

template <decltype(&fmm::gpu::TreeCell2d::source_range) range_ptr> 
struct cell2int {
	__device__ __host__ int operator()(const fmm::gpu::TreeCell2d& cell) { 
		return (cell.*range_ptr).second - (cell.*range_ptr).first; }
};

void FastMultipole::ComputeForces()
{
	auto& leaves = tree->dev_levels.back();
	auto& leaves_inner = inner_expansions.back();
	auto& particles = tree->dev_particles;
	auto& neighbours = tree->dev_closeneighbours.back();
	auto& nbr_sizes = tree->dev_closeneighbours_sizes.back();

	const size_t num_leaves = tree->level_sizes.back();
	int max_particles;
	if (tree->targets_num == 0)
		max_particles = thrust::transform_reduce(thrust::device, tree->dev_levels.back(), tree->dev_levels.back() + tree->level_sizes.back(),
		cell2int<&fmm::gpu::TreeCell2d::source_range>(), 0, thrust::maximum<int>());
	else
		max_particles = thrust::transform_reduce(thrust::device, tree->dev_levels.back(), tree->dev_levels.back() + tree->level_sizes.back(),
		cell2int<&fmm::gpu::TreeCell2d::target_range>(), 0, thrust::maximum<int>());
	std::cout << "max particles in cell = " << max_particles << std::endl;
	const size_t num_blocks = num_leaves;
	const size_t num_threads = std::min(512, (max_particles + 32 - 1) / 32 * 32);
	if (tree->targets_num == 0)
		Compute_Forces<true><<<num_blocks, num_threads>>>(leaves, num_leaves, particles, leaves_inner, N, neighbours, nbr_sizes, dev_forces);
	else
		Compute_Forces<false><<<num_blocks, num_threads>>>(leaves, num_leaves, particles, leaves_inner, N, neighbours, nbr_sizes, dev_forces);
	cudaDeviceSynchronize();
}

void FastMultipole::ComputePotentials()
{
	auto& leaves = tree->dev_levels.back();
	auto& leaves_inner = inner_expansions.back();
	auto& particles = tree->dev_particles;
	auto& neighbours = tree->dev_closeneighbours.back();
	auto& nbr_sizes = tree->dev_closeneighbours_sizes.back();

	const size_t num_leaves = tree->level_sizes.back();
	int max_particles;
	if (tree->targets_num == 0)
		max_particles = thrust::transform_reduce(thrust::device, tree->dev_levels.back(), tree->dev_levels.back() + tree->level_sizes.back(),
		cell2int<&fmm::gpu::TreeCell2d::source_range>(), 0, thrust::maximum<int>());
	else
		max_particles = thrust::transform_reduce(thrust::device, tree->dev_levels.back(), tree->dev_levels.back() + tree->level_sizes.back(),
		cell2int<&fmm::gpu::TreeCell2d::target_range>(), 0, thrust::maximum<int>());
	std::cout << "max particles in cell = " << max_particles << std::endl;
	const size_t num_blocks = num_leaves;
	const size_t num_threads = std::min(512, (max_particles + 32 - 1) / 32 * 32);
	if (tree->targets_num == 0)
		Compute_Potentials<true><<<num_blocks, num_threads>>>(leaves, num_leaves, particles, leaves_inner, N, neighbours, nbr_sizes, dev_potentials);
	else
		Compute_Potentials<false><<<num_blocks, num_threads>>>(leaves, num_leaves, particles, leaves_inner, N, neighbours, nbr_sizes, dev_potentials);
	cudaDeviceSynchronize();
}

template <typename T>
__global__ void map_sort(const T* values, T* buf, const size_t* positions_map, size_t num)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num)
		buf[idx] = values[positions_map[idx]];
}

template <typename T, typename T2>
void Map_Sort(const T* dev_values, const size_t* dev_positions_map, std::vector<T2>& values, size_t num_particles)
{
	static_assert(sizeof(T) == sizeof(T2));
	T* dev_buf;
	cudaMalloc(&dev_buf, num_particles * sizeof(T));
	const size_t num_blocks = (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
	map_sort<<<num_blocks, BLOCK_SIZE>>>(dev_values, dev_buf, dev_positions_map, num_particles);
	cudaDeviceSynchronize();
	cudaMemcpy(values.data(), dev_buf, num_particles * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(&dev_buf);
}

FastMultipole::FastMultipole(const std::vector<fmm::particle2d>& particles, double eps, int N_, int tree_depth_)
{
	targets_num = particles.size();
	Solve(particles, eps, N_, tree_depth_);
}

FastMultipole::FastMultipole(const std::vector<fmm::particle2d>& source_particles, const std::vector<fmm::particle2d>& target_particles, double eps, int N_, int tree_depth_)
{
	std::vector<fmm::particle2d> particles(source_particles.size() + target_particles.size());
	memcpy(particles.data(), target_particles.data(), target_particles.size() * sizeof(fmm::particle2d));
	memcpy(particles.data() + target_particles.size(), source_particles.data(), source_particles.size() * sizeof(fmm::particle2d));
	targets_num = target_particles.size();
	Solve(particles, eps, N_, tree_depth_);
}

void FastMultipole::Solve(const std::vector<fmm::particle2d>& particles, double eps, int N_, int tree_depth_)
{
	double T = omp_get_wtime();
	std::cout << "\n***************** Start FMM **************" << std::endl;
	num_particles = particles.size();
	if (N_ == FMM_AUTO)
		N = std::min(fmm::detail::_2d_MAX_MULTIPOLE_NUM, int(-1.18 * log(1.65 * eps)));
	else
		N = std::min(fmm::detail::_2d_MAX_MULTIPOLE_NUM, N_);
	std::cout << "multipole num = " << N << std::endl;
	if (tree_depth_ == FMM_AUTO)
		tree_depth = std::max(3.0, 2 + log(double(num_particles) / 100.0) / log(4.0));
	else
		tree_depth = tree_depth_;
	std::cout << "tree depth = " << tree_depth << std::endl;

	tree = std::make_shared<MortonTree2d>(particles, tree_depth);

	double t = omp_get_wtime();
	outer_expansions.resize(tree_depth);
	inner_expansions.resize(tree_depth);
	for (int i = 0; i < tree_depth; ++i)
	{
		cudaMalloc(&outer_expansions[i], tree->level_sizes[i] * N * sizeof(cuda_complex));
		cudaMalloc(&inner_expansions[i], tree->level_sizes[i] * N * sizeof(cuda_complex));
	}
	cudaMalloc(&dev_forces, num_particles * sizeof(cuda_complex));
	cudaMalloc(&dev_potentials, num_particles * sizeof(double));
	forces.resize(targets_num);
	potentials.resize(targets_num);
	std::cout << "allocation time: " << omp_get_wtime() - t << std::endl;

	Upward();
	
	t = omp_get_wtime();
	Downward();
	std::cout << "m2l+l2l time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	ComputePotentials();
	Map_Sort(dev_potentials, tree->dev_positions_map, potentials, targets_num);
	std::cout << "leaf potential time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	ComputeForces();
	Map_Sort(dev_forces, tree->dev_positions_map, forces, targets_num);
	std::cout << "leaf force time: " << omp_get_wtime() - t << std::endl;

	std::cout << "total fmm time: " << omp_get_wtime() - T << std::endl;
	std::cout << "***************** End FMM **************\n" << std::endl;
}

FastMultipole::~FastMultipole()
{
	for (int i = 0; i < tree_depth; ++i)
	{
		cudaFree(&outer_expansions[i]);
		cudaFree(&inner_expansions[i]);
	}
	cudaFree(&dev_forces);
	cudaFree(&dev_potentials);
}

} // fmm