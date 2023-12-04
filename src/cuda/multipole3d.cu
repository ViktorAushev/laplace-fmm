#include "multipole3d.h"
#include "omp.h"
#include "../common/special_functions.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

namespace fmm::gpu {

const int MAX_MULTIPOLE_NUM = fmm::detail::_3d_MAX_MULTIPOLE_NUM;
const int MAX_MULTIPOLE_NUMx2 = MAX_MULTIPOLE_NUM * (MAX_MULTIPOLE_NUM + 1) / 2;

#ifdef FMM_CONSTEXPR_MATH
__constant__ auto Knm = fmm::detail::Knm.Knm;
__constant__ auto Anm = fmm::detail::Anm.Anm;
__constant__ auto m2lcoef = fmm::detail::m2lcoef.m2lcoef;
#define Knm_arg
#define Anm_arg
#define m2lcoef_arg
#define __Knm__
#define __Anm__
#define __m2lcoef__
#else
#define Knm_arg , double* Knm
#define Anm_arg , double* Anm
#define m2lcoef_arg , double* m2lcoef
#define __Knm__ , fmm::detail::dev_Knm
#define __Anm__ , fmm::detail::dev_Anm
#define __m2lcoef__ , fmm::detail::dev_m2lcoef
#endif

template <typename TreeCell_t, typename particle_t, typename complex_value>
__global__ void multipole(TreeCell_t* leaves, particle_t* particles, complex_value* outer, int N, int Nx2 Knm_arg)
{
	size_t leaf_idx = blockIdx.x;
	const auto [s, e] = leaves[leaf_idx].source_range;
	const int particle_num = e - s;
	const int particles_per_thread = (particle_num + blockDim.x - 1) / blockDim.x;

	const Vector3d center = leaves[leaf_idx].center;
	complex_value local_outer[MAX_MULTIPOLE_NUMx2];
	double Pnm[MAX_MULTIPOLE_NUMx2];
	cuda_complex eim[MAX_MULTIPOLE_NUM];
	Vector3d rho;
	double r, rn, ss, cc;
	for (int i = 0; i < MAX_MULTIPOLE_NUMx2; ++i)
	{
		if constexpr (std::is_same_v<complex_value, cuda_complex>)
			local_outer[i] = 0;
		else
			local_outer[i] = { 0,0,0 };
	}

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			auto local_particle = particles[particle_idx];
			rho = DecToSph(local_particle.center - center);
			rn = r = rho[0];
			ComputePnm(N, rho[1], Pnm);
			for (int m = 0; m < N; ++m)
			{
				sincos(m * rho[2], &ss, &cc);
				eim[m] = cuda_complex(cc, -ss);
			}

			for (int n = 1; n < N; ++n)
			{
				for (int m = 0; m <= n; ++m)
				{
					local_outer[n * (n + 1) / 2 + m] += local_particle.q * rn * Knm[n * (n + 1) / 2 + m] * Pnm[n * (n + 1) / 2 + m] * eim[m];
				}
				rn *= r;
			}
			local_outer[0] += local_particle.q;
		}
	}

	assert(blockDim.x <= 32);
	__shared__ complex_value shared_outer[32];
	for (int i = 0; i < Nx2; ++i)
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
			outer[leaf_idx * Nx2 + i] = shared_outer[0];
	}
}

template <typename complex_value>
__device__ void rotate_z(complex_value* a, double phi, int N)
{
	cuda_complex Exp[MAX_MULTIPOLE_NUM];
	double s, c;
	for (int n = 0; n < N; ++n)
	{
		sincos(n * phi, &s, &c);
		Exp[n] = cuda_complex(c, s);
	}

	int idx;
	for (int n = 0; n < N; ++n)
	{
		idx = n * (n + 1) / 2;
		for (int m = 0; m <= n; ++m)
		{
			a[idx + m] *= Exp[m];
		}
	}
}

template <typename complex_value>
__device__ void rotate_z_plus(const complex_value* a, double phi, complex_value* b, int N)
{
	cuda_complex Exp[MAX_MULTIPOLE_NUM];
	double s, c;
	for (int n = 0; n < N; ++n)
	{
		sincos(n * phi, &s, &c);
		Exp[n] = cuda_complex(c, s);
	}

	int idx;
	for (int n = 0; n < N; ++n)
	{
		idx = n * (n + 1) / 2;
		for (int m = 0; m <= n; ++m)
		{
			b[idx + m] += a[idx + m] * Exp[m];
		}
	}
}

template <typename complex_value>
__device__ void rotate_z_equal(const complex_value* a, double phi, complex_value* b, int N)
{
	cuda_complex Exp[MAX_MULTIPOLE_NUM];
	double s, c;
	for (int n = 0; n < N; ++n)
	{
		sincos(n * phi, &s, &c);
		Exp[n] = cuda_complex(c, s);
	}

	int idx;
	for (int n = 0; n < N; ++n)
	{
		idx = n * (n + 1) / 2;
		for (int m = 0; m <= n; ++m)
		{
			b[idx + m] = a[idx + m] * Exp[m];
		}
	}
}

template <typename complex_value>
__device__ void rotate_y(const complex_value* a, const double* dmatrix, complex_value* res, int N)
{
	int idx0, idx1, idx2;
	complex_value val;
	for (int n = 0; n < N; ++n)
	{
		idx0 = (n * (5 + n * (3 + 4 * n))) / 6;
		idx1 = 2 * n + 1;
		for (int m = 0; m <= n; ++m)
		{
			idx2 = idx0 + idx1 * m;
			if constexpr (std::is_same_v<complex_value, cuda_complex>)
				val = 0.0;
			else
				val = { 0.0, 0.0, 0.0 };
			for (int k = 1; k <= n; ++k)
			{
				const auto& z1 = dmatrix[idx2 + k];
				const auto& z2 = dmatrix[idx2 - k];
				const auto& w = a[n * (n + 1) / 2 + k];
				if constexpr (std::is_same_v<complex_value, cuda_complex>)
				{
					const double& wre = w.real();
					const double& wim = w.imag();
					val += cuda_complex(wre * (z1 + z2), wim * (z1 - z2));
				}
				else
				{
					for (int s = 0; s < 3; ++s)
					{
						const auto& wre = w[s].real();
						const auto& wim = w[s].imag();
						val[s] += cuda_complex(wre * (z1 + z2), wim * (z1 - z2));
					}
				}
			}
			val += dmatrix[idx2] * a[n * (n + 1) / 2];
			res[n * (n + 1) / 2 + m] = val;
		}
	}
}

__device__ int binary_search(int begin, int end, int* vec, int key)
{
	int middle = (begin + end) / 2;
	for (;;)
	{
		auto cmp = (key <=> vec[middle]);
		if (std::is_lt(cmp)) {
			end = middle;
		}
		else {
			if (std::is_gt(cmp))
				begin = middle;
			else
				break;
		}
		middle = (begin + end) / 2;
	}
	return middle;
}

template <typename TreeCell_t, typename complex_value>
__global__ void M2M(TreeCell_t* top_level, size_t top_level_size, TreeCell_t* bottom_level, size_t bottom_level_size, complex_value* top_outer, complex_value* bottom_outer,
	double* dmatrix, size_t dm_size, size_t dm_count, int* dm_map, int N, int Nx2 Anm_arg)
{
	size_t idx = blockIdx.x;
	const auto& cell = top_level[idx];
	const auto [s, e] = cell.source_range;
	const Vector3d center = cell.center;

	complex_value work1[MAX_MULTIPOLE_NUMx2], work2[MAX_MULTIPOLE_NUMx2];
	complex_value other_outer[MAX_MULTIPOLE_NUMx2];
	double rn[MAX_MULTIPOLE_NUM];

	size_t k = s + threadIdx.x;
	if (k < e)
	{
		const auto& child_cell = bottom_level[k];
		auto rho = DecToSph(child_cell.center - center);
		const auto& theta = rho[1];
		const auto& phi = rho[2];
		for (int j = 0; j < Nx2; ++j)
		{
			other_outer[j] = bottom_outer[k * Nx2 + j];
		}
			
		int key1{ doublehash(theta) };
		int shift1 = binary_search(0, dm_count, dm_map, key1);

		rotate_z_equal(other_outer, phi, work2, N);
		rotate_y(work2, dmatrix + shift1 * dm_size, work1, N);

		for (int j = 0; j < Nx2; ++j)
		{
			if constexpr (std::is_same_v<complex_value, cuda_complex>)
				work2[j] = 0;
			else
				work2[j] = { 0, 0, 0 };
		}

		rn[0] = 1.0;
		for (int i = 1; i < N; ++i)
			rn[i] = rn[i - 1] * rho[0];

		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k <= j; ++k)
			{
				for (int n = 0; j - n >= k; ++n)
				{
					int j_minus_n = j - n;
					work2[j * (j + 1) / 2 + k] += Anm[n * (n + 1) / 2]* Anm[j_minus_n * (j_minus_n + 1) / 2 + k] * rn[n]
						/ Anm[j * (j + 1) / 2 + k] * work1[j_minus_n * (j_minus_n + 1) / 2 + k];
				}
			}
		}

		int key2{ doublehash(-theta) };
		int shift2 = binary_search(0, dm_count, dm_map, key2);

		rotate_y(work2, dmatrix + shift2 * dm_size, work1, N);
		rotate_z(work1, -phi, N);
	}

	auto outer = top_outer + idx * Nx2;

	__shared__ complex_value shared_outer[8];
	for (int i = 0; i < Nx2; ++i)
	{
		shared_outer[threadIdx.x] = work1[i];
		__syncthreads();
		for (int s = blockDim.x / 2; s >= 1; s >>= 1)
		{
			if (threadIdx.x < s)
				shared_outer[threadIdx.x] += shared_outer[s + threadIdx.x];
			__syncthreads();
		}

		if (threadIdx.x == 0)
			outer[i] = shared_outer[0];
	}
}

template <typename TreeCell_t, typename complex_value>
__global__ void M2L(TreeCell_t* level, size_t level_size, complex_value* level_inner, complex_value* level_outer, size_t* farneighbours, size_t* farneighbours_sizes,
	double* dmatrix, size_t dm_size, size_t dm_count, int* dm_map, int N, int Nx2 m2lcoef_arg)
{
	size_t idx = blockIdx.x;

	const TreeCell_t& cell = level[idx];
	const Vector3d center = cell.center;
	complex_value work1[MAX_MULTIPOLE_NUMx2], work2[MAX_MULTIPOLE_NUMx2];
	complex_value other_outer[MAX_MULTIPOLE_NUMx2];
	double rn[2 * MAX_MULTIPOLE_NUM + 1];

	if (threadIdx.x < farneighbours_sizes[idx])
	{
		size_t nbr_idx = farneighbours[idx * 189 + threadIdx.x];
		const TreeCell_t& nbr_cell = level[nbr_idx];
		auto rho = DecToSph(nbr_cell.center - center);
		const auto& theta = rho[1];
		const auto& phi = rho[2];
		for (int j = 0; j < Nx2; ++j)
		{
			other_outer[j] = level_outer[nbr_idx * Nx2 + j];
		}

		rn[0] = 1.0;
		for (int i = 1; i < 2 * N + 1; ++i)
			rn[i] = rn[i - 1] / rho[0];
						
		int key1{ doublehash(theta) };
		int shift1 = binary_search(0, dm_count, dm_map, key1);

		rotate_z_equal(other_outer, phi, work2, N);
		rotate_y(work2, dmatrix + shift1 * dm_size, work1, N);

		for (int j = 0; j < Nx2; ++j)
		{
			if constexpr (std::is_same_v<complex_value, cuda_complex>)
				work2[j] = 0;
			else
				work2[j] = { 0, 0, 0 };
		}

		int idx1, idx2, idx3;
		for (int j = 0; j < N; ++j)
		{
			idx1 = (j + 2) * (j + 1) / 2;
			for (int k = 0; k <= j; ++k)
			{
				idx2 = MAX_MULTIPOLE_NUM * (idx1 + k + 1) + 1;
				for (int n = k; n < N; ++n)
				{
					idx3 = idx2 + n;
					work2[j * (j + 1) / 2 + k] += m2lcoef[idx3] * rn[j + n + 1] * work1[n * (n + 1) / 2 + k];
				}
			}
		}

		int key2{ doublehash(-theta) };
		int shift2 = binary_search(0, dm_count, dm_map, key2);

		rotate_y(work2, dmatrix + shift2 * dm_size, work1, N);
		rotate_z(work1, -phi, N);
	}

	__shared__ complex_value shared_inner[256];
	for (int i = 0; i < Nx2; ++i)
	{
		shared_inner[threadIdx.x] = work1[i];
		__syncthreads();
		for (int s = blockDim.x / 2; s >= 1; s >>= 1)
		{
			if (threadIdx.x < s)
				shared_inner[threadIdx.x] += shared_inner[s + threadIdx.x];
			__syncthreads();
		}

		if (threadIdx.x == 0)
			level_inner[idx * Nx2 + i] += shared_inner[0];
	}
}

template <typename TreeCell_t, typename complex_value>
__global__ void L2L(TreeCell_t* top_level, size_t top_level_size, TreeCell_t* bottom_level, size_t bottom_level_size, complex_value* top_inner, complex_value* bottom_inner,
	double* dmatrix, size_t dm_size, size_t dm_count, int* dm_map, int N, int Nx2 Anm_arg)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < bottom_level_size)
	{
		const TreeCell_t& cell = bottom_level[idx];
		const TreeCell_t& parent_cell = top_level[cell.parent];

		auto rho = DecToSph(parent_cell.center - cell.center);
		const auto& theta = rho[1];
		const auto& phi = rho[2];

		complex_value work1[MAX_MULTIPOLE_NUMx2], work2[MAX_MULTIPOLE_NUMx2];
		complex_value other_inner[MAX_MULTIPOLE_NUMx2];
		complex_value inner[MAX_MULTIPOLE_NUMx2];
		double rn[MAX_MULTIPOLE_NUM];

		for (int j = 0; j < Nx2; ++j)
		{
			other_inner[j] = top_inner[cell.parent * Nx2 + j];
		}

		int key1{ doublehash(theta) };
		int shift1 = binary_search(0, dm_count, dm_map, key1);

		rotate_z_equal(other_inner, phi, work2, N);
		rotate_y(work2, dmatrix + shift1 * dm_size, work1, N);

		for (int j = 0; j < Nx2; ++j)
		{
			if constexpr (std::is_same_v<complex_value, cuda_complex>)
				work2[j] = 0;
			else
				work2[j] = { 0, 0, 0 };
		}

		rn[0] = 1;
		for (int i = 1; i < N; ++i)
			rn[i] = rn[i - 1] * rho[0];

		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k <= j; ++k)
			{
				for (int n = j; n < N; ++n)
				{
					work2[j * (j + 1) / 2 + k] += Anm[(n - j) * (n - j + 1) / 2] * Anm[j * (j + 1) / 2 + k] * rn[n - j] * ni(n + j) / Anm[n * (n + 1) / 2 + k] * work1[n * (n + 1) / 2 + k];
				}
			}
		}

		int key2{ doublehash(-theta) };
		int shift2 = binary_search(0, dm_count, dm_map, key2);

		rotate_y(work2, dmatrix + shift2 * dm_size, inner, N);
		rotate_z_plus(inner, -phi, bottom_inner + idx * Nx2, N);
	}
}

template <typename TreeCell_t, decltype(&TreeCell_t::source_range) range_ptr>
struct cell2int {
	__device__ __host__ int operator()(const TreeCell_t& cell) {
		return (cell.*range_ptr).second - (cell.*range_ptr).first;
	}
};

template <typename value>
void FastMultipole3d<value>::Upward()
{
	double t1 = omp_get_wtime();

	auto& leaves = tree->dev_levels.back();
	auto& leaves_outer = outer_expansions.back();
	const auto& particles = tree->dev_particles;


	const size_t num_leaves = tree->level_sizes.back();
	multipole<<<num_leaves, 32>>>(leaves, particles, leaves_outer, N, Nx2 __Knm__);
	cudaDeviceSynchronize();

	std::cout << "multipole time: " << omp_get_wtime() - t1 << std::endl;

	t1 = omp_get_wtime();

	const size_t dm_count = fmm::detail::dmatrix.size();
	const size_t dm_size = fmm::detail::dmatrix.begin()->second.size();

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = tree->dev_levels[i];
		auto& top_outer = outer_expansions[i];
		const auto& bottom_level = tree->dev_levels[i + 1];
		const auto& bottom_outer = outer_expansions[i + 1];

		//const size_t num_blocks = (tree->level_sizes[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
		M2M<<<tree->level_sizes[i], 8>>>(top_level, tree->level_sizes[i], bottom_level, tree->level_sizes[i + 1], top_outer, bottom_outer,
			fmm::detail::dev_dmatrix, dm_size, dm_count, fmm::detail::dev_dm_map, N, Nx2 __Anm__);
		cudaDeviceSynchronize();
	}

	std::cout << "m2m time: " << omp_get_wtime() - t1 << std::endl;
}

template <typename value>
void FastMultipole3d<value>::Downward()
{
	const size_t dm_count = fmm::detail::dmatrix.size();
	const size_t dm_size = fmm::detail::dmatrix.begin()->second.size();

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
		L2L << <num_blocks, BLOCK_SIZE >> > (top_level, tree->level_sizes[i - 1], bottom_level, tree->level_sizes[i], top_inner, bottom_inner, 
			fmm::detail::dev_dmatrix, dm_size, dm_count, fmm::detail::dev_dm_map, N, Nx2 __Anm__);
		cudaDeviceSynchronize();
		M2L << <tree->level_sizes[i], 256>> > (bottom_level, tree->level_sizes[i], bottom_inner, bottom_outer, neighbours, nbr_sizes,
			fmm::detail::dev_dmatrix, dm_size, dm_count, fmm::detail::dev_dm_map, N, Nx2 __m2lcoef__);
		cudaDeviceSynchronize();
	}
}

template <bool USE_SOURCE, typename TreeCell_t, typename particle_t, typename complex_value>
__global__ void compute_leaves(TreeCell_t* leaves, size_t num_leaves, particle_t* particles, complex_value* leaves_inner, size_t* close_neighbours, size_t* close_neighbours_sizes, Vector3d* dev_forces, double* dev_potentials, int N, int Nx2 Knm_arg)

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

	__shared__ complex_value inner[MAX_MULTIPOLE_NUMx2];
	if (blockDim.x < Nx2)
	{
		if (threadIdx.x == 0)
			for (int i = 0; i < Nx2; ++i)
				inner[i] = leaves_inner[leaf_idx * Nx2 + i];
	}
	else
	{
		if (threadIdx.x < Nx2)
			inner[threadIdx.x] = leaves_inner[leaf_idx * Nx2 + threadIdx.x];
	}
	__syncthreads();

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			Vector3d force{ 0.0, 0.0, 0.0 };
			double potential = 0.0;
			const particle_t p1 = particles[particle_idx];

			for (size_t k = 0; k < close_neighbours_sizes[leaf_idx]; ++k)
			{
				size_t nbr_idx = close_neighbours[leaf_idx * 27 + k];
				const auto [ns, ne] = leaves[nbr_idx].source_range;
				for (size_t j = ns; j < ne; ++j) {
					force += Force3d(p1, particles[j]);
					potential += Potential3d(p1, particles[j]);
				}
			}

			auto dr = DecToSph(p1.center - cell.center);
			double Pnm[(MAX_MULTIPOLE_NUM + 1) * (MAX_MULTIPOLE_NUM + 2) / 2];
			cuda_complex eim[MAX_MULTIPOLE_NUM + 1];
			ComputePnm(N + 1, dr[1], Pnm);
			double x, y;
			for (int m = 0; m <= N; ++m)
			{
				sincos(m * dr[2], &y, &x);
				eim[m] = cuda_complex(x, y);
			}
			double rn = 1.0 / dr[0];
			sincos(dr[1], &y, &x);
			cuda_complex imag_one(0, 1), coef, Ynm, SphDTheta, SphDPhi;

			Vector3d temp{ 0,0,0 };
			for (int n = 0; n < N; ++n)
			{
				for (int m = 0; m <= n; ++m)
				{
					coef = inner[n * (n + 1) / 2 + m] * rn;
					Ynm = Pnm[n * (n + 1) / 2 + m] * eim[m] * Knm[n * (n + 1) / 2 + m];

					potential += (dr[0] * Ynm * coef).real();
					if (m != 0)
						potential += (dr[0] * Ynm * coef).real();

					SphDTheta = Knm[n * (n + 1) / 2 + m] * eim[m] * ((-1 - n) * x * Pnm[n * (n + 1) / 2 + m] + (1 - m + n) * Pnm[(n + 1) * (n + 2) / 2 + m]) / y;
					SphDPhi = double(m) * imag_one * Ynm;

					temp[0] -= (double(n) * coef * Ynm).real();
					temp[1] -= (coef * SphDTheta).real();
					temp[2] -= (coef * SphDPhi).real();
					if (m != 0)
					{
						temp[0] -= (double(n) * coef * Ynm).real();
						temp[1] -= (coef * SphDTheta).real();
						temp[2] -= (coef * SphDPhi).real();
					}

				}
				rn *= dr[0];
			}
			double s1 = y;
			double c1 = x;
			double s2, c2;
			sincos(dr[2], &s2, &c2);
			force[0] += temp[0] * s1 * c2 + temp[1] * c1 * c2 - temp[2] * s2 / s1;
			force[1] += temp[0] * s1 * s2 + temp[1] * c1 * s2 + temp[2] * c2 / s1;
			force[2] += temp[0] * c1 - temp[1] * s1;
			dev_forces[particle_idx] += force;
			dev_potentials[particle_idx] += potential;
		}
	}
}

template <bool USE_SOURCE, typename TreeCell_t, typename particle_t, typename complex_value>
__global__ void Compute_Forces_p2p(TreeCell_t* leaves, size_t num_leaves, particle_t* particles, complex_value* leaves_inner, size_t* close_neighbours, size_t* close_neighbours_sizes, Vector3d* dev_forces, int N, int Nx2)
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

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			Vector3d force{ 0.0, 0.0, 0.0 };
			const particle_t p1 = particles[particle_idx];

			for (size_t k = 0; k < close_neighbours_sizes[leaf_idx]; ++k)
			{
				size_t nbr_idx = close_neighbours[leaf_idx * 27 + k];
				const auto [ns, ne] = leaves[nbr_idx].source_range;
				for (size_t j = ns; j < ne; ++j) {
					force += Force3d(p1, particles[j]);
				}
			}
			dev_forces[particle_idx] += force;
		}
	}
}

template <bool USE_SOURCE, typename TreeCell_t, typename particle_t, typename complex_value>
__global__ void Compute_Forces_l2p(TreeCell_t* leaves, size_t num_leaves, particle_t* particles, complex_value* leaves_inner, size_t* close_neighbours, size_t* close_neighbours_sizes, Vector3d* dev_forces, int N, int Nx2, int num Knm_arg)
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

	__shared__ complex_value inner[MAX_MULTIPOLE_NUMx2];
	if (blockDim.x < Nx2)
	{
		if (threadIdx.x == 0)
			for (int i = 0; i < Nx2; ++i)
				inner[i] = leaves_inner[leaf_idx * Nx2 + i];
	}
	else
	{
		if (threadIdx.x < Nx2)
			inner[threadIdx.x] = leaves_inner[leaf_idx * Nx2 + threadIdx.x];
	}
	__syncthreads();

	for (int local_part = 0; local_part < particles_per_thread; ++local_part)
	{
		size_t particle_idx = s + local_part * blockDim.x + threadIdx.x;

		if (particle_idx < e)
		{
			Vector3d force{ 0.0, 0.0, 0.0 };
			const particle_t p1 = particles[particle_idx];

			auto dr = DecToSph(p1.center - cell.center);
			double Pnm[(MAX_MULTIPOLE_NUM + 1) * (MAX_MULTIPOLE_NUM + 2) / 2];
			cuda_complex eim[MAX_MULTIPOLE_NUM + 1];
			ComputePnm(N + 1, dr[1], Pnm);
			double x, y;
			for (int m = 0; m <= N; ++m)
			{
				sincos(m * dr[2], &y, &x);
				eim[m] = cuda_complex(x, y);
			}
			double rn = 1;
			sincos(dr[1], &y, &x);
			cuda_complex imag_one(0, 1);

			Vector3d temp{ 0,0,0 };
			for (int n = 1; n < N; ++n)
			{
				for (int m = 0; m <= n; ++m)
				{
					auto coef = inner[n * (n + 1) / 2 + m][num] * rn;
					const auto& Ynm = Pnm[n * (n + 1) / 2 + m] * eim[m] * Knm[n * (n + 1) / 2 + m];
					auto SphDTheta = Knm[n * (n + 1) / 2 + m] * eim[m] * ((-1 - n) * x * Pnm[n * (n + 1) / 2 + m] + (1 - m + n) * Pnm[(n + 1) * (n + 2) / 2 + m]) / y;
					auto SphDPhi = double(m) * imag_one * Ynm;

					temp[0] -= (double(n) * coef * Ynm).real();
					temp[1] -= (coef * SphDTheta).real();
					temp[2] -= (coef * SphDPhi).real();
					if (m != 0)
					{
						temp[0] -= (double(n) * coef * Ynm).real();
						temp[1] -= (coef * SphDTheta).real();
						temp[2] -= (coef * SphDPhi).real();
					}

				}
				rn *= dr[0];
			}
			double s1 = y;
			double c1 = x;
			double s2, c2;
			sincos(dr[2], &s2, &c2);
			force[0] += temp[0] * s1 * c2 + temp[1] * c1 * c2 - temp[2] * s2 / s1;
			force[1] += temp[0] * s1 * s2 + temp[1] * c1 * s2 + temp[2] * c2 / s1;
			force[2] += temp[0] * c1 - temp[1] * s1;
			dev_forces[particle_idx] += force;
		}
	}
}

__global__ void forces_cross(Vector3d* f1, Vector3d* f2, Vector3d* f3, Vector3d* dev_forces, size_t num_particles)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_particles)
	{
		Vector3d force{ 0,0,0 };
		force[0] = f2[idx][2] - f3[idx][1];
		force[1] = f3[idx][0] - f1[idx][2];
		force[2] = f1[idx][1] - f2[idx][0];
		dev_forces[idx] += force;
	}
}

template <typename value>
void FastMultipole3d<value>::ComputeLeaves()
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
		cell2int<TreeCell_t, &TreeCell_t::source_range>(), 0, thrust::maximum<int>());
	else
		max_particles = thrust::transform_reduce(thrust::device, tree->dev_levels.back(), tree->dev_levels.back() + tree->level_sizes.back(),
		cell2int<TreeCell_t, &TreeCell_t::target_range>(), 0, thrust::maximum<int>());
	std::cout << "max particles in cell = " << max_particles << std::endl;
	const size_t num_blocks = num_leaves;
	const size_t num_threads = std::min(512, (max_particles + 32 - 1) / 32 * 32);
	if constexpr (std::is_same_v<value, double>)
	{
		if (tree->targets_num == 0) {
			//Compute_Potentials<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_potentials, N, Nx2 __Knm__);
			compute_leaves<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_forces, dev_potentials, N, Nx2 __Knm__);
		}
		else {
			//Compute_Potentials<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_potentials, N, Nx2 __Knm__);
			compute_leaves<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_forces, dev_potentials, N, Nx2 __Knm__);
		}
	}
	else
	{
		Vector3d* temp_forces1, *temp_forces2, *temp_forces3;
		cudaMalloc(&temp_forces1, num_particles * sizeof(Vector3d));
		cudaMalloc(&temp_forces2, num_particles * sizeof(Vector3d));
		cudaMalloc(&temp_forces3, num_particles * sizeof(Vector3d));

		if (tree->targets_num == 0)
		{
			Compute_Forces_p2p<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_forces, N, Nx2);
			Compute_Forces_l2p<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces1, N, Nx2, 0 __Knm__);
			Compute_Forces_l2p<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces2, N, Nx2, 1 __Knm__);
			Compute_Forces_l2p<true> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces3, N, Nx2, 2 __Knm__);
			cudaDeviceSynchronize();
			forces_cross << < (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (temp_forces1, temp_forces2, temp_forces3, dev_forces, num_particles);
		}
		else
		{
			Compute_Forces_p2p<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, dev_forces, N, Nx2);
			cudaDeviceSynchronize();
			Compute_Forces_l2p<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces1, N, Nx2, 0 __Knm__);
			Compute_Forces_l2p<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces2, N, Nx2, 1 __Knm__);
			Compute_Forces_l2p<false> << <num_blocks, num_threads >> > (leaves, num_leaves, particles, leaves_inner, neighbours, nbr_sizes, temp_forces3, N, Nx2, 2 __Knm__);
			cudaDeviceSynchronize();
			forces_cross << < (num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> > (temp_forces1, temp_forces2, temp_forces3, dev_forces, num_particles);
		}

		cudaFree(&temp_forces1);
		cudaFree(&temp_forces2);
		cudaFree(&temp_forces3);
	}
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
	map_sort << <num_blocks, BLOCK_SIZE >> > (dev_values, dev_buf, dev_positions_map, num_particles);
	cudaDeviceSynchronize();
	cudaMemcpy(values.data(), dev_buf, num_particles * sizeof(T), cudaMemcpyDeviceToHost);
	cudaFree(&dev_buf);
}

template <typename value>
FastMultipole3d<value>::FastMultipole3d(const std::vector<fmm::particle<point3d, value>>& particles, double eps, int N_, int tree_depth_)
{
	targets_num = particles.size();
	Solve(particles, eps, N_, tree_depth_);
}

template <typename value>
FastMultipole3d<value>::FastMultipole3d(const std::vector<fmm::particle<point3d, value>>& source_particles, const std::vector<fmm::particle<point3d, value>>& target_particles, double eps, int N_, int tree_depth_)
{
	std::vector<fmm::particle<point3d, value>> particles(source_particles.size() + target_particles.size());
	memcpy(particles.data(), target_particles.data(), target_particles.size() * sizeof(fmm::particle<point3d, value>));
	memcpy(particles.data() + target_particles.size(), source_particles.data(), source_particles.size() * sizeof(fmm::particle<point3d, value>));
	targets_num = target_particles.size();
	Solve(particles, eps, N_, tree_depth_);
}

template <typename value>
void FastMultipole3d<value>::Solve(const std::vector<fmm::particle<point3d, value>>& particles, double eps, int N_, int tree_depth_)
{
	double T = omp_get_wtime();
	std::cout << "\n***************** Start FMM **************" << std::endl;
	num_particles = particles.size();
	eps = std::max(eps, 1.e-11);
	if (N_ == FMM_AUTO)
		N = std::min(fmm::detail::_3d_MAX_MULTIPOLE_NUM, int(54.2 - 10.73 * sqrt(log(2.4 * eps) + 25.5)));
	else
		N = std::min(fmm::detail::_3d_MAX_MULTIPOLE_NUM, N_);
	Nx2 = N * (N + 1) / 2;
	std::cout << "multipole num = " << N << std::endl;
	if (tree_depth_ == FMM_AUTO)
		tree_depth = std::max(2.0, 1 + log(double(num_particles) / 100.0) / log(8.0));
	else
		tree_depth = tree_depth_;
	std::cout << "tree depth = " << tree_depth << std::endl;

	tree = std::make_shared<MortonTree_t>(particles, tree_depth);

	double t = omp_get_wtime();
	outer_expansions.resize(tree_depth);
	inner_expansions.resize(tree_depth);
	for (int i = 0; i < tree_depth; ++i)
	{
		cudaMalloc(&outer_expansions[i], tree->level_sizes[i] * Nx2 * sizeof(complex_value));
		cudaMalloc(&inner_expansions[i], tree->level_sizes[i] * Nx2 * sizeof(complex_value));
	}
	cudaMalloc(&dev_forces, num_particles * sizeof(Vector3d));
	forces.resize(targets_num);
	if constexpr (std::is_same_v<value, double>)
	{
		cudaMalloc(&dev_potentials, num_particles * sizeof(double));
		potentials.resize(targets_num);
	}
	std::cout << "allocation time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	std::cout << "prepare time: " << omp_get_wtime() - t << std::endl;

	Upward();

	t = omp_get_wtime();
	Downward();
	std::cout << "m2l+l2l time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	ComputeLeaves();
	Map_Sort(dev_forces, tree->dev_positions_map, forces, targets_num);
	if constexpr (std::is_same_v<value, double>)
		Map_Sort(dev_potentials, tree->dev_positions_map, potentials, targets_num);

	std::cout << "leaf time: " << omp_get_wtime() - t << std::endl;

	std::cout << "total fmm time: " << omp_get_wtime() - T << std::endl;
	std::cout << "***************** End FMM **************\n" << std::endl;
}

template <typename value>
FastMultipole3d<value>::~FastMultipole3d()
{
	for (int i = 0; i < tree_depth; ++i)
	{
		cudaFree(&outer_expansions[i]);
		cudaFree(&inner_expansions[i]);
	}
	cudaFree(&dev_forces);
	cudaFree(&dev_potentials);
}

template class FastMultipole3d<double>;
template class FastMultipole3d<Vector3d>;

} // fmm::gpu