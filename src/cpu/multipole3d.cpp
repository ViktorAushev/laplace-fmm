#include "multipole3d.h"

#include <algorithm>
#include <numeric>
#include "omp.h"
#include <iostream>
#include "../common/simple_math.h"
#include "../common/avx.h"
#include "../common/special_functions.h"
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <cmath>
#include "../common/mpi_utils.h"

namespace fmm {

using namespace detail;

template <typename value>
void FastMultipole3d<value>::Multipole(complex_value* a, const std::pair<size_t, size_t>& particle_range, const Vector3d& p0, double* Pnm, std::complex<double>* eim)
{
	auto [p_begin, p_end] = particle_range;
	Vector3d rho;
	double r, rn, mphi;

	for (int i = p_begin; i < p_end; ++i)
	{
		auto& particle = tree->particles[i];
		rho = DecToSph(particle.center - p0);
		rn = r = rho[0];
		ComputePnm(N, rho[1], Pnm);
		for (int m = 0; m < N; ++m)
		{
			mphi = m * rho[2];
			eim[m] = std::complex<double>(cos(mphi), -sin(mphi));
		}
		
		for (int n = 1; n < N; ++n)
		{
			for (int m = 0; m <= n; ++m)
			{
				a[n * (n + 1) / 2 + m] += particle.q * rn * Knm(n, m) * Pnm[n * (n + 1)/ 2 + m]* eim[m];
			}
			rn *= r;
		}
		a[0] += particle.q;
	}	
}

template <typename value>
void FastMultipole3d<value>::RotateZ(complex_value* a, const std::vector<std::complex<double>>& rotation_exp)
{
	int idx;
	for (int n = 0; n < N; ++n)
	{
		idx = n * (n + 1) / 2;
		for (int m = 0; m <= n; ++m)
		{
			a[idx + m] *= rotation_exp[m];
		}
	}
}

template <typename value>
void FastMultipole3d<value>::RotateZ(const complex_value* a, const std::vector<std::complex<double>>& rotation_exp, complex_value* b)
{
	int idx;
	for (int n = 0; n < N; ++n)
	{
		idx = n * (n + 1) / 2;
		for (int m = 0; m <= n; ++m)
		{
			b[idx + m] += a[idx + m] * rotation_exp[m];
		}
	}
}

template <typename value>
void FastMultipole3d<value>::RotateY(const complex_value* a, const std::vector<double>& dmatrix, complex_value* res)
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
			if constexpr (std::is_same_v<value, double>)
				val = 0.0;
			else
				val = { 0,0,0 };
			for (int k = 1; k <= n; ++k)
			{
				const auto z1 = dmatrix[idx2 + k];
				const auto z2 = dmatrix[idx2 - k];				
				const auto w = a[n * (n + 1) / 2 + k];
				if constexpr (std::is_same_v<value, double>)
				{
					val.real(w.real() * (z1 + z2) + val.real());
					val.imag(w.imag() * (z1 - z2) + val.imag());
				}
				else
					for (int s = 0; s < 3; ++s)
						val[s] += std::complex<double>(w[s].real() * (z1 + z2), w[s].imag() * (z1 - z2));
			}
			val += dmatrix[idx2] * a[n * (n + 1) / 2];
			res[n * (n + 1) / 2 + m] = val;
		}
	}
}

template <typename value>
void FastMultipole3d<value>::M2M(const complex_value* a, const Vector3d& p0, complex_value* b, std::vector<complex_value>& work1, std::vector<complex_value>& work2)
{
	auto rho = DecToSph(p0);
	const auto& theta = rho[1];
	const auto& phi = rho[2];

	const auto& dm1 = dmatrix.at(doublehash(theta));
	const auto& dm2 = dmatrix.at(doublehash(-theta));
	const auto& exp1 = rotation_exponents.at(doublehash(phi));
	const auto& exp2 = rotation_exponents.at(doublehash(-phi));

	memset(work2.data(), 0, work2.size() * sizeof(complex_value));
	RotateZ(a, exp1, work2.data());
	RotateY(work2.data(), dm1, work1.data());
	memset(work2.data(), 0, work2.size() * sizeof(complex_value));

	std::vector<double> rn(N, 1);
	for (int i = 1; i < N; ++i)
		rn[i] = rn[i - 1] * rho[0];

	for (int j = 0; j < N; ++j)
	{
		for (int k = 0; k <= j; ++k)
		{
			for (int n = 0; j - n >= k; ++n)
			{
				int j_minus_n = j - n;
				work2[j * (j + 1) / 2 + k] += Anm(n, 0) * Anm(j_minus_n, k) * rn[n] / Anm(j, k) * work1[j_minus_n * (j_minus_n + 1) / 2 + k];
			}
		}
	}

	RotateY(work2.data(), dm2, work1.data());
	RotateZ(work1.data(), exp2, b);
}

template <typename value>
void FastMultipole3d<value>::M2L(const complex_value* a, const Vector3d& p0, complex_value* b, std::vector<complex_value>& work1, std::vector<complex_value>& work2)
{
	auto rho = DecToSph(p0);
	const auto& theta = rho[1];
	const auto& phi = rho[2];

	const auto& dm1 = dmatrix.at(doublehash(theta));
	const auto& dm2 = dmatrix.at(doublehash(-theta));
	const auto& exp1 = rotation_exponents.at(doublehash(phi));
	const auto& exp2 = rotation_exponents.at(doublehash(-phi));

	memset(work2.data(), 0, work2.size() * sizeof(complex_value));
	RotateZ(a, exp1, work2.data());
	RotateY(work2.data(), dm1, work1.data());
	memset(work2.data(), 0, work2.size() * sizeof(complex_value));

	std::vector<double> rn(2 * N + 1, 1);
	for (int i = 1; i < 2 * N + 1; ++i)
		rn[i] = rn[i - 1] / rho[0];

	int idx1, idx2, idx3;
	for (int j = 0; j < N; ++j)
	{
		idx1 = (j + 2) * (j + 1) / 2;
		for (int k = 0; k <= j; ++k)
		{
			idx2 = detail::_3d_MAX_MULTIPOLE_NUM * (idx1 + k + 1) + 1;
			for (int n = k; n < N; ++n)
			{
				idx3 = idx2 + n;
				work2[j * (j + 1) / 2 + k] += (m2lcoef.m2lcoef[idx3] * rn[j + n + 1]) * work1[n * (n + 1) / 2 + k];
			}
		}
	}

	RotateY(work2.data(), dm2, work1.data());
	RotateZ(work1.data(), exp2, b);
}

template <typename value>
void FastMultipole3d<value>::L2L(const complex_value* a, const Vector3d& p0, complex_value* b, std::vector<complex_value>& work1, std::vector<complex_value>& work2)
{
	auto rho = DecToSph(p0);
	const auto& theta = rho[1];
	const auto& phi = rho[2];

	const auto& dm1 = dmatrix.at(doublehash(theta));
	const auto& dm2 = dmatrix.at(doublehash(-theta));
	const auto& exp1 = rotation_exponents.at(doublehash(phi));
	const auto& exp2 = rotation_exponents.at(doublehash(-phi));

	memset(work2.data(), 0, work2.size() * sizeof(complex_value));
	RotateZ(a, exp1, work2.data());
	RotateY(work2.data(), dm1, work1.data());
	memset(work2.data(), 0, work2.size() * sizeof(complex_value));

	std::vector<double> rn(N, 1);
	for (int i = 1; i < N; ++i)
		rn[i] = rn[i - 1] * rho[0];

	for (int j = 0; j < N; ++j)
	{
		for (int k = 0; k <= j; ++k)
		{
			for (int n = j; n < N; ++n)
			{
				work2[j * (j + 1) / 2 + k] += Anm(n - j, 0) * Anm(j, k) * ni(n + j) * rn[n - j] / Anm(n, k) * work1[n * (n + 1) / 2 + k];
			}
		}
	}

	RotateY(work2.data(), dm2, work1.data());
	RotateZ(work1.data(), exp2, b);
}

template <typename value>
void FastMultipole3d<value>::Upward()
{
	double t1 = omp_get_wtime();

	auto& leaves = tree->levels.back();
	auto& leaves_outer = outer_expansions.back();
#ifdef FMM_MPI
	auto [Begin, End] = LocalPart(0, tree->level_sizes[tree_depth - 1]);
#else
	size_t Begin = 0, End = tree->level_sizes[tree_depth - 1];
#endif
	tbb::enumerable_thread_specific<std::vector<std::complex<double>>> eim_ets((std::vector<std::complex<double>>(N)));
	tbb::enumerable_thread_specific<treevector<double>> Pnm_ets(treevector<double>(N + 1));
	tbb::parallel_for(Begin, End, [&](size_t i) {
		TreeCell_t& cell = leaves[i];
		auto& eim = eim_ets.local();
		auto& Pnm = Pnm_ets.local();
		Multipole(leaves_outer.data() + Nx2 * i, cell.source_range, cell.center, Pnm.data(), eim.data());
		});

#ifdef FMM_MPI
	std::vector<complex_value> buf(leaves_outer.size());
	std::vector<int> sizes(NProc());
	std::vector<int> displs(NProc());
	for (int i = 0; i < NProc(); ++i)
	{
		sizes[i] = Nx2 * LocalPart(0, tree->level_sizes[tree_depth - 1], i);
	}
	for (int i = 1; i < NProc(); ++i)
	{
		displs[i] = displs[i - 1] + sizes[i - 1];
	}

	MPI_Datatype MPI_COMPLEX_VALUE;
	MPI_Type_contiguous(sizeof(complex_value) / sizeof(std::complex<double>), MPI_COMPLEX16, &MPI_COMPLEX_VALUE);
	MPI_Type_commit(&MPI_COMPLEX_VALUE);

	MPI_Allgatherv(leaves_outer.data() + displs[MyID()], sizes[MyID()], MPI_COMPLEX_VALUE,
		buf.data(), sizes.data(), displs.data(), MPI_COMPLEX_VALUE, MPI_COMM_WORLD);
	std::swap(buf, leaves_outer);
#endif

	if (IAmRoot()) std::cout << "multipole time: " << omp_get_wtime() - t1 << std::endl;

	t1 = omp_get_wtime();

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = tree->levels[i];
		auto& top_outer = outer_expansions[i];
		const auto& bottom_level = tree->levels[i + 1];
		const auto& bottom_outer = outer_expansions[i + 1];
#ifdef FMM_MPI
		auto [Begin, End] = LocalPart(0, tree->level_sizes[i]);
#else
		size_t Begin = 0, End = tree->level_sizes[i];
#endif
		tbb::parallel_for(Begin, End, [&](size_t j) {
			TreeCell_t& cell = top_level[j];
			auto& work1 = local_work1.local();
			auto& work2 = local_work2.local();
			const auto& [begin, end] = cell.source_range;

			for (int k = begin; k < end; ++k)
			{
				const auto& children_cell = bottom_level[k];
				M2M(bottom_outer.data() + k * Nx2, children_cell.center - cell.center, top_outer.data() + j * Nx2, work1, work2);
			}
		});

#ifdef FMM_MPI
		{
			std::vector<complex_value> buf(top_outer.size());
			for (int j = 0; j < NProc(); ++j)
			{
				sizes[j] = Nx2 * LocalPart(0, tree->level_sizes[i], j);
			}
			for (int j = 1; j < NProc(); ++j)
			{
				displs[j] = displs[j - 1] + sizes[j - 1];
			}
			MPI_Allgatherv(top_outer.data() + displs[MyID()], sizes[MyID()], MPI_COMPLEX_VALUE,
				buf.data(), sizes.data(), displs.data(), MPI_COMPLEX_VALUE, MPI_COMM_WORLD);
			std::swap(buf, top_outer);
		}
#endif
	}

	if (IAmRoot()) std::cout << "m2m time: " << omp_get_wtime() - t1 << std::endl;
}

template <typename value>
void FastMultipole3d<value>::Downward()
{
#ifdef FMM_MPI
	std::vector<int> sizes(NProc());
	std::vector<int> displs(NProc());
#endif

	for (int i = 2; i < tree_depth; ++i)
	{
		auto& bottom_level = tree->levels[i];
		auto& bottom_inner = inner_expansions[i];
		const auto& bottom_outer = outer_expansions[i];
#ifdef FMM_MPI
		auto [Begin, End] = LocalPart(0, tree->level_sizes[i]);
#else
		size_t Begin = 0, End = tree->level_sizes[i];
#endif
		tbb::parallel_for(Begin, End, [&](size_t j) {
			TreeCell_t& cell = bottom_level[j];
			auto& work1 = local_work1.local();
			auto& work2 = local_work2.local();

			for (const auto& idx : cell.farneighbours)
			{
				const auto& far_neighbour = bottom_level[idx];
				M2L(bottom_outer.data() + idx * Nx2, far_neighbour.center - cell.center, bottom_inner.data() + Nx2 * j, work1, work2);
			}
		});

#ifdef FMM_MPI
		{
			MPI_Datatype MPI_COMPLEX_VALUE;
			MPI_Type_contiguous(sizeof(complex_value) / sizeof(std::complex<double>), MPI_COMPLEX16, &MPI_COMPLEX_VALUE);
			MPI_Type_commit(&MPI_COMPLEX_VALUE);

			std::vector<complex_value> buf(bottom_inner.size());
			for (int j = 0; j < NProc(); ++j)
			{
				sizes[j] = Nx2 * LocalPart(0, tree->level_sizes[i], j);
			}
			for (int j = 1; j < NProc(); ++j)
			{
				displs[j] = displs[j - 1] + sizes[j - 1];
			}
			MPI_Allgatherv(bottom_inner.data() + displs[MyID()], sizes[MyID()], MPI_COMPLEX_VALUE,
				buf.data(), sizes.data(), displs.data(), MPI_COMPLEX_VALUE, MPI_COMM_WORLD);
			std::swap(buf, bottom_inner);
		}
#endif
	}

	for (int i = 3; i < tree_depth; ++i)
	{
		const auto& top_level = tree->levels[i - 1];
		const auto& top_inner = inner_expansions[i - 1];
		auto& bottom_level = tree->levels[i];
		auto& bottom_inner = inner_expansions[i];
#ifdef FMM_MPI
		auto [Begin, End] = LocalPart(0, tree->level_sizes[i]);
#else
		size_t Begin = 0, End = tree->level_sizes[i];
#endif
		tbb::parallel_for(Begin, End, [&](size_t j) {
			TreeCell_t& cell = bottom_level[j];
			auto& work1 = local_work1.local();
			auto& work2 = local_work2.local();
			auto& parent_cell = top_level[cell.parent];
			L2L(top_inner.data() + Nx2 * cell.parent, parent_cell.center - cell.center, bottom_inner.data() + Nx2 * j, work1, work2);
		});

#ifdef FMM_MPI
		{
			MPI_Datatype MPI_COMPLEX_VALUE;
			MPI_Type_contiguous(sizeof(complex_value) / sizeof(std::complex<double>), MPI_COMPLEX16, &MPI_COMPLEX_VALUE);
			MPI_Type_commit(&MPI_COMPLEX_VALUE);

			std::vector<complex_value> buf(bottom_inner.size());
			for (int j = 0; j < NProc(); ++j)
			{
				sizes[j] = Nx2 * LocalPart(0, tree->level_sizes[i], j);
			}
			for (int j = 1; j < NProc(); ++j)
			{
				displs[j] = displs[j - 1] + sizes[j - 1];
			}
			MPI_Allgatherv(bottom_inner.data() + displs[MyID()], sizes[MyID()], MPI_COMPLEX_VALUE,
				buf.data(), sizes.data(), displs.data(), MPI_COMPLEX_VALUE, MPI_COMM_WORLD);
			std::swap(buf, bottom_inner);
		}
#endif
	}
}

template <>
void FastMultipole3d<Vector3d>::ComputeLeaves()
{
	using namespace std::literals;

	tbb::enumerable_thread_specific<Vector3d> tmp_force_ets, dr_ets;
	tbb::enumerable_thread_specific<Matrix3d> matrix_ets;
	tbb::enumerable_thread_specific<std::vector<std::complex<double>>> eim_ets(std::vector<std::complex<double>>(N + 1));
	tbb::enumerable_thread_specific<treevector<double>> Pnm_ets(treevector<double>(N + 1));
	tbb::enumerable_thread_specific<std::vector<Vector3d>> buf_forces_ets((std::vector<Vector3d>(num_particles)));

	const auto& leaves = tree->levels.back();
	const auto& leaves_inner = inner_expansions.back();
	const auto& particles = tree->particles;

#ifdef FMM_MPI
	auto [Begin, End] = LocalPart(0, tree->level_sizes.back());
#else
	size_t Begin = 0, End = tree->level_sizes.back();
#endif

	decltype(&TreeCell_t::source_range) range_ptr;
	tree->targets_num == 0 ? range_ptr = &TreeCell_t::source_range : range_ptr = &TreeCell_t::target_range;

	using simd_vec_t = std::conditional_t<detail::avx_vec_length == 2, __m128d, std::conditional_t<detail::avx_vec_length == 4, __m256d, __m512d>>;
	simd_vec_t oneVec, epsVec;
	avx_set_constant(epsVec, FORCE_EPS2);
	avx_set_constant(oneVec, 1.0);

	std::unique_ptr<size_t[]> shifts(new size_t[tree->level_sizes.back() + 1]);
	shifts[0] = 0;
	for (size_t i = 0; i < tree->level_sizes.back(); ++i)
	{
		auto [a, b] = leaves[i].source_range;
		shifts[i + 1] = shifts[i] + (b - a + detail::avx_vec_length - 1) / detail::avx_vec_length;
	}
	size_t num_vecs = shifts[tree->level_sizes.back()];
	std::unique_ptr<simd_vec_t[]> x_simd(new simd_vec_t[num_vecs]),
								  y_simd(new simd_vec_t[num_vecs]),
								  z_simd(new simd_vec_t[num_vecs]),
								  qx_simd(new simd_vec_t[num_vecs]),
								  qy_simd(new simd_vec_t[num_vecs]),
								  qz_simd(new simd_vec_t[num_vecs]);

	tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i)
		{
			auto [a, b] = leaves[i].source_range;
			auto xptr = x_simd.get() + shifts[i];
			auto yptr = y_simd.get() + shifts[i];
			auto zptr = z_simd.get() + shifts[i];
			auto qxptr = qx_simd.get() + shifts[i];
			auto qyptr = qy_simd.get() + shifts[i];
			auto qzptr = qz_simd.get() + shifts[i];
			for (int idx = a, k = 0; idx < b; idx += detail::avx_vec_length, ++k)
			{
#if defined(AVX128)
				xptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qxptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].q[0], particles[idx].q[0]);
				qyptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].q[1], particles[idx].q[1]);
				qzptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].q[2], particles[idx].q[2]);
#elif defined(AVX256)
				xptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[0],
					idx + 3 > b ? 0 : particles[idx + 2].center[0],
					idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[1],
					idx + 3 > b ? 0 : particles[idx + 2].center[1],
					idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[2],
					idx + 3 > b ? 0 : particles[idx + 2].center[2],
					idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qxptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].q[0],
					idx + 3 > b ? 0 : particles[idx + 2].q[0],
					idx + 2 > b ? 0 : particles[idx + 1].q[0], particles[idx].q[0]);
				qyptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].q[1],
					idx + 3 > b ? 0 : particles[idx + 2].q[1],
					idx + 2 > b ? 0 : particles[idx + 1].q[1], particles[idx].q[1]);
				qzptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].q[2],
					idx + 3 > b ? 0 : particles[idx + 2].q[2],
					idx + 2 > b ? 0 : particles[idx + 1].q[2], particles[idx].q[2]);
#elif defined(AVX512)
				xptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[0],
					idx + 7 > b ? 0 : particles[idx + 6].center[0],
					idx + 6 > b ? 0 : particles[idx + 5].center[0],
					idx + 5 > b ? 0 : particles[idx + 4].center[0],
					idx + 4 > b ? 0 : particles[idx + 3].center[0],
					idx + 3 > b ? 0 : particles[idx + 2].center[0],
					idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[1],
					idx + 7 > b ? 0 : particles[idx + 6].center[1],
					idx + 6 > b ? 0 : particles[idx + 5].center[1],
					idx + 5 > b ? 0 : particles[idx + 4].center[1],
					idx + 4 > b ? 0 : particles[idx + 3].center[1],
					idx + 3 > b ? 0 : particles[idx + 2].center[1],
					idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[2],
					idx + 7 > b ? 0 : particles[idx + 6].center[2],
					idx + 6 > b ? 0 : particles[idx + 5].center[2],
					idx + 5 > b ? 0 : particles[idx + 4].center[2],
					idx + 4 > b ? 0 : particles[idx + 3].center[2],
					idx + 3 > b ? 0 : particles[idx + 2].center[2],
					idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qxptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].q[0],
					idx + 7 > b ? 0 : particles[idx + 6].q[0],
					idx + 6 > b ? 0 : particles[idx + 5].q[0],
					idx + 5 > b ? 0 : particles[idx + 4].q[0],
					idx + 4 > b ? 0 : particles[idx + 3].q[0],
					idx + 3 > b ? 0 : particles[idx + 2].q[0],
					idx + 2 > b ? 0 : particles[idx + 1].q[0], particles[idx].q[0]);
				qyptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].q[1],
					idx + 7 > b ? 0 : particles[idx + 6].q[1],
					idx + 6 > b ? 0 : particles[idx + 5].q[1],
					idx + 5 > b ? 0 : particles[idx + 4].q[1],
					idx + 4 > b ? 0 : particles[idx + 3].q[1],
					idx + 3 > b ? 0 : particles[idx + 2].q[1],
					idx + 2 > b ? 0 : particles[idx + 1].q[1], particles[idx].q[1]);
				qzptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].q[2],
					idx + 7 > b ? 0 : particles[idx + 6].q[2],
					idx + 6 > b ? 0 : particles[idx + 5].q[2],
					idx + 5 > b ? 0 : particles[idx + 4].q[2],
					idx + 4 > b ? 0 : particles[idx + 3].q[2],
					idx + 3 > b ? 0 : particles[idx + 2].q[2],
					idx + 2 > b ? 0 : particles[idx + 1].q[2], particles[idx].q[2]);
#endif
			}
		});

	tbb::enumerable_thread_specific<std::vector<simd_vec_t>> fx_simd_ets((std::vector<simd_vec_t>(num_vecs))),
															 fy_simd_ets((std::vector<simd_vec_t>(num_vecs))),
															 fz_simd_ets((std::vector<simd_vec_t>(num_vecs)));

	tbb::parallel_for(Begin, End, [&](size_t leaf_idx) {
		auto& tmp_force = tmp_force_ets.local();
		auto& dr = dr_ets.local();
		auto& eim = eim_ets.local();
		auto& Pnm = Pnm_ets.local();
		auto& m_temp = matrix_ets.local();
		auto& buf_forces = buf_forces_ets.local();
		auto& fx_simd = fx_simd_ets.local();
		auto& fy_simd = fy_simd_ets.local();
		auto& fz_simd = fz_simd_ets.local();

		const auto& cell = leaves[leaf_idx];
		const auto& [p1_begin, p1_end] = cell.*range_ptr;

		simd_vec_t x_target, y_target, z_target, qx_target, qy_target, qz_target,
				   x_source, y_source, z_source, qx_source, qy_source, qz_source, dx, dy, dz, fx, fy, fz, invdr, invdr2, invdr3;

		auto x_leaf = x_simd.get() + shifts[leaf_idx];
		auto y_leaf = y_simd.get() + shifts[leaf_idx];
		auto z_leaf = z_simd.get() + shifts[leaf_idx];
		auto qx_leaf = qx_simd.get() + shifts[leaf_idx];
		auto qy_leaf = qy_simd.get() + shifts[leaf_idx];
		auto qz_leaf = qz_simd.get() + shifts[leaf_idx];

		auto fx_leaf = fx_simd.data() + shifts[leaf_idx];
		auto fy_leaf = fy_simd.data() + shifts[leaf_idx];
		auto fz_leaf = fz_simd.data() + shifts[leaf_idx];

		for (int j = p1_begin; j < p1_end; ++j)
		{
			auto& particle = particles[j];

			if (range_ptr == &TreeCell_t::target_range)
			{
				avx_set_constant(x_target, particle.center[0]);
				avx_set_constant(y_target, particle.center[1]);
				avx_set_constant(z_target, particle.center[2]);
				avx_set_constant(qx_target, particle.q[0]);
				avx_set_constant(qy_target, particle.q[1]);
				avx_set_constant(qz_target, particle.q[2]);
				fx = avx_zero_vec<simd_vec_t>();
				fy = avx_zero_vec<simd_vec_t>();
				fz = avx_zero_vec<simd_vec_t>();

				{
					const auto& [p2_begin, p2_end] = cell.source_range;

					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_leaf[i];
						y_source = y_leaf[i];
						z_source = z_leaf[i];
						qx_source = qx_leaf[i];
						qy_source = qy_leaf[i];
						qz_source = qz_leaf[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						invdr3 = avx_mul(invdr2, invdr);

						fx = avx_add(avx_mul(avx_sub(avx_mul(dz, qy_source), avx_mul(dy, qz_source)), invdr3), fx);
						fy = avx_add(avx_mul(avx_sub(avx_mul(dx, qz_source), avx_mul(dz, qx_source)), invdr3), fy);
						fz = avx_add(avx_mul(avx_sub(avx_mul(dy, qx_source), avx_mul(dx, qy_source)), invdr3), fz);
					}
				}
				for (const auto& idx : cell.closeneighbours)
				{
					const auto& neighbour_cell = leaves[idx];
					const auto& [p2_begin, p2_end] = neighbour_cell.source_range;

					auto x_nbr = x_simd.get() + shifts[idx];
					auto y_nbr = y_simd.get() + shifts[idx];
					auto z_nbr = z_simd.get() + shifts[idx];
					auto qx_nbr = qx_simd.get() + shifts[idx];
					auto qy_nbr = qy_simd.get() + shifts[idx];
					auto qz_nbr = qz_simd.get() + shifts[idx];

					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_nbr[i];
						y_source = y_nbr[i];
						z_source = z_nbr[i];
						qx_source = qx_nbr[i];
						qy_source = qy_nbr[i];
						qz_source = qz_nbr[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						invdr3 = avx_mul(invdr2, invdr);

						fx = avx_add(avx_mul(avx_sub(avx_mul(dz, qy_source), avx_mul(dy, qz_source)), invdr3), fx);
						fy = avx_add(avx_mul(avx_sub(avx_mul(dx, qz_source), avx_mul(dz, qx_source)), invdr3), fy);
						fz = avx_add(avx_mul(avx_sub(avx_mul(dy, qx_source), avx_mul(dx, qy_source)), invdr3), fz);
					}
				}
			}
			else
			{
				avx_set_constant(x_target, particle.center[0]);
				avx_set_constant(y_target, particle.center[1]);
				avx_set_constant(z_target, particle.center[2]);
				avx_set_constant(qx_target, particle.q[0]);
				avx_set_constant(qy_target, particle.q[1]);
				avx_set_constant(qz_target, particle.q[2]);
				fx = avx_zero_vec<simd_vec_t>();
				fy = avx_zero_vec<simd_vec_t>();
				fz = avx_zero_vec<simd_vec_t>();

				int pletnum = (j - p1_begin) / detail::avx_vec_length;
				{
					x_source = x_leaf[pletnum];
					y_source = y_leaf[pletnum];
					z_source = z_leaf[pletnum];
					qx_source = qx_leaf[pletnum];
					qy_source = qy_leaf[pletnum];
					qz_source = qz_leaf[pletnum];
					avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
					invdr3 = avx_mul(invdr2, invdr);

					fx = avx_add(avx_mul(avx_sub(avx_mul(dz, qy_source), avx_mul(dy, qz_source)), invdr3), fx);
					fy = avx_add(avx_mul(avx_sub(avx_mul(dx, qz_source), avx_mul(dz, qx_source)), invdr3), fy);
					fz = avx_add(avx_mul(avx_sub(avx_mul(dy, qx_source), avx_mul(dx, qy_source)), invdr3), fz);
				}
				pletnum++;
				for (int k = p1_begin + pletnum * detail::avx_vec_length, i = pletnum; k < p1_end; k += detail::avx_vec_length, ++i)
				{
					x_source = x_leaf[i];
					y_source = y_leaf[i];
					z_source = z_leaf[i];
					qx_source = qx_leaf[i];
					qy_source = qy_leaf[i];
					qz_source = qz_leaf[i];
					avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
					invdr3 = avx_mul(invdr2, invdr);

					fx = avx_add(avx_mul(avx_sub(avx_mul(dz, qy_source), avx_mul(dy, qz_source)), invdr3), fx);
					fy = avx_add(avx_mul(avx_sub(avx_mul(dx, qz_source), avx_mul(dz, qx_source)), invdr3), fy);
					fz = avx_add(avx_mul(avx_sub(avx_mul(dy, qx_source), avx_mul(dx, qy_source)), invdr3), fz);

					fx_leaf[i] = avx_sub(fx_leaf[i], avx_mul(avx_sub(avx_mul(dz, qy_target), avx_mul(dy, qz_target)), invdr3));
					fy_leaf[i] = avx_sub(fy_leaf[i], avx_mul(avx_sub(avx_mul(dx, qz_target), avx_mul(dz, qx_target)), invdr3));
					fz_leaf[i] = avx_sub(fz_leaf[i], avx_mul(avx_sub(avx_mul(dy, qx_target), avx_mul(dx, qy_target)), invdr3));
				}

				for (const auto& idx : cell.closeneighbours)
				{
					const auto& neighbour_cell = leaves[idx];
					const auto& [p2_begin, p2_end] = neighbour_cell.source_range;

					auto x_nbr = x_simd.get() + shifts[idx];
					auto y_nbr = y_simd.get() + shifts[idx];
					auto z_nbr = z_simd.get() + shifts[idx];
					auto qx_nbr = qx_simd.get() + shifts[idx];
					auto qy_nbr = qy_simd.get() + shifts[idx];
					auto qz_nbr = qz_simd.get() + shifts[idx];

					auto fx_nbr = fx_simd.data() + shifts[idx];
					auto fy_nbr = fy_simd.data() + shifts[idx];
					auto fz_nbr = fz_simd.data() + shifts[idx];

					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_nbr[i];
						y_source = y_nbr[i];
						z_source = z_nbr[i];
						qx_source = qx_nbr[i];
						qy_source = qy_nbr[i];
						qz_source = qz_nbr[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						invdr3 = avx_mul(invdr2, invdr);

						fx = avx_add(avx_mul(avx_sub(avx_mul(dz, qy_source), avx_mul(dy, qz_source)), invdr3), fx);
						fy = avx_add(avx_mul(avx_sub(avx_mul(dx, qz_source), avx_mul(dz, qx_source)), invdr3), fy);
						fz = avx_add(avx_mul(avx_sub(avx_mul(dy, qx_source), avx_mul(dx, qy_source)), invdr3), fz);

						fx_nbr[i] = avx_sub(fx_nbr[i], avx_mul(avx_sub(avx_mul(dz, qy_target), avx_mul(dy, qz_target)), invdr3));
						fy_nbr[i] = avx_sub(fy_nbr[i], avx_mul(avx_sub(avx_mul(dx, qz_target), avx_mul(dz, qx_target)), invdr3));
						fz_nbr[i] = avx_sub(fz_nbr[i], avx_mul(avx_sub(avx_mul(dy, qx_target), avx_mul(dx, qy_target)), invdr3));
					}
				}
			}
			tmp_force[0] = avx_hsum(fx);
			tmp_force[1] = avx_hsum(fy);
			tmp_force[2] = avx_hsum(fz);
			buf_forces[j] += tmp_force;
		}

		const auto& inner = leaves_inner.data() + leaf_idx * Nx2;
		std::complex<double> Ynm, SphDTheta, SphDPhi;
		Vector3cd coef;
		double x, y, rn, s1, s2, c1, c2;

		for (int j = p1_begin; j < p1_end; ++j)
		{
			auto& particle = particles[j];
			memset(m_temp.data(), 0, sizeof(Matrix3d));
			dr = DecToSph(particle.center - cell.center);

			ComputePnm(N + 1, dr[1], Pnm.data()); // +1 for computing derivative

			for (int m = 0; m <= N; ++m)
			{
				x = m * dr[2];
				eim[m] = std::complex<double>(cos(x), sin(x));
			}
			x = cos(dr[1]);
			y = sin(dr[1]);;
			rn = 1.;


			for (int n = 1; n < N; ++n)
			{
				for (int m = 0; m <= n; ++m)
				{
					Ynm = Pnm(n, m) * eim[m] * Knm(n, m);
					coef = inner[n * (n + 1) / 2 + m] * rn;
					SphDTheta = Knm(n, m) * eim[m] * ((1 - m + n) * Pnm(n + 1, m) - (n + 1) * x * Pnm(n, m)) / y;
					SphDPhi = double(m) * 1.0i * Ynm;

					for (int s = 0; s < 3; ++s)
					{
						m_temp[s][0] -= (double(n) * coef[s] * Ynm).real();
						m_temp[s][1] -= (coef[s] * SphDTheta).real();
						m_temp[s][2] -= (coef[s] * SphDPhi).real();
						if (m != 0)
						{
							m_temp[s][0] -= (double(n) * coef[s] * Ynm).real();
							m_temp[s][1] -= (coef[s] * SphDTheta).real();
							m_temp[s][2] -= (coef[s] * SphDPhi).real();
						}
					}
				}
				rn *= dr[0];
			}

			s1 = sin(dr[1]); c2 = cos(dr[2]); c1 = cos(dr[1]); s2 = sin(dr[2]);
			auto& f = buf_forces[j];
			
			Matrix3d m_copy;
			for (int s = 0; s < 3; ++s)
			{
				m_copy[s][0] = m_temp[s][0] * s1 * c2 + m_temp[s][1] * c1 * c2 - m_temp[s][2] * s2 / s1;
				m_copy[s][1] = m_temp[s][0] * s1 * s2 + m_temp[s][1] * c1 * s2 + m_temp[s][2] * c2 / s1;
				m_copy[s][2] = m_temp[s][0] * c1 - m_temp[s][1] * s1;
			}
			f[0] += m_copy[1][2] - m_copy[2][1];
			f[1] += m_copy[2][0] - m_copy[0][2];
			f[2] += m_copy[0][1] - m_copy[1][0];
		}
	});

	auto& buf_forces = *buf_forces_ets.begin();
	for (auto it = buf_forces_ets.begin() + 1; it < buf_forces_ets.end(); it++)
	{
		const auto& x = *it;
		tbb::parallel_for(tbb::blocked_range<size_t>(0, num_particles),
			[&](tbb::blocked_range<size_t> r) {
				for (int i = r.begin(); i < r.end(); ++i)
					buf_forces[i] += x[i];
			});
	}

	auto& fx_simd = *fx_simd_ets.begin();
	auto& fy_simd = *fy_simd_ets.begin();
	auto& fz_simd = *fz_simd_ets.begin();

	tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i) {
		auto [a, b] = leaves[i].source_range;
		auto sz = (b - a + detail::avx_vec_length - 1) / detail::avx_vec_length;

		auto fx = fx_simd.data() + shifts[i];
		auto fy = fy_simd.data() + shifts[i];
		auto fz = fz_simd.data() + shifts[i];
		double bufVec[detail::avx_vec_length];
		for (int j = 0; j < sz; ++j)
		{
			for (auto it = fx_simd_ets.begin() + 1; it < fx_simd_ets.end(); it++) {
				auto x = it->data() + shifts[i];
				fx[j] = avx_add(fx[j], x[j]);
			}
			for (auto it = fy_simd_ets.begin() + 1; it < fy_simd_ets.end(); it++) {
				auto x = it->data() + shifts[i];
				fy[j] = avx_add(fy[j], x[j]);
			}
			for (auto it = fz_simd_ets.begin() + 1; it < fz_simd_ets.end(); it++) {
				auto x = it->data() + shifts[i];
				fz[j] = avx_add(fz[j], x[j]);
			}

			int k = a + detail::avx_vec_length * j;
			avx_store(bufVec, fx[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][0] += bufVec[s];
			avx_store(bufVec, fy[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][1] += bufVec[s];
			avx_store(bufVec, fz[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][2] += bufVec[s];
		}
		});

#ifdef FMM_MPI
	AllReduce(buf_forces.data(), buf_forces.size());
#endif

	const auto& index_mapping = tree->positions_map;

	tbb::parallel_for(size_t(0), targets_num, [&](size_t i) {
		forces[i] = buf_forces[index_mapping[i]];
	});
}

template <>
void FastMultipole3d<double>::ComputeLeaves()
{
	using namespace std::literals;

	tbb::enumerable_thread_specific<Vector3d> tmp_force_ets, dr_ets;
	tbb::enumerable_thread_specific<Vector3d> matrix_ets;
	tbb::enumerable_thread_specific<std::vector<std::complex<double>>> eim_ets(std::vector<std::complex<double>>(N + 1));
	tbb::enumerable_thread_specific<treevector<double>> Pnm_ets(treevector<double>(N + 1));
	tbb::enumerable_thread_specific<std::vector<Vector3d>> buf_forces_ets((std::vector<Vector3d>(num_particles)));
	tbb::enumerable_thread_specific<std::vector<double>> buf_potentials_ets((std::vector<double>(num_particles)));

	const auto& leaves = tree->levels.back();
	const auto& leaves_inner = inner_expansions.back();
	const auto& particles = tree->particles;

#ifdef FMM_MPI
	auto [Begin, End] = LocalPart(0, tree->level_sizes.back());
#else
	size_t Begin = 0, End = tree->level_sizes.back();
#endif

	decltype(&TreeCell_t::source_range) range_ptr;
	tree->targets_num == 0 ? range_ptr = &TreeCell_t::source_range : range_ptr = &TreeCell_t::target_range;

	using simd_vec_t = std::conditional_t<detail::avx_vec_length == 2, __m128d, std::conditional_t<detail::avx_vec_length == 4, __m256d, __m512d>>;
	simd_vec_t oneVec, epsVec;
	avx_set_constant(epsVec, FORCE_EPS2);
	avx_set_constant(oneVec, 1.0);

	std::unique_ptr<size_t[]> shifts(new size_t[tree->level_sizes.back() + 1]);
	shifts[0] = 0;
	for (size_t i = 0; i < tree->level_sizes.back(); ++i)
	{
		auto [a, b] = leaves[i].source_range;
		shifts[i + 1] = shifts[i] + (b - a + detail::avx_vec_length - 1) / detail::avx_vec_length;
	}
	size_t num_vecs = shifts[tree->level_sizes.back()];
	std::unique_ptr<simd_vec_t[]> x_simd(new simd_vec_t[num_vecs]),
								  y_simd(new simd_vec_t[num_vecs]),
								  z_simd(new simd_vec_t[num_vecs]),
								  q_simd(new simd_vec_t[num_vecs]);
	
	double t1 = omp_get_wtime();
	tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i)
		{
			auto [a, b] = leaves[i].source_range;
			auto xptr = x_simd.get() + shifts[i];
			auto yptr = y_simd.get() + shifts[i];
			auto zptr = z_simd.get() + shifts[i];
			auto qptr = q_simd.get() + shifts[i];
			for (int idx = a, k = 0; idx < b; idx += detail::avx_vec_length, ++k)
			{
#if defined(AVX128)
				xptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qptr[k] = _mm_set_pd(idx + 2 > b ? 0 : particles[idx + 1].q, particles[idx].q);
#elif defined(AVX256)
				xptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[0],
					idx + 3 > b ? 0 : particles[idx + 2].center[0],
					idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[1],
					idx + 3 > b ? 0 : particles[idx + 2].center[1],
					idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].center[2],
					idx + 3 > b ? 0 : particles[idx + 2].center[2],
					idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qptr[k] = _mm256_set_pd(idx + 4 > b ? 0 : particles[idx + 3].q,
					idx + 3 > b ? 0 : particles[idx + 2].q,
					idx + 2 > b ? 0 : particles[idx + 1].q, particles[idx].q);
#elif defined(AVX512)
				xptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[0],
					idx + 7 > b ? 0 : particles[idx + 6].center[0],
					idx + 6 > b ? 0 : particles[idx + 5].center[0],
					idx + 5 > b ? 0 : particles[idx + 4].center[0],
					idx + 4 > b ? 0 : particles[idx + 3].center[0],
					idx + 3 > b ? 0 : particles[idx + 2].center[0],
					idx + 2 > b ? 0 : particles[idx + 1].center[0], particles[idx].center[0]);
				yptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[1],
					idx + 7 > b ? 0 : particles[idx + 6].center[1],
					idx + 6 > b ? 0 : particles[idx + 5].center[1],
					idx + 5 > b ? 0 : particles[idx + 4].center[1],
					idx + 4 > b ? 0 : particles[idx + 3].center[1],
					idx + 3 > b ? 0 : particles[idx + 2].center[1],
					idx + 2 > b ? 0 : particles[idx + 1].center[1], particles[idx].center[1]);
				zptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].center[2],
					idx + 7 > b ? 0 : particles[idx + 6].center[2],
					idx + 6 > b ? 0 : particles[idx + 5].center[2],
					idx + 5 > b ? 0 : particles[idx + 4].center[2],
					idx + 4 > b ? 0 : particles[idx + 3].center[2],
					idx + 3 > b ? 0 : particles[idx + 2].center[2],
					idx + 2 > b ? 0 : particles[idx + 1].center[2], particles[idx].center[2]);
				qptr[k] = _mm512_set_pd(idx + 8 > b ? 0 : particles[idx + 7].q,
					idx + 7 > b ? 0 : particles[idx + 6].q,
					idx + 6 > b ? 0 : particles[idx + 5].q,
					idx + 5 > b ? 0 : particles[idx + 4].q,
					idx + 4 > b ? 0 : particles[idx + 3].q,
					idx + 3 > b ? 0 : particles[idx + 2].q,
					idx + 2 > b ? 0 : particles[idx + 1].q, particles[idx].q);
#endif
		}
	});

	tbb::enumerable_thread_specific<std::vector<simd_vec_t>> fx_simd_ets((std::vector<simd_vec_t>(num_vecs))),
															 fy_simd_ets((std::vector<simd_vec_t>(num_vecs))),
															 fz_simd_ets((std::vector<simd_vec_t>(num_vecs))),
															 phi_simd_ets((std::vector<simd_vec_t>(num_vecs)));
	
	tbb::parallel_for(Begin, End, [&](size_t leaf_idx) {
		auto& tmp_force = tmp_force_ets.local();
		auto& dr = dr_ets.local();
		auto& eim = eim_ets.local();
		auto& Pnm = Pnm_ets.local();
		auto& m_temp = matrix_ets.local();
		auto& buf_forces = buf_forces_ets.local();
		auto& buf_potentials = buf_potentials_ets.local();
		auto& fx_simd = fx_simd_ets.local();
		auto& fy_simd = fy_simd_ets.local();
		auto& fz_simd = fz_simd_ets.local();
		auto& phi_simd = phi_simd_ets.local();
		double phi = 0.0;

		const auto& cell = leaves[leaf_idx];
		const auto& [p1_begin, p1_end] = cell.*range_ptr;

		simd_vec_t x_target, y_target, z_target, q_target, 
				   x_source, y_source, z_source, q_source, dx, dy, dz, fx, fy, fz, pot, invdr, invdr2, q_invdr, q_invdr3;

		auto x_leaf = x_simd.get() + shifts[leaf_idx];
		auto y_leaf = y_simd.get() + shifts[leaf_idx];
		auto z_leaf = z_simd.get() + shifts[leaf_idx];
		auto q_leaf = q_simd.get() + shifts[leaf_idx];

		auto fx_leaf = fx_simd.data() + shifts[leaf_idx];
		auto fy_leaf = fy_simd.data() + shifts[leaf_idx];
		auto fz_leaf = fz_simd.data() + shifts[leaf_idx];
		auto phi_leaf = phi_simd.data() + shifts[leaf_idx];

		for (int j = p1_begin; j < p1_end; ++j)
		{
			auto& particle = particles[j];

			if (range_ptr == &TreeCell_t::target_range)
			{
				avx_set_constant(x_target, particle.center[0]);
				avx_set_constant(y_target, particle.center[1]);
				avx_set_constant(z_target, particle.center[2]);
				avx_set_constant(q_target, particle.q);
				fx = avx_zero_vec<simd_vec_t>();
				fy = avx_zero_vec<simd_vec_t>();
				fz = avx_zero_vec<simd_vec_t>();
				pot = avx_zero_vec<simd_vec_t>();

				{
					const auto& [p2_begin, p2_end] = cell.source_range;
					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_leaf[i];
						y_source = y_leaf[i];
						z_source = z_leaf[i];
						q_source = q_leaf[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						q_invdr = avx_mul(q_source, invdr);
						q_invdr3 = avx_mul(invdr2, q_invdr);

						fx = avx_add(avx_mul(dx, q_invdr3), fx);
						fy = avx_add(avx_mul(dy, q_invdr3), fy);
						fz = avx_add(avx_mul(dz, q_invdr3), fz);
						pot = avx_add(q_invdr, pot);
					}
				}

				for (const auto& idx : cell.closeneighbours)
				{
					const auto& neighbour_cell = leaves[idx];
					const auto& [p2_begin, p2_end] = neighbour_cell.source_range;

					auto x_nbr = x_simd.get() + shifts[idx];
					auto y_nbr = y_simd.get() + shifts[idx];
					auto z_nbr = z_simd.get() + shifts[idx];
					auto q_nbr = q_simd.get() + shifts[idx];

					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_nbr[i];
						y_source = y_nbr[i];
						z_source = z_nbr[i];
						q_source = q_nbr[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						q_invdr = avx_mul(q_source, invdr);
						q_invdr3 = avx_mul(invdr2, q_invdr);

						fx = avx_add(avx_mul(dx, q_invdr3), fx);
						fy = avx_add(avx_mul(dy, q_invdr3), fy);
						fz = avx_add(avx_mul(dz, q_invdr3), fz);
						pot = avx_add(q_invdr, pot);
					}
				}
			}
			else
			{
				avx_set_constant(x_target, particle.center[0]);
				avx_set_constant(y_target, particle.center[1]);
				avx_set_constant(z_target, particle.center[2]);
				avx_set_constant(q_target, particle.q);
				fx = avx_zero_vec<simd_vec_t>();
				fy = avx_zero_vec<simd_vec_t>();
				fz = avx_zero_vec<simd_vec_t>();
				pot = avx_zero_vec<simd_vec_t>();

				int pletnum = (j - p1_begin) / detail::avx_vec_length;
				{
					x_source = x_leaf[pletnum];
					y_source = y_leaf[pletnum];
					z_source = z_leaf[pletnum];
					q_source = q_leaf[pletnum];
					avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
					q_invdr = avx_mul(q_source, invdr);
					q_invdr3 = avx_mul(invdr2, q_invdr);

					fx = avx_add(avx_mul(dx, q_invdr3), fx);
					fy = avx_add(avx_mul(dy, q_invdr3), fy);
					fz = avx_add(avx_mul(dz, q_invdr3), fz);
					pot = avx_add(q_invdr, pot);
				}
				pletnum++;
				for (int k = p1_begin + pletnum * detail::avx_vec_length, i = pletnum; k < p1_end; k += detail::avx_vec_length, ++i)
				{
					x_source = x_leaf[i];
					y_source = y_leaf[i];
					z_source = z_leaf[i];
					q_source = q_leaf[i];
					avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
					q_invdr = avx_mul(q_source, invdr);
					q_invdr3 = avx_mul(invdr2, q_invdr);

					fx = avx_add(avx_mul(dx, q_invdr3), fx);
					fy = avx_add(avx_mul(dy, q_invdr3), fy);
					fz = avx_add(avx_mul(dz, q_invdr3), fz);
					pot = avx_add(q_invdr, pot);

					q_invdr = avx_mul(q_target, invdr);
					q_invdr3 = avx_mul(invdr2, q_invdr);
						
					fx_leaf[i] = avx_sub(fx_leaf[i], avx_mul(dx, q_invdr3));
					fy_leaf[i] = avx_sub(fy_leaf[i], avx_mul(dy, q_invdr3));
					fz_leaf[i] = avx_sub(fz_leaf[i], avx_mul(dz, q_invdr3));
					phi_leaf[i] = avx_add(q_invdr, phi_leaf[i]);
				}

				for (const auto& idx : cell.closeneighbours)
				{
					const auto& neighbour_cell = leaves[idx];
					const auto& [p2_begin, p2_end] = neighbour_cell.source_range;

					auto x_nbr = x_simd.get() + shifts[idx];
					auto y_nbr = y_simd.get() + shifts[idx];
					auto z_nbr = z_simd.get() + shifts[idx];
					auto q_nbr = q_simd.get() + shifts[idx];

					auto fx_nbr = fx_simd.data() + shifts[idx];
					auto fy_nbr = fy_simd.data() + shifts[idx];
					auto fz_nbr = fz_simd.data() + shifts[idx];
					auto phi_nbr = phi_simd.data() + shifts[idx];

					for (int k = p2_begin, i = 0; k < p2_end; k += detail::avx_vec_length, ++i)
					{
						x_source = x_nbr[i];
						y_source = y_nbr[i];
						z_source = z_nbr[i];
						q_source = q_nbr[i];
						avx_inv_dr(oneVec, epsVec, x_target, y_target, z_target, x_source, y_source, z_source, dx, dy, dz, invdr, invdr2);
						q_invdr = avx_mul(q_source, invdr);
						q_invdr3 = avx_mul(invdr2, q_invdr);
				
						fx = avx_add(avx_mul(dx, q_invdr3), fx);
						fy = avx_add(avx_mul(dy, q_invdr3), fy);
						fz = avx_add(avx_mul(dz, q_invdr3), fz);
						pot = avx_add(q_invdr, pot);

						q_invdr = avx_mul(q_target, invdr);
						q_invdr3 = avx_mul(invdr2, q_invdr);
					
						fx_nbr[i] = avx_sub(fx_nbr[i], avx_mul(dx, q_invdr3));
						fy_nbr[i] = avx_sub(fy_nbr[i], avx_mul(dy, q_invdr3));
						fz_nbr[i] = avx_sub(fz_nbr[i], avx_mul(dz, q_invdr3));
						phi_nbr[i] = avx_add(q_invdr, phi_nbr[i]);
					}
				}	
			}
			tmp_force[0] = avx_hsum(fx);
			tmp_force[1] = avx_hsum(fy);
			tmp_force[2] = avx_hsum(fz);
			buf_forces[j] += tmp_force;
			buf_potentials[j] += avx_hsum(pot);
		}
		const auto& inner = leaves_inner.data() + leaf_idx * Nx2;
		std::complex<double> Ynm, coef, SphDTheta, SphDPhi;
		double x, y, rn, s1, s2, c1, c2;

		for (int j = p1_begin; j < p1_end; ++j)
		{
			auto& particle = particles[j];
			memset(m_temp.data(), 0, sizeof(Vector3d));
			dr = DecToSph(particle.center - cell.center);
		
			ComputePnm(N + 1, dr[1], Pnm.data()); // +1 for computing derivative
		
			for (int m = 0; m <= N; ++m)
			{
				x = m * dr[2];
				eim[m] = std::complex<double>(cos(x), sin(x));
			}
			x = cos(dr[1]);
			y = sin(dr[1]);;
			rn = 1. / dr[0];
		
			phi = 0.0;
					
			for (int n = 0; n < N; ++n)
			{
				for (int m = 0; m <= n; ++m)
				{
					Ynm = Pnm(n, m) * eim[m] * Knm(n, m);
					coef = inner[n * (n + 1) / 2 + m] * rn;
		
					phi += (dr[0] * Ynm * coef).real();
					if (m != 0)
						phi += (dr[0] * Ynm * coef).real();
		
					SphDTheta = Knm(n, m) * eim[m] * ((1 - m + n) * Pnm(n + 1, m) - (n + 1) * x * Pnm(n, m)) / y;
					SphDPhi = double(m) * 1.0i * Ynm;
		
					m_temp[0] -= (double(n) * coef * Ynm).real();
					m_temp[1] -= (coef * SphDTheta).real();
					m_temp[2] -= (coef * SphDPhi).real();
					if (m != 0)
					{
						m_temp[0] -= (double(n) * coef * Ynm).real();
						m_temp[1] -= (coef * SphDTheta).real();
						m_temp[2] -= (coef * SphDPhi).real();
					}
				}
				rn *= dr[0];
			}
			buf_potentials[j] += phi;
		
			s1 = sin(dr[1]); c2 = cos(dr[2]); c1 = cos(dr[1]); s2 = sin(dr[2]);
			auto& f = buf_forces[j];
			f[0] += m_temp[0] * s1 * c2 + m_temp[1] * c1 * c2 - m_temp[2] * s2 / s1;
			f[1] += m_temp[0] * s1 * s2 + m_temp[1] * c1 * s2 + m_temp[2] * c2 / s1;
			f[2] += m_temp[0] * c1 - m_temp[1] * s1;
		}
	});
	std::cout << omp_get_wtime() - t1 << std::endl;
	
	auto& buf_forces = *buf_forces_ets.begin();
	for (auto it = buf_forces_ets.begin() + 1; it < buf_forces_ets.end(); it++)
	{
		const auto& x = *it;
		tbb::parallel_for(tbb::blocked_range<size_t>(0, num_particles),
			[&](tbb::blocked_range<size_t> r) {
				for (int i = r.begin(); i < r.end(); ++i)
					buf_forces[i] += x[i];
			});
	}

	auto& buf_potentials = *buf_potentials_ets.begin();
	for (auto it = buf_potentials_ets.begin() + 1; it < buf_potentials_ets.end(); it++)
	{
		const auto& x = *it;
		tbb::parallel_for(tbb::blocked_range<size_t>(0, num_particles),
			[&](tbb::blocked_range<size_t> r) {
				for (int i = r.begin(); i < r.end(); ++i)
					buf_potentials[i] += x[i];
			});
	}

	auto& fx_simd = *fx_simd_ets.begin();
	auto& fy_simd = *fy_simd_ets.begin();
	auto& fz_simd = *fz_simd_ets.begin();
	auto& phi_simd = *phi_simd_ets.begin();

	tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i) {
		auto [a, b] = leaves[i].source_range;
		auto sz = (b - a + detail::avx_vec_length - 1) / detail::avx_vec_length;

		auto fx = fx_simd.data() + shifts[i];
		auto fy = fy_simd.data() + shifts[i];
		auto fz = fz_simd.data() + shifts[i];
		auto phi = phi_simd.data() + shifts[i];
		double bufVec[detail::avx_vec_length];
		for (int j = 0; j < sz; ++j)
		{
			for (auto it = fx_simd_ets.begin() + 1; it < fx_simd_ets.end(); it++) {
				const auto& x = it->data() + shifts[i];
				fx[j] = avx_add(fx[j], x[j]);
			}
			for (auto it = fy_simd_ets.begin() + 1; it < fy_simd_ets.end(); it++) {
				const auto& x = it->data() + shifts[i];
				fy[j] = avx_add(fy[j], x[j]);
			}
			for (auto it = fz_simd_ets.begin() + 1; it < fz_simd_ets.end(); it++) {
				const auto& x = it->data() + shifts[i];
				fz[j] = avx_add(fz[j], x[j]);
			}
			for (auto it = phi_simd_ets.begin() + 1; it < phi_simd_ets.end(); it++) {
				const auto& x = it->data() + shifts[i];
				phi[j] = avx_add(phi[j], x[j]);
			}

			int k = a + detail::avx_vec_length * j;
			avx_store(bufVec, fx[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][0] += bufVec[s];
			avx_store(bufVec, fy[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][1] += bufVec[s];
			avx_store(bufVec, fz[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_forces[k + s][2] += bufVec[s];
			avx_store(bufVec, phi[j]);
			for (int s = 0; s < detail::avx_vec_length; ++s)
				if (k + s < b) buf_potentials[k + s] += bufVec[s];
		}
	});

#ifdef FMM_MPI
	AllReduce(buf_forces.data(), buf_forces.size());
	AllReduce(buf_potentials.data(), buf_potentials.size());
#endif

	const auto& index_mapping = tree->positions_map;

	tbb::parallel_for(size_t(0), targets_num, [&](size_t i) {
		forces[i] = buf_forces[index_mapping[i]];
		potentials[i] = buf_potentials[index_mapping[i]];
	});
}

template <typename value>
FastMultipole3d<value>::FastMultipole3d(const std::vector<particle_t>& particles, double eps, int N_, int tree_depth_)
{
	auto particles_copy = particles;
	targets_num = particles.size();
	Solve(std::move(particles_copy), eps, N_, tree_depth_);
}

template <typename value>
FastMultipole3d<value>::FastMultipole3d(std::vector<particle_t>&& particles, double eps, int N_, int tree_depth_)
{
	targets_num = particles.size();
	Solve(std::move(particles), eps, N_, tree_depth_);
}

template <typename value>
FastMultipole3d<value>::FastMultipole3d(const std::vector<particle_t>& source_particles, const std::vector<particle_t>& target_particles, double eps, int N_, int tree_depth_)
{
	std::vector<particle_t> particles(source_particles.size() + target_particles.size());
	memcpy(particles.data(), target_particles.data(), target_particles.size() * sizeof(particle_t));
	memcpy(particles.data() + target_particles.size(), source_particles.data(), source_particles.size() * sizeof(particle_t));
	targets_num = target_particles.size();
	Solve(std::move(particles), eps, N_, tree_depth_);
}

template <typename value>
void FastMultipole3d<value>::Solve(std::vector<particle_t>&& particles, double eps, int N_, int tree_depth_)
{
	double T = omp_get_wtime();
	if (IAmRoot()) std::cout << "\n***************** Start FMM **************" << std::endl;
	num_particles = particles.size();
	eps = std::max(eps, 1.e-11);
	if (N_ == FMM_AUTO)
		N = std::min(fmm::detail::_3d_MAX_MULTIPOLE_NUM, int(54.2 - 10.73 * sqrt(log(2.4 * eps) + 25.5)));
	else
		N = std::min(detail::_3d_MAX_MULTIPOLE_NUM, N_);
	Nx2 = N * (N + 1) / 2;
	if (IAmRoot()) std::cout << "multipole num = " << N << std::endl;
	if (tree_depth_ == FMM_AUTO)
		tree_depth = std::max(2.0, 1 + log(double(num_particles) / 100.0) / log(8.0));
	else
		tree_depth = tree_depth_;
	if (IAmRoot()) std::cout << "tree depth = " << tree_depth << std::endl;

	tree = std::make_shared<MortonTree_t>(std::move(particles), tree_depth);
	
	double t = omp_get_wtime();
	outer_expansions.resize(tree_depth);
	inner_expansions.resize(tree_depth);
	for (int i = 0; i < tree_depth; ++i)
	{
		auto& outer = outer_expansions[i];
		auto& inner = inner_expansions[i];
		outer.resize(tree->level_sizes[i] * Nx2);
		inner.resize(tree->level_sizes[i] * Nx2);
	}
	forces.resize(targets_num);
	if constexpr (std::is_same_v<value, double>)
		potentials.resize(targets_num);

	if (IAmRoot()) std::cout << "allocation time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	Prepare();
	if (IAmRoot()) std::cout << "prepare time: " << omp_get_wtime() - t << std::endl;

	Upward();

	t = omp_get_wtime();
	Downward();
	if (IAmRoot()) std::cout << "m2l+l2l time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	ComputeLeaves();
	if (IAmRoot()) std::cout << "leaf time: " << omp_get_wtime() - t << std::endl;
	
	if (IAmRoot()) std::cout << "total fmm time: " << omp_get_wtime() - T << std::endl;
	if (IAmRoot()) std::cout << "***************** End FMM **************\n" << std::endl;
}

template <typename value>
void FastMultipole3d<value>::Prepare()
{
	local_work1 = tbb::enumerable_thread_specific<std::vector<complex_value>>{ std::vector<complex_value>(Nx2) };
	local_work2 = tbb::enumerable_thread_specific<std::vector<complex_value>>{ std::vector<complex_value>(Nx2) };

	/*m2lcoef.resize(2 * N * N * N);
	for (int j = 0; j < N; ++j)
	{
		for (int k = 0; k <= j; ++k)
		{
			for (int n = k; n < N; ++n)
			{
				int idx = N * (j + 2) * (j + 1) / 2 + N * (k + 1) + n + 1;
				m2lcoef[idx] = Anm(n, k) * Anm(j, k) * ni(j + k) / Anm(j + n, 0);
			}
		}
	}*/
}

template class FastMultipole3d<double>;
template class FastMultipole3d<Vector3d>;

} // fmm