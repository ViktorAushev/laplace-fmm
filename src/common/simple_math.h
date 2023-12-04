#pragma once
#ifndef __NVCC__
#include <tbb/parallel_for.h>
#endif
#include <vector>
#include <complex>
#include <numbers>
#include <functional>
#include "cuda_utils.h"
#include "mpi_utils.h"
#include <iostream>

namespace fmm {

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

namespace detail {

template<int n>
struct BinomNewton_wrapper
{
	std::array<double, n * n> cft;
	int dim;

	constexpr BinomNewton_wrapper() : cft(), dim(n)
	{
		cft[0 * dim + 0] = 1.0;

		for (int i = 1; i < n; ++i)
		{
			cft[i * dim + 0] = 1.0;
			cft[i * dim + i] = 1.0;
			for (int j = 1; j < i; ++j)
				cft[i * dim + j] = cft[(i - 1) * dim + j] + cft[(i - 1) * dim + (j - 1)];
		}
	}
	constexpr double operator()(int p, int q) const
	{
		return cft[p * dim + q];
	}
};

constexpr BinomNewton_wrapper<2 * _2d_MAX_MULTIPOLE_NUM> binom;

}

class BinomNewton
{
public:
	std::vector<double> cft;
	int dim;

public:
	constexpr BinomNewton(int n)
		: dim(n + 1)
	{
		cft.resize(dim * dim, 0.0);
		cft[0 * dim + 0] = 1.0;

		for (int i = 1; i <= n; ++i)
		{
			cft[i * dim + 0] = 1.0;
			cft[i * dim + i] = 1.0;
			for (int j = 1; j < i; ++j)
				cft[i * dim + j] = cft[(i - 1) * dim + j] + cft[(i - 1) * dim + (j - 1)];
		}
	}

	constexpr double operator()(int p, int q) const
	{
		return cft[p * dim + q];
	}
};

template <typename T>
__DEVICE__ __HOST__ constexpr T MyPow(T base, unsigned int exp)
{
	T res = 1;
	while (exp) {
		if (exp & 1)
		{
			res *= base;
		}
		exp >>= 1;
		base *= base;
	}
	return res;
}

__DEVICE__ __HOST__ constexpr inline double ni(int n)
{
	return ((n & 1) == 1) ? -1.0 : 1.0;
}

__DEVICE__ __HOST__ inline Vector3d DecToSph(const Vector3d& vec)
{
	const double eps = 1e-12;
	Vector3d c;
	c[0] = abs(vec) + eps;
	c[1] = acos(vec[2] / c[0]);
	if (fabs(vec[0]) + fabs(vec[1]) < eps) {
		c[2] = 0;
	}
	else if (fabs(vec[0]) < eps) {
		c[2] = vec[1] / fabs(vec[1]) * std::numbers::pi * 0.5;
	}
	else {
		c[2] = atan2(vec[1], vec[0]);
	}
	return c;
}

inline double Potential2d(const particle2d& p1, const particle2d& p2)
{
	return p2.q * 0.5 * log(std::max(norm(p1.center - p2.center), FORCE_EPS2));
}

inline std::complex<double> Force2d(const particle2d& p1, const particle2d& p2)
{
	auto dz = p1.center - p2.center;
	return p2.q * std::conj(dz) / std::max(norm(dz), FORCE_EPS2);
}

inline double Potential3d(const particle3d& p1, const particle3d& p2)
{
	return p2.q / std::max(abs(p1.center - p2.center), FORCE_EPS);
}

inline Vector3d Force3d(const particle3d& p1, const particle3d& p2)
{
	auto dr = p1.center - p2.center;
	return p2.q * dr / MyPow(std::max(abs(dr), FORCE_EPS), 3);
}

inline Vector3d Force3d(const particle3d3& p1, const particle3d3& p2)
{
	auto dr = p1.center - p2.center;
	return cross(p2.q, dr) / MyPow(std::max(abs(dr), FORCE_EPS), 3);
}

inline void Potential3dMutual(const particle3d& p1, const particle3d& p2, double& potential1, double& potential2)
{
	auto invdr = 1.0 / std::max(abs(p1.center - p2.center), FORCE_EPS);
	potential1 += p2.q * invdr;
	potential2 += p1.q * invdr;
}

inline void Force3dMutual(const particle3d& p1, const particle3d& p2, Vector3d& force1, Vector3d& force2)
{
	auto dr = p1.center - p2.center;
	auto invdr = dr / MyPow(std::max(abs(dr), FORCE_EPS), 3);
	force1 += p2.q * invdr;
	force2 -= p1.q * invdr;
}

inline void Force3dMutual(const particle3d3& p1, const particle3d3& p2, Vector3d& force1, Vector3d& force2)
{
	auto dr = p1.center - p2.center;
	auto invdr = dr / MyPow(std::max(abs(dr), FORCE_EPS), 3);
	force1 += cross(p2.q, invdr);
	force2 -= cross(p1.q, invdr);
}

inline void Potential2dMutual(const particle2d& p1, const particle2d& p2, double& potential1, double& potential2)
{
	double dz = 0.5 * log(std::max(norm(p1.center - p2.center), FORCE_EPS2));
	potential1 += p2.q * dz;
	potential2 += p1.q * dz;
}

inline void Force2dMutual(const particle2d& p1, const particle2d& p2, std::complex<double>& force1, std::complex<double>& force2)
{
	auto dz = p1.center - p2.center;
	auto invdz = std::conj(dz) / std::max(norm(dz), FORCE_EPS2);
	force1 += p2.q * invdz;
	force2 -= p1.q * invdz;
}

#ifdef __NVCC__
__device__ inline double Potential2d(const gpu::particle2d& p1, const gpu::particle2d& p2)
{
	return p2.q * 0.5 * log(max(cuda::std::norm(p1.center - p2.center), CUDA_FORCE_EPS2));
}

__device__ inline gpu::cuda_complex Force2d(const gpu::particle2d& p1, const gpu::particle2d& p2)
{
	auto dz = p1.center - p2.center;
	return p2.q * cuda::std::conj(dz) / max(cuda::std::norm(dz), CUDA_FORCE_EPS2);
}

__device__ inline double Potential3d(const gpu::particle3d& p1, const gpu::particle3d& p2)
{
	double dr2 = max(norm(p1.center - p2.center), CUDA_FORCE_EPS2);
	return p2.q * rsqrt(dr2);
}

__device__ inline Vector3d Force3d(const gpu::particle3d& p1, const gpu::particle3d& p2)
{
	auto dr = p1.center - p2.center;
	auto dr2 = max(norm(dr), CUDA_FORCE_EPS2);
	return p2.q * dr * MyPow(rsqrt(dr2), 3);
}

__device__ inline Vector3d Force3d(const gpu::particle3d3& p1, const gpu::particle3d3& p2)
{
	auto dr = p1.center - p2.center;
	auto dr2 = max(norm(dr), CUDA_FORCE_EPS2);
	return cross(p2.q, dr) * MyPow(rsqrt(dr2), 3);
}
#endif

#ifndef __NVCC__
template <typename point_type, typename interaction_type, typename value_type>
void ComputeExact(const std::vector<particle<point_type, value_type>>& source_particles,
	const std::vector<particle<point_type, value_type>>& target_particles,
	std::function<interaction_type(const particle<point_type, value_type>&, const particle<point_type, value_type>&)> func, std::string filename)
{
	size_t num_particles = target_particles.size();
#ifdef FMM_MPI
	auto [shift, end_part] = LocalPart(0, num_particles);
	size_t local_size = end_part - shift;
#else
	size_t shift = 0, local_size = num_particles;
#endif
	std::vector<interaction_type> exact(local_size);

	auto particles_ptr = target_particles.data() + shift;
	tbb::parallel_for(size_t(0), local_size, [&](size_t i)
	{
		const auto& p1 = particles_ptr[i];
		for (const auto& p2 : source_particles)
		{
			exact[i] += func(p1, p2);
		}
	});

#ifdef FMM_MPI
	std::vector<int> sizes(NProc());
	std::vector<int> displs(NProc());
	std::vector<interaction_type> buf(num_particles);
	for (int j = 0; j < NProc(); ++j)
	{
		sizes[j] = LocalPart(0, num_particles, j);
	}
	for (int j = 1; j < NProc(); ++j)
	{
		displs[j] = displs[j - 1] + sizes[j - 1];
	}
	if constexpr (std::is_same_v<interaction_type,double>)
		MPI_Allgatherv(exact.data(), local_size, MPI_DOUBLE,
			buf.data(), sizes.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	if constexpr (std::is_same_v<interaction_type,std::complex<double>>)
		MPI_Allgatherv(exact.data(), local_size, MPI_COMPLEX16,
			buf.data(), sizes.data(), displs.data(), MPI_COMPLEX16, MPI_COMM_WORLD);
	if constexpr (std::is_same_v<interaction_type, Vector3d>)
	{
		MPI_Datatype MPI_VECTOR3;
		MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VECTOR3);
		MPI_Type_commit(&MPI_VECTOR3);
		MPI_Allgatherv(exact.data(), local_size, MPI_VECTOR3,
			buf.data(), sizes.data(), displs.data(), MPI_VECTOR3, MPI_COMM_WORLD);
	}
#endif

	std::ofstream fout(filename);
	fout.precision(20);
#ifdef FMM_MPI
	for (const auto& x : buf)
		fout << x << "\n";
#else
	for (const auto& x : exact)
		fout << x << "\n";
#endif
}
#endif

namespace gpu {

namespace detail {

template <InteractionType it, typename interaction_type, typename point_type, typename value_type, typename gpu_interaction_type, typename gpu_point_type>
struct ComputeExact {
	static void Compute(const std::vector<fmm::particle<point_type, value_type>>& source_particles, const std::vector<fmm::particle<point_type, value_type>>& target_particles);
}; }

template <InteractionType it, typename point_type, typename value_type>
void ComputeExact(const std::vector<fmm::particle<point_type, value_type>>& source_particles, const std::vector<fmm::particle<point_type, value_type>>& target_particles)
{
	if constexpr (it == fmm::InteractionType::Potential2d)
		fmm::gpu::detail::ComputeExact<fmm::InteractionType::Potential2d, double, fmm::point2d, value_type, double, fmm::gpu::point2d>::Compute(source_particles, target_particles);
	if constexpr (it == fmm::InteractionType::Force2d)
		fmm::gpu::detail::ComputeExact<fmm::InteractionType::Force2d, fmm::point2d, fmm::point2d, value_type, fmm::gpu::point2d, fmm::gpu::point2d>::Compute(source_particles, target_particles);
	if constexpr (it == fmm::InteractionType::Potential3d)
		fmm::gpu::detail::ComputeExact<fmm::InteractionType::Potential3d, double, fmm::point3d, value_type, double, fmm::gpu::point3d>::Compute(source_particles, target_particles);
	if constexpr (it == fmm::InteractionType::Force3d)
		fmm::gpu::detail::ComputeExact<fmm::InteractionType::Force3d, fmm::point3d, fmm::point3d, value_type, fmm::gpu::point3d, fmm::gpu::point3d>::Compute(source_particles, target_particles);
}

template <InteractionType it, typename point_type, typename value_type>
void ComputeExact(const std::vector<fmm::particle<point_type, value_type>>& particles)
{
	fmm::gpu::ComputeExact<it, point_type, value_type>(particles, particles);
}

}

template <InteractionType it, typename point_type, typename value_type>
void ComputeExact(const std::vector<particle<point_type, value_type>>& source_particles, const std::vector<particle<point_type, value_type>>& target_particles)
{
	if constexpr (it == fmm::InteractionType::Potential2d) {
		auto foo = [](const particle2d& p1, const particle2d& p2) {return Potential2d(p1, p2); };
		ComputeExact<point2d, double, value_type>(source_particles, target_particles, foo, "potential2d_exact.txt");
	}
	if constexpr (it == fmm::InteractionType::Force2d) {
		auto foo = [](const particle2d& p1, const particle2d& p2) {return Force2d(p1, p2); };
		ComputeExact<point2d, point2d, value_type>(source_particles, target_particles, foo, "force2d_exact.txt");
	}
	if constexpr (it == fmm::InteractionType::Potential3d)
	{
		auto foo = [](const particle3d& p1, const particle3d& p2) {return Potential3d(p1, p2); };
		ComputeExact<point3d, double, value_type>(source_particles, target_particles, foo, "potential3d_exact.txt");
	}
	if constexpr (it == fmm::InteractionType::Force3d)
	{
		auto foo = [](const particle<point_type, value_type>& p1, const particle<point_type, value_type>& p2) {return Force3d(p1, p2); };
		ComputeExact<point3d, Vector3d, value_type>(source_particles, target_particles, foo, "force3d_exact.txt");
	}
}

template <InteractionType it, typename point_type, typename value_type>
void ComputeExact(const std::vector<particle<point_type, value_type>>& particles)
{
	ComputeExact<it, point_type, value_type>(particles, particles);
}

template <InteractionType it, typename T>
void ReadError(const std::vector<T>& res)
{
	size_t num_particles = res.size();
	std::vector<T> exact(num_particles);
	
	std::string filename;
	switch (it)
	{
	case fmm::InteractionType::Potential2d:
		filename = "potential2d_exact.txt";
		break;
	case fmm::InteractionType::Force2d:
		filename = "force2d_exact.txt";
		break;
	case fmm::InteractionType::Potential3d:
		filename = "potential3d_exact.txt";
		break;
	case fmm::InteractionType::Force3d:
		filename = "force3d_exact.txt";
		break;
	default:
		break;
	}
	
	std::ifstream fin(filename);
	for (auto& x : exact)
		fin >> x;
	
	double max_error = 0.0;
	double err1 = 0.0, err2 = 0.0;
	double l2error = 0.0;
#pragma omp parallel for reduction(+: err1, err2, l2error) reduction(max: max_error)
	for (int i = 0; i < num_particles; ++i)
	{
		double err = 0;
		if constexpr (it != InteractionType::Force3d)
			err = std::abs(exact[i] - res[i]);
		else
			err = abs(exact[i] - res[i]);
		max_error = std::max(max_error, err);
		err1 += err;
		if constexpr (it != InteractionType::Force3d)
		{
			err2 += std::abs(exact[i]);
			l2error += (err / std::abs(exact[i])) * (err / std::abs(exact[i]));
		}
		else
		{
			err2 += abs(exact[i]);
			l2error += (err / abs(exact[i])) * (err / abs(exact[i]));
		}

	}
	l2error /= num_particles;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "max error = " << max_error << std::endl;
	std::cout << "l2 error = " << sqrt(l2error) << std::endl;
	std::cout << "relative error = " << err1 / err2 << std::endl;
	std::cout << "--------------------------------" << std::endl;
}

} // fmm

