#pragma once

#define FMM_MPI // enable MPI
#define FMM_CONSTEXPR_MATH // fast CUDA solver, but limits max multipole num (min error 10^-7)

#ifdef FMM_CONSTEXPR_MATH
#define FMM_CONSTEXPR constexpr
#else
#define FMM_CONSTEXPR
#endif

#ifdef __NVCC__
#define __HOST__ __host__
#define __DEVICE__ __device__
#else
#define __HOST__
#define __DEVICE__
#endif

namespace fmm {

	const int FMM_AUTO = std::numeric_limits<int>::max();

	constexpr double FORCE_EPS = 1.e-4;
	constexpr double FORCE_EPS2 = FORCE_EPS * FORCE_EPS;
	__DEVICE__ constexpr double CUDA_FORCE_EPS = FORCE_EPS;
	__DEVICE__ constexpr double CUDA_FORCE_EPS2 = CUDA_FORCE_EPS * CUDA_FORCE_EPS;

namespace detail {

	const int _2d_MAX_MULTIPOLE_NUM = 30; // set max 30 if constexpr math
	const int _3d_MAX_MULTIPOLE_NUM = 18; // set max 18 if constexpr math

} // detail

} // fmm