#pragma once
#include <cuda/std/complex>
#include <fstream>
#include "utils.h"

namespace fmm::gpu
{
    const int BLOCK_SIZE = 32;

    using cuda_complex = cuda::std::complex<double>;

    inline std::ostream& operator<<(std::ostream& ss, const cuda_complex& val)
    {
        ss << "(" << val.real() << ", " << val.imag() << ")";
        return ss;
    }

    template <typename point, typename value>
    struct particle
    {
        point center;
        value q;
        size_t morton_code;
    };

    using point2d = cuda_complex;
    using point3d = Vector3d;
    using particle2d = particle<point2d, double>;
    using particle3d = particle<point3d, double>;
    using particle3d3 = particle<point3d, point3d>;
} // fmm::gpu


