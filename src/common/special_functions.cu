#include "special_functions.h"
#include <cuda_runtime.h>
#include <map>
#include <algorithm>

namespace fmm {

namespace detail {

    void cudaCopyMathConstants()
    {
#ifndef FMM_CONSTEXPR_MATH
        cudaMalloc(&dev_Knm, Knm.Knm.size() * sizeof(double));
        cudaMemcpy(dev_Knm, Knm.Knm.data(), Knm.Knm.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_Anm, Anm.Anm.size() * sizeof(double));
        cudaMemcpy(dev_Anm, Anm.Anm.data(), Anm.Anm.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&dev_m2lcoef, m2lcoef.m2lcoef.size() * sizeof(double));
        cudaMemcpy(dev_m2lcoef, m2lcoef.m2lcoef.data(), m2lcoef.m2lcoef.size() * sizeof(double), cudaMemcpyHostToDevice);
#endif // !FMM_CONSTEXPR_MATH

        size_t dm_size = dmatrix.size();
        cudaMalloc(&dev_dmatrix, dm_size * dmatrix.begin()->second.size() * sizeof(double));
        cudaMalloc(&dev_dm_map, dm_size * sizeof(int));

        std::vector<int> keys(dm_size), keymap(dm_size);
        for (int i = 0; const auto& [key, value] : dmatrix)
        {
            keys[i++] = key;
        }
        std::sort(keys.begin(), keys.end());
        for (int i = 0; i < dm_size; ++i)
        {
            keymap[i] = keys[i];
        }
        cudaMemcpy(dev_dm_map, keymap.data(), dm_size * sizeof(int), cudaMemcpyHostToDevice);
        for (int i = 0; const auto& key : keys)
        {
            const auto& value = dmatrix[key];
            cudaMemcpy(dev_dmatrix + i * value.size(), value.data(), value.size() * sizeof(double), cudaMemcpyHostToDevice);
            ++i;
        }
    }

    void cudaClearMathConstants()
    {
#ifndef FMM_CONSTEXPR_MATH
        cudaFree(&dev_Knm);
        cudaFree(&dev_Anm);
        cudaFree(&dev_m2lcoef);
#endif
        cudaFree(&dev_dm_map);
        cudaFree(&dev_dmatrix);
    }

} // detail

} // fmm