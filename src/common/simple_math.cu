#include "simple_math.h"
#include "mpi_utils.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

namespace fmm::gpu::detail {

template <typename interaction_type, typename func, typename particle_t>
__global__ void compute_exact(particle_t* source_particles, size_t num_sources, particle_t* target_particles, interaction_type* res,  func foo, size_t local_size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < local_size)
	{
		auto p1 = target_particles[idx];
		interaction_type val = res[idx];
		for (size_t i = 0; i < num_sources; ++i)
		{
			val += foo(p1, source_particles[i]);
		}
		res[idx] = val;
	}
}

template <InteractionType it, typename interaction_type, typename point_type, typename value_type, typename gpu_interaction_type, typename gpu_point_type>
void ComputeExact<it, interaction_type, point_type, value_type, gpu_interaction_type, gpu_point_type>::Compute(const std::vector<fmm::particle<point_type, value_type>>& source_particles, const std::vector<fmm::particle<point_type, value_type>>& target_particles)
{
	size_t num_sources = source_particles.size();
	size_t num_targets = target_particles.size();
#ifdef FMM_MPI
	auto [shift, end_part] = LocalPart(0, num_targets);
	size_t local_size = end_part - shift;
#else
	size_t shift = 0, local_size = num_targets;
#endif
	std::vector<interaction_type> exact(local_size);
	gpu::particle<gpu_point_type, value_type>* dev_source_particles;
	gpu::particle<gpu_point_type, value_type>* dev_target_particles;
	gpu_interaction_type* dev_exact;
	cudaMalloc(&dev_exact, local_size * sizeof(gpu_interaction_type));
	cudaMemset(&dev_exact, 0, local_size * sizeof(gpu_interaction_type));
	cudaMalloc(&dev_source_particles, num_sources * sizeof(gpu::particle<gpu_point_type, value_type>));
	cudaMalloc(&dev_target_particles, local_size * sizeof(gpu::particle<gpu_point_type, value_type>));
	cudaMemcpy(dev_source_particles, source_particles.data(), num_sources * sizeof(gpu::particle<gpu_point_type, value_type>), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_target_particles, target_particles.data() + shift, local_size * sizeof(gpu::particle<gpu_point_type, value_type>), cudaMemcpyHostToDevice);

	const size_t num_blocks = (local_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
	if constexpr (it == fmm::InteractionType::Potential2d)
	{
		auto foo = [] __device__(const gpu::particle<gpu_point_type, value_type>&p1, const gpu::particle<gpu_point_type, value_type>&p2) { return Potential2d(p1, p2); };
		compute_exact << <num_blocks, BLOCK_SIZE >> > (dev_source_particles, num_sources, dev_target_particles, dev_exact, foo, local_size);
	}
	if constexpr (it == fmm::InteractionType::Force2d)
	{
		auto foo = [] __device__(const gpu::particle<gpu_point_type, value_type>&p1, const gpu::particle<gpu_point_type, value_type>&p2) { return Force2d(p1, p2); };
		compute_exact << <num_blocks, BLOCK_SIZE >> > (dev_source_particles, num_sources, dev_target_particles, dev_exact, foo, local_size);
	}
	if constexpr (it == fmm::InteractionType::Potential3d)
	{
		auto foo = [] __device__(const gpu::particle<gpu_point_type, value_type>&p1, const gpu::particle<gpu_point_type, value_type>&p2) { return Potential3d(p1, p2); };
		compute_exact << <num_blocks, BLOCK_SIZE >> > (dev_source_particles, num_sources, dev_target_particles, dev_exact, foo, local_size);
	}
	if constexpr (it == fmm::InteractionType::Force3d)
	{
		auto foo = [] __device__(const gpu::particle<gpu_point_type, value_type>&p1, const gpu::particle<gpu_point_type, value_type>&p2) { return Force3d(p1, p2); };
		compute_exact << <num_blocks, BLOCK_SIZE >> > (dev_source_particles, num_sources, dev_target_particles, dev_exact, foo, local_size);
	}

	cudaMemcpy(exact.data(), dev_exact, local_size * sizeof(interaction_type), cudaMemcpyDeviceToHost);
	cudaFree(&dev_source_particles);
	cudaFree(&dev_target_particles);
	cudaFree(&dev_exact);

#ifdef FMM_MPI
	std::vector<int> sizes(NProc());
	std::vector<int> displs(NProc());
	std::vector<interaction_type> buf(num_targets);
	for (int j = 0; j < NProc(); ++j)
	{
		sizes[j] = LocalPart(0, num_targets, j);
	}
	for (int j = 1; j < NProc(); ++j)
	{
		displs[j] = displs[j - 1] + sizes[j - 1];
	}
	if constexpr (it == fmm::InteractionType::Potential2d || it == fmm::InteractionType::Potential3d)
		MPI_Allgatherv(exact.data(), local_size, MPI_DOUBLE,
			buf.data(), sizes.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
	if constexpr (it == fmm::InteractionType::Force2d)
		MPI_Allgatherv(exact.data(), local_size, MPI_COMPLEX16,
			buf.data(), sizes.data(), displs.data(), MPI_COMPLEX16, MPI_COMM_WORLD);
	if constexpr (it == fmm::InteractionType::Force3d)
	{
		MPI_Datatype MPI_VECTOR3;
		MPI_Type_contiguous(3, MPI_DOUBLE, &MPI_VECTOR3);
		MPI_Type_commit(&MPI_VECTOR3);
		MPI_Allgatherv(exact.data(), local_size, MPI_VECTOR3,
			buf.data(), sizes.data(), displs.data(), MPI_VECTOR3, MPI_COMM_WORLD);
	}
#endif

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

template class ComputeExact<InteractionType::Potential2d, double, fmm::point2d, double, double, gpu::point2d>;
template class ComputeExact<InteractionType::Force2d, fmm::point2d, fmm::point2d, double, gpu::point2d, gpu::point2d>;
template class ComputeExact<InteractionType::Potential3d, double, fmm::point3d, double, double, gpu::point3d>;
template class ComputeExact<InteractionType::Force3d, fmm::point3d, fmm::point3d, double, gpu::point3d, gpu::point3d>;
template class ComputeExact<InteractionType::Force3d, fmm::point3d, fmm::point3d, fmm::point3d, gpu::point3d, gpu::point3d>;
} // fmm::gpu::detail