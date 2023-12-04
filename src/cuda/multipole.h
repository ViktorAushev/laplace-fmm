#pragma once

#include "morton_tree.h"
#include <memory>

namespace fmm::gpu {

class FastMultipole
{
public:
	FastMultipole(const std::vector<fmm::particle2d>& particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	FastMultipole(const std::vector<fmm::particle2d>& source_particles, const std::vector<fmm::particle2d>& target_particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	
	~FastMultipole();
	
	std::vector<std::complex<double>> forces;
	std::vector<double> potentials;
private:
	void Solve(const std::vector<fmm::particle2d>& particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	void Upward();
	void Downward();

	void ComputeForces();
	void ComputePotentials();

	std::shared_ptr<MortonTree2d> tree;
	std::vector<cuda_complex*> outer_expansions; // for all tree levels
	std::vector<cuda_complex*> inner_expansions; // for all tree levels
	cuda_complex* dev_forces;
	double* dev_potentials;

	size_t tree_depth;
	int N; // multipole_num
	size_t num_particles;
	size_t targets_num;
};

} // fmm