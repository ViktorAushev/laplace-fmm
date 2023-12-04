#pragma once
#include "morton_tree.h"
#include <memory>

namespace fmm::gpu {

template <typename value>
class FastMultipole3d
{
public:
	using complex_value = std::conditional_t<std::is_same_v<value, double>, cuda_complex, cuda_Vector3cd>;
	using particle_t = particle<Vector3d, value>;
	using MortonTree_t = MortonTree<3, point3d, point3d, value>;
	using TreeCell_t = typename MortonTree_t::TreeCell_t;

	FastMultipole3d(const std::vector<fmm::particle<point3d, value>>& particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	FastMultipole3d(const std::vector<fmm::particle<point3d, value>>& source_particles, const std::vector<fmm::particle<point3d, value>>& target_particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	~FastMultipole3d();

	std::vector<Vector3d> forces;
	std::vector<double> potentials;
private:
	void Solve(const std::vector<fmm::particle<point3d, value>>& particles, double eps = 1.e-8, int N = FMM_AUTO, int tree_depth = FMM_AUTO);
	void Upward();
	void Downward();

	void ComputeLeaves();

	std::shared_ptr<MortonTree_t> tree;
	std::vector<complex_value*> outer_expansions; // for all tree levels
	std::vector<complex_value*> inner_expansions; // for all tree levels
	Vector3d* dev_forces;
	double* dev_potentials;

	//double* m2lcoef;

	size_t tree_depth;
	int N; // multipole_num
	int Nx2; // multipole_num
	size_t num_particles;
	size_t targets_num;
};

} // fmm