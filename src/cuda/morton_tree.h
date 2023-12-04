#pragma once
#include <utility>
#include <vector>
#include "../common/utils.h"
#include "../common/cuda_utils.h"
#include <iostream>

namespace fmm::gpu {

template <typename point, typename value>
struct TreeCell
{
	using particle_t = particle<point, value>;

	__DEVICE__ __HOST__ TreeCell& operator=(const particle_t& particle);
	__DEVICE__ __HOST__ TreeCell& operator=(const TreeCell& cell);

	point center;

	std::pair<size_t, size_t> source_range;
	std::pair<size_t, size_t> target_range;
	size_t parent;
	size_t morton_code;
};

template <int dim, typename point, typename host_point, typename value>
class MortonTree
{
public:
	using particle_t = particle<point, value>;
	using TreeCell_t = TreeCell<point, value>;

	MortonTree(const std::vector<fmm::particle<host_point, value>>& particles, size_t tree_depth);
	~MortonTree();
	std::vector<TreeCell_t*> dev_levels;
	std::vector<size_t*> dev_closeneighbours;
	std::vector<size_t*> dev_closeneighbours_sizes;
	std::vector<size_t*> dev_farneighbours;
	std::vector<size_t*> dev_farneighbours_sizes;

	std::vector<size_t> level_sizes;
	size_t tree_depth;
	particle_t* dev_particles;
	size_t num_particles;
	size_t targets_num;
	size_t* dev_positions_map;

private:
	point* shift;
	double* scale;

};

using TreeCell2d = TreeCell<point2d, double>;
using TreeCell3d = TreeCell<point3d, double>;
using TreeCell3d3 = TreeCell<point3d, point3d>;
using MortonTree2d = MortonTree<2, cuda_complex, fmm::point2d, double>;
using MortonTree3d = MortonTree<3, point3d, fmm::point3d, double>;
using MortonTree3d3 = MortonTree<3, point3d, fmm::point3d, point3d>;

} // fmm::cuda