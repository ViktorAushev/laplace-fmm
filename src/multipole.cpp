#include "multipole.h"
#include "../common/simple_math.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include "omp.h"
#include <oneapi/tbb/parallel_for_each.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/blocked_range.h>
#include <cmath>
#include <fstream>
#include <ranges>
#include <Eigen/Dense>

namespace fmm {

using std::complex;
using std::vector;
using namespace std::complex_literals;

BinomNewton binom(100);

void FastMultipole::Multipole(std::complex<double>* a, const std::pair<size_t, size_t>& particle_range, const std::complex<double>& z0)
{
	auto p_begin = tree->particles.begin() + particle_range.first;
	auto p_end = tree->particles.begin() + particle_range.second;
	for (int i = 1; i < N; ++i)
	{
		auto temp = std::accumulate(p_begin, p_end, 0.0i, [&z0, i](const complex<double>& x, const particle2d& y)
			{
				return x + y.q * MyPow(y.center - z0, i);
			});
		a[i] = -temp / double(i);
	}

	a[0] = std::accumulate(p_begin, p_end, 0.0i, [](const complex<double>& x, const particle2d& y) { return x + y.q; });
}

void FastMultipole::Multipole2Particle(const std::complex<double>* a, double r, std::complex<double>* b, const Eigen::MatrixXcd& M)
{
	Eigen::VectorXcd A(N);
	for (int i = 0; i < N; ++i)
	{
		A[i] = a[i];
	}
	Eigen::VectorXcd B = M * A;
	for (int i = 0; i < N; ++i)
		b[i] = B[i];
}

void FastMultipole::Particle2Multipole(const std::complex<double>* a, const complex<double>& dz, double r, std::complex<double>* b)
{
	for (int i = 1; i < N; ++i)
	{
		std::complex<double> temp = 0.0;
		for (int j = 0; j < N; ++j)
		{
			temp += a[j] * MyPow(r, i) * MyPow(dz + std::complex<double>(cos(2.0 * std::numbers::pi * j / N), sin(2.0 * std::numbers::pi * j / N)), i);
		}
		b[i] += -temp / double(i);
	}

	b[0] += std::accumulate(a, a + N, 0.0i, [](const complex<double>& x, const complex<double>& y) { return x + y; });
}

void FastMultipole::M2M(const std::complex<double>* a, const complex<double>& z0, std::complex<double>* b, vector<complex<double>>& work)
{
	work[0] = 1.0;
	for (int i = 1; i < N; ++i)
		work[i] = work[i - 1] * z0;

	for (int j = 1; j < N; ++j)
	{
		for (int i = 1; i <= j; ++i)
		{
			b[j] += a[i] * work[j - i] * binom(j - 1, i - 1);
		}
		b[j] -= a[0] * work[j] / double(j);
	}
	b[0] += a[0];
}

inline std::complex<double> divComp(const std::complex<double>& a, const std::complex<double>& b)
{
	const double& ax = a.real();
	const double& bx = b.real();
	const double& ay = a.imag();
	const double& by = b.imag();

	double zn = bx * bx + by * by;
	return { (ax * bx + ay * by) / zn, (ay * bx - ax * by) / zn };
}

void FastMultipole::M2L(const std::complex<double>* a, const complex<double>& z0, std::complex<double>* b, vector<complex<double>>& work)
{
	work[0] = a[0];
	b[0] += a[0] * std::conj(log(-z0));
	for (int i = 1; i < N; ++i)
	{
		work[i] = ni(i) * a[i] / MyPow(z0, i);
		b[0] += work[i];
	}
	for (int i = 1; i < N; ++i)
	{
		complex<double> tmp{ 0.0 };
		for (int j = 1; j < N; ++j)
		{
			tmp += work[j] * double(binom(i + j - 1, j - 1));
		}
		b[i] += divComp(-a[0] / double(i) + tmp, MyPow(z0, i));
	}
}

void FastMultipole::L2L(const std::complex<double>* a, const complex<double>& z0, std::complex<double>* b)
{
	for (int i = 0; i < N; ++i)
		b[i] = a[i];
	for (int j = 0; j < N; ++j)
	{
		for (int k = N - j - 1; k < N - 1; ++k)
		{
			b[k] -= z0 * b[k + 1];
		}
	}
}

void FastMultipole::Upward()
{
	double t1 = omp_get_wtime();

	auto& leaves = tree->levels.back();
	auto& leaves_outer = outer_expansions.back();
	tbb::parallel_for(size_t(0), tree->level_sizes[tree_depth - 1], [&](size_t i) {
		TreeCell2d& cell = leaves[i];
		Multipole(leaves_outer.data() + N * i, cell.children_range, cell.center);
	});

	std::cout << "multipole time: " << omp_get_wtime() - t1 << std::endl;

	for (int i = 0; i < tree->level_sizes[tree_depth - 1] * N; ++i)
	{
		std::cout << leaves_outer[i] << std::endl;
	}
	t1 = omp_get_wtime();
	Eigen::MatrixXcd A(N,N);
	double radius = 1 * tree->cell_size(tree_depth);
	double rn = radius;
	for (int i = 1; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			A(i, j) = -rn * std::complex<double>(cos(2.0 * std::numbers::pi * i * j / N), sin(2.0 * std::numbers::pi * i * j / N)) / double(i);
		}
		rn *= radius;
	}
	for (int j = 0; j < N; ++j)
	{
		A(0, j) = 1;
	}
	Eigen::MatrixXcd B = A.inverse();

	auto& pseudo_outer = particles_expansions.back();
	tbb::parallel_for(size_t(0), tree->level_sizes[tree_depth - 1], [&](size_t i) {
		Multipole2Particle(leaves_outer.data() + N * i, radius, pseudo_outer.data() + N * i, B);
		});
	std::cout << "M2P time: " << omp_get_wtime() - t1 << std::endl;
	
	for (int i = 0; i < tree->level_sizes[tree_depth - 1] * N; ++i)
	{
		leaves_outer[i] = 0;
	}

	tbb::parallel_for(size_t(0), tree->level_sizes[tree_depth - 1], [&](size_t i) {
		Particle2Multipole(pseudo_outer.data() + N * i, {0,0}, radius, leaves_outer.data() + N * i);
		});
	for (int i = 0; i < tree->level_sizes[tree_depth - 1] * N; ++i)
	{
		std::cout << leaves_outer[i] << std::endl;
	}
	t1 = omp_get_wtime();

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = tree->levels[i];
		auto& top_outer = outer_expansions[i];
		const auto& bottom_level = tree->levels[i + 1];
		const auto& bottom_outer = outer_expansions[i + 1];

		tbb::parallel_for(size_t(0), tree->level_sizes[i], [&](size_t j)
			{
				TreeCell2d& cell = top_level[j];
				auto& work = local_work.local();
				const auto& [begin, end] = cell.children_range;

				for (int k = begin; k < end; ++k)
				{
					const auto& children_cell = bottom_level[k];
					M2M(bottom_outer.data() + k * N, children_cell.center - cell.center, top_outer.data() + j * N, work);
				}
			});
	}
	std::cout << "----------" << std::endl;

	for (int i = 0; i < tree->level_sizes[0] * N; ++i)
	{
		std::cout << outer_expansions[0][i] << std::endl;
		outer_expansions[0][i] = 0;
	}

	std::cout << "m2m time: " << omp_get_wtime() - t1 << std::endl;

	for (int i = tree_depth - 2; i >= 0; --i)
	{
		auto& top_level = tree->levels[i];
		auto& top_outer = outer_expansions[i];
		const auto& bottom_level = tree->levels[i + 1];
		const auto& bottom_outer = outer_expansions[i + 1];

		for (size_t j = 0; j < tree->level_sizes[i]; ++j)
		{
				TreeCell2d& cell = top_level[j];
				const auto& [begin, end] = cell.children_range;

				for (int k = begin; k < end; ++k)
				{
					const auto& children_cell = bottom_level[k];
					Particle2Multipole(pseudo_outer.data() + k * N, children_cell.center - cell.center, radius, top_outer.data() + j * N);
				}
		}
	}
	for (int i = 0; i < tree->level_sizes[0] * N; ++i)
	{
		std::cout << outer_expansions[0][i] << std::endl;
	}
	std::cout << "----------" << std::endl;
}

void FastMultipole::Downward()
{
	for (int i = 2; i < tree_depth; ++i)
	{
		const auto& top_level = tree->levels[i - 1];
		const auto& top_inner = inner_expansions[i - 1];
		auto& bottom_level = tree->levels[i];
		auto& bottom_inner = inner_expansions[i];
		const auto& bottom_outer = outer_expansions[i];


		tbb::parallel_for(size_t(0), tree->level_sizes[i], [&](size_t j)
			{
				TreeCell2d& cell = bottom_level[j];
				auto& work = local_work.local();
				auto& parent_cell = top_level[cell.parent];
				L2L(top_inner.data() + N * cell.parent, parent_cell.center - cell.center, bottom_inner.data() + N * j);

				for (const auto& idx : cell.farneighbours)
				{
					const auto& far_neighbour = bottom_level[idx];
					M2L(bottom_outer.data() + idx * N, far_neighbour.center - cell.center, bottom_inner.data() + N * j, work);
				}
			});
	}
}

void FastMultipole::ComputePotentials()
{
	 tbb::enumerable_thread_specific<std::complex<double>> dz_local;
	 tbb::enumerable_thread_specific<double> phi_local;

	 const auto& leaves = tree->levels.back();
	 const auto& leaves_inner = inner_expansions.back();
	 auto& particles = tree->particles;
	 tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i) {
	 	auto& dz = dz_local.local();
	 	auto& phi = phi_local.local();
	 	const auto& cell = leaves[i];
	 	const auto& inner = leaves_inner.data() + i * N;

	 	const auto& [p1_begin, p1_end] = cell.children_range;
	 	for (const auto& idx : cell.closeneighbours)
	 	{
	 		const auto& neighbour_cell = leaves[idx];
	 		const auto& [p2_begin, p2_end] = neighbour_cell.children_range;
	 		for (int j = p1_begin; j < p1_end; ++j)
	 		{
	 			auto& p1 = particles[j];
	 			auto& phi1 = potentials[j];
	 			for (int k = p2_begin; k < p2_end; ++k)
	 			{
					phi1 += Potential2d(p1, particles[k]);
	 			}
	 		}
	 	}

	 	for (int j = p1_begin; j < p1_end; ++j)
	 	{
	 		auto& particle = particles[j];
	 		phi = 0.0;
	 		dz = particle.center - cell.center;
	 		complex<double> dzpow{ 1.0 };
	 		for (int k = 0; k < N; ++k)
	 		{
	 			phi += inner[k].real() * dzpow.real() - inner[k].imag() * dzpow.imag();
	 			dzpow *= dz;
	 		}
	 		potentials[j] += phi;
	 	}
	 	});
}

void FastMultipole::ComputeForces()
{
	tbb::enumerable_thread_specific<std::complex<double>> dz_local;
	tbb::enumerable_thread_specific<std::complex<double>> force_local;

	const auto& leaves = tree->levels.back();
	const auto& leaves_inner = inner_expansions.back();
	auto& particles = tree->particles;
	tbb::parallel_for(size_t(0), tree->level_sizes.back(), [&](size_t i) {
		auto& dz = dz_local.local();
		auto& force = force_local.local();
		const auto& cell = leaves[i];
		const auto& inner = leaves_inner.data() + i * N;
		const auto& [p1_begin, p1_end] = cell.children_range;
		for (const auto& idx : cell.closeneighbours)
		{
			const auto& neighbour_cell = leaves[idx];
			const auto& [p2_begin, p2_end] = neighbour_cell.children_range;
			for (int j = p1_begin; j < p1_end; ++j)
			{
				auto& p1 = particles[j];
				auto& force1 = forces[j];
				for (int k = p2_begin; k < p2_end; ++k)
				{
					force1 += Force2d(p1, particles[k]);
				}
			}
		}

		for (int j = p1_begin; j < p1_end; ++j)
		{
			auto& particle = particles[j];
			force = 0.0;
			dz = particle.center - cell.center;
			complex<double> dzpow{ 1.0 };
			for (int k = 1; k < N; ++k)
			{
				force += double(k) * inner[k] * dzpow;
				dzpow *= dz;
			}
			forces[j] += force;
		}
		});
}

FastMultipole::FastMultipole(std::vector<particle2d>& particles, double eps)
{
	double T = omp_get_wtime();
	std::cout << "\n***************** Start FMM **************" << std::endl;
	num_particles = particles.size();
	N = 2;// -log(eps);
	std::cout << "multipole num = " << N << std::endl;
	tree_depth = 2;// std::max(3.0, 2 + log(double(num_particles) / 100.0) / log(4.0));
	std::cout << "tree depth = " << tree_depth << std::endl;
	
	tree = std::make_shared<MortonTree2d>(particles, tree_depth);

	double t = omp_get_wtime();
	outer_expansions.resize(tree_depth);
	inner_expansions.resize(tree_depth);
	particles_expansions.resize(tree_depth);
	for (int i = 0; i < tree_depth; ++i)
	{
		auto& outer = outer_expansions[i];
		auto& inner = inner_expansions[i];
		auto& pseudo_particles = particles_expansions[i];
		outer.resize(tree->level_sizes[i] * N);
		inner.resize(tree->level_sizes[i] * N);
		pseudo_particles.resize(tree->level_sizes[i] * N);
	}
	forces.resize(num_particles);
	potentials.resize(num_particles);
	std::cout << "allocation time: " << omp_get_wtime() - t << std::endl;
	local_work = tbb::enumerable_thread_specific<vector<complex<double>>>{ vector<complex<double>>(N) };

	Upward();
	
	t = omp_get_wtime();
	Downward();
	std::cout << "m2l+l2l time: " << omp_get_wtime() - t << std::endl;

	t = omp_get_wtime();
	ComputePotentials();
	//ComputeError<fmm::InteractionType::Potential>();
	MapSort(potentials, tree->positions_map);
	std::cout << "Leaf potential time: " << omp_get_wtime() - t << std::endl;
	
	t = omp_get_wtime();
	ComputeForces();
	//ComputeError<fmm::InteractionType::Force>();
	MapSort(forces, tree->positions_map);
	std::cout << "leaf force time: " << omp_get_wtime() - t << std::endl;

	std::cout << "total fmm time: " << omp_get_wtime() - T << std::endl;
	std::cout << "***************** End FMM **************\n" << std::endl;
}

} // fmm