#include <iostream>
#include <random>
#include <fstream>
#include "omp.h"
#include "../cpu/multipole.h"
#include "../cpu/multipole3d.h"
#include "../cuda/multipole.h"
#include "../cuda/multipole3d.h"
#include <cmath>
#include <oneapi/tbb/global_control.h>
#include "../cpu/morton_tree.h"
#include "../cuda/morton_tree.h"
#include <ranges>
#include <numbers>

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_for_each.h>
#include "../common/special_functions.h"
#include "../common/mpi_utils.h"
#include "../common/avx.h"

using namespace std;
using fmm::IAmRoot;

//points in [xmin,xmax]x[ymin,ymax] with particles in [-1.0;1.0]
vector<fmm::particle2d> GenerateParticles(double xmin, double xmax, double ymin, double ymax, int N)
{
	random_device rd;  
	mt19937 gen(rd()); 
	uniform_real_distribution xdis(xmin, xmax);
	uniform_real_distribution ydis(ymin, ymax);
	uniform_real_distribution qdis(-1.0, 1.0);
	vector<fmm::particle2d> res(N);
	if (IAmRoot())
	{
		tbb::parallel_for_each(res.begin(), res.end(), [&](auto& x) {
			x = { xdis(gen) + 1.0i * ydis(gen), qdis(gen) };
			});
	}
#ifdef FMM_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(res.data(), N * sizeof(fmm::particle2d), MPI_BYTE, fmm::RootID(), MPI_COMM_WORLD);
#endif
	return res;
}

vector<fmm::particle3d> GenerateParticles(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int N)
{
	random_device rd;  
	mt19937 gen(rd()); 
	uniform_real_distribution xdis(xmin, xmax);
	uniform_real_distribution ydis(ymin, ymax);
	uniform_real_distribution zdis(zmin, zmax);
	uniform_real_distribution qdis(0.0, 1.0);
	vector<fmm::particle3d> res(N);
	if (IAmRoot())
	{
		tbb::parallel_for_each(res.begin(), res.end(), [&](auto& x) {
			x = { {xdis(gen), ydis(gen), zdis(gen)}, qdis(gen) };
			});
	}
#ifdef FMM_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(res.data(), N * sizeof(fmm::particle3d), MPI_BYTE, fmm::RootID(), MPI_COMM_WORLD);
#endif
	return res;
}

vector<fmm::particle3d3> GenerateParticles3(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int N)
{
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution xdis(xmin, xmax);
	uniform_real_distribution ydis(ymin, ymax);
	uniform_real_distribution zdis(zmin, zmax);
	uniform_real_distribution qdis(-1.0, 1.0);
	vector<fmm::particle3d3> res(N);
	if (IAmRoot())
	{
		tbb::parallel_for_each(res.begin(), res.end(), [&](auto& x) {
			x = { {xdis(gen), ydis(gen), zdis(gen)}, {qdis(gen), qdis(gen), qdis(gen)} };
			});
	}
#ifdef FMM_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(res.data(), N * sizeof(fmm::particle3d3), MPI_BYTE, fmm::RootID(), MPI_COMM_WORLD);
#endif
	return res;
}

vector<fmm::particle3d> GenerateParticlesD()
{
	vector<fmm::particle3d> res;
	//if (IAmRoot())
	//{
	//	for (int i = 0; i < 3; ++i)
	//		for (int j = 0; j < 3; ++j)
	//			for (int k = 0; k < 2; ++k)
	//				res.push_back({ {i * 0.1, j * 0.1, k * 0.1}, double(1 + i + j + k) });
	//}
	double h = 0.01;
	for (int i = 1; i < 40; ++i)
		for (int j = 1; j < 40; ++j)
			for (int k = 1; k < 40; ++k)
				res.push_back({ { i * h,j * h,k * h },double(1 + i + j + k) });
	return res;
}

int main(int argc, char* argv[])
{
#ifdef FMM_MPI
	MPI_Init(&argc, &argv);
#endif
	//oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
	if (IAmRoot()) std::cout << "max threads = " << omp_get_max_threads() << std::endl;
	if (IAmRoot()) std::cout << "vec length = " << fmm::detail::avx_vec_length << std::endl;

	fmm::detail::InitMathConstants(fmm::detail::_3d_MAX_MULTIPOLE_NUM); // must call for 3d cpu fmm
	fmm::detail::cudaCopyMathConstants(); // must call for 3d gpu fmm

	// example with 2d fmm
	{
		auto source_points = GenerateParticles(0, 1, 0, 1, 100'000);
		auto target_points = GenerateParticles(-1, 2, -1, 2, 100'000);
		for (auto& x : target_points) // target points must have q = 0
			x.q = 0;

		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Potential2d>(source_points, target_points); // or fmm::ComputeExact... for cpu version
		fmm::gpu::ComputeExact<fmm::InteractionType::Force2d>(source_points, target_points); // or fmm::ComputeExact... for cpu version
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<double> p;
		std::vector<std::complex<double>> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole fmm2(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			p = fmm2.potentials;
			f = fmm2.forces;
		}
		fmm::FastMultipole fmm(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force2d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force2d>(f);
			fmm::ReadError<fmm::InteractionType::Potential2d>(fmm.potentials);
			fmm::ReadError<fmm::InteractionType::Potential2d>(p);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	{
		auto source_points = GenerateParticles(0, 1, 0, 1, 100'000);
		
		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Potential2d>(source_points); // or fmm::ComputeExact... for cpu version
		fmm::gpu::ComputeExact<fmm::InteractionType::Force2d>(source_points); // or fmm::ComputeExact... for cpu version
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<double> p;
		std::vector<std::complex<double>> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole fmm2(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			p = fmm2.potentials;
			f = fmm2.forces;
		}
		fmm::FastMultipole fmm(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force2d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force2d>(f);
			fmm::ReadError<fmm::InteractionType::Potential2d>(fmm.potentials);
			fmm::ReadError<fmm::InteractionType::Potential2d>(p);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	//// example with 3d fmm, force = q*\vec{r}/r^3
	{
		auto source_points = GenerateParticles(0, 1, 0, 1, 0, 1, 100'000);
		auto target_points = GenerateParticles(-1, 2, -1, 2, -1, 2, 100'000);
		for (auto& x : target_points)
			x.q = 0;

		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Force3d>(source_points, target_points);
		fmm::gpu::ComputeExact<fmm::InteractionType::Potential3d>(source_points, target_points);
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<double> p;
		std::vector<fmm::Vector3d> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole3d fmm2(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			p = fmm2.potentials;
			f = fmm2.forces;
		}
		fmm::FastMultipole3d fmm(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force3d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force3d>(f);
			fmm::ReadError<fmm::InteractionType::Potential3d>(fmm.potentials);
			fmm::ReadError<fmm::InteractionType::Potential3d>(p);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	{
		auto source_points = GenerateParticles(0, 1, 0, 1, 0, 1, 100'000);
		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Force3d>(source_points);
		fmm::gpu::ComputeExact<fmm::InteractionType::Potential3d>(source_points);
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<double> p;
		std::vector<fmm::Vector3d> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole3d fmm2(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			p = fmm2.potentials;
			f = fmm2.forces;
		}
		fmm::FastMultipole3d fmm(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force3d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force3d>(f);
			fmm::ReadError<fmm::InteractionType::Potential3d>(fmm.potentials);
			fmm::ReadError<fmm::InteractionType::Potential3d>(p);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	// example with 3d fmm, computes only force = \vec{q} x \vec{r}/r^3
	{
		auto source_points = GenerateParticles3(0, 1, 0, 1, 0, 1, 100'000);
		auto target_points = GenerateParticles3(-1, 2, -1, 2, -1, 2, 100'000);
		for (auto& x : target_points)
			x.q = { 0,0,0 };

		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Force3d>(source_points, target_points);
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<fmm::Vector3d> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole3d fmm2(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			f = fmm2.forces;
		}
		fmm::FastMultipole3d fmm(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force3d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force3d>(f);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	{
		auto source_points = GenerateParticles3(0, 1, 0, 1, 0, 1, 100'000);

		double t = omp_get_wtime();
		fmm::gpu::ComputeExact<fmm::InteractionType::Force3d>(source_points);
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<fmm::Vector3d> f;
		if (IAmRoot()) {
			fmm::gpu::FastMultipole3d fmm2(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			f = fmm2.forces;
		}
		fmm::FastMultipole3d fmm(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force3d>(fmm.forces);
			fmm::ReadError<fmm::InteractionType::Force3d>(f);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	fmm::detail::cudaClearMathConstants();
#ifdef FMM_MPI
	MPI_Finalize();
#endif
	return 0;
}