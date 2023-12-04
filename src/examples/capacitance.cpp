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

#include "mkl.h"

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
	uniform_real_distribution qdis(-1.0, 1.0);
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

namespace fmm {

	struct tri_mesh
	{
		tri_mesh() = default;
		tri_mesh(std::string filename)
		{
			std::ifstream input(filename);
			read(input);
		}

		std::vector<Vector3d> vertices;
		std::vector<std::array<size_t, 3>> elems;
		std::vector<double> areas;
		std::vector<Vector3d> centers;

		size_t n_vertices() const { return nv; }
		size_t n_elems() const { return ne; }

		void read(std::istream& stream)
		{
			stream >> nv;
			vertices.resize(nv);
			for (size_t i = 0; i < nv; ++i)
				stream >> vertices[i];

			stream >> ne;
			elems.resize(ne);
			for (size_t i = 0; i < ne; ++i)
				stream >> elems[i][0] >> elems[i][1] >> elems[i][2];
		}
		void precalc()
		{
			areas.resize(ne);
			centers.resize(ne);
			tbb::parallel_for(size_t(0), ne, [&](size_t e)
			{
				const auto [i, j, k] = elems[e];
				areas[e] = abs(cross(vertices[j] - vertices[i], vertices[k] - vertices[j])) / 2.0;
				centers[e] = (vertices[i] + vertices[j] + vertices[k]) / 3.0;
			});
		}

		bool is_adjacent(int elem1, int elem2) const
		{
			for (auto i : elems[elem1])
				for (auto j : elems[elem2])
					if (i == j) return true;
			return false;
		}

	private:
		size_t nv;
		size_t ne;
	};

	struct CapacitanceSolver
	{
		CapacitanceSolver(const tri_mesh& mesh_) : mesh(mesh_) 
		{

		}

		void AssembleMatrix()
		{
			const size_t ne = mesh.n_elems();
			M.reset(new double[ne * ne]);
			tbb::parallel_for(size_t(0), ne, [&](size_t i) {
				for (size_t j = 0; j < ne; ++j)
				{
					//if (mesh.is_adjacent(i, j))
					if (true)
						M[j * ne + i] = DirectEval(mesh.centers[i], j);
					else
						M[j * ne + i] = mesh.areas[j] / abs(mesh.centers[i] - mesh.centers[j]);
				}
			});
		}

		void Solve()
		{
			double t = omp_get_wtime();
			AssembleMatrix();
			std::cout << "assemble matrix time: " << omp_get_wtime() - t << std::endl;

			const int ne = mesh.n_elems();
			sol.reset(new double[ne]);
			std::fill(sol.get(), sol.get() + ne, 1.0);

			std::unique_ptr<int[]> ipiv(new int[ne]);
			t = omp_get_wtime();
			int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, ne, 1, M.get(), ne, ipiv.get(), sol.get(), ne);
			std::cout << "solve time: " << omp_get_wtime() - t << std::endl;

			double Q = 0.0;
			for (size_t i = 0; i < ne; ++i)
				Q += sol[i] * mesh.areas[i];
			std::cout << "Q total = " << Q << std::endl;
		}

		double DirectEval(const Vector3d& source, size_t elem)
		{
			const auto [i, j, k] = mesh.elems[elem];
			const auto& v1 = mesh.vertices[i];
			const auto& v2 = mesh.vertices[j];
			const auto& v3 = mesh.vertices[k];

			auto va = source - v1;
			auto vb = source - v2;
			auto vc = source - v3;
			
			double lva = abs(va);
			double lvb = abs(vb);
			double lvc = abs(vc);

			const auto ta = v3 - v2;
			const auto tb = v1 - v3;
			const auto tc = v2 - v1;

			double La = abs(ta);
			double Lb = abs(tb);
			double Lc = abs(tc);

			auto n = cross(ta, tb);
			n /= abs(n);
			va /= lva;
			vb /= lvb;
			vc /= lvc;

			const double fca = -dot(vc, ta) / La;
			const double fcb = dot(vc, tb) / Lb;
			const double fab = -dot(va, tb) / Lb;
			const double fac = dot(va, tc) / Lc;
			const double fbc = -dot(vb, tc) / Lc;
			const double fba = dot(vb, ta) / La;

			Vector3d Phi{ 0.0,0.0,0.0 };
			Vector3d temp;

			temp = cross(va, vb);
			if (abs(temp) > 1.e-7) Phi += temp * log(lva * (1.0 + fac) / lvb / (1.0 - fbc)) / (Lc * lvc);
			temp = cross(vb, vc);
			if (abs(temp) > 1.e-7) Phi += temp * log(lvb * (1.0 + fba) / lvc / (1.0 - fca)) / (La * lva);
			temp = cross(vc, va);
			if (abs(temp) > 1.e-7) Phi += temp * log(lvc * (1.0 + fcb) / lva / (1.0 - fab)) / (Lb * lvb);
			Phi *= (lva * lvb * lvc);
			const double Theta = 2 * atan2(dot(cross(va, vb), vc), 1 + dot(va, vb) + dot(vb, vc) + dot(vc, va));
			return dot(Phi - (source - mesh.centers[elem]) * Theta, n);
		}
	private:
		const tri_mesh& mesh;
		std::unique_ptr<double[]> M;
		std::unique_ptr<double[]> sol;
	};
}

int main(int argc, char* argv[])
{
#ifdef FMM_MPI
	MPI_Init(&argc, &argv);
#endif
	oneapi::tbb::global_control global_limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
	if (IAmRoot()) std::cout << "max threads = " << omp_get_max_threads() << std::endl;
	fmm::detail::InitMathConstants(fmm::detail::_3d_MAX_MULTIPOLE_NUM); // must call for 3d cpu fmm
	fmm::detail::cudaCopyMathConstants(); // must call for 3d gpu fmm

	/*fmm::tri_mesh mesh("sphere.txt");
	mesh.precalc();
	fmm::CapacitanceSolver csolver(mesh);
	csolver.Solve();*/

	// example with 2d fmm
	//{
	//	auto source_points = GenerateParticles(0, 1, 0, 1, 100000);
	//	auto target_points = GenerateParticles(-1, 2, -1, 2, 100000);
	//	for (auto& x : target_points) // target points must have q = 0
	//		x.q = 0;

	//	double t = omp_get_wtime();
	//	fmm::gpu::ComputeExact<fmm::InteractionType::Potential2d>(source_points, target_points); // or fmm::ComputeExact... for cpu version
	//	fmm::gpu::ComputeExact<fmm::InteractionType::Force2d>(source_points, target_points); // or fmm::ComputeExact... for cpu version
	//	if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
	//	t = omp_get_wtime();
	//	std::vector<double> p;
	//	std::vector<std::complex<double>> f;
	//	if (IAmRoot()) {
	//		fmm::gpu::FastMultipole fmm2(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
	//		p = fmm2.potentials;
	//		f = fmm2.forces;
	//	}
	//	fmm::FastMultipole fmm(source_points, target_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
	//	if (IAmRoot()) {
	//		fmm::ReadError<fmm::InteractionType::Force2d>(fmm.forces);
	//		fmm::ReadError<fmm::InteractionType::Force2d>(f);
	//		fmm::ReadError<fmm::InteractionType::Potential2d>(fmm.potentials);
	//		fmm::ReadError<fmm::InteractionType::Potential2d>(p);
	//		cout << "Total time = " << omp_get_wtime() - t << endl;
	//	}
	//}

	// example with 3d fmm, force = q*\vec{r}/r^3
	/*{
		auto source_points = GenerateParticles(0, 1, 0, 1, 0, 1, 100000);
		auto target_points = GenerateParticles(-1, 2, -1, 2, -1, 2, 100000);
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
	}*/

	{
		auto source_points = GenerateParticles(0, 1, 0, 1, 0, 1, 1000000);
		
		double t = omp_get_wtime();
		//fmm::gpu::ComputeExact<fmm::InteractionType::Force3d>(source_points);
		//fmm::gpu::ComputeExact<fmm::InteractionType::Potential3d>(source_points);
		if (IAmRoot()) cout << "Compute exact time = " << omp_get_wtime() - t << endl;
		t = omp_get_wtime();
		std::vector<double> p;
		std::vector<fmm::Vector3d> f;
		/*if (IAmRoot()) {
			fmm::gpu::FastMultipole3d fmm2(source_points, 1.e-8, fmm::FMM_AUTO, fmm::FMM_AUTO);
			p = fmm2.potentials;
			f = fmm2.forces;
		}*/
		fmm::FastMultipole3d fmm(source_points, 1.e-8, 10, fmm::FMM_AUTO);
		if (IAmRoot()) {
			fmm::ReadError<fmm::InteractionType::Force3d>(fmm.forces);
			//fmm::ReadError<fmm::InteractionType::Force3d>(f);
			fmm::ReadError<fmm::InteractionType::Potential3d>(fmm.potentials);
			//fmm::ReadError<fmm::InteractionType::Potential3d>(p);
			cout << "Total time = " << omp_get_wtime() - t << endl;
		}
	}

	// example with 3d fmm, computes only force = \vec{q} x \vec{r}/r^3
	/*{
		auto source_points = GenerateParticles3(0, 1, 0, 1, 0, 1, 100000);
		auto target_points = GenerateParticles3(-1, 2, -1, 2, -1, 2, 100000);
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
	}*/

	fmm::detail::cudaClearMathConstants();
#ifdef FMM_MPI
	MPI_Finalize();
#endif
	return 0;
}