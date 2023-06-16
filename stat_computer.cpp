#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "ft.hpp"
#include <random>
#include <thread>
#include "stochastic_gaussian.hpp"
#include "sequential_gaussian_1.hpp"

constexpr size_t n_chunks = 8;

class Statistics{
public:
	Statistics(const PhysicalSpace& space):
		_space(space), _n(0),
		_rxx(space.N3, 0),
		_rxy(space.N3, 0),
		_rxz(space.N3, 0),
		_ryy(space.N3, 0),
		_ryz(space.N3, 0),
		_rzz(space.N3, 0){}

	void add_velocity(
		const std::vector<double>& ux,
		const std::vector<double>& uy={},
		const std::vector<double>& uz={});

	void save_r(const std::string& filename)const;
private:
	PhysicalSpace _space;
	size_t _n = 0;
	std::vector<double> _rxx, _rxy, _rxz, _ryy, _ryz, _rzz;

	std::vector<double> compute_r(const std::vector<double>& u, const std::vector<double>& v) const;
	void add_to_r(const std::vector<double>& r, std::vector<double>& result);
};


std::vector<double> Statistics::compute_r(const std::vector<double>& u, const std::vector<double>& v) const{
	size_t center_index = _space.lin_index(_space.N/2, _space.N/2, _space.N/2);
	double u0 = u[center_index];

	std::vector<double> ret(v);
	for (double& v: ret) v*= u0;

	return ret;
}

void Statistics::add_to_r(const std::vector<double>& r, std::vector<double>& result){
	for (size_t i=0; i<r.size(); ++i){
		result[i] = (result[i] * _n + r[i])/(_n+1);
	}
};

void Statistics::add_velocity(
		const std::vector<double>& ux,
		const std::vector<double>& uy,
		const std::vector<double>& uz){
	// R11
	if (ux.size() > 0) {
		std::vector<double> r = compute_r(ux, ux);
		add_to_r(r, _rxx);
	}
	// R12
	if (ux.size() > 0 && uy.size() > 0) {
		std::vector<double> r = compute_r(ux, uy);
		add_to_r(r, _rxy);
	}
	// R13
	if (ux.size() > 0 && uz.size() > 0) {
		std::vector<double> r = compute_r(ux, uz);
		add_to_r(r, _rxz);
	}
	// R22
	if (uy.size() > 0) {
		std::vector<double> r = compute_r(uy, uy);
		add_to_r(r, _ryy);
	}
	// R23
	if (uy.size() > 0 && uz.size() > 0) {
		std::vector<double> r = compute_r(uy, uz);
		add_to_r(r, _ryz);
	}
	// R33
	if (uz.size() > 0) {
		std::vector<double> r = compute_r(uz, uz);
		add_to_r(r, _rzz);
	}
	_n += 1;
}

void Statistics::save_r(const std::string& filename) const {
	_space.tovtk(filename,
	             {"r11", "r12", "r13", "r22", "r23", "r33"},
	             {&_rxx, &_rxy, &_rxz, &_ryy, &_ryz, &_rzz});
}



// Given energy spectrum
double E(double kappa){
	double logkappa = log10(kappa);
	double logE;
	if (logkappa < 0.0){
		logE = 2 * logkappa - 1;
	} else if (logkappa < 3.0){
		logE = -5.0/3.0 * logkappa - 1;
	} else {
		logE = -3 * logkappa + 3;
	}
	return std::pow(10, logE);
};

// Velocity spectrum tensor
double Phi(int i, int j, double kappa_x, double kappa_y, double kappa_z){
	double kappa2 = kappa_x * kappa_x + kappa_y * kappa_y + kappa_z * kappa_z;
	if (kappa2 == 0) return 0;
	double kappa = std::sqrt(kappa2);
	double e = E(kappa);
	double delta = (i == j) ? 1 : 0;
	double kappa_i = i == 1 ? kappa_x : ( i == 2 ? kappa_y : kappa_z);
	double kappa_j = j == 1 ? kappa_x : ( j == 2 ? kappa_y : kappa_z);
	return e / (4 * M_PI * kappa2) * (delta - kappa_i * kappa_j / kappa2);
};

void gaussian1(size_t n_tries){
	// 1. build correlations and variance
	std::cout << "Computing spatial correlations" << std::endl;
	size_t N = 51;
	double L = 20;
	PhysicalSpace ps(N, L);
	FourierSpace fs = ps.fourier_space();

	std::vector<double> phi11(N*N*N, 0.0);
	for (int k=0; k<N; ++k)
	for (int j=0; j<N; ++j)
	for (int i=0; i<N; ++i){
		phi11[fs.lin_index(i, j, k)] = Phi(1, 1, fs.coo[i], fs.coo[j], fs.coo[k]);
	}
	fs.tovtk("phi_11.vtk", phi11);
	std::vector<double> r11 = inverse_fourier3(fs, phi11);
	ps.tovtk("r_11.vtk", r11);
	GridVariance1 var(ps, std::move(r11));

	// 2. Stochastic solver
	size_t N1 = 31;
	double L1 = 20;
	PhysicalSpace velspace(N1, L1);
	StochasticGaussian<1>::Params params;
	params.eigen_cut = 1000;
	params.variance_cut = 0.05;
	StochasticGaussian<1> solver(velspace, var, params);

	// 3. generate
	Statistics stats(velspace);
	for (size_t itry=0; itry<n_tries; ++itry){
		std::vector<double> u = solver.generate(itry)[0];
		stats.add_velocity(u);
	}
	stats.save_r("rcomputed.vtk");

	std::cout << "DONE" << std::endl;
}

void gaussian2(size_t n_tries){
	// ==== PARAMETERS
	// -- 1d partition and linear size of the discretized variance scalar field
	size_t N = 51;
	double L = 20;
	// -- 1d partition and linear size of the resulting velocity vector field
	size_t N1 = 21;
	double L1 = 20;
	// -- number of eigen vectors with largest eigen values that will be taken into account
	size_t eigen_cut = 500;
	// -- set variance to zero if it is lower than max_variance * variance_cut
	double variance_cut = 0.05;
	
	// 1. build correlations and variance
	std::cout << "Computing spatial correlations" << std::endl;
	PhysicalSpace ps(N, L);
	FourierSpace fs = ps.fourier_space();

	std::vector<double> phi11(N*N*N, 0.0);
	std::vector<double> phi12(N*N*N, 0.0);
	std::vector<double> phi22(N*N*N, 0.0);

	for (int k=0; k<N; ++k)
	for (int j=0; j<N; ++j)
	for (int i=0; i<N; ++i){
		phi11[fs.lin_index(i, j, k)] = Phi(1, 1, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi12[fs.lin_index(i, j, k)] = Phi(1, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi22[fs.lin_index(i, j, k)] = Phi(2, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
	}
	fs.tovtk("phi_11.vtk", phi11);
	fs.tovtk("phi_12.vtk", phi12);
	fs.tovtk("phi_22.vtk", phi22);
	std::vector<double> r11 = inverse_fourier3(fs, phi11);
	std::vector<double> r12 = inverse_fourier3(fs, phi12);
	std::vector<double> r22 = inverse_fourier3(fs, phi22);
	ps.tovtk("r_11.vtk", r11);
	ps.tovtk("r_12.vtk", r12);
	ps.tovtk("r_22.vtk", r22);

	GridVariance2 var(ps,
		std::move(r11),
		std::move(r12),
		std::move(r22));


	// 2. Stochastic solver
	PhysicalSpace velspace(N1, L1);
	StochasticGaussian<2>::Params params;
	params.eigen_cut = eigen_cut;
	params.variance_cut = variance_cut;
	StochasticGaussian<2> solver(velspace, var, params);

	// 3. generate
	Statistics stats(velspace);
	for (size_t itry=0; itry<n_tries; ++itry){
		std::array<std::vector<double>, 2> u = solver.generate(itry);
		stats.add_velocity(u[0], u[1]);
	}
	stats.save_r("rcomputed_2.vtk");

	std::cout << "DONE" << std::endl;
}

void gaussian3(){
	// ==== PARAMETERS
	// -- 1d partition and linear size of the discretized variance scalar field
	size_t N = 51;
	double L = 20;
	// -- 1d partition and linear size of the resulting velocity vector field
	size_t N1 = 21;
	double L1 = 10;
	// -- number of eigen vectors with largest eigen values that will be taken into account
	size_t eigen_cut = 200;
	// -- set variance to zero if it is lower than max_variance * variance_cut
	double variance_cut = 0.05;
	// -- generate n_tries resulting fields
	size_t n_tries = 10;
	
	// 1. build correlations and variance
	std::cout << "Computing spatial correlations" << std::endl;
	PhysicalSpace ps(N, L);
	FourierSpace fs = ps.fourier_space();

	std::vector<double> phi11(N*N*N, 0.0);
	std::vector<double> phi12(N*N*N, 0.0);
	std::vector<double> phi13(N*N*N, 0.0);
	std::vector<double> phi22(N*N*N, 0.0);
	std::vector<double> phi23(N*N*N, 0.0);
	std::vector<double> phi33(N*N*N, 0.0);

	for (int k=0; k<N; ++k)
	for (int j=0; j<N; ++j)
	for (int i=0; i<N; ++i){
		phi11[fs.lin_index(i, j, k)] = Phi(1, 1, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi12[fs.lin_index(i, j, k)] = Phi(1, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi13[fs.lin_index(i, j, k)] = Phi(1, 3, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi22[fs.lin_index(i, j, k)] = Phi(2, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi23[fs.lin_index(i, j, k)] = Phi(2, 3, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi33[fs.lin_index(i, j, k)] = Phi(3, 3, fs.coo[i], fs.coo[j], fs.coo[k]);
	}
	fs.tovtk("phi_11.vtk", phi11);
	fs.tovtk("phi_12.vtk", phi12);
	fs.tovtk("phi_13.vtk", phi13);
	fs.tovtk("phi_22.vtk", phi22);
	fs.tovtk("phi_23.vtk", phi23);
	fs.tovtk("phi_33.vtk", phi33);
	std::vector<double> r11 = inverse_fourier3(fs, phi11);
	std::vector<double> r12 = inverse_fourier3(fs, phi12);
	std::vector<double> r13 = inverse_fourier3(fs, phi13);
	std::vector<double> r22 = inverse_fourier3(fs, phi22);
	std::vector<double> r23 = inverse_fourier3(fs, phi23);
	std::vector<double> r33 = inverse_fourier3(fs, phi33);
	ps.tovtk("r_11.vtk", r11);
	ps.tovtk("r_12.vtk", r12);
	ps.tovtk("r_13.vtk", r13);
	ps.tovtk("r_22.vtk", r22);
	ps.tovtk("r_23.vtk", r23);
	ps.tovtk("r_33.vtk", r33);
	throw;

	GridVariance3 var(ps,
		std::move(r11),
		std::move(r12),
		std::move(r13),
		std::move(r22),
		std::move(r23),
		std::move(r33));


	// 2. Stochastic solver
	PhysicalSpace velspace(N1, L1);
	StochasticGaussian<3>::Params params;
	params.eigen_cut = eigen_cut;
	params.variance_cut = variance_cut;
	StochasticGaussian<3> solver(velspace, var, params);

	// 3. generate
	for (size_t itry=0; itry<n_tries; ++itry){
		std::array<std::vector<double>, 3> u = solver.generate(itry);
		std::string fout = solver.dstr() + "_try" + std::to_string(itry) + ".vtk";
		velspace.tovtk(fout, u[0], u[1], u[2]);
		std::cout << "Data saved into " << fout << std::endl;
	}

	std::cout << "DONE" << std::endl;
}


void gaussian_sequential1(size_t n_tries){
	// ==== PARAMETERS
	// -- 1d partition and linear size of the discretized variance scalar field
	size_t N = 51;
	double L = 20;
	// -- 1d partition and linear size of the resulting velocity vector field
	size_t N1 = 31;
	double L1 = 20;
	// -- maximum number of adjacent points to be used in kriging solver
	size_t n_adjacent_points = 10;
	
	// 1. build correlations and variance
	std::cout << "Computing spatial correlations" << std::endl;
	PhysicalSpace ps(N, L);
	FourierSpace fs = ps.fourier_space();

	std::vector<double> phi11(N*N*N, 0.0);

	for (int k=0; k<N; ++k)
	for (int j=0; j<N; ++j)
	for (int i=0; i<N; ++i){
		phi11[fs.lin_index(i, j, k)] = Phi(1, 1, fs.coo[i], fs.coo[j], fs.coo[k]);
	}
	fs.tovtk("phi_11.vtk", phi11);
	std::vector<double> r11 = inverse_fourier3(fs, phi11);
	ps.tovtk("r_11.vtk", r11);

	GridVariance1 var(ps,
		std::move(r11));


	SequentialGaussian<1>::Params params;
	params.n_adjacent_points = n_adjacent_points;
	PhysicalSpace velspace(N1, L1);

	Statistics stats(velspace);

	//for (size_t itry=0; itry<n_tries; ++itry){
		//std::cout << itry << " / " << n_tries << std::endl;
		//SequentialGaussian<1> solver(velspace, var, params);
		//auto result=solver.generate(itry);
		//stats.add_velocity(result[0]);
	//}

	std::array<std::shared_ptr<std::thread>, n_chunks> threads;
	std::array<std::array<std::vector<double>, 1>, n_chunks> chunk_result;

	auto worker = [&](size_t seed, std::array<std::vector<double>, 1>& result){
		SequentialGaussian<1> solver(velspace, var, params);
		result = solver.generate(seed);
	};

	for (size_t itry=0; itry<n_tries; itry += n_chunks){
		std::cout << itry << " / " << n_tries << std::endl;

		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk].reset(new std::thread(worker, itry + ichunk, std::ref(chunk_result[ichunk])));
		}
		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk]->join();
		}
		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk].reset();
			stats.add_velocity(chunk_result[ichunk][0]);
		}
	}

	stats.save_r("rcomputed_sequential.vtk");
	std::cout << "DONE" << std::endl;
}

void gaussian_sequential2(size_t n_tries){
	// ==== PARAMETERS
	// -- 1d partition and linear size of the discretized variance scalar field
	size_t N = 51;
	double L = 20;
	// -- 1d partition and linear size of the resulting velocity vector field
	size_t N1 = 31;
	double L1 = 10;
	// -- maximum number of adjacent points to be used in kriging solver
	size_t n_adjacent_points = 10;
	
	// 1. build correlations and variance
	std::cout << "Computing spatial correlations" << std::endl;
	PhysicalSpace ps(N, L);
	FourierSpace fs = ps.fourier_space();

	std::vector<double> phi11(N*N*N, 0.0);
	std::vector<double> phi12(N*N*N, 0.0);
	std::vector<double> phi22(N*N*N, 0.0);

	for (int k=0; k<N; ++k)
	for (int j=0; j<N; ++j)
	for (int i=0; i<N; ++i){
		phi11[fs.lin_index(i, j, k)] = Phi(1, 1, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi12[fs.lin_index(i, j, k)] = Phi(1, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
		phi22[fs.lin_index(i, j, k)] = Phi(2, 2, fs.coo[i], fs.coo[j], fs.coo[k]);
	}
	fs.tovtk("phi_11.vtk", phi11);
	fs.tovtk("phi_12.vtk", phi12);
	fs.tovtk("phi_22.vtk", phi22);
	std::vector<double> r11 = inverse_fourier3(fs, phi11);
	std::vector<double> r12 = inverse_fourier3(fs, phi12);
	std::vector<double> r22 = inverse_fourier3(fs, phi22);
	ps.tovtk("r_11.vtk", r11);
	ps.tovtk("r_12.vtk", r12);
	ps.tovtk("r_22.vtk", r22);

	GridVariance2 var(ps,
		std::move(r11),
		std::move(r12),
		std::move(r22));


	SequentialGaussian<2>::Params params;
	params.n_adjacent_points = n_adjacent_points;
	PhysicalSpace velspace(N1, L1);

	Statistics stats(velspace);

	std::array<std::shared_ptr<std::thread>, n_chunks> threads;
	std::array<std::array<std::vector<double>, 2>, n_chunks> chunk_result;

	auto worker = [&](size_t seed, std::array<std::vector<double>, 2>& result){
		SequentialGaussian<2> solver(velspace, var, params);
		result = solver.generate(seed);
	};

	for (size_t itry=0; itry<n_tries; itry += n_chunks){
		std::cout << itry << " / " << n_tries << std::endl;

		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk].reset(new std::thread(worker, itry + ichunk, std::ref(chunk_result[ichunk])));
		}
		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk]->join();
		}
		for (size_t ichunk = 0; ichunk < n_chunks; ++ichunk){
			threads[ichunk].reset();
			stats.add_velocity(chunk_result[ichunk][0], chunk_result[ichunk][1]);
		}
	}

	stats.save_r("rcomputed_sequential2.vtk");
	std::cout << "DONE" << std::endl;
}

int main(){
	//gaussian1(1000);
	//gaussian2(1000);
	//gaussian3();
	//gaussian_sequential1(1000);
	gaussian_sequential2(10000);
}
