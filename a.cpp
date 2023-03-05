#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include "ft.hpp"
#include <random>
#include "stochastic_gaussian.hpp"

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

void gaussian1(){
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
	StochasticGaussian1::Params params;
	params.eigen_cut = 100;
	params.variance_cut = 0.05;
	StochasticGaussian1 solver(velspace, var, params);

	// 3. generate
	for (size_t itry=0; itry<10; ++itry){
		arma::Col<double> u = solver.generate(itry);
		std::string fout = solver.dstr() + "_try" + std::to_string(itry) + ".vtk";
		velspace.tovtk(fout, u.begin(), u.end());
		std::cout << "Data saved into " << fout << std::endl;
	}

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
	size_t eigen_cut = 1000;
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

	GridVariance3 var(ps,
		std::move(r11),
		std::move(r12),
		std::move(r13),
		std::move(r22),
		std::move(r23),
		std::move(r33));


	// 2. Stochastic solver
	PhysicalSpace velspace(N1, L1);
	StochasticGaussian3::Params params;
	params.eigen_cut = eigen_cut;
	params.variance_cut = variance_cut;
	StochasticGaussian3 solver(velspace, var, params);

	// 3. generate
	for (size_t itry=0; itry<n_tries; ++itry){
		std::array<std::vector<double>, 3> u = solver.generate(itry);
		std::string fout = solver.dstr() + "_try" + std::to_string(itry) + ".vtk";
		velspace.tovtk(fout, u[0], u[1], u[2]);
		std::cout << "Data saved into " << fout << std::endl;
	}

	std::cout << "DONE" << std::endl;
}

int main(){
	//gaussian1();
	gaussian3();
}
