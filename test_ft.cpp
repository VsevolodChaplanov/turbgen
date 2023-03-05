#include "ft.hpp"
#include <iostream>
#include <cmath>
#include <sstream>

void check_equal(double a, double b){
	if (std::abs(a - b) > 1e-6){
		std::ostringstream oss;
		oss << "Check failed: " << a << " != " << b << std::endl;
		throw std::runtime_error(oss.str());
	}
}

double func(double x, double y, double z){
	double v1 = 1.0/(std::abs(x) + 1);
	double v2 = 1.0/(std::abs(y) + 1);
	double v3 = 1.0/(std::abs(z) + 1);

	return std::pow(v1, 6) * std::pow(v2, 4) * std::pow(v3, 5);
}

void test01(){
	// 1d partition
	size_t N = 11;

	// physical space boundaries
	double Lx = 2;

	PhysicalSpace ps(N, Lx);
	FourierSpace fs = ps.fourier_space();

	// fill 3d function array
	std::vector<double> fx(N*N*N);

	for (size_t k=0; k<N; ++k)
	for (size_t j=0; j<N; ++j)
	for (size_t i=0; i<N; ++i){
		fx[ps.lin_index(i, j, k)] = func(ps.coo[i], ps.coo[j], ps.coo[k]);
	}

	// compute forward transform
	std::vector<double> fk = fourier3(ps, fx);
	check_equal(fk[5 + 5*N + 5*N*N], 0.0005079819819736192);
	check_equal(fk[6 + 5*N + 5*N*N], 0.00036019289899672674);
	check_equal(fk[7 + 5*N + 5*N*N], 0.000216620583015753);
	check_equal(fk[0], 1.7043779074776304e-06);
	check_equal(fk[4 + 4*N + 4*N*N], 0.00011965752189560347);
	check_equal(fk[4 + 5*N + 7*N*N], 0.0001205415104388087);
	
	// compute inverse transform
	std::vector<double> fx2 = inverse_fourier3(fs, fk);
	for (size_t i=0; i<N*N*N; ++i){
		check_equal(fx2[i], fx[i]);
	}
}


int main(){
	test01();
	std::cout << "DONE" << std::endl;
}
