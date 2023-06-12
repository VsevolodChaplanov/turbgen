#include "space.hpp"
#include <cmath>
#include <fstream>
#include <sstream>

namespace{

std::vector<double> build_coo(size_t N, double L){
	std::vector<double> ret(N);
	double h = L/N;

	for (size_t i=0; i<N; ++i){
		ret[i] = -L/2 + h/2 + i*h;
	}

	return ret;
}

}

Space::Space(size_t N, double L)
		: N(N), N3(N*N*N), L(L), 
		  A(-L/2), B(L/2),
		  h(L/N), coo(build_coo(N, L)){

	if (N % 2 != 1){
		throw std::runtime_error("N should be odd number");
	}
	if (N < 5){
		throw std::runtime_error("N should be >= 5");
	}
	if (L <= 0){
		throw std::runtime_error("L should be > 0");
	}
}

PhysicalSpace::PhysicalSpace(size_t N, double L): Space(N, L){}

FourierSpace PhysicalSpace::fourier_space() const{
	double hk = 2*M_PI/L;
	double Lk = hk*N;
	return FourierSpace(N, Lk);
};

FourierSpace::FourierSpace(size_t N, double L): Space(N, L){}

PhysicalSpace FourierSpace::physical_space() const{
	double Lx = 2*M_PI/h;
	return PhysicalSpace(N, Lx);
};

point_t Space::point(size_t i, size_t j, size_t k) const{
	return {coo[i], coo[j], coo[k]};
}

point_t Space::point(size_t ind) const{
	std::array<size_t, 3> ind3 = tri_index(ind);
	return {coo[ind3[0]], coo[ind3[1]], coo[ind3[2]]};
}

bool Space::point_within(const point_t& p) const{
	if (p[0] < coo[0] || p[0] > coo.back()) return false;
	if (p[1] < coo[0] || p[1] > coo.back()) return false;
	if (p[2] < coo[0] || p[2] > coo.back()) return false;
	return true;
}

double Space::interpolate_at(const point_t& p, const std::vector<double>& f) const{
	std::array<size_t, 8> indices;
	std::array<double, 8> bases;
	interpolation_polynom(p, indices, bases);

	return use_interpolation_polynom(indices, bases, f);
}

std::array<double, 3> Space::interpolate_at3(const point_t& p,
		const std::vector<double>& f1,
		const std::vector<double>& f2,
		const std::vector<double>& f3) const{

	std::array<size_t, 8> indices;
	std::array<double, 8> bases;
	interpolation_polynom(p, indices, bases);

	return {use_interpolation_polynom(indices, bases, f1),
	        use_interpolation_polynom(indices, bases, f2),
	        use_interpolation_polynom(indices, bases, f3)};
}

std::array<double, 6> Space::interpolate_at6(const point_t& p,
		const std::vector<double>& f1,
		const std::vector<double>& f2,
		const std::vector<double>& f3,
		const std::vector<double>& f4,
		const std::vector<double>& f5,
		const std::vector<double>& f6) const{

	std::array<size_t, 8> indices;
	std::array<double, 8> bases;
	interpolation_polynom(p, indices, bases);

	return {use_interpolation_polynom(indices, bases, f1),
	        use_interpolation_polynom(indices, bases, f2),
	        use_interpolation_polynom(indices, bases, f3),
	        use_interpolation_polynom(indices, bases, f4),
	        use_interpolation_polynom(indices, bases, f5),
	        use_interpolation_polynom(indices, bases, f6)};
}


double Space::use_interpolation_polynom(const std::array<size_t, 8>& indices, const std::array<double, 8>& bases, const std::vector<double>& v) const{
	return v[indices[0]] * bases[0]
	       + v[indices[1]] * bases[1]
	       + v[indices[2]] * bases[2]
	       + v[indices[3]] * bases[3]
	       + v[indices[4]] * bases[4]
	       + v[indices[5]] * bases[5]
	       + v[indices[6]] * bases[6]
	       + v[indices[7]] * bases[7];
}

void Space::interpolation_polynom(const point_t& p, std::array<size_t, 8>& indices, std::array<double, 8>& bases) const{
	double kx = (p[0] - coo[0])/h;
	double ky = (p[1] - coo[0])/h;
	double kz = (p[2] - coo[0])/h;

	size_t i = std::max(0, std::min((int)N-2, int(std::floor(kx))));
	size_t j = std::max(0, std::min((int)N-2, int(std::floor(ky))));
	size_t k = std::max(0, std::min((int)N-2, int(std::floor(kz))));

	double xi = kx - i;
	double eta = ky - j;
	double zeta = kz - k;

	indices[0] = lin_index(i, j, k);
	indices[1] = lin_index(i+1, j, k);
	indices[2] = lin_index(i+1, j+1, k);
	indices[3] = lin_index(i, j+1, k);
	indices[4] = lin_index(i, j, k+1);
	indices[5] = lin_index(i+1, j, k+1);
	indices[6] = lin_index(i+1, j+1, k+1);
	indices[7] = lin_index(i, j+1, k+1);

	bases[0] = (1-xi)*(1-eta)*(1-zeta);
	bases[1] = (xi)*(1-eta)*(1-zeta);
	bases[2] = (xi)*(eta)*(1-zeta);
	bases[3] = (1-xi)*(eta)*(1-zeta);
	bases[4] = (1-xi)*(1-eta)*zeta;
	bases[5] = (xi)*(1-eta)*zeta;
	bases[6] = (xi)*(eta)*zeta;
	bases[7] = (1-xi)*(eta)*zeta;
};

size_t Space::lin_index(size_t i, size_t j, size_t k) const{
	return i + j*N + k*N*N;
}

std::array<size_t, 3> Space::tri_index(size_t ind) const{
	size_t k = ind / (N*N);
	size_t ij = ind % (N*N);
	size_t j = ij / N;
	size_t i = ij % N;

	return {i, j, k};
}

void Space::tovtk_init(std::ostream& ofs, int data_dim) const {
	double h = L/N;

	ofs << "# vtk DataFile Version 2.0" << std::endl;
	ofs << "Func" << std::endl;
	ofs << "ASCII" << std::endl;
	ofs << "DATASET RECTILINEAR_GRID" << std::endl;
	ofs << "DIMENSIONS " << N << " " << N << " " << N << std::endl;
	ofs << "X_COORDINATES " << N << " double" << std::endl;
	for (size_t i=0; i<N; ++i){
		ofs << coo[i] << std::endl;
	}
	ofs << "Y_COORDINATES " << N << " double" << std::endl;
	for (size_t i=0; i<N; ++i){
		ofs << coo[i] << std::endl;
	}
	ofs << "Z_COORDINATES " << N << " double" << std::endl;
	for (size_t i=0; i<N; ++i){
		ofs << coo[i] << std::endl;
	}
	ofs << "POINT_DATA " << N*N*N << std::endl;
	ofs << "SCALARS data double " << data_dim << std::endl;
	ofs << "LOOKUP_TABLE default" << std::endl;
}
