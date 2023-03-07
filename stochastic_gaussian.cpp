#include "stochastic_gaussian.hpp"
#include <vector>
#include <set>

namespace{

class TmpSpMat{
public:
	TmpSpMat(size_t n, double threshold): _n(n), _threshold(threshold){}

	void set_value(int irow, int icol, double value){
		if (std::abs(value) > _threshold){
			if (icol < irow) std::swap(irow, icol);
			if (icol == irow){
				_rows.push_back(irow);
				_cols.push_back(irow);
				_values.push_back(value);
			} else {
				_rows.push_back(irow);
				_cols.push_back(icol);
				_rows.push_back(icol);
				_cols.push_back(irow);
				_values.push_back(value);
				_values.push_back(value);
			}
		}
	}
	arma::SpMat<double> assemble_sym(){
		arma::umat locations(2, _rows.size());
		for (size_t i=0; i<_rows.size(); ++i){
			locations(0, i) = _rows[i];
			locations(1, i) = _cols[i];
		}
		_rows.clear();
		_cols.clear();

		arma::Col<double> values(_values.size());
		for (size_t i=0; i<_values.size(); ++i){
			values(i) = _values[i];
		}
		_values.clear();

		return arma::SpMat<double>(std::move(locations), std::move(values));
	}
private:
	std::vector<int> _rows, _cols;
	std::vector<double> _values;
	const size_t _n;
	const double _threshold;

	size_t _n_data = 0;
};

// ================================= 1

arma::SpMat<double> build_covariance_matrix(const IVarFun<1>& varfun, double cut_share, PhysicalSpace outspace){
	arma::SpMat<double> ret(outspace.N3, outspace.N3);

	double threshold = cut_share * varfun.max_value();

	for (int irow=0; irow<outspace.N3; ++irow){
		point_t p0 = outspace.point(irow); 

		// diagonal
		double c0 = varfun.variance0();
		ret(irow, irow) = c0;

		// non-diagonal
		for (int icol=irow+1; icol<outspace.N3; ++icol){
			point_t p1 = outspace.point(icol); 
			p1[0] -= p0[0];
			p1[1] -= p0[1];
			p1[2] -= p0[2];

			double c = varfun.variance(p1);
			if (std::abs(c) > threshold){
				ret(irow, icol) = c;
				ret(icol, irow) = c;
			}
		}
	}

	std::cout << ret.n_nonzero << " non zero entries in the covariance matrix from " << ret.n_elem << std::endl;
	return ret;
}

// ================================= 3

arma::SpMat<double> build_covariance_matrix(const IVarFun<3>& varfun, double cut_share, PhysicalSpace outspace){
	double threshold = cut_share * varfun.max_value();
	TmpSpMat tmp(3*outspace.N3, threshold);

	size_t n = outspace.N3;
	auto ex = [n](size_t i)->size_t{ return i; };
	auto ey = [n](size_t i)->size_t{ return i + n; };
	auto ez = [n](size_t i)->size_t{ return i + 2*n; };

	for (int irow=0; irow<outspace.N3; ++irow){
		point_t p0 = outspace.point(irow); 

		// diagonal
		std::array<double, 6> c0 = varfun.variance0();
		tmp.set_value(ex(irow), ex(irow), c0[0]);
		tmp.set_value(ex(irow), ey(irow), c0[1]);
		tmp.set_value(ex(irow), ez(irow), c0[2]);
		tmp.set_value(ey(irow), ey(irow), c0[3]);
		tmp.set_value(ey(irow), ez(irow), c0[4]);
		tmp.set_value(ez(irow), ez(irow), c0[5]);

		// non-diagonal
		for (int icol=irow+1; icol<outspace.N3; ++icol){
			point_t p1 = outspace.point(icol); 
			p1[0] -= p0[0];
			p1[1] -= p0[1];
			p1[2] -= p0[2];

			std::array<double, 6> c = varfun.variance(p1);
			// xx
			if (std::abs(c[0]) > threshold){
				tmp.set_value(ex(irow), ex(icol), c[0]);
			}
			// xy
			if (std::abs(c[1]) > threshold){
				tmp.set_value(ex(irow), ey(icol), c[1]);
				tmp.set_value(ex(icol), ey(irow), c[1]);
			}
			//xz
			if (std::abs(c[2]) > threshold){
				tmp.set_value(ex(irow), ez(icol), c[2]);
				tmp.set_value(ez(irow), ex(icol), c[2]);
			}
			// yy
			if (std::abs(c[3]) > threshold){
				tmp.set_value(ey(irow), ey(icol), c[3]);
			}
			//yz
			if (std::abs(c[4]) > threshold){
				tmp.set_value(ey(irow), ez(icol), c[4]);
				tmp.set_value(ez(irow), ey(icol), c[4]);
			}
			// zz
			if (std::abs(c[5]) > threshold){
				tmp.set_value(ez(irow), ez(icol), c[5]);
			}
		}
	}
	arma::SpMat<double> ret = tmp.assemble_sym();

	std::cout << ret.n_nonzero << " non zero entries in the covariance matrix from " << ret.n_elem << std::endl;
	return ret;
}

}

template<size_t Dim>
std::string StochasticGaussian<Dim>::dstr() const{
	std::ostringstream oss;
	oss << "sg" << std::to_string(Dim) << "_L" << round(_space.L) << "_N" <<_space.N << "_eigcut" << _params.eigen_cut;
	return oss.str();
}

template<size_t Dim>
std::array<std::vector<double>, Dim> StochasticGaussian<Dim>::generate(size_t seed) const{
	std::mt19937_64 generator;
	generator.seed(seed);
	size_t n_vals = _eigval.n_elem;

	arma::Col<double> random_normals(n_vals);
	std::normal_distribution<double> ndist(0, 1);
	for (size_t i=0; i<n_vals; ++i){
		random_normals(i) = ndist(generator) * std::sqrt(_eigval(i));
	}
	arma::Col<double> raw = _eigvec * random_normals;

	size_t n = space().N3;
	size_t n_start = 0;
	std::array<std::vector<double>, Dim> ret;
	for (size_t d=0; d<Dim; ++d){
		ret[d] = std::vector<double>(raw.begin()+n_start, raw.begin()+n_start+n);
		n_start += n;
	}
	return ret;
}

template<size_t Dim>
void StochasticGaussian<Dim>::initialize(const IVarFun<Dim>& varfun){
	std::cout << "Building covariance matrix" << std::endl;
	arma::SpMat<double> cov = build_covariance_matrix(varfun, params().variance_cut, space());

	std::cout << "Calculating eigen vectors" << std::endl;
	arma::eigs_sym(_eigval, _eigvec, cov, params().eigen_cut);
	
	// check for positive values only
	std::vector<size_t> not_needed;
	for (size_t i=0; i<_eigval.n_elem; ++i){
		if (_eigval(i) < 0){
			not_needed.push_back(i);
		}
	}
	if (not_needed.size() > 0){
		std::cout << "WARNING: " << not_needed.size() << " eigen values are negative: ";
		for (size_t i: not_needed) std::cout << _eigval(i) << " ";
		std::cout << std::endl;

		arma::uvec nn(not_needed.size());
		for (size_t i=0; i<not_needed.size(); ++i){
			nn(i) = not_needed[i];
		}
		_eigvec.shed_cols(nn);
		_eigval.shed_rows(nn);
	}

	std::cout << _eigval.n_elem << " eigen vectors calculated." << std::endl;
	std::cout << "Eigen values range = [" << _eigval(0) << ", " << _eigval(_eigval.n_elem-1) << "]" << std::endl;
}

template class StochasticGaussian<1>;
template class StochasticGaussian<3>;
