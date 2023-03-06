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

}

// ================================= 1

namespace {

arma::SpMat<double> build_covariance_matrix(const IVarFun1& varfun, double cut_share, PhysicalSpace outspace){
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

}

StochasticGaussian1::StochasticGaussian1(
		PhysicalSpace space,
		const IVarFun1& varfun,
		Params params): _space(space), _params(params){
	initialize(varfun);
}

void StochasticGaussian1::initialize(const IVarFun1& varfun){
	std::cout << "Building covariance matrix" << std::endl;
	arma::SpMat<double> cov = build_covariance_matrix(varfun, _params.variance_cut, _space);

	std::cout << "Calculating eigen vectors" << std::endl;
	arma::eigs_sym(eigval, eigvec, cov, _params.eigen_cut);
	std::cout << eigval.n_elem << " eigen vectors calculated" << std::endl;
	std::cout << "Eigen values range = [" << eigval(0) << ", " << eigval(eigval.n_elem-1) << "]" << std::endl;
}

arma::Col<double> StochasticGaussian1::generate(size_t seed) const{
	std::mt19937_64 generator;
	generator.seed(seed);
	size_t n_vals = eigval.n_elem;

	arma::Col<double> random_normals(n_vals);
	std::normal_distribution<double> ndist(0, 1);
	for (size_t i=0; i<n_vals; ++i){
		random_normals(i) = ndist(generator) * std::sqrt(eigval(i));
	}
	return eigvec * random_normals;
}

std::string StochasticGaussian1::dstr() const{
	std::ostringstream oss;
	oss << "sg1_L" << round(_space.L) << "_N" <<_space.N << "_eigcut" << _params.eigen_cut;
	return oss.str();
}


// ================================= 3

namespace {
arma::SpMat<double> build_covariance_matrix(const IVarFun3& varfun, double cut_share, PhysicalSpace outspace){
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


StochasticGaussian3::StochasticGaussian3(
		PhysicalSpace space,
		const IVarFun3& varfun,
		Params params): _space(space), _params(params){
	initialize(varfun);
}

void StochasticGaussian3::initialize(const IVarFun3& varfun){
	std::cout << "Building covariance matrix" << std::endl;
	arma::SpMat<double> cov = build_covariance_matrix(varfun, _params.variance_cut, _space);

	std::cout << "Calculating eigen vectors" << std::endl;
	arma::eigs_sym(eigval, eigvec, cov, _params.eigen_cut);
	
	// check for positive values only
	std::set<size_t> not_needed;
	for (size_t i=0; i<eigval.n_elem; ++i){
		if (eigval(i) < 0){
			not_needed.insert(i);
		}
	}
	if (not_needed.size() > 0){
		std::cout << "WARNING: " << not_needed.size() << " eigen values are negative: ";
		for (size_t i: not_needed) std::cout << eigval(i) << " ";
		std::cout << std::endl;

		arma::Mat<double> eigvec2 = arma::Mat<double>(eigvec.n_rows, eigvec.n_cols - not_needed.size());
		arma::Col<double> eigval2 = arma::Col<double>(eigvec.n_cols - not_needed.size());

		int kcol = 0;
		for (size_t icol=0; icol < eigvec.n_cols; ++icol){
			if (not_needed.find(icol) != not_needed.end()){
				continue;
			}
			eigval2(kcol) = eigval(icol);
			for (size_t irow=0; irow < eigvec.n_rows; ++irow){
				eigvec2(irow, kcol) = eigvec(irow, icol);
			}
			++kcol;
		}
		std::swap(eigval2, eigval);
		std::swap(eigvec2, eigvec);
	}

	std::cout << eigval.n_elem << " eigen vectors calculated." << std::endl;
	std::cout << "Eigen values range = [" << eigval(0) << ", " << eigval(eigval.n_elem-1) << "]" << std::endl;
}

std::array<std::vector<double>, 3> StochasticGaussian3::generate(size_t seed) const{
	std::mt19937_64 generator;
	generator.seed(seed);
	size_t n_vals = eigval.n_elem;

	arma::Col<double> random_normals(n_vals);
	std::normal_distribution<double> ndist(0, 1);
	for (size_t i=0; i<n_vals; ++i){
		random_normals(i) = ndist(generator) * std::sqrt(eigval(i));
	}
	arma::Col<double> raw = eigvec * random_normals;

	size_t n = _space.N3;
	return {
		std::vector<double>(raw.begin(), raw.begin()+n),
		std::vector<double>(raw.begin()+n, raw.begin()+2*n),
		std::vector<double>(raw.begin()+2*n, raw.end())
	};
}

std::string StochasticGaussian3::dstr() const{
	std::ostringstream oss;
	oss << "sg3_L" << round(_space.L) << "_N" <<_space.N << "_eigcut" << _params.eigen_cut;
	return oss.str();
}


