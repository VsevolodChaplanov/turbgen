#include "sequential_gaussian_1.hpp"
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include "boost/geometry.hpp"
#include <boost/geometry/geometries/adapted/std_array.hpp>
#include <armadillo>

BOOST_GEOMETRY_REGISTER_STD_ARRAY_CS(boost::geometry::cs::cartesian)

namespace{

// 1d
double value_from_sym(double sym, size_t i, size_t j){
	return sym;
}

// 2d
double value_from_sym(const std::array<double, 3>& sym, size_t i, size_t j){
	if (i == j){
		if (i == 0) return sym[0];
		else return sym[2];
	} else {
		return sym[1];
	}
}

// 3d
double value_from_sym(const std::array<double, 6>& sym, size_t i, size_t j){
	switch (i){
		case 0:
			switch (j){
				case 0: return sym[0];
				case 1: return sym[1];
				case 2: return sym[2];
				default: throw std::runtime_error("invalid value from sym");
			}
		case 1:
			switch (j){
				case 0: return sym[1];
				case 1: return sym[3];
				case 2: return sym[4];
				default: throw std::runtime_error("invalid value from sym");
			}
		case 2:
			switch (j){
				case 0: return sym[2];
				case 1: return sym[4];
				case 2: return sym[5];
				default: throw std::runtime_error("invalid value from sym");
			}
		default: throw std::runtime_error("invalid value from sym");
	}
}

}



template<size_t Dim>
std::array<std::vector<double>, Dim> SequentialGaussian<Dim>::generate(size_t seed, bool cout_progress) const{
	std::vector<double> vel(Dim*_space.N3, 0);

	// random engine
	std::mt19937_64 generator(seed);

	// 1. build indicies and shuffle
	std::vector<size_t> indices(Dim*_space.N3);
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), generator);

	// 2. create point finder
	rtree_t rt;

	// 3 start indices loop
	size_t progress_step = indices.size() / 100;
	for (size_t i_indices=0; i_indices<indices.size(); ++i_indices){
		size_t index = indices[i_indices];
		size_t dim = index / _space.N3;
		size_t pindex = index - dim*_space.N3;
		point_t p = _space.point(pindex);

		// 1. find closest points
		std::vector<rtree_entry_t> adj;
		rt.query(boost::geometry::index::nearest(p, _params.n_adjacent_points),
			  std::back_inserter(adj));
		size_t n_adj = adj.size();

		// 2. compute mean and variance
		double mean, variance;
		std::tie(mean, variance) = kriging_solver(index, adj, vel);
		
		// 3. compute velocity
		double value = std::normal_distribution<double>(mean, std::sqrt(variance))(generator);

		// 4. add to solution
		rt.insert(std::make_pair(p, index));
		vel[index] = value;

		if (cout_progress && i_indices % progress_step == 0){
			std::cout << std::round((float)i_indices/indices.size() * 10000)/100 << "% done" << std::endl;
		}
	}

	// to result
	std::array<std::vector<double>, Dim> ret;
	for (size_t i=0; i<Dim; ++i){
		ret[i].resize(_space.N3);
		std::copy(vel.begin() + i*_space.N3, vel.begin() + (i+1)*_space.N3, ret[i].begin());
	}
	return ret;
}

template<size_t Dim>
std::pair<double, double> SequentialGaussian<Dim>::kriging_solver(size_t index, const std::vector<rtree_entry_t>& adj, const std::vector<double>& vel) const {
	auto c0 = _varfun.variance0();
	size_t n_adj = adj.size();
	size_t dim = index / _space.N3;
	size_t pindex = index - dim*_space.N3;
	point_t p = _space.point(pindex);

	if (n_adj == 0){
		return {0, value_from_sym(c0, dim, dim)};
	}
	
	// 1. assemble matrix
	arma::Mat<double> mat(n_adj, n_adj, arma::fill::zeros);
	arma::Col<double> rhs(n_adj, arma::fill::zeros);
	for (size_t i=0; i<n_adj; ++i){
		size_t dim_i = adj[i].second / _space.N3;
		size_t pindex_i = adj[i].second - dim_i*_space.N3;
		point_t pi = _space.point(pindex_i);

		// rhs
		point_t h1 {pi[0] - p[0], pi[1] - p[1], pi[2] - p[2]};
		rhs(i) = value_from_sym(_varfun.variance(h1), dim, dim_i);

		// diagonal
		mat(i, i) = value_from_sym(c0, dim_i, dim_i);

		// non-diagonal
		for (size_t j=i+1; j<n_adj; ++j){
			size_t dim_j = adj[j].second / _space.N3;
			size_t pindex_j = adj[j].second - dim_j*_space.N3;
			point_t pj = _space.point(pindex_j);
			point_t h2 {pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]};
			double v = value_from_sym(_varfun.variance(h2), dim_i, dim_j);
			mat(i, j) = v;
			mat(j, i) = v;
		}
	}

	// 2. solve matrix
	arma::Col<double> solution = arma::solve(mat, rhs);

	// 3. build mean and variance
	double mean = 0;
	double var = value_from_sym(c0, dim, dim);
	for (size_t i=0; i<n_adj; ++i){
		mean += solution(i) * vel[adj[i].second];
		var -= solution(i) * rhs(i);
	}

	return std::make_pair(mean, var);
}


template<size_t Dim>
std::string SequentialGaussian<Dim>::dstr() const{
	std::ostringstream oss;
	oss << "seqg" << std::to_string(Dim) << "_L" << round(_space.L) << "_N" <<_space.N << "_Adj" << _params.n_adjacent_points;
	return oss.str();
}


template class SequentialGaussian<1>;
template class SequentialGaussian<2>;
template class SequentialGaussian<3>;
