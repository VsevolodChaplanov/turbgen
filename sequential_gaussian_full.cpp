#include "sequential_gaussian_full.hpp"
#include <random>
#include <algorithm>
#include <vector>
#include <set>
#include "boost/geometry.hpp"
#include <boost/geometry/geometries/adapted/std_array.hpp>
#include <armadillo>

BOOST_GEOMETRY_REGISTER_STD_ARRAY_CS(boost::geometry::cs::cartesian)

namespace{

template<size_t Dim>
arma::Mat<double> to_nonsym(const std::array<double, (Dim*Dim + Dim)/2>& sym){
	arma::Mat<double> ret(Dim, Dim, arma::fill::zeros);
	size_t k = 0;
	for (size_t i=0; i<Dim; ++i)
	for (size_t j=i; j<Dim; ++j){
		ret(i, j) = sym[k];
		ret(j, i) = sym[k];
		++k;
	}
	return ret;
}

}


template<size_t Dim>
std::array<std::vector<double>, Dim> SequentialGaussianFull<Dim>::generate(size_t seed) const{
	std::array<std::vector<double>, Dim> ret;
	for (size_t i=0; i<ret.size(); ++i){
		ret[i] = std::vector<double>(_space.N3, 0);
	}

	// 0. initialize

	// random engine
	std::mt19937_64 generator(seed);
	// C0
	arma::Mat<double> c0 = to_nonsym<Dim>(_varfun.variance0());
	
	// 1. build indicies and shuffle
	std::vector<size_t> indices(_space.N3);
	std::iota(indices.begin(), indices.end(), 0);
	std::shuffle(indices.begin(), indices.end(), generator);

	// 2. create point finder
	using rtree_entry_t = std::pair<point_t, int>;
	using rtree_t = boost::geometry::index::rtree<rtree_entry_t, boost::geometry::index::rstar<8>>;
	std::vector<point_t> rtree_data;
	rtree_t rt;

	// 3 start indices loop
	for (size_t i_indices=0; i_indices<indices.size(); ++i_indices){
		size_t index = indices[i_indices];
		point_t p = _space.point(index);
		
		// 1. find closest points
		std::vector<rtree_entry_t> adj;
		rt.query(boost::geometry::index::nearest(p, _params.n_adjacent_points),
			  std::back_inserter(adj));
		size_t n_adj = adj.size();

		// 2. assemble matrix
		arma::Mat<double> mat(Dim*n_adj, Dim*n_adj, arma::fill::zeros);
		arma::Col<double> rhs(Dim*n_adj, arma::fill::zeros);
		for (size_t i=0; i<n_adj; ++i){
			point_p pi = _space.point(adj[i].second);

			// rhs
			point_p h {pj[0] - pi[0], pj[1] - pi[1], pj[2] - pj[0]};
			auto ch = _varfun.variance();

			for (size_t j=i+1; j<n_adj; ++j){
				point_p pj = _space.point(adj[j].second);
				point_p h {pj[0] - pi[0], pj[1] - pi[1], pj[2] - pj[0]};
				auto c = _varfun.variance(h);
			}
		}

		// 3. solve matrix
		arma::Col<double> solution = arma::solve(mat, rhs);

		// 4. build mean and variance
		arma::Col<double> mean(Dim, arma::fill::zeros);
		arma::Mat<double> variance(c0);
		for (size_t i=0; i<n_adj; ++i){
			size_t ret_ind = adj[i].second;
			for(size_t j=0; j<Dim; ++j){
				for (size_t k=0; k<Dim; ++k){
					double slv = solution(i*Dim*Dim + j*Dim + k);
					mean(j) += slv * ret[k][ret_ind];
					variance(j, k) -= slv * rhs(i*Dim*Dim + k*Dim + j);
				}
			}
		}

		// 5. compute velocity
		arma::Col<double> eigval;
		arma::Mat<double> eigvec;
		// TODO: is variance symmetric here?
		arma::eig_sym(eigval, eigvec, variance);
		arma::Col<double> random_normals(Dim);
		std::normal_distribution<double> ndist(0, 1);
		for (size_t i=0; i<Dim; ++i){
			random_normals(i) = ndist(generator) * std::sqrt(eigval(i));
		}
		arma::Col<double> vel = mean + eigvec * random_normals;

		// 6. add to solution
		rt.insert(std::make_pair(p, index));
		for (size_t i=0; i<ret.size(); ++i){
			ret[i][index] = vel(i);
		}
	}

	return ret;
}

template class SequentialGaussianFull<2>;
