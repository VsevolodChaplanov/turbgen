#ifndef _SEQUENTIAL_GAUSSIAN_1_HPP_
#define _SEQUENTIAL_GAUSSIAN_1_HPP_

#include <functional>
#include <string>
#include <array>
#include "space.hpp"
#include "varfun.hpp"
#include "boost/geometry.hpp"

template<size_t Dim>
class SequentialGaussian{
public:
	struct Params{
		size_t n_adjacent_points = 10;
	};

	SequentialGaussian(
		PhysicalSpace space,
		const IVarFun<Dim>& varfun,
		Params params): _space(space), _params(params), _varfun(varfun){}

	const PhysicalSpace& space() const { return _space; }
	const Params& params() const { return _params; }

	std::array<std::vector<double>, Dim> generate(size_t seed, bool cout_progress=false) const;

	std::string dstr() const;
private:
	using rtree_entry_t = std::pair<point_t, int>;
	using rtree_t = boost::geometry::index::rtree<rtree_entry_t, boost::geometry::index::rstar<8>>;

	// -> mean, variance
	std::pair<double, double> kriging_solver(size_t index, const std::vector<rtree_entry_t>& adj, const std::vector<double>& vel) const;

	PhysicalSpace _space;
	Params _params;
	const IVarFun<Dim>& _varfun;
};

#endif
