#ifndef _SEQUENTIAL_GAUSSIAN_FULL_HPP_
#define _SEQUENTIAL_GAUSSIAN_FULL_HPP_

#include <functional>
#include <string>
#include <array>
#include "space.hpp"
#include "varfun.hpp"

template<size_t Dim>
class SequentialGaussianFull{
public:
	struct Params{
		size_t n_adjacent_points = 10;
	};

	SequentialGaussianFull(
		PhysicalSpace space,
		const IVarFun<Dim>& varfun,
		Params params): _space(space), _params(params), _varfun(varfun){}

	const PhysicalSpace& space() const { return _space; }
	const Params& params() const { return _params; }

	std::array<std::vector<double>, Dim> generate(size_t seed) const;

	std::string dstr() const;
private:
	void initialize(const IVarFun<Dim>& varfun);

	PhysicalSpace _space;
	Params _params;
	const IVarFun<Dim>& _varfun;
};

#endif
