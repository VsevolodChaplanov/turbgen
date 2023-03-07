#ifndef _STOCHASTIC_GAUSSIAN_HPP_
#define _STOCHASTIC_GAUSSIAN_HPP_

#include <functional>
#include <string>
#include <array>
#include <armadillo>
#include "space.hpp"
#include "varfun.hpp"

template<size_t Dim>
class StochasticGaussian{
public:
	struct Params{
		size_t eigen_cut = 1000;
		double variance_cut = 0.05;
	};

	StochasticGaussian(
		PhysicalSpace space,
		const IVarFun<Dim>& varfun,
		Params params): _space(space), _params(params){ initialize(varfun); }

	const PhysicalSpace& space() const { return _space; }
	const Params& params() const { return _params; }

	std::array<std::vector<double>, Dim> generate(size_t seed) const;

	std::string dstr() const;
private:
	PhysicalSpace _space;
	Params _params;
	arma::Col<double> _eigval;
	arma::Mat<double> _eigvec;

	void initialize(const IVarFun<Dim>& varfun);
};

#endif
