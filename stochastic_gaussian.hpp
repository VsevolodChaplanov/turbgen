#ifndef _STOCHASTIC_GAUSSIAN_HPP_
#define _STOCHASTIC_GAUSSIAN_HPP_

#include <functional>
#include <string>
#include <array>
#include <armadillo>
#include "space.hpp"
#include "varfun.hpp"

class StochasticGaussian1{
public:
	struct Params{
		size_t eigen_cut = 1000;
		double variance_cut = 0.05;
	};

	StochasticGaussian1(
		PhysicalSpace space,
		const IVarFun1& varfun,
		Params params);

	const PhysicalSpace& space() const;
	arma::Col<double> generate(size_t seed) const;

	void save_state(std::string fn) const;
	static StochasticGaussian1 load_state(std::string fn);

	std::string dstr() const;
private:
	const PhysicalSpace _space;
	const Params _params;
	arma::Col<double> eigval;
	arma::Mat<double> eigvec;

	void initialize(const IVarFun1& varfun);
};

class StochasticGaussian3{
public:
	struct Params{
		size_t eigen_cut = 1000;
		double variance_cut = 0.05;
	};

	StochasticGaussian3(
		PhysicalSpace space,
		const IVarFun3& varfun,
		Params params);

	const PhysicalSpace& space() const;
	std::array<std::vector<double>, 3> generate(size_t seed) const;

	void save_state(std::string fn) const;
	static StochasticGaussian1 load_state(std::string fn);

	std::string dstr() const;
private:
	const PhysicalSpace _space;
	const Params _params;
	arma::Col<double> eigval;
	arma::Mat<double> eigvec;

	void initialize(const IVarFun3& varfun);
};


#endif
