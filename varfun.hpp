#ifndef _VARFUN_HPP_
#define _VARFUN_HPP_

#include "space.hpp"

class IVarFun1{
public:
	virtual ~IVarFun1() = default;

	using point_t = std::array<double, 3>;

	virtual double variance(const point_t& direction) const = 0;
	virtual double variance0() const = 0;
	virtual double max_value() const = 0;
};

class GridVariance1: public IVarFun1{
public:
	using IVarFun1::point_t;

	GridVariance1(PhysicalSpace space, std::vector<double>&& val);

	double variance(const point_t& direction) const override;
	double variance0() const override;
	double max_value() const override;
private:
	const PhysicalSpace _space;
	const std::vector<double> _value;
	double _center;
};


class IVarFun3{
public:
	virtual ~IVarFun3() = default;

	using point_t = std::array<double, 3>;

	virtual std::array<double, 6> variance(const point_t& direction) const = 0;
	virtual std::array<double, 6> variance0() const = 0;
	virtual double max_value() const = 0;
};


class GridVariance3: public IVarFun3{
public:
	using IVarFun3::point_t;

	GridVariance3(PhysicalSpace space,
			std::vector<double>&& val11,
			std::vector<double>&& val12,
			std::vector<double>&& val13,
			std::vector<double>&& val22,
			std::vector<double>&& val23,
			std::vector<double>&& val33);

	std::array<double, 6> variance(const point_t& direction) const override;
	std::array<double, 6> variance0() const override;
	double max_value() const override;
private:
	const PhysicalSpace _space;
	const std::vector<double> _value11, _value12, _value13, _value22, _value23, _value33;
	std::array<double, 6> _center;
};

#endif
