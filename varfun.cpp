#include "varfun.hpp"
#include <algorithm>

namespace{

size_t find_center_index(const PhysicalSpace& space){
	size_t i = size_t((space.N-1)/2);
	return space.lin_index(i, i, i);
}

}

GridVariance1::GridVariance1(PhysicalSpace space, std::vector<double>&& val):
	_space(space),
	_value(std::move(val)),
	_center(_value[find_center_index(space)]){}

double GridVariance1::variance(const point_t& direction) const{
	if (_space.point_within(direction)){
		return _space.interpolate_at(direction, _value);
	} else {
		return 0;
	}
}
double GridVariance1::variance0() const{
	return _center;

}
double GridVariance1::max_value() const{
	double vmax = *std::max_element(_value.begin(), _value.end());
	double vmin = *std::min_element(_value.begin(), _value.end());
	return std::max(std::abs(vmax), std::abs(vmin));
}

GridVariance3::GridVariance3(PhysicalSpace space,
		std::vector<double>&& val11,
		std::vector<double>&& val12,
		std::vector<double>&& val13,
		std::vector<double>&& val22,
		std::vector<double>&& val23,
		std::vector<double>&& val33):
	_space(space),
	_value11(std::move(val11)),
	_value12(std::move(val12)),
	_value13(std::move(val13)),
	_value22(std::move(val22)),
	_value23(std::move(val23)),
	_value33(std::move(val33)),
	_center({_value11[find_center_index(space)],
	         _value12[find_center_index(space)],
	         _value13[find_center_index(space)],
	         _value22[find_center_index(space)],
	         _value23[find_center_index(space)],
	         _value33[find_center_index(space)]})
{}

std::array<double, 6> GridVariance3::variance(const point_t& direction) const{
	if (_space.point_within(direction)){
		return _space.interpolate_at6(direction,
				_value11, _value12, _value13,
				_value22, _value23, _value33);
	} else {
		return {0, 0, 0, 0, 0, 0};
	}
}

std::array<double, 6> GridVariance3::variance0() const{
	return _center;
}

double GridVariance3::max_value() const{
	return *std::max_element(_center.begin(), _center.end());
}
