#ifndef _SEQSIM_HPP_
#define _SEQSIM_HPP_

#include <array>
#include <boost/geometry.hpp>
#include "space.hpp"

class KrigingInterpolator{
public:
	using point_t = std::array<double, 3>;
	using varfun_t = std::function<double(const point_t&)>;

	KrigingInterpolator(
		PhysSpace space,
		varfun_t varfun,
		int maxadj=30);

	void add_point(const point_t& point, double value);
	std::pair<double, double> calculate_mean_and_variance(const point_t& point) const;
private:
	const int _maxadj;

	typedef std::pair<TPnt, size_t> TRval;
	typedef boost::geometry::index::rtree<TRval, boost::geometry::index::rstar<8>> RTree;
	std::shared_ptr<RTree> rt;

};


#endif
