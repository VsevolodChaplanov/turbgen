#ifndef _SPACE_HPP
#define _SPACE_HPP

#include <array>
#include <vector>
#include <string>
#include <fstream>

using point_t = std::array<double, 3>;

// 3d symmetric cube [-L/2, L/2]
// divided by N segments in each direction
struct Space{
	Space(size_t N, double L);

	point_t point(size_t i, size_t j, size_t k) const;
	point_t point(size_t ind) const;

	size_t lin_index(size_t i, size_t j, size_t k) const;
	std::array<size_t, 3> tri_index(size_t ind) const;

	bool point_within(const point_t& p) const;
	double interpolate_at(const point_t& p, const std::vector<double>& f) const;
	std::array<double, 3> interpolate_at3(const point_t& p,
			const std::vector<double>& f1,
			const std::vector<double>& f2,
			const std::vector<double>& f3) const;
	std::array<double, 6> interpolate_at6(const point_t& p,
			const std::vector<double>& f1,
			const std::vector<double>& f2,
			const std::vector<double>& f3,
			const std::vector<double>& f4,
			const std::vector<double>& f5,
			const std::vector<double>& f6) const;

	const size_t N;
	const size_t N3;
	const double L;
	const double A;
	const double B;
	const double h;
	const std::vector<double> coo;

	void tovtk(std::string fn, const std::vector<double>& f) const{
		return tovtk(fn, f.begin(), f.end());
	}

	void tovtk(std::string fn, const std::vector<std::string>& data_fieldnames,
	           const std::vector<const std::vector<double>*>& data_fields) const{
		std::ofstream fs(fn);
		tovtk_init(fs, -1);
		for (size_t i=0; i<data_fieldnames.size(); ++i){
			tovtk_init_data(fs, data_fieldnames[i], 1);
			auto begin = data_fields[i]->begin();
			auto end = data_fields[i]->end();
			for (auto it=begin; it != end; ++it){
				fs << *it << std::endl;
			}
		}
	}

	template<typename DataIter>
	void tovtk(std::string fn, DataIter begin, DataIter end) const{
		std::ofstream fs(fn);
		tovtk_init(fs);
		for (DataIter it=begin; it != end; ++it){
			fs << *it << std::endl;
		}
	}

	void tovtk(std::string fn,
			const std::vector<double>& u,
			const std::vector<double>& v) const{
		return tovtk(fn, u.begin(), u.end(), v.begin(), v.end());
	}

	template<typename DataIter>
	void tovtk(std::string fn,
			DataIter ubegin, DataIter uend,
			DataIter vbegin, DataIter vend) const{
		std::ofstream fs(fn);
		tovtk_init(fs, 2);
		DataIter uit = ubegin;
		DataIter vit = vbegin;
		while (uit != uend && vit != vend){
			fs << *uit++ << " " << *vit++ << std::endl;
		}
	}

	void tovtk(std::string fn,
			const std::vector<double>& u,
			const std::vector<double>& v,
			const std::vector<double>& w) const{
		return tovtk(fn, u.begin(), u.end(), v.begin(), v.end(), w.begin(), w.end());
	}

	template<typename DataIter>
	void tovtk(std::string fn,
			DataIter ubegin, DataIter uend,
			DataIter vbegin, DataIter vend,
			DataIter wbegin, DataIter wend) const{
		std::ofstream fs(fn);
		tovtk_init(fs, 3);
		DataIter uit = ubegin;
		DataIter vit = vbegin;
		DataIter wit = wbegin;
		while (uit != uend && vit != vend && wit != wend){
			fs << *uit++ << " " << *vit++ << " " << *wit++ << std::endl;
		}
	}
private:
	void tovtk_init(std::ostream& fs, int datadim=1) const;
	void tovtk_init_data(std::ostream& ofs, std::string data_name, int data_dim) const;
	double use_interpolation_polynom(const std::array<size_t, 8>& indices, const std::array<double, 8>& bases, const std::vector<double>& v) const;
	void interpolation_polynom(const point_t& p, std::array<size_t, 8>& indices, std::array<double, 8>& bases) const;
};

struct PhisicalSpace;
struct FourierSpace;

struct PhysicalSpace: public Space{
	PhysicalSpace(size_t N, double L);
	FourierSpace fourier_space() const;
};

struct FourierSpace: public Space{
	FourierSpace(size_t N, double Lk);
	PhysicalSpace physical_space() const;
};

#endif
