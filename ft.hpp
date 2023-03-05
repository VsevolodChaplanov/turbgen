#ifndef _FT_HPP
#define _FT_HPP

#include "space.hpp"
#include <vector>
#include <array>

// N - partition in single direction. Total number of cells is N*N*N. Sould be odd.
// f is given in physical space on the cube [-L/2, L/2] as cell_centered values
// !! Only for real valued transformations
//
// => (1/2pi)^3 * int3{-L/2;L/2} ( f(x)*exp(-i*dot(x, k)*dx )
std::vector<double> fourier3(PhysicalSpace ps, const std::vector<double>& f);

// N - partition in single direction. Total number of cells is N*N*N. Sould be odd.
// fk is given in fourier space on the cube [-Lk/2, Lk/2] as cell_centered values
// !! Only for real valued transformations
//
// => int3{-Lk/2;Lk/2} ( fk(k)*exp(i*dot(x, k)*dk )
std::vector<double> inverse_fourier3(FourierSpace fs, const std::vector<double>& fk);


#endif
