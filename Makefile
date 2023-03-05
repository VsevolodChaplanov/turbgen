FILELIST= a.cpp \
	ft.cpp \
	space.cpp \
	stochastic_gaussian.cpp \
	varfun.cpp

all: $(FILELIST)
	g++ -O3 -o a $(FILELIST) -std=c++17 -lfftw3 -lm -larmadillo

test_ft: test_ft.cpp ft.cpp space.cpp
	g++ -O3 -o test_ft test_ft.cpp ft.cpp space.cpp -std=c++17 -lfftw3 -lm
