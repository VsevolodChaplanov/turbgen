FILELIST=\
	ft.cpp \
	space.cpp \
	stochastic_gaussian.cpp \
	varfun.cpp \
	sequential_gaussian_1.cpp \

#MAINFILE=a.cpp
MAINFILE=stat_computer.cpp

all: $(FILELIST)
	g++ -O3 -o a $(MAINFILE) $(FILELIST) -std=c++17 -lfftw3 -lm -larmadillo -lpthread

test_ft: test_ft.cpp ft.cpp space.cpp
	g++ -O3 -o test_ft test_ft.cpp ft.cpp space.cpp -std=c++17 -lfftw3 -lm
