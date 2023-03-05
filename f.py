import numpy as np
import scipy as sp


def func(x, y):
    # return np.cos(2*np.pi * (17*x + 8*y)) + np.cos(2*np.pi * (4*x + 5*y)) + 1
    return 1 - x - y + x*y


def tovtk(func, h, fn):
    fd = open(fn, 'w')
    fd.write(f"# vtk DataFile Version 2.0\n")
    fd.write(f"Func\n")
    fd.write(f"ASCII\n")
    fd.write(f"DATASET RECTILINEAR_GRID\n")
    nx = func.shape[1]
    ny = func.shape[0]
    fd.write(f"DIMENSIONS {nx} {ny} 1\n")
    fd.write(f"X_COORDINATES {nx} double\n")
    for i in range(nx):
        fd.write(f"{i*h}\n")
    fd.write(f"Y_COORDINATES {ny} double\n")
    for i in range(ny):
        fd.write(f"{i*h}\n")
    fd.write(f"Z_COORDINATES 1 double\n")
    fd.write(f"0\n")
    fd.write(f"POINT_DATA {nx*ny}\n")
    fd.write(f"SCALARS data double 1\n")
    fd.write(f"LOOKUP_TABLE default\n")
    for j in range(ny):
        for i in range(nx):
            fd.write(f"{func[j, i]}\n")
    fd.close()


L = 1.0
N = 101
h = L/N

f = np.zeros(shape=(N, N))

for j in range(N):
    for i in range(N):
        f[j, i] = func(i*h, j*h)


r = (h*h)*np.real(sp.fft.fftn(f))

tovtk(f, h, "f.vtk")
tovtk(r, 2*np.pi/L, "r.vtk")
print(r[0][0])
print(r[0][1])
print(r[1][0])
print(r[1][1])
print(np.where(np.abs(r) > 0.1))
print(r[5][4])
print(r[8][17])
print(r[93][84])
print(r[96][97])
