import numpy as np
import scipy as sp

N = 11
A = -1.0
B = 1.0
L = B - A
hx = L/N
hk = 2*np.pi/L


def rfun(r):
    return np.power(1/(np.abs(r)+1), 6)


x = np.arange(0, N)*hx + A + hx/2
k = np.arange(0, N)*hk - hk*N/2 + hk/2
r = rfun(x)

print("== manual computing of ft")
phi = np.ones(N) * 0j
for i, kj in enumerate(k):
    v = np.exp(-1j * x * kj)
    phi_j = np.sum(r * v) * hx / (2*np.pi)
    print("i=", i, "=>", phi_j)
    phi[i] = phi_j

print("== restructure fft")
phi_fft = sp.fft.fft(r)
for i, fft_j in enumerate(phi_fft):
    i_kappa = int((i + (N - 1)/2) % N)
    kappa = k[i_kappa]
    num = np.exp(-1j*kappa*x[0])
    phi_j = fft_j * hx / (2*np.pi) * num
    print("i=", i_kappa, "=>", phi_j)

print("== manual computing of ift")
r2 = np.ones(N) * 0j
for i, xj in enumerate(x):
    v = np.exp(1j*k*xj)
    r_j = np.sum(phi*v) * hk
    print("i=", i, "=>", r_j, "vs", r[i])
    r2[i] = r_j

print("== restructure ifft")
r2_ifft = sp.fft.ifft(phi)
for i, ifft_j in enumerate(r2_ifft):
    i_x = int((i + (N - 1)/2) % N)
    xt = x[i_x]
    num = np.exp(1j*k[0]*xt)
    r_j = ifft_j * N * hk * num
    print("i=", i_x, "=>", r_j, "vs", r[i_x])
