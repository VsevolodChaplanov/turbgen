import numpy as np
import scipy as sp

N = 11
N2 = N*N
A = -1.0
B = 1.0
L = B - A
hx = L/N
hk = 2*np.pi/L


def rfun(x, y):
    return np.power(1/(np.abs(x)+1), 6) * np.power(1/(np.abs(y)+1), 4)


x = np.arange(0, N)*hx + A + hx/2
k = np.arange(0, N)*hk - hk*N/2 + hk/2
xtile = np.tile(x, (N, 1))
txtile = np.transpose(xtile)
ktile = np.tile(k, (N, 1))
tktile = np.transpose(ktile)
r = rfun(xtile, txtile)

print("== manual computing of ft")
phi = np.ones((N, N)) * 0j
for j, kj in enumerate(k):
    for i, ki in enumerate(k):
        v = np.exp(-1j * (xtile*ki + np.transpose(xtile)*kj))
        phi_ij = np.sum(r * v) * hx * hx / (2*np.pi) / (2*np.pi)
        # print("j,i=", j, i, "=>", phi_ij)
        phi[j, i] = phi_ij

print("== restructure fft")
phi_fft = sp.fft.fftn(r)
for j in range(phi_fft.shape[0]):
    for i in range(phi_fft.shape[1]):
        fft_ij = phi_fft[j, i]
        i_kappa = int((i + (N - 1)/2) % N)
        j_kappa = int((j + (N - 1)/2) % N)
        kappa = k[i_kappa]
        num = np.exp(-1j*(k[i_kappa] + k[j_kappa])*x[0])
        phi_ij = fft_ij * hx * hx / (2*np.pi) / (2*np.pi) * num
        diff = np.abs(phi_ij - phi[j_kappa, i_kappa])
        ok = diff < 1e-16
        print("j,i=", j_kappa, i_kappa, "=>", phi_ij, "vs", phi[j_kappa, i_kappa], "diff", ok)

print("== manual computing of ift")
for i, xi in enumerate(x):
    for j, xj in enumerate(x):
        v = np.exp(1j * (ktile*xi + tktile*xj))
        r_ij = np.sum(phi*v) * hk * hk
        print("j,i=", j, i, "=>", r_ij, "vs", r[j, i])

print("== restructure ifft")
r2_ifft = sp.fft.ifftn(phi)
for j in range(r2_ifft.shape[0]):
    for i in range(r2_ifft.shape[1]):
        i_x = int((i + (N - 1)/2) % N)
        i_y = int((j + (N - 1)/2) % N)
        xt = x[i_x]
        yt = x[i_y]
        num = np.exp(1j*k[0]*(xt + yt))
        r_ij = r2_ifft[j, i] * N * N * hk * hk * num
        print("j,i=", i_y, i_x, "=>", r_ij, "vs", r[i_y, i_x])
