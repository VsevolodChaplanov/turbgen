import numpy as np
import scipy as sp

N = 11
A = -1.0
B = 1.0
L = B - A
hx = L/N
hk = 2*np.pi/L


def rfun(x, y, z):
    return np.power(1/(np.abs(x)+1), 6)\
            * np.power(1/(np.abs(y)+1), 4)\
            * np.power(1/(np.abs(z)+1), 5)


x = np.arange(0, N)*hx + A + hx/2
k = np.arange(0, N)*hk - hk*N/2 + hk/2
xtile1 = np.tile(x, (N, N, 1))
xtile2 = np.transpose(xtile1, (1, 2, 0))
xtile3 = np.transpose(xtile1, (2, 0, 1))
ktile1 = np.tile(k, (N, N, 1))
ktile2 = np.transpose(ktile1, (1, 2, 0))
ktile3 = np.transpose(ktile1, (2, 0, 1))
r = rfun(xtile1, xtile2, xtile3)

print("== manual computing of ft")
phi = np.ones((N, N, N)) * 0j
for s, ks in enumerate(k):
    for j, kj in enumerate(k):
        for i, ki in enumerate(k):
            v = np.exp(-1j * (xtile1*ki + xtile2*kj + xtile3*ks))
            phi_ijs = np.sum(r * v) * (hx/(2*np.pi))**3
            # print("j,i=", j, i, "=>", phi_ij)
            phi[s, j, i] = phi_ijs

print("== restructure fft")
phi_fft = sp.fft.fftn(r)
for s in range(phi_fft.shape[0]):
    for j in range(phi_fft.shape[1]):
        for i in range(phi_fft.shape[2]):
            fft_ijs = phi_fft[s, j, i]
            i_kappa = int((i + (N - 1)/2) % N)
            j_kappa = int((j + (N - 1)/2) % N)
            s_kappa = int((s + (N - 1)/2) % N)
            num = np.exp(-1j*(k[i_kappa] + k[j_kappa] + k[s_kappa])*x[0])
            phi_ijs = fft_ijs * (hx/(2*np.pi))**3 * num
            diff = np.abs(phi_ijs - phi[s_kappa, j_kappa, i_kappa])
            ok = diff < 1e-16
            if phi_ijs > 1e-4:
                print("s,j,i=", s_kappa, j_kappa, i_kappa, "=>", phi_ijs, "vs", phi[s_kappa, j_kappa, i_kappa], "diff", ok)


print("== manual computing of ift")
for s, xs in enumerate(x):
    for j, xj in enumerate(x):
        for i, xi in enumerate(x):
            v = np.exp(1j * (ktile1*xi + ktile2*xj + ktile3*xs))
            r_ijs = np.sum(phi*v) * hk * hk * hk
            print("s,j,i=", s, j, i, "=>", r_ijs, "vs", r[s, j, i])

print("== restructure ifft")
r2_ifft = sp.fft.ifftn(phi)
for s in range(r2_ifft.shape[0]):
    for j in range(r2_ifft.shape[0]):
        for i in range(r2_ifft.shape[1]):
            i_x = int((i + (N - 1)/2) % N)
            i_y = int((j + (N - 1)/2) % N)
            i_z = int((s + (N - 1)/2) % N)
            xt = x[i_x]
            yt = x[i_y]
            zt = x[i_z]
            num = np.exp(1j*k[0]*(xt + yt + zt))
            r_ijs = r2_ifft[s, j, i] * N * N * N * hk * hk * hk * num
            print("s,j,i=", i_z, i_y, i_x, "=>", r_ijs, "vs", r[i_z, i_y, i_x])

            print(num)
            print(r2_ifft[s, j, i] * N * N * N)
            if i >= 1:
                quit()


def rfun_spherical(r):
    return np.power(1/(r+1), 6);
