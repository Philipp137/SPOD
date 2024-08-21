#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 06 15:08:42 2024

@author: Philipp Krah, Beata Zorawski, Arthur Marmin

The file provides numerical examples.
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys

sys.path.append("../lib")
import numpy as np
from numpy import mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from sPOD_tools import trunc_svd
from transforms import Transform
from plot_utils import save_fig
from timeit import default_timer as timer
# ============================================================================ #


# ============================================================================ #
#                              Auxiliary Functions                             #
# ============================================================================ #
def generate_data(Nx, Nt, case, noise_percent=0.2):
    Tmax = 0.5  # total time
    L = 1  # total domain size
    sigma = 0.015  # standard diviation of the puls
    x = np.arange(0, Nx) / Nx * L
    t = np.arange(0, Nt) / Nt * Tmax
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    c = 1
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T

    if case == "crossing_waves":
        nmodes = 1
        fun = lambda x, t: exp(-((mod((x - c * t), L) - 0.1) ** 2) / sigma**2) + exp(
            -((mod((x + c * t), L) - 0.9) ** 2) / sigma**2
        )

        # Define your field as a list of fields:
        # For example the first element in the list can be the density of a
        # flow quantity and the second element could be the velocity in 1D
        density = fun(X, T)
        velocity = fun(X, T)
        shifts1 = np.asarray(-c * t)
        shifts2 = np.asarray(c * t)
        Q = density  # , velocity]
        shift_list = [shifts1, shifts2]
    elif case == "sine_waves":
        delta = 0.0125
        # First frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # Second frame
        c2 = dx / dt
        shifts2 = -c2 * t
        q2 = np.zeros_like(X)

        x2 = 0.2 - shifts2
        q2 = exp(-((X - x2) ** 2) / delta**2)

        Q = q1 + q2
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "sine_waves_noise":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        shifts1 = -0.25 * cos(7 * pi * t)
        for r in np.arange(1, 5):
            x1 = 0.25 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t

        x2 = 0.2 - shifts2
        q2 = exp(-((X - x2) ** 2) / delta**2)
        Q = q1 + q2  # + E
        indices = np.random.choice(
            np.arange(Q.size), replace=False, size=int(Q.size * noise_percent)
        )
        Q = Q.flatten()
        Q[indices] = 1
        Q = np.reshape(Q, np.shape(q1))
        nmodes = [4, 1]
        shift_list = [shifts1, shifts2]

    elif case == "multiple_ranks":
        delta = 0.0125
        # first frame
        q1 = np.zeros_like(X)
        c2 = dx / dt
        shifts1 = c2 * t
        for r in np.arange(1, 5):
            x1 = 0.5 + 0.1 * r - shifts1
            q1 = q1 + sin(2 * pi * r * T / Tmax) * exp(-((X - x1) ** 2) / delta**2)
        # second frame
        c2 = dx / dt
        shifts2 = -c2 * t
        q2 = np.zeros_like(X)
        for r in np.arange(1, 3):
            x2 = 0.2 + 0.1 * r - shifts2
            q2 = q2 + cos(2 * pi * r * T / Tmax) * exp(-((X - x2) ** 2) / delta**2)

        Q = q1 + q2
        nmodes = [4, 2]
        shift_list = [shifts1, shifts2]

    return Q, shift_list, nmodes, L, dx


# ============================================================================ #


# ============================================================================ #
#                              CONSTANT DEFINITION                             #
# ============================================================================ #
PIC_DIR = "../images/"
SAVE_FIG = False
CASE = "multiple_ranks"
# CASE = "sine_waves"
Nx = 400  # number of grid points in x
Nt = Nx // 2  # number of time intervals
Niter = 20  # number of sPOD iterations
# METHOD = "ALM"
# METHOD = "BFB"
# METHOD = "JFB"
METHOD = "ALM"
# ============================================================================ #

Nt_list = np.logspace(3,11,9, base=2,dtype=np.int32)
Nx_list = np.logspace(7,16,10, base=2,dtype=np.int32)

# ============================================================================ #
# Test scaling with number of grid points
# ============================================================================ #
print(Nx_list)
tcpu_list =[]
tcpu_list_SVD = []
mu0 = 0.01
lambd0 = 0.01
for k,Nx in enumerate(Nx_list):
    print("\n\nNumber of grid points:", Nx)
    # Data Deneration
    fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, CASE)
    Nframes = 1 #len(nmodes) # Number
    data_shape = [Nx, 1, 1, Nt]
    transfos = [
        Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
        Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    ]

    qmat = np.reshape(fields, [Nx, Nt])
  
    myparams = sPOD_Param()
    myparams.maxit = Niter
    param_alm = None

    if METHOD == "ALM":
        param_alm = mu0  # adjust for case
    elif METHOD == "BFB":
        myparams.lambda_s = 0.3  # adjust for case
    elif METHOD == "JFB":
        myparams.lambda_s = 0.4  # adjust for case


    start = timer()
    ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm)
    tcpu_list.append( timer()- start)

    # timing for trunctated SVD
    start = timer()
    for i in range(Niter*Nframes):
        [U,S,V] = trunc_svd(qmat, nmodes_max=None, use_rSVD=False)
    tcpu_list_SVD.append( timer()- start)

fig = plt.figure(3)
coefs = np.polyfit(np.log(Nx_list), np.log(tcpu_list), 1)
handl, = plt.loglog(Nx_list, tcpu_list,'*')
plt.loglog(Nx_list, np.exp(coefs[1])*Nx_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], '*--', color=handl.get_color(), label=r"$t_\mathrm{cpu}(M)=\mathcal{O}(M^{%.1f})$" % coefs[0])




# ============================================================================ #
# Test scaling with number of snapshots
# ============================================================================ #
Nx = 2**11

tcpu_list_N = []
tcpu_list_SVD_N = []
for k,Nt in enumerate(Nt_list):
    print()
    print()
    print("Nt:", Nt)
    # Data Deneration
    fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, CASE)
    data_shape = [Nx, 1, 1, Nt]
    transfos = [
        Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
        Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    ]

    qmat = np.reshape(fields, [Nx, Nt])
  #  mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))*0.001
  #  lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
    myparams = sPOD_Param()
    myparams.maxit = Niter
    param_alm = None

    if METHOD == "ALM":
        param_alm = mu0  # adjust for case
    elif METHOD == "BFB":
        myparams.lambda_s = 0.3  # adjust for case
    elif METHOD == "JFB":
        myparams.lambda_s = 0.4  # adjust for case


    start = timer()
    ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm)
    tcpu_list_N.append( timer()- start)

    # timing for trunctated SVD
    start = timer()
    for i in range(Niter*Nframes):
        [U,S,V] = trunc_svd(qmat, nmodes_max=None, use_rSVD=False)
    tcpu_list_SVD_N.append( timer()- start)


coefs = np.polyfit(np.log(Nt_list), np.log(tcpu_list_N), 1)
handl, = plt.loglog(Nt_list, tcpu_list_N,'o')
plt.loglog(Nt_list, np.exp(coefs[1])*Nt_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], 'o-', color=handl.get_color(), label=r"$t_\mathrm{cpu}(N)=\mathcal{O}(N^{%.1f})$" % coefs[0])





# ============================================================================ #
# Test scaling with rSVD
# ============================================================================ #
tcpu_list_N =[]
tcpu_list_rSVD_N = []
for k,Nt in enumerate(Nt_list):
    print()
    print()
    print("Nt:", Nt)
    # Data Deneration
    fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt, CASE)
    data_shape = [Nx, 1, 1, Nt]
    transfos = [
        Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
        Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    ]

    qmat = np.reshape(fields, [Nx, Nt])
  #  mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))*0.001
  #  lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
    myparams = sPOD_Param()
    myparams.use_rSVD = True
    myparams.maxit = Niter
    param_alm = None

    if METHOD == "ALM":
        param_alm = mu0  # adjust for case
    elif METHOD == "BFB":
        myparams.lambda_s = 0.3  # adjust for case
    elif METHOD == "JFB":
        myparams.lambda_s = 0.4  # adjust for case


    start = timer()
    ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes=10)
    tcpu_list_N.append( timer()- start)

    # timing for trunctated SVD
    start = timer()
    for i in range(Niter*Nframes):
        [U,S,V] = trunc_svd(qmat, nmodes_max=10, use_rSVD=True)
    tcpu_list_rSVD_N.append( timer()- start)

coefs = np.polyfit(np.log(Nt_list), np.log(tcpu_list_N), 1)
handl, = plt.loglog(Nt_list, tcpu_list_N,'x')
plt.loglog(Nt_list, np.exp(coefs[1])*Nt_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], 'x-', color=handl.get_color(), label=r"$t_\mathrm{cpu}(N)=\mathcal{O}(N^{%.1f})$" % coefs[0])
plt.xlabel(r"dimension")
plt.ylabel(r"$t_\mathrm{cpu}$ [sec]")
plt.legend()
save_fig("timing_ADM",fig)



fig = plt.figure(54)
coefs = np.polyfit(np.log(Nx_list), np.log(tcpu_list_SVD), 1)
handl, = plt.loglog(Nx_list, tcpu_list_SVD,'*')
plt.loglog(Nx_list, np.exp(coefs[1])*Nx_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], '*--', color=handl.get_color(), label=r"$t_\mathrm{cpu}(M)=\mathcal{O}(M^{%.1f})$" % coefs[0])


coefs = np.polyfit(np.log(Nt_list), np.log(tcpu_list_SVD_N), 1)
handl, = plt.loglog(Nt_list, tcpu_list_SVD_N,'o')
plt.loglog(Nt_list, np.exp(coefs[1])*Nt_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], 'x-', color=handl.get_color(), label=r"$t_\mathrm{cpu}(N)=\mathcal{O}(N^{%.1f})$" % coefs[0])

coefs = np.polyfit(np.log(Nt_list), np.log(tcpu_list_rSVD_N), 1)
handl, = plt.loglog(Nt_list, tcpu_list_rSVD_N,'x')
plt.loglog(Nt_list, np.exp(coefs[1])*Nt_list**coefs[0],'-', color=handl.get_color())
plt.plot([], [], 'x-', color=handl.get_color(), label=r"$t_\mathrm{cpu}(N)=\mathcal{O}(N^{%.1f})$" % coefs[0])
plt.legend()

save_fig("timing_SVD",fig)

plt.show()