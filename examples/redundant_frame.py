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
import os
from numpy import mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

# ============================================================================ #


# ============================================================================ #
#                              Auxiliary Functions                             #
# ============================================================================ #
def generate_data(Nx, Nt):
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
    nmodes = [4, 2, 0]
    shift_list = [shifts1, shifts2,t**2]

    return Q, shift_list, nmodes, L, dx


# ============================================================================ #


# ============================================================================ #
#                              CONSTANT DEFINITION                             #
# ============================================================================ #
PIC_DIR = "../images/redundat_frame/"
SAVE_FIG = True
# CASE = "sine_waves"
Nx = 400  # number of grid points in x
Nt = Nx // 2  # number of time intervals
Niter = 500  # number of sPOD iterations
METHOD = "ALM"
# METHOD = "BFB"
# METHOD = "JFB"
# METHOD = "J2"
# ============================================================================ #

# ============================================================================ #
#                                 Main Program                                 #
# ============================================================================ #
# Clean-up
plt.close("all")
# Data Deneration
fields, shift_list, nmodes, L, dx = generate_data(Nx, Nt)
############################################
# %% CALL THE SPOD algorithm
############################################
data_shape = [Nx, 1, 1, Nt]
transfos = [
        Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
        Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
        Transform(data_shape, [L], shifts=shift_list[2], dx=[dx], interp_order=5),
    ]

interp_err = np.max([give_interpolation_error(fields, transfo) for transfo in transfos])
print("interpolation error: {:1.2e}".format(interp_err))
# %%
qmat = np.reshape(fields, [Nx, Nt])
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))
lambd0 = 1 / np.sqrt(np.maximum(Nx, Nt))
myparams = sPOD_Param()
myparams.maxit =Niter
param_alm = None

if METHOD == "ALM":
    param_alm = mu0  # adjust for case
elif METHOD == "BFB":
    myparams.lambda_s = 0.3  # adjust for case
elif METHOD == "JFB":
    myparams.lambda_s = 0.4  # adjust for case
ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm)

sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(transfo.apply(frame.build_field()), data_shape))
    for transfo, frame in zip(transfos, ret.frames)
]
############################################
# %% 1. visualize your results: sPOD frames
############################################
# first we plot the resulting field
gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
fig, ax = plt.subplots(1, 5, figsize=(12, 4), gridspec_kw=gridspec, num=101)
mycmap = "viridis"
vmin = np.min(qtilde) * 0.6
vmax = np.max(qtilde) * 0.6

ax[0].pcolormesh(qmat, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[0].set_title(r"$\mathbf{Q}$")
# ax[0].axis("image")
ax[0].axis("off")

ax[1].pcolormesh(qtilde, vmin=vmin, vmax=vmax, cmap=mycmap)
ax[1].set_title(r"$\tilde{\mathbf{Q}}$")
# ax[0].axis("image")
ax[1].axis("off")
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.
# If you want to plot the k-th frame use:
# 1. frame
plot_shifted = True
k_frame = 0
if plot_shifted:
    ax[2].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[2].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    ax[2].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[2].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[2].axis("off")
# ax[1].axis("image")
# 2. frame
k_frame = 1
if plot_shifted:
    im2 = ax[3].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[3].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[3].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[3].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[3].axis("off")

# 3. frame
k_frame = 2
if plot_shifted:
    im2 = ax[4].pcolormesh(qf[k_frame], vmin=vmin, vmax=vmax, cmap=mycmap)
    ax[4].set_title(r"$T^" + str(k_frame + 1) + "\mathbf{Q}^" + str(k_frame + 1) + "$")
else:
    im2 = ax[4].pcolormesh(sPOD_frames[k_frame].build_field())
    ax[4].set_title(r"$\mathbf{Q}^" + str(k_frame + 1) + "$")
ax[4].axis("off")

# ax[2].axis("image")

for axes in ax[:5]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

if SAVE_FIG:
    os.makedirs(PIC_DIR,exist_ok=True)
    save_fig(PIC_DIR + "01_traveling_wave_1D_Frames_redundant.png", fig)
#plt.show()

############################################
# %% convergence co-moving ranks
############################################
xlims = [-1, Niter]
plt.close(11)
fig, ax = plt.subplots(num=11)
plt.plot(ret.ranks_hist[0], "+", label="$\mathrm{rank}(\mathbf{Q}^1)$")
plt.plot(ret.ranks_hist[1], "x", label="$\mathrm{rank}(\mathbf{Q}^2)$")
plt.plot(ret.ranks_hist[2], "o", label="$\mathrm{rank}(\mathbf{Q}^3)$")
plt.plot(xlims, [nmodes[0], nmodes[0]], "k-", label="exact rank $r_1=%d$" % nmodes[0])
plt.plot(xlims, [nmodes[1], nmodes[1]], "k-", label="exact rank $r_2=%d$" % nmodes[1])
plt.plot(xlims, [nmodes[2], nmodes[2]], "k-", label="exact rank $r_3=%d$" % nmodes[2])
plt.xlim(xlims)
plt.xlabel("iterations")
plt.ylabel("rank $r_k$")
plt.legend()

# left, bottom, width, height = [0.2, 0.45, 0.3, 0.35]
# ax2 = fig.add_axes([left, bottom, width, height])
# ax2.pcolormesh(qmat)
# ax2.axis("off")
# ax2.set_title(r"$\mathbf{Q}$")

if SAVE_FIG:
    os.makedirs(PIC_DIR,exist_ok=True)
    save_fig(PIC_DIR + "/convergence_ranks_redundant.png", fig)

plt.show()