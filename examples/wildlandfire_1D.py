# -*- coding: utf-8 -*-
"""
@author: Philipp Krah, Beata Zorawski, Arthur Marmin
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys

sys.path.append("../lib")
import numpy as np
from numpy import meshgrid
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform
from plot_utils import save_fig

# ============================================================================ #


def generate_wildlandfire_data(grid, time, snapshot, shifts):
    x = np.load(grid, allow_pickle=True)[0]
    t = np.load(time)
    L = x[-1]
    Nx = len(x)
    Nt = len(t)
    q = np.load(snapshot)
    Q = q[:Nx, :]  # Temperature
    # Q = q[Nx:, :] # Supply mass
    shift_list = np.load(shifts)
    [X, T] = meshgrid(x, t)
    X = X.T
    T = T.T
    dx = x[1] - x[0]
    nmodes = [1, 1, 1]

    return Q, shift_list, L, dx, Nx, Nt, nmodes


SAVE_FIG = False
PIC_DIR = "../images/"
Niter = 4

fields, shift_list, L, dx, Nx, Nt, nmodes = generate_wildlandfire_data(
    f"../examples/data/Wildlandfire_1d/1D_Grid.npy",
    f"../examples/data/Wildlandfire_1d/Time.npy",
    f"../examples/data/Wildlandfire_1d/SnapShotMatrix558.49.npy",
    f"../examples/data/Wildlandfire_1d/Shifts558.49.npy",
)

data_shape = [Nx, 1, 1, Nt]
transfos = [
    Transform(data_shape, [L], shifts=shift_list[0], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[1], dx=[dx], interp_order=5),
    Transform(data_shape, [L], shifts=shift_list[2], dx=[dx], interp_order=5),
]

interp_err = np.max([give_interpolation_error(fields, trafo) for trafo in transfos])
print("interpolation error: %1.2e " % interp_err)

qmat = np.reshape(fields, [Nx, Nt])

# METHOD = "ALM"
METHOD = "ALM"
# METHOD = "BFB"
lambda0 = 4000  # for Temperature
# lambda0 = 27  # for supply mass
myparams = sPOD_Param()
myparams.maxit = Niter
param_alm = None
mu0 = Nx * Nt / (4 * np.sum(np.abs(qmat)))

if METHOD == "ALM":
    param_alm = mu0 * 0.01
elif METHOD == "BFB":
    myparams.lambda_s = lambda0
elif METHOD == "JFB":
    myparams.lambda_s = lambda0

ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm)

sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
qf = [
    np.squeeze(np.reshape(trafo.apply(frame.build_field()), data_shape))
    for trafo, frame in zip(transfos, ret.frames)
]


# %% 1. visualize your results: sPOD frames
##########################################
# first we plot the resulting field
gridspec = {"width_ratios": [1, 1, 1, 1, 1]}
fig, ax = plt.subplots(1, 5, figsize=(12, 5), gridspec_kw=gridspec, num=101)
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
# ax[2].axis("image")
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

for axes in ax[:4]:
    axes.set_aspect(0.6)

plt.colorbar(im2)
plt.tight_layout()

if SAVE_FIG:
    save_fig(PIC_DIR, fig)
plt.show()


# ####### Visualize convergence of co moving ranks ###################
xlims = [-1, Niter]
plt.close(11)
fig, ax = plt.subplots(num=11)
plt.plot(ret.ranks_hist[0], "+", label="$\mathrm{rank}(\mathbf{Q}^1)$")
plt.plot(ret.ranks_hist[1], "x", label="$\mathrm{rank}(\mathbf{Q}^2)$")
plt.plot(ret.ranks_hist[2], "*", label="$\mathrm{rank}(\mathbf{Q}^3)$")
plt.plot(xlims, [nmodes[0], nmodes[0]], "k--", label="exact rank $r_1=%d$" % nmodes[0])
plt.plot(xlims, [nmodes[1], nmodes[1]], "k-", label="exact rank $r_2=%d$" % nmodes[1])
plt.plot(xlims, [nmodes[2], nmodes[2]], "k-.", label="exact rank $r_3=%d$" % nmodes[2])
plt.xlim(xlims)
plt.xlabel("iterations")
plt.ylabel("rank $r_k$")
plt.legend()

left, bottom, width, height = [0.5, 0.45, 0.3, 0.35]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.pcolormesh(qmat)
ax2.axis("off")
ax2.set_title(r"$\mathbf{Q}$")

plt.show()
if SAVE_FIG:
    save_fig(PIC_DIR + "ranks_wildlandfire_S.png", fig)
