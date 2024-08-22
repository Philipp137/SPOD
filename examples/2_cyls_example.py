#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOVING CYLINDERS VORTEX STREET OPTIMIZATION
@author: Philipp Krah
"""

###############################################################################
# %% IMPORTED MODULES
###############################################################################
import sys

sys.path.append("../lib")
import numpy as np
from numpy import mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    save_frames,
    load_frames
)

from IO import read_ACM_dat
from transforms import Transform
from plot_utils import save_fig
from farge_colormaps import farge_colormap_multi
import os
from os.path import expanduser
from utils import *

fc = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)

plt.rcParams["text.usetex"] = False
ROOT_DIR = os.path.dirname(os.path.abspath("README.md"))
home = expanduser("~")

###############################################################################
cm = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)
###############################################################################
def path(mu_vec, time, freq):
    return mu_vec[0] * cos(2 * pi * freq * time)

def smooth_sign_distance(x,L):
#                ____________1     
#               /          
#            __/         
#           /
# 0________/
#         0  L/2 L     x->
################################
    result = np.where(x <= 0, 0, np.where(x > L, 1, 1 - 0.5 * (1 + np.cos(np.pi * (x) / (L)))))
    return result


def dpath(mu_vec, time, freq):
    return pi * freq * (-2 * mu_vec[0] * sin(2 * pi * freq * time))


def give_shift(time, x, mu_vec, freq):
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        shift[..., it] = path(mu_vec, np.heaviside(x, 0) * (x) - t, freq)
    return shift

def give_shift_ALE(time, x, mu_vec, freq, L):
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        y1 = path(mu_vec, t, freq)
        y0 = 0
        shift[..., it] = y0 + smooth_sign_distance(x,L) * (y1 - y0)
    return shift



# ============================================================================ #
#                                 Main Program                                 #
# ============================================================================ #
plt.close("all")
# ddir = ROOT_DIR+"/../data"
# ddir = "../data/1params_opt/"
ddir = "./data/2cylinder"
case = "vortex_street"
load_existing = True        # Load existing data if available
frac = 1  # fraction of grid points to use
time_frac = 1
METHOD = "ALM"              # Method to use (ALM,BFB,JFB)
idir = "./images/2cylinder/"+METHOD # Directory to save images
myparams = sPOD_Param()
myparams.maxit = 500        # maximal number of iterations
myparams.use_rSVD =True    # use rSVD instead of SVD in sPOD algorithm
myparams.isError = True     # make sure sPOD uses noise term
mu = 16.0
# read data
ux, uy, mask, p, time, Ngrid, dx, L = read_ACM_dat(
    ddir + "/ALL_2cyls_mu16.mat", sample_fraction=frac, time_sample_fraction=time_frac
)


# %% Extract data information
Nt = np.size(ux, -1)  # Number of time intervalls
Nvar = 1  # data.shape[0]                    # Number of variables
nmodes_max = [200,200 ] # reduction of singular values
Ngrid = np.shape(ux[..., 0])

# number of grid points in x
data_shape = [*Ngrid, Nvar, Nt]
print("Nx Ny Nvar Nt :", data_shape)
# size of time intervall
freq0 = 0.01 / 5
Radius = 1
T = time[-1]
C_eta = 2.5e-3
x, y = (np.linspace(0, L[i] - dx[i], Ngrid[i]) for i in range(2))
dX = (x[1] - x[0], y[1] - y[0])
dt = time[1] - time[0]
[Y, X] = meshgrid(y, x)
fd = finite_diffs(Ngrid, dX)

vort = np.asarray(
    [
        fd.rot(ux[..., nt], uy[..., nt]) for nt in range(np.size(ux, 2))
    ]  # ux as shifsted truncated
)
vort = np.moveaxis(vort, 0, -1)



# ============================================================================ #
#                               Shift calculation                              #
# ============================================================================ #

# %% calculate the shifts
shift1 = np.zeros([2, Nt])
shift2 = np.zeros([2, Nt])
y_shifts = []
dy_shifts = []

y_shifts.append(-path([mu], time, freq0))  # frame 2, shift in y
dy_shifts.append(dpath([mu], time, freq0))  # frame 2, shift in y
dy_shifts = np.concatenate(dy_shifts, axis=0)
shift2[1, :] = np.concatenate(y_shifts, axis=0)
shift_trafo_1 = Transform(
    data_shape,
    L,
    shifts=shift1,
    transfo_type="identity",
    dx=dX,
    use_scipy_transform=False,
)
shift_trafo_2 = Transform(
    data_shape, L, shifts=shift2, dx=dX, use_scipy_transform=False, interp_order=5
)
trafos = [shift_trafo_1, shift_trafo_2]

vort_shift = shift_trafo_2.reverse(vort)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        sPOD - Method
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("START "+ METHOD)

####################
# Perform sPOD algorithm
####################
q_ux = np.reshape(ux, [-1, Nt])
q_uy = np.reshape(uy, [-1, Nt])
[N, M] = np.shape(q_ux)
Nframes = len(trafos)

myparams.lambda_E = 1 / np.sqrt(np.maximum(M, N)) * 20
param_alm_ux = [None]
param_alm_uy = [None]
if METHOD == "ALM":
    param_alm_ux = N * M / (4 * np.sum(np.abs(q_ux))) * 5.0e-03,  # adjust for case
    param_alm_uy = N * M / (4 * np.sum(np.abs(q_uy))) * 4.0e-04,  # adjust for case
elif METHOD == "BFB":
    myparams.lambda_s = 1 # adjust for case
elif METHOD == "JFB":
    myparams.lambda_s = 1  # adjust for case

########################################################################
# RUN sPOD
if os.path.exists(ddir+"/Ux_"+METHOD+"_frames_00.pkl") and load_existing:
    print("--------------------------------------------------------")
    print("Loading sPOD Frames")
    print("--------------------------------------------------------")
    qframes_ux = load_frames(ddir+"/Ux_"+METHOD+"_frames", Nframes)
    qframes_uy = load_frames(ddir+"/Uy_"+METHOD+"_frames", Nframes)
else:
    ret_vort_ux= shifted_POD(q_ux, trafos, myparams, METHOD, param_alm = param_alm_ux[0], nmodes = nmodes_max)
    ret_vort_uy= shifted_POD(q_uy, trafos, myparams, METHOD, param_alm = param_alm_uy[0], nmodes = nmodes_max)
    save_frames(ddir+"/Ux_"+METHOD+"_frames", ret_vort_ux.frames, ret_vort_ux.error_matrix)
    save_frames(ddir+"/Uy_"+METHOD+"_frames", ret_vort_uy.frames, ret_vort_uy.error_matrix)
    qframes_ux = ret_vort_ux.frames
    qframes_uy = ret_vort_uy.frames
########################################################################

# compute error of ux

qtilde_ux = np.zeros_like(q_ux)
qk_ux = []
ranks_ux = [40, 31]
for k, (trafo, frame) in enumerate(zip(trafos, qframes_ux)):
    qtilde_ux += trafo.apply(frame.build_field(ranks_ux[k]))
    qk_ux.append(np.reshape(trafo.apply(frame.build_field(ranks_ux[k])), ux.shape))

qtilde_ux = np.reshape(qtilde_ux, ux.shape)

error_ux = np.linalg.norm(ux - qtilde_ux) / np.linalg.norm(ux)

####################
# UY component
####################

qtilde_uy = np.zeros_like(q_uy)
qk_uy = []
ranks_uy = [119, 128]
for k, (trafo, frame) in enumerate(zip(trafos, qframes_uy)):
    qtilde_uy += trafo.apply(frame.build_field(ranks_uy[k]))
    qk_uy.append(np.reshape(trafo.apply(frame.build_field(ranks_uy[k])), uy.shape))

qtilde_uy = np.reshape(qtilde_uy, uy.shape)

error_uy = np.linalg.norm(uy - qtilde_uy) / np.linalg.norm(uy)

####################
# visualization
####################


frame = 0
vort_1 = np.asarray(
    [
        fd.rot(qk_ux[frame][..., nt], qk_uy[frame][..., nt])
        for nt in range(np.size(qtilde_ux, 2))
    ]  # ux as shifsted truncated
)
vort_1 = np.moveaxis(vort_1, 0, -1)

frame = 1
vort_2 = np.asarray(
    [
        fd.rot(qk_ux[frame][..., nt], qk_uy[frame][..., nt])
        for nt in range(np.size(qtilde_ux, 2))
    ]  # ux as shifsted truncated
)
vort_2 = np.moveaxis(vort_2, 0, -1)


vort_tilde = np.asarray(
    [
        fd.rot(qtilde_ux[..., nt], qtilde_uy[..., nt])
        for nt in range(np.size(qtilde_ux, 2))
    ]
)
vort_tilde = np.moveaxis(vort_tilde, 0, -1)

ntime = vort.shape[-1]
hs = []
save_path = idir + "/video/"
os.makedirs(save_path,exist_ok=True)
vmin, vmax = -1,1
Xgrid = [x,y]
cmap = fc
vort = vort.swapaxes(0,1)
vort_tilde = vort_tilde.swapaxes(0,1)
vort_1 = vort_1.swapaxes(0,1)
vort_2 = vort_2.swapaxes(0,1)
for t in range(0, ntime):
            fig, axs = plt.subplots(1,4)
            if vmin is None:
                vmin = np.min(q)
            if vmax is None:
                vmax = np.max(q)
            
            h = axs[0].pcolormesh(
                Xgrid[0], Xgrid[1], vort[..., t % ntime], cmap=cmap, vmin=vmin, vmax=vmax
            )
            hs.append(h)
            h = axs[1].pcolormesh(
                Xgrid[0], Xgrid[1], vort_tilde[..., t % ntime], cmap=cmap, vmin=vmin, vmax=vmax
            )
            hs.append(h)
            h = axs[2].pcolormesh(
                Xgrid[0], Xgrid[1], vort_1[..., t % ntime], cmap=cmap, vmin=vmin, vmax=vmax
            )
            hs.append(h)
            h = axs[3].pcolormesh(
                Xgrid[0], Xgrid[1], vort_2[..., t % ntime], cmap=cmap, vmin=vmin, vmax=vmax
            )
            hs.append(h)
            for ax in axs:
                ax.axis("image")
                ax.set_xticks([])
                ax.set_yticks([])
                #ax.set_xlabel(r"$x$")
                #ax.set_ylabel(r"$y$")
            
            # for h in hs:
            #     fig.colorbar(h)

            if save_path is not None:
                fig.savefig(f"{save_path}/vid_{t:03d}.png",dpi=600)
            #plt.show()
            plt.close(fig)  # Close the figure to free memory


plt.show()

print("EROR UX:", error_ux)
print("EROR UY:", error_uy)