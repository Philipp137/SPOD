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
import os
from os.path import expanduser

import numpy as np
from numpy import mod, meshgrid, cos, sin, exp, pi
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Importing custom modules from the project library
sys.path.append("../lib")
from IO import read_ACM_dat
from transforms import Transform
from plot_utils import save_fig
from farge_colormaps import farge_colormap_multi
from utils import finite_diffs

# Initialize colormap for plotting
fc = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)

# Define paths
ROOT_DIR = os.path.dirname(os.path.abspath("README.md"))
home = expanduser("~")

###############################################################################
# %% FUNCTION DEFINITIONS
###############################################################################

def path(mu_vec, time, freq):
    """Compute the path of the moving object based on mu_vec, time, and frequency."""
    return mu_vec[0] * cos(2 * pi * freq * time)

def smooth_sign_distance(x, L):
    """Smooth step function for ALE method."""
    return np.where(x <= 0, 0, np.where(x > L, 1, 1 - 0.5 * (1 + np.cos(np.pi * x / L))))

def dpath(mu_vec, time, freq):
    """Compute the derivative of the path."""
    return pi * freq * (-2 * mu_vec[0] * sin(2 * pi * freq * time))

def give_shift(time, x, mu_vec, freq):
    """Calculate shift for the ALE method."""
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        shift[..., it] = path(mu_vec, np.heaviside(x, 0) * x - t, freq)
    return shift

def give_shift_ALE(time, x, mu_vec, freq, L):
    """Calculate ALE shift with a smooth transition."""
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        y1 = path(mu_vec, t, freq)
        y0 = 0
        shift[..., it] = y0 + smooth_sign_distance(x, L) * (y1 - y0)
    return shift

def show_animation_headless(q, Xgrid=None, cycles=1, frequency=1, cmap=fc, vmin=None, vmax=None, save_path=None):
    """Generate and save animation frames headlessly."""
    ntime = q.shape[-1]
    if Xgrid is not None:
        for t in range(0, cycles * ntime, frequency):
            fig, ax = plt.subplots()
            if vmin is None:
                vmin = np.min(q)
            if vmax is None:
                vmax = np.max(q)
            h = ax.pcolormesh(Xgrid[0], Xgrid[1], q[..., t % ntime], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("image")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(r"x")
            ax.set_ylabel(r"y")
            fig.colorbar(h)

            if save_path is not None:
                fig.savefig(f"{save_path}/vid_{t:03d}.png")
            plt.close(fig)
    else:
        x = np.arange(0, np.size(q, 0))
        for t in range(0, cycles * ntime, frequency):
            fig, ax = plt.subplots()
            (h,) = ax.plot(x, q[:, t % ntime])
            ax.set_ylim(np.min(q), np.max(q))
            ax.set_xlabel(r"x")

            if save_path is not None:
                fig.savefig(f"{save_path}/vid_{t:03d}.png")
            plt.close(fig)


###############################################################################
# %% MAIN SCRIPT STARTS HERE
###############################################################################

# Close all open figures
plt.close("all")

# Define paths and settings
ddir = "./2cylinder"
idir = "./images/"
case = "vortex_street_ALE"
fpath = idir + case
shift_type = "ALE"
skip_existing = False
frac = 8  # Fraction of grid points to use
time_frac = 8
use_general_shift = True

time_sum = []
ux_list = []
uy_list = []
mu_vec_list = [[16]]

# Load ACM data
ux, uy, mask, p, time, Ngrid, dx, L = read_ACM_dat(ddir + "/ALL_2cyls_mu16.mat", sample_fraction=frac, time_sample_fraction=time_frac)
time_sum = time

# Define parameters
Nt = np.size(ux, -1)  # Number of time intervals
Nvar = 1  # Number of variables
nmodes = [40, 40]  # Reduction of singular values
data_shape = [*Ngrid, Nvar, Nt]
freq0 = 0.01 / 5
Radius = 1
T = time[-1]
C_eta = 2.5e-3
x, y = (np.linspace(0, L[i] - dx[i], Ngrid[i]) for i in range(2))
dX = (x[1] - x[0], y[1] - y[0])
dt = time[1] - time[0]
[Y, X] = meshgrid(y, x)
xext = np.linspace(0, L[0] - dx[0], Ngrid[0])
yext = np.linspace(-L[1], 2 * L[1] - dx[1], 3 * Ngrid[1])
[Yext, Xext] = meshgrid(yext, xext)
fd = finite_diffs(Ngrid, dX)

# Calculate vorticity
vort = np.asarray([fd.rot(ux[..., nt], uy[..., nt]) for nt in range(np.size(ux, 2))])
vort = np.moveaxis(vort, 0, -1)

# Plot full animation if needed
plot_full = False
if plot_full:
    os.makedirs("./images/full", exist_ok=True)
    show_animation_headless(vort.swapaxes(0, 1), Xgrid=[x, y], vmin=-1, vmax=1, save_path="images/full")

# Shift calculation based on shift type
if shift_type == "ALE":
    shift1 = np.zeros([2, np.prod(Ngrid), Nt])
    shift2_general = np.zeros([2, np.prod(Ngrid), Nt])
    y_shifts = []
    dy_shifts = []

    print(L)
    for mu_vec in mu_vec_list:
        print(f"mu = {mu_vec}")
        y_shifts.append(-give_shift_ALE(time, X.flatten() - (L[0] / 4 + Radius), mu_vec, freq0, (L[0] / 4 - 2 * Radius)))
        dy_shifts.append(dpath(mu_vec, time, freq0))
    dy_shifts = np.concatenate([*dy_shifts], axis=-1)
    shift2_general[1, ...] = np.concatenate([*y_shifts], axis=-1)

    shift_trafo_1 = Transform(data_shape, L, shifts=shift1, transfo_type="identity", dx=dX, use_scipy_transform=False)
    shift_trafo_2 = Transform(data_shape, L, shifts=shift2_general, dx=dX, use_scipy_transform=False, interp_order=5)
    trafos = [shift_trafo_1, shift_trafo_2]

# Reverse transformation to apply shifts
vort_shift = shift_trafo_2.reverse(vort)

# SVD and approximation
Nrank_max = 65
u1 = ux
u2 = uy
u1_shift = shift_trafo_2.reverse(u1)
u2_shift = shift_trafo_2.reverse(u2)

[U1,S1,VT1] = np.linalg.svd(np.reshape(u1_shift,[np.prod(Ngrid),Nt]), full_matrices=False)
[U2,S2,VT2] = np.linalg.svd(np.reshape(u2_shift,[np.prod(Ngrid),Nt]), full_matrices=False)

error = np.zeros([Nrank_max,2])
for rank in range(Nrank_max):
    u1_shift_tilde = U1[:,:rank]@np.diag(S1[:rank])@VT1[:rank,:]
    u2_shift_tilde = U2[:,:rank]@np.diag(S2[:rank])@VT2[:rank,:]

    u1_tilde = shift_trafo_2.apply(np.reshape(u1_shift_tilde,[*Ngrid,Nt]))
    u2_tilde = shift_trafo_2.apply(np.reshape(u2_shift_tilde,[*Ngrid,Nt]))

    error[rank,0] = np.linalg.norm(u1-u1_tilde)/np.linalg.norm(u1)
    error[rank,1] = np.linalg.norm(u2-u2_tilde)/np.linalg.norm(u2)
    
fig = plt.figure(44)
plt.semilogy(error[:,0],'x')
plt.xlabel("rank $r$")
plt.ylabel("relative error")
save_fig(fpath+"_rel_err_ux" ,fig)

fig = plt.figure(45)
plt.semilogy(error[:,1],'+')
plt.xlabel("rank $r$")
plt.ylabel("relative error")
save_fig(fpath+"_rel_err_uy" ,fig)

################################################################
# show approximation vs ale vs reference mesh
################################################################
vort_tilde = np.asarray(
    [
        fd.rot(u1_tilde[..., nt], u2_tilde[..., nt]) for nt in range(np.size(ux, 2))
    ]  # ux as shifsted truncated
)
vort_tilde = np.moveaxis(vort_tilde, 0, -1)

fpathvid = fpath+'/ALE_video/'
os.makedirs(fpathvid,exist_ok=True)

fpathtikz = fpath+'/ALE_tikz/'
os.makedirs(fpathtikz,exist_ok=True)

vort = vort.swapaxes(0, 1)
vort_shift = vort_shift.swapaxes(0, 1)
for it in range(Nt):
    fig = plt.figure(it)
    dix = Ngrid[0]//(30)
    #it =30
    opacity = 0.3
    fig.clf()
    print("time = "+str(time[it]))
    plt.subplot(1,3,1)
    
    x_min, x_max = 0, np.max(X)
    y_min, y_max = 0, np.max(Y)
    extent = [x_min, x_max, y_min, y_max]
    plt.imshow( vort[:,:,it],extent=extent, origin='lower',cmap=fc, vmin=-1, vmax=1)
    Xdeformed= Xext
    Ydeformed= Yext+ smooth_sign_distance(Xdeformed - (L[0]/4+Radius),(1*L[0]/4-2*Radius)) * (path(mu_vec_list[0], time[it], freq0))
    plt.plot(Xdeformed[::,(3*Ngrid[1])//2],Ydeformed[::,(3*Ngrid[1])//2],'b-', lw=0.5)
    plt.plot(Xdeformed[::,::dix],Ydeformed[::,::dix],'k-', lw=0.3,alpha=opacity)
    plt.plot(Xdeformed[::dix,::dix].T,Ydeformed[::dix,::dix].T,'k-', lw=0.3,alpha=opacity)
    #plt.axis('off')  # Hide axes
    plt.ylim([0,y_max])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title("ALE - mesh", fontsize=10)


    plt.subplot(133)
    x_min, x_max = 0, np.max(X)
    y_min, y_max = 0, np.max(Y)
    extent = [x_min, x_max, y_min, y_max]
    plt.imshow( vort_shift[:,:,it],extent=extent, origin='lower',cmap=fc, vmin=-1, vmax=1)
    Xdeformed= Xext
    Ydeformed= Yext
    plt.plot(Xdeformed[::,(3*Ngrid[1])//2],Ydeformed[::,(3*Ngrid[1])//2],'b-', lw=0.5)
    plt.plot(Xdeformed[::,::dix],Ydeformed[::,::dix],'k-', lw=0.3,alpha=opacity)
    plt.plot(Xdeformed[::dix,::dix].T,Ydeformed[::dix,::dix].T,'k-', lw=0.3,alpha=opacity)
    #plt.axis('off')  # Hide axes
    plt.ylim([0,y_max])
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.xlabel(r'$x$')
    #plt.ylabel(r'$y$')
    plt.title("Reference - mesh", fontsize=10)

    plt.subplot(132)
    x_min, x_max = 0, np.max(X)
    y_min, y_max = 0, np.max(Y)
    extent = [x_min, x_max, y_min, y_max]
    plt.imshow( vort_tilde[:,:,it].swapaxes(0,1),extent=extent, origin='lower',cmap=fc, vmin=-1, vmax=1)
    Xdeformed= Xext
    Ydeformed= Yext+ smooth_sign_distance(Xdeformed - (L[0]/4+Radius),(1*L[0]/4-2*Radius)) * (path(mu_vec_list[0], time[it], freq0))
    plt.plot(Xdeformed[::,(3*Ngrid[1])//2],Ydeformed[::,(3*Ngrid[1])//2],'b-', lw=0.5)
    plt.plot(Xdeformed[::,::dix],Ydeformed[::,::dix],'k-', lw=0.3,alpha=opacity)
    plt.plot(Xdeformed[::dix,::dix].T,Ydeformed[::dix,::dix].T,'k-', lw=0.3,alpha=opacity)
    #plt.axis('off')  # Hide axes
    plt.ylim([0,y_max])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$x$')
    #plt.ylabel(r'$y$')
    plt.title("ALE - approximation", fontsize=10)

    fig.savefig(f"{fpathvid}/vid_{it:03d}.png", dpi=800, transparent=True)
    
#    save_fig(f"{fpathtikz}/vid_{it:03d}.png",fig)
    plt.close(fig)

#    




plt.show()


#exit(1)





os.makedirs("./images/ALE", exist_ok=True)
show_animation_headless(
    vort_shift.swapaxes(0, 1),
    Xgrid=[x, y],
    vmin=-1,
    vmax=1,
    save_path="images/ALE",
)

exit(0)


