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
from numpy import exp, mod, meshgrid, pi, sin, size, cos
import scipy
import matplotlib.pyplot as plt
from plot_utils import *
import glob

from IO import read_ACM_dat
from transforms import Transform
from tabulate import tabulate
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from sPOD_tools import (
    Frame,
    reconstruction_error,
    build_all_frames,
)
from farge_colormaps import farge_colormap_multi
from plot_utils import show_animation, save_fig

import matplotlib
from os.path import expanduser
import os
from farge_colormaps import farge_colormap_multi
from IPython.display import HTML

fc = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)

ROOT_DIR = os.path.dirname(os.path.abspath("README.md"))
home = expanduser("~")

font = {"family": "normal", "weight": "bold", "size": 24}

matplotlib.rc("font", **font)
###############################################################################
cm = farge_colormap_multi(etalement_du_zero=0.2, limite_faible_fort=0.5)


###############################################################################
def save_frames(fname, frames, error_matrix=None):
    fname_base, old_ext = os.path.splitext(fname)
    ext = ".pkl"
    for k, frame in enumerate(frames):
        fname_frame = fname_base + "_%.2d" % k + ext
        print("frame %2d saved to: " % k, fname_frame)
        frame.save(fname_frame)
    if error_matrix is not None:
        fname_error_matrix = fname_base + "_error_mat.npy"
        np.save(fname_error_matrix, error_matrix)


def load_frames(fname, Nframes, load_ErrMat=False):
    fname_base, old_ext = os.path.splitext(fname)
    ext = ".pkl"

    # load frames
    frame_list = []
    for k in range(Nframes):
        fname_frame = fname_base + "_%.2d" % k + ext
        print("frame %2d loaded: " % k, fname_frame)
        newframe = frame(fname=fname_frame)
        frame_list.append(newframe)

    # load sparse error matrix
    fname_error_matrix = fname_base + "_error_mat.npy"
    if load_ErrMat and os.path.isfile(fname_error_matrix):
        E = np.load(fname_error_matrix)
        return frame_list, E
    else:
        return frame_list


def path(mu_vec, time, freq):
    return mu_vec[0] * cos(2 * pi * freq * time)


def dpath(mu_vec, time, freq):
    return pi * freq * (-2 * mu_vec[0] * sin(2 * pi * freq * time))


def give_shift(time, x, mu_vec, freq):
    shift = np.zeros([len(x), len(time)])
    for it, t in enumerate(time):
        shift[..., it] = path(mu_vec, np.heaviside(x, 0) * (x) - t, freq)
    return shift


def my_interpolated_state(
    sPOD_frames, frame_amplitude_list, mu_points, Ngrid, Nt, mu_vec
):
    from scipy.interpolate import griddata

    print(mu_vec)

    shift1 = np.zeros([2, Nt])
    shift2 = np.zeros([2, Nt])
    shift2[1, :] = -path(mu_vec)
    shiftsnew = [shift1, shift2]

    qtilde = 0
    for shift, frame, amplitudes in zip(shiftsnew, sPOD_frames, frame_amplitude_list):

        frame = Frame()
        Modes = frame.modal_system["U"]
        VT = []
        for k in range(frame.Nmodes):
            a = griddata(mu_points.T, amplitudes[k].T, mu_vec)
            VT.append(np.squeeze(a))
        VT = np.asarray(VT)
        Q = Modes[:, : frame.Nmodes] @ VT
        qframe = np.reshape(Q, [*Ngrid, 1, Nt])
        trafo = Transform(
            [*Ngrid, 1, Nt],
            frame.trafo.domain_size,
            shifts=shift,
            dx=frame.trafo.dx,
            use_scipy_transform=True,
        )
        qtilde += trafo.apply(qframe)

    return qtilde


def compute_offline_errors(
    q,
    transfos,
    methods=["ALM"],
    maxmodes=None,
    Niter=40,
    eps=1e-13,
    skip_existing=True,
    mu0=None,
    lambd0=None,
    case="vortex_street",
):

    [N, M] = np.shape(q)
    # lambd0 = mu0*5e2
    if mu0 is None:
        mu0 = N * M / (4 * np.sum(np.abs(q)))
    if lambd0 is None:
        lambd0 = 1 / np.sqrt(np.maximum(M, N))
    myparams = sPOD_Param()
    myparams.isError = True
    myparams.maxit = Niter
    myparams.eps = eps
    param_alm = None
    err_dict = {}
    for method in methods:
        if method == "ALM":
            sPOD_log_dir = ddir + "/ALM/" + case + "_mu0_%.3e" % mu0 + "/"
            if skip_existing and os.path.exists(sPOD_log_dir):
                Nframes = len(transfos)
                frame = Frame(Nmodes=Nframes, fname=sPOD_log_dir + "frame.npy")
                qframes, E = load_frames(
                    sPOD_log_dir + "frame.npy", Nframes, load_ErrMat=True
                )
                qtilde = build_all_frames(qframes)
            else:
                param_alm = mu0
                myparams.lambda_E = lambd0
                ret = shifted_POD(q, transfos, nmodes, myparams, method, param_alm)
                os.makedirs(
                    sPOD_log_dir, exist_ok=True
                )  # succeeds even if directory exists.
                qframes, qtilde, rel_err_list = (
                    ret.frames[:],
                    np.reshape(ret.data_approx, data_shape),
                    ret.rel_err_hist,
                )
                E = np.reshape(ret.error_matrix, data_shape)
                save_frames(sPOD_log_dir + "frame.npy", qframes, error_matrix=E)

            log_save = sPOD_log_dir + "err_mat_ALM.out"
            nModes = np.asarray([frame.Nmodes for frame in qframes])
            if maxmodes is None:
                nmodes_max = nModes
            else:
                nmodes_max = np.asarray([maxmodes for frame in qframes])
            # err_mat = reconstruction_error(q, qframes, transfos, max_ranks=nModes)
            # ret = force_constraint(qframes.copy(), transfos, q, Niter=1, alphas = [1,0])
            err_mat_rPCA = reconstruction_error(
                q, ret.frames, transfos, max_ranks=nmodes_max
            )

            err_dict[method] = err_mat_rPCA
            np.savetxt(log_save, err_mat_rPCA, delimiter=",")

        if method == "J2":
            if maxmodes is None:
                nmodes_max = int(np.max(nModes))
            else:
                nmodes_max = maxmodes
            norm_q = np.linalg.norm(q, ord="fro")
            max_dof = np.sum(np.asarray(nmodes_max) * 2 + 1)
            err_mat = 2 * np.ones_like(err_mat_rPCA)
            for k in range(0, np.shape(err_mat_rPCA)[0]):
                nModes = np.asarray(err_mat_rPCA[k, :2], dtype=np.int32)
                print("mode combi: [", *nModes, "]\n")
                rank_text = "_".join(map(str, nModes))
                sPOD_log_dir = ddir + "/sPOD/" + case + "_ranks_" + rank_text + "/"
                if skip_existing and os.path.exists(sPOD_log_dir):
                    Nframes = len(transfos)
                    qframes = load_frames(
                        sPOD_log_dir + "frame.npy", Nframes, load_ErrMat=False
                    )
                    qtilde = build_all_frames(qframes)
                else:
                    ret = shifted_POD(q, transfos, nModes, myparams, method, param_alm)
                    print(
                        "\n\nmodes: [",
                        *nModes,
                        "] error = %4.4e \n\n" % ret.rel_err_hist[-1]
                    )
                    os.makedirs(
                        sPOD_log_dir, exist_ok=True
                    )  # succeeds even if directory exists.
                    qframes, qtilde, rel_err_list = (
                        ret.frames,
                        np.reshape(ret.data_approx, data_shape),
                        ret.rel_err_hist,
                    )
                    save_frames(sPOD_log_dir + "frame.npy", qframes)

                rel_err = (
                    np.linalg.norm(np.reshape(qtilde, np.shape(q)) - q, ord="fro")
                    / norm_q
                )
                dof = np.sum(nModes)
                if rel_err < err_mat[dof, -1]:
                    err_mat[dof, -1] = rel_err
                    for ir, r in enumerate(nModes):
                        err_mat[dof, ir] = r

            log_save = sPOD_log_dir + "err_mat_J2.out"

            err_dict[method] = err_mat
            np.savetxt(log_save, err_mat, delimiter=",")

        if method == "BFB":
            if method == "BFB":
                BFB_log_dir = ddir + "/BFB/" + case + "_ranks_" + rank_text + "/"
            if skip_existing and os.path.exists(BFB_log_dir):
                Nframes = len(transfos)
                qframes, E = load_frames(
                    BFB_log_dir + "frame.npy", Nframes, load_ErrMat=True
                )
                qtilde = build_all_frames(qframes)
            else:
                myparams.lambda_s = 1
                ret = shifted_POD(q, transfos, nModes, myparams, method, param_alm)
                os.makedirs(
                    BFB_log_dir, exist_ok=True
                )  # succeeds even if directory exists.
                qframes, qtilde, rel_err_list = (
                    ret.frames[:],
                    np.reshape(ret.data_approx, data_shape),
                    ret.rel_err_hist,
                )

            log_save = BFB_log_dir + "err_mat_ALM.out"
            nModes = np.asarray([frame.Nmodes for frame in qframes])
            if maxmodes is None:
                nmodes_max = nModes
            else:
                nmodes_max = np.asarray([maxmodes for frame in qframes])
            # err_mat = reconstruction_error(q, qframes, transfos, max_ranks=nModes)
            # ret = force_constraint(qframes.copy(), transfos, q, Niter=1, alphas = [1,0])
            err_mat_BFB = reconstruction_error(
                q, ret.frames, transfos, max_ranks=nmodes_max
            )

            err_dict[method] = err_mat_BFB
            np.savetxt(log_save, err_mat_BFB, delimiter=",")
    return err_dict


##########################################
# %%% Define your DATA:
##########################################
plt.close("all")
# ddir = ROOT_DIR+"/../data"
# ddir = "../data/1params_opt/"
ddir = "./2cylinder"
idir = "./images/2cylinder"
case = "vortex_street"
shift_type = "general"
skip_existing = False  # True #False
frac = 8  # fraction of grid points to use
time_frac = 2
use_general_shift = True

time_sum = []
ux_list = []
uy_list = []
mu_vec_list = [[16]]
ux, uy, mask, p, time, Ngrid, dx, L = read_ACM_dat(
    ddir + "/ALL_2cyls_mu16.mat", sample_fraction=8, time_sample_fraction=4
)  # scipy.io.loadmat(ddir + "/ALL_2cyls_mu16.mat")
time_sum = time
# %%

Nt = np.size(ux, -1)  # Number of time intervalls
Nvar = 1  # data.shape[0]                    # Number of variables
nmodes = [40, 40]  # reduction of singular values
Ngrid = np.shape(ux[..., 0])

# number of grid points in x
data_shape = [*Ngrid, Nvar, Nt]
# size of time intervall
freq0 = 0.01 / 5
Radius = 1
T = time[-1]
C_eta = 2.5e-3
x, y = (np.linspace(0, L[i] - dx[i], Ngrid[i]) for i in range(2))
dX = (x[1] - x[0], y[1] - y[0])
dt = time[1] - time[0]
[Y, X] = meshgrid(y, x)
# fd = finite_diffs(Ngrid, dX)

# if True:
#    vort = np.asarray(
#        [fd.rot(ux[..., nt], uy[..., nt]) for nt in range(np.size(ux, 2))]
#    )
#    vort = np.moveaxis(vort, 0, -1)
if False:
    show_animation(
        vort.swapaxes(0, 1),
        Xgrid=[x, y],
        use_html=False,
        vmin=-1,
        vmax=1,
        save_path="images",
    )

# %% general_shift with wake correction:
if shift_type == "general":
    shift1 = np.zeros([2, np.prod(Ngrid), Nt])
    shift2_general = np.zeros([2, np.prod(Ngrid), Nt])
    # shift1 = np.zeros([2,Nsnapshots])
    # shift2 = np.zeros([2,Nsnapshots])

    # shift1[0,:] = 0 * time_joint                      # frame 1, shift in x
    # shift1[1,:] = 0 * time_joint                      # frame 1, shift in y
    # shift2[0,:] = 0 * time_joint                      # frame 2, shift in x
    y_shifts = []
    dy_shifts = []
    for mu_vec in mu_vec_list:
        print("mu = " + str(mu_vec))
        y_shifts.append(
            -give_shift(time, X.flatten() - L[0] / 2 - Radius, mu_vec, freq0)
        )
        # y_shifts.append(-path(mu_vec, time, freq0)) # frame 2, shift in y
        dy_shifts.append(dpath(mu_vec, time, freq0))  # frame 2, shift in y
    dy_shifts = np.concatenate([*dy_shifts], axis=-1)
    shift2_general[1, ...] = np.concatenate([*y_shifts], axis=-1)

    shift_trafo_1 = Transform(
        data_shape,
        L,
        shifts=shift1,
        transfo_type="identity",
        dx=dX,
        use_scipy_transform=False,
    )
    shift_trafo_2 = Transform(
        data_shape,
        L,
        shifts=shift2_general,
        dx=dX,
        use_scipy_transform=False,
        interp_order=5,
    )
    transfos = [shift_trafo_1, shift_trafo_2]

else:
    # %% naive shift
    shift1 = np.zeros([2, Nt])
    shift2 = np.zeros([2, Nt])
    shift1[0, :] = 0 * time_sum  # frame 1, shift in x
    shift1[1, :] = 0 * time_sum  # frame 1, shift in y
    shift2[0, :] = 0 * time_sum  # frame 2, shift in x
    y_shifts = []
    dy_shifts = []
    for mu_vec in mu_vec_list:
        y_shifts.append(-path(mu_vec, time, freq0))  # frame 2, shift in y
        dy_shifts.append(dpath(mu_vec, time, freq0))  # frame 2, shift in y
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
    transfos = [shift_trafo_1, shift_trafo_2]
# %% interp err

# interp_err_list_naive = []
# for q in [ux,uy,p,mask]:
#     qmat = np.reshape(q,[-1,Nt])
#     err = give_interpolation_error(qmat,transfos_naive[1])
#     interp_err_list_naive.append(err)

# interp_err_list_wake = []
# for q in [ux, uy, p, mask]:
#     qmat = np.reshape(q, [-1, Nt])
#     err = give_interpolation_error(qmat, transfos_wake[1])
#     interp_err_list_wake.append(err)

# table = [["interp. err. naive", *interp_err_list_naive],["interp. err. wake corr.", *interp_err_list_wake]]
# table_string = tabulate(table,headers=["","$u_1$","$u_2$","$p$", "$\chi$"], tablefmt='latex_booktabs',floatfmt=("", "1.1e", "1.1e", "1.1e","1.1e"), numalign="center")
# print(table_string)


# vort_shift = shift_trafo_2.reverse(vort)
# %%

# show_animation(vort_shift.swapaxes(0,1),Xgrid = [x,y],use_html=False,vmin=-1,vmax=1,save_path="../images/")
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                        velocity ux
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = np.reshape(ux, [-1, Nt])
Niter = 10
[N, M] = np.shape(q)
mu0 = N * M / (4 * np.sum(np.abs(q))) * 0.001
lambd0 = 1 / np.sqrt(np.maximum(M, N)) * 20
rel_err_dict = compute_offline_errors(
    q,
    transfos,
    methods=["ALM", "J2", "BFB"],
    eps=1e-9,
    Niter=Niter,
    skip_existing=skip_existing,
    mu0=mu0,
    lambd0=lambd0,
    case="ux",
)
err_mat_ALM = rel_err_dict["ALM"]
err_mat_J2 = rel_err_dict["J2"]
err_mat_BFB = rel_err_dict["BFB"]
# %%
nModes = err_mat_ALM[-1, :2]
maxrank = int(np.sum(nModes))
[_, ss, _] = np.linalg.svd(q, full_matrices=False)
err_POD = list(np.sqrt(1 - (np.cumsum(ss[:-1] ** 2)) / np.sum(ss[:-1] ** 2)))
err_POD.insert(0, 1)
# plot it
# fig, ax = plt.subplots(num=201)
# ax.semilogy(err_mat_ALM[:-1, -1], "*", label="sPOD-$\mathcal{J}_1$")
# ax.semilogy(err_mat_ALM2[:-1,-1], '*', label="sPOD-$\mathcal{J}_1$")
# ax.semilogy(err_mat_sPOD[:-1, -1], "+", label="sPOD-$\mathcal{J}_2$")
# ax.semilogy(err_POD[: maxrank - 1], "x", label="POD")
# ax.semilogy(err_mat_BFB[:-1, -1], "o", label="BFB")
# plt.legend(loc=1)
# plt.show()

nmodes_max_show = 190

err_mat_ALM = rel_err_dict["ALM"]
err_mat_J2 = rel_err_dict["J2"]
err_mat_BFB = rel_err_dict["BFB"]
nModes = (np.max(err_mat_J2[:, :2]) + 1) * 2
maxrank = int(np.sum(nModes))
[_, ss, _] = np.linalg.svd(q, full_matrices=False)
err_POD = list(np.sqrt(1 - (np.cumsum(ss[:-1] ** 2)) / np.sum(ss[:-1] ** 2)))
err_POD.insert(0, 1)
fig, ax = plt.subplots(num=201)
print(err_mat_ALM)
print(err_mat_J2)
err_mat_ALM[1, -1] = err_mat_J2[1, -1]
ax.semilogy(err_mat_ALM[:-1, -1], "*", label="ADM")
# ax.semilogy(err_mat_ALM2[:-1,-1], '*', label="sPOD-$\mathcal{J}_1$")
ax.semilogy(err_mat_J2[:-1, -1], "+", label="J2")
ax.semilogy(err_POD[: maxrank - 1], "x", label="POD")
ax.semilogy(err_mat_BFB[:-1, -1], "o", label="BFB")
plt.legend(loc=1)
plt.xlim([-1, nmodes_max_show])
plt.ylim([1e-2, 1.1])
plt.grid(which="both", linestyle="--")
save_fig("images/error_ux_two_cycl.png", fig)
plt.show()


DOFs = np.arange(0, nmodes_max_show - 1)

idx_show = [
    1,
    2,
    3,
    8,
    9,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    170,
    180,
]

dofs_show = [DOFs[i] for i in idx_show]
ePOD = [err_POD[i] for i in idx_show]
eBFB = [err_mat_BFB[i, -1] for i in idx_show]
esPOD = [err_mat_J2[i, -1] for i in idx_show]
eALM = [err_mat_ALM[i, -1] for i in idx_show]

table = np.stack((dofs_show, esPOD, eALM, eBFB, ePOD), axis=1)
table_string = tabulate(
    table,
    headers=["DOF", "shifted POD", "shifted ALM", "BFB", "POD"],
    tablefmt="latex_booktabs",
    floatfmt=(".0f", ".3f", ".3f", ".3f"),
)
print(table_string)

with open("tabl/error-ux-two-cyls-BFB-3000.tabl.tex", "w") as text_file:
    text_file.write(table_string)
