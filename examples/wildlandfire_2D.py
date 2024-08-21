# -*- coding: utf-8 -*-
"""
@author: Philipp Krah, Beata Zorawski, Arthur Marmin

"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import sys
from sklearn.utils.extmath import randomized_svd
import os

sys.path.append("../lib")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from sPOD_algo import (
    shifted_POD,
    sPOD_Param,
    give_interpolation_error,
)
from transforms import Transform

# ============================================================================ #


# ============================================================================ #
#                              AUXILIARY FUNCTIONS                             #
# ============================================================================ #
def cartesian_to_polar(cartesian_data, X, Y, t, fill_val=0):

    Nx = np.size(X)
    Ny = np.size(Y)
    Nt = np.size(t)
    X_grid, Y_grid = np.meshgrid(X, Y)
    X_c = X.shape[-1] // 2
    Y_c = Y.shape[-1] // 2
    aux = []

    X_new = X_grid - X_c  # Shift the origin to the center of the image
    Y_new = Y_grid - Y_c
    r = np.sqrt(X_new**2 + Y_new**2).flatten()  # polar coordinate r
    theta = np.arctan2(Y_new, X_new).flatten()  # polar coordinate theta

    # Make a regular (in polar space) grid based on the min and max r & theta
    N_r = Nx
    N_theta = Ny
    r_i = np.linspace(np.min(r), np.max(r), N_r)
    theta_i = np.linspace(np.min(theta), np.max(theta), N_theta)
    polar_data = np.zeros((N_r, N_theta, 1, Nt))

    import polarTransform

    for k in range(Nt):
        data, ptSettings = polarTransform.convertToPolarImage(
            cartesian_data[..., 0, k],
            radiusSize=N_r,
            angleSize=N_theta,
            initialRadius=np.min(r_i),
            finalRadius=np.max(r_i),
            initialAngle=np.min(theta_i),
            finalAngle=np.max(theta_i),
            center=(X_c, Y_c),
            borderVal=fill_val,
        )
        polar_data[..., 0, k] = data.transpose()
        aux.append(ptSettings)

    return polar_data, theta_i, r_i, aux


def polar_to_cartesian(polar_data, t, aux=None):
    Nt = len(t)
    cartesian_data = np.zeros_like(polar_data)

    for k in range(Nt):
        cartesian_data[..., 0, k] = aux[k].convertToCartesianImage(
            polar_data[..., 0, k].transpose()
        )

    return cartesian_data

# ============================================================================ #


# ============================================================================ #
#                                  MAIN PROGRAM                                #
# ============================================================================ #
impath = "../examples//data/Wildlandfire_2d/"
os.makedirs(impath, exist_ok=True)

# %% Read the data
SnapShotMatrix = np.load(impath + "SnapShotMatrix558.49.npy")
XY_1D = np.load(impath + "1D_Grid.npy", allow_pickle=True)
t = np.load(impath + "Time.npy")
XY_2D = np.load(impath + "2D_Grid.npy", allow_pickle=True)
delta = np.load(impath + "Shifts558.49.npy")
X = XY_1D[0]
Y = XY_1D[1]
X_2D = XY_2D[0]
Y_2D = XY_2D[1]

Nx = np.size(X)
Ny = np.size(Y)
Nt = np.size(t)
X_c = X[-1] // 2
Y_c = Y[-1] // 2


SnapShotMatrix = np.reshape(
    np.transpose(SnapShotMatrix), newshape=[Nt, 2, Nx, Ny], order="F"
)
T = np.transpose(
    np.reshape(np.squeeze(SnapShotMatrix[:, 0, :, :]), newshape=[Nt, -1], order="F")
)
Q = T

# Reshape the variable array to suit the dimension of the input for the sPOD
q = np.reshape(T, newshape=[Nx, Ny, 1, Nt], order="F")

# Map the field variable from cartesian to polar coordinate system
q_polar, theta_i, r_i, aux = cartesian_to_polar(q, X, Y, t, fill_val=0)

# Check the transformation back and forth error between polar and cartesian coordinates (Checkpoint)
q_cartesian = polar_to_cartesian(q_polar, t, aux=aux)
res = q - q_cartesian
err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
print(
    "Transformation back and forth error (cartesian - polar - cartesian) =  %4.4e "
    % err
)


data_shape = [Nx, Ny, 1, Nt]
dr = r_i[1] - r_i[0]
dtheta = theta_i[1] - theta_i[0]
d_del = np.asarray([dr, dtheta])
L = np.asarray([r_i[-1], theta_i[-1]])

# Create the transformations
trafo_1 = Transform(
    data_shape,
    L,
    shifts=np.reshape(delta[0], newshape=[2, -1, Nt]),
    dx=d_del,
    use_scipy_transform=False,
)
trafo_2 = Transform(
    data_shape,
    L,
    shifts=np.reshape(delta[1], newshape=[2, -1, Nt]),
    trafo_type="identity",
    dx=d_del,
    use_scipy_transform=False,
)

# Check the transformation interpolation error
err = give_interpolation_error(q_polar, trafo_1)
print("Transformation interpolation error =  %4.4e " % err)

#METHOD = "J2"
METHOD = "ALM"
# METHOD = "JFB"
# METHOD = "BFB"


transfos = [trafo_1, trafo_2]
qmat = np.reshape(q_polar, [-1, Nt])
param_alm = None
myparams = sPOD_Param()
if METHOD == "J2":
    print("START J2")
    myparams.maxit = 40
    modes_list = [5, 5]
    ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm)

    qframes, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
else:
    if METHOD == "ALM":
        print("START ALM")
        myparams.maxit = 8
        myparams.lambda_E = 1 / np.sqrt(np.max([Nx, Ny])) * 100
        myparams.use_rSVD = True
        param_alm = np.prod(np.size(qmat, 0)) / (4 * np.sum(np.abs(qmat))) * 0.5
        myparams.isError = True
        modes_list = [5, 5] + 45

    elif METHOD == "BFB":
        print("START BFB")
        myparams.maxit = 18
        myparams.lambda_s = 3e4
        modes_list = [5, 5] + 45

    elif METHOD == "JFB":
        print("START JFB")
        myparams.maxit = 15
        myparams.lambda_s = 3e4
        modes_list = [5, 5] +45

    ret = shifted_POD(qmat, transfos, myparams, METHOD, param_alm, nmodes= modes_list)
    qframes, qtilde, rel_err, ranks = (
        ret.frames,
        ret.data_approx,
        ret.rel_err_hist,
        ret.ranks,
    )
    modes_list = ranks

# Deduce the frames
q_frame_1 = np.reshape(qframes[0].build_field(), newshape=data_shape)
q_frame_2 = np.reshape(qframes[1].build_field(), newshape=data_shape)
qtilde = np.reshape(qtilde, newshape=data_shape)

# Transform the frame wise snapshots into lab frame (moving frame)
q_frame_1_lab = transfos[0].apply(q_frame_1)
q_frame_2_lab = transfos[1].apply(q_frame_2)

# Shift the pre-transformed polar data to cartesian grid to visualize
q_frame_1_cart_lab = polar_to_cartesian(q_frame_1_lab, t, aux=aux)
q_frame_2_cart_lab = polar_to_cartesian(q_frame_2_lab, t, aux=aux)
qtilde_cart = polar_to_cartesian(qtilde, t, aux=aux)

# Relative reconstruction error for sPOD
res = q - qtilde_cart
err_full = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
print("Error for full sPOD recons: {}".format(err_full))

# Relative reconstruction error for POD
U, S, VT = randomized_svd(Q, n_components=sum(modes_list), random_state=None)
Q_POD = U.dot(np.diag(S).dot(VT))
err_full = np.linalg.norm(np.reshape(Q - Q_POD, -1)) / np.linalg.norm(np.reshape(Q, -1))
print("Error for full POD recons: {}".format(err_full))
q_POD = np.reshape(Q_POD, newshape=[Nx, Ny, 1, Nt], order="F")

# Save the frame results when doing large computations
impath = "./2D_data/result_srPCA_2D/"
os.makedirs(impath, exist_ok=True)
np.save(impath + "q1_frame_lab.npy", q_frame_1_cart_lab)
np.save(impath + "q2_frame_lab.npy", q_frame_2_cart_lab)
np.save(impath + "qtilde.npy", qtilde_cart)
np.save(impath + "q_POD.npy", q_POD)
np.save(impath + "frame_modes.npy", modes_list, allow_pickle=True)


cmap = "YlOrRd"
plot_every = 5
immpath = f"./plots/{METHOD}/mixed/"
os.makedirs(immpath, exist_ok=True)
for n in range(Nt):
    if n % plot_every == 0:
        min = np.min(q[..., 0, n])
        max = np.max(q[..., 0, n])
        fig, ax = plt.subplots(2, 3, figsize=(15, 11))
        ax[0, 0].pcolormesh(
            X_2D,
            Y_2D,
            np.squeeze(qtilde_cart[:, :, 0, n]),
            vmin=min,
            vmax=max,
            cmap=cmap,
        )
        ax[0, 0].axis("scaled")
        ax[0, 0].set_title("sPOD")
        ax[0, 0].axhline(y=Y[Ny // 2 - 1], linestyle="--", color="g")
        ax[0, 0].set_yticks([], [])
        ax[0, 0].set_xticks([], [])

        ax[0, 1].pcolormesh(
            X_2D,
            Y_2D,
            np.squeeze(q_frame_1_cart_lab[:, :, 0, n]),
            vmin=min,
            vmax=max,
            cmap=cmap,
        )
        ax[0, 1].axis("scaled")
        ax[0, 1].set_title("Frame 1")
        ax[0, 1].axhline(y=Y[Ny // 2 - 1], linestyle="--", color="g")
        ax[0, 1].set_yticks([], [])
        ax[0, 1].set_xticks([], [])

        ax[0, 2].pcolormesh(
            X_2D,
            Y_2D,
            np.squeeze(q_frame_2_cart_lab[:, :, 0, n]),
            vmin=min,
            vmax=max,
            cmap=cmap,
        )
        ax[0, 2].axis("scaled")
        ax[0, 2].set_title("Frame 2")
        ax[0, 2].axhline(y=Y[Ny // 2 - 1], linestyle="--", color="g")
        ax[0, 2].set_yticks([], [])
        ax[0, 2].set_xticks([], [])

        ax[1, 0].plot(
            X,
            np.squeeze(q[:, Ny // 2, 0, n]),
            color="green",
            linestyle="-",
            label="actual",
        )
        ax[1, 0].plot(
            X,
            np.squeeze(qtilde_cart[:, Ny // 2, 0, n]),
            color="yellow",
            linestyle="--",
            label="sPOD",
        )
        ax[1, 0].plot(
            X,
            np.squeeze(q_POD[:, Ny // 2, 0, n]),
            color="black",
            linestyle="-.",
            label="POD",
        )
        ax[1, 0].set_ylim(bottom=min - max / 10, top=max + max / 10)
        ax[1, 0].legend()
        ax[1, 0].grid()

        ax[1, 1].plot(
            X,
            np.squeeze(q[:, Ny // 2, 0, n]),
            color="green",
            linestyle="-",
            label="actual",
        )
        ax[1, 1].plot(
            X,
            np.squeeze(q_frame_1_cart_lab[:, Ny // 2, 0, n]),
            color="blue",
            linestyle="--",
            label="Frame 1",
        )
        ax[1, 1].set_ylim(bottom=min - max / 10, top=max + max / 10)
        ax[1, 1].legend()
        ax[1, 1].grid()

        ax[1, 2].plot(
            X,
            np.squeeze(q[:, Ny // 2, 0, n]),
            color="green",
            linestyle="-",
            label="actual",
        )
        ax[1, 2].plot(
            X,
            np.squeeze(q_frame_2_cart_lab[:, Ny // 2, 0, n]),
            color="red",
            linestyle="--",
            label="Frame 2",
        )
        ax[1, 2].set_ylim(bottom=min - max / 10, top=max + max / 10)
        ax[1, 2].legend()
        ax[1, 2].grid()

        fig.savefig(immpath + "mixed" + str(n), dpi=200, transparent=True)
        plt.close(fig)
