# -*- coding: utf-8 -*-
"""
Created on Wed Mar 06 15:08:42 2024

@author: Philipp Krah, Beata Zorawski, Arthur Marmin

This file provides the algorithms that solve the optimization problem associated
with the robust shifter proper orthogonal decomposition (sPOD).
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import os
import time
from timeit import default_timer as timer

import numpy as np
from scipy.linalg import sqrtm, khatri_rao
from numpy import reshape
from numpy.linalg import norm
from dataclasses import dataclass
from warnings import warn
from sPOD_tools import Frame, SVT, trunc_svd, shrink
# ============================================================================ #


# ============================================================================ #
#                            CLASS of Return Values                            #
# ============================================================================ #
class ReturnValue:
    """
    This class inherits all return values of the shifted POD routines
    """

    def __init__(
        self,
        frames,
        approximation,
        relaltive_error_hist=None,
        ranks=None,
        ranks_hist=None,
        error_matrix=None,
        tcpu_SVD = None,
    ):
        """
        Constructor.

        :param frames:
        :type frames:

        :param approximation:
        :type approximation:

        :param relative_err_hist:
        :type relative_err_hist: , optional

        :param ranks:
        :type ranks: , optional

        :param ranks_hist:
        :type ranks_hist: , optional

        :param error_matrix:
        :type error_matrix: , optional

        :param tcpu_SVD
        :type tcpu_SVD: , optional
        """
        self.frames = frames  # List of all frames
        self.data_approx = approximation  # Approximation of the snapshot data
        if relaltive_error_hist is not None:
            self.rel_err_hist = relaltive_error_hist
        if error_matrix is not None:
            self.error_matrix = error_matrix
        if ranks is not None:
            self.ranks = ranks
        if ranks_hist is not None:
            self.ranks_hist = ranks_hist
        if tcpu_SVD is not None:
            self.tcpu_SVD= tcpu_SVD

# ============================================================================ #


# ============================================================================ #
#                                sPOD ALGORITHMS                               #
# ============================================================================ #
def shifted_POD(snapshot_matrix, transforms, myparams, method, param_alm=None,
                nmodes=None, qt_frames=None):
    """
    This function aggregates all the different shifted_POD_Algo() methods to
    provide a unique interface.

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the JFB algorithm
    :type myparams: sPOD_Param

    :param method: Name of the method
    :type myparams: string

    :param param_alm: Parameter mu for ALM algorithm
    :type myparams: float

    :return:
    :rtype: :class:`ReturnValue`
    """
    if method == "ALM":
        return shifted_POD_ALM(
            snapshot_matrix,
            transforms,
            myparams,
            nmodes_max=nmodes,
            mu=param_alm,
            qt_frames=qt_frames
        )
    elif method == "BFB":
        return shifted_POD_FB(
            snapshot_matrix,
            transforms,
            myparams,
            nmodes_max=nmodes,
            method="BFB"
        )
    elif method == "JFB":
        return shifted_POD_FB(
            snapshot_matrix,
            transforms,
            myparams,
            nmodes_max=nmodes,
            method="JFB"
        )
    elif method == "J2":
        return shifted_POD_J2(snapshot_matrix, transforms, nmodes, myparams)


def shifted_POD_J2(
    snapshot_matrix,
    transforms,
    nmodes,
    myparams,
):
    """
    This function implements the J2 algorithm.

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the JFB algorithm
    :type myparams: sPOD_Param

    :return:
    :rtype: :class:`ReturnValue`
    """

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    Nframes = len(transforms)
    if np.size(nmodes) != Nframes:
        nmodes = list([nmodes]) * Nframes
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))

    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    rel_err = 1
    rel_err_list = []
    ranks_hist = [[] for r in range(Nframes)]
    sum_elapsed = 0
    while rel_err > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        #############################
        # 2.Step: Calculate Residual
        #############################
        res = q - qtilde
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        qtilde = np.zeros_like(q)
        ranks = [0] * Nframes

        ###########################
        # 3. Step: update frames
        ##########################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            res_shifted = trafo.reverse(res)
            q_frame_field = q_frame.build_field()
            stepsize = 1 / Nframes
            q_frame.set_orthonormal_system(
                q_frame_field + stepsize * res_shifted, myparams.use_rSVD
            )
            if myparams.total_variation_iterations > 0:
                q_frame.smoothen_time_amplitudes(
                    TV_iterations=myparams.tv_niter
                )
            qtilde += trafo.apply(q_frame.build_field())
            S = q_frame.modal_system["sigma"]
            U = q_frame.modal_system["U"]
            VT = q_frame.modal_system["VT"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        print(
            "it=%d rel_err= %4.4e t_cpu = %2.2f, ranks_frame ="
            % (current_it, rel_err, elapsed),
            *ranks
        )
        if (current_it > 5) and (
            np.abs(rel_err_list[-1] - rel_err_list[-4])
            < myparams.gtol * abs(rel_err_list[-1])
        ):
            break
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_BFBTV(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
):
    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    if myparams.isError:
        E = np.zeros_like(snapshot_matrix)
    Nframes = len(transforms)
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0
    if np.size(nmodes_max) != Nframes:
        nmodes = list([nmodes_max]) * Nframes
    else:
        nmodes = nmodes_max
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))
    D = generate_discr_diff_mat(q.shape[1])
    spnorm_D = np.linalg.norm(D, 2)
    
    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    objective_0 = 0.5 * norm(q, ord="fro") ** 2
    objective_list = [objective_0]
    rel_decrease = 1
    rel_decrease_list = [1]
    rel_err_list = []
    ranks_hist = [[] for r in range(Nframes)]
    sum_elapsed = 0

    V = []
    for k in range(Nframes):
        V.append(np.zeros(q.shape))
    
    while rel_decrease > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        ###################################
        #      2. Calculate residual      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        ranks = [0] * Nframes

        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            # Set the two step sizes
            tau = 1 / Nframes
            sigma = 0.99 / (tau*spnorm_D**2)
            # Set the constant along the sub-iterations
            res_shifted = trafo.reverse(res) 
            for iter in range(myparams.tv_niter):
                q_frame_field_old = q_frame.build_field()
                qtilde -= trafo.apply(q_frame_field_old)
                q_frame.set_orthonormal_system_svt(
                    q_frame_field_old + tau * res_shifted - tau*sigma*V[k]@D.T,
                    tau * myparams.lambda_s
                )
                # Dual update
                q_frame_field_new = q_frame.build_field()
                tmp = V[k] + (2*q_frame_field_new-q_frame_field_old) @ D
                V[k] = tmp - shrink(tmp, myparams.mu/sigma)

            S = q_frame.modal_system["sigma"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
            qtilde += trafo.apply(q_frame.build_field())
            if myparams.isError:
                res = q - qtilde - E
            else:
                res = q - qtilde
        if myparams.isError:
            E = shrink(E + stepsize * res, stepsize * myparams.lambda_E)
            objective = (
                0.5 * norm(res, ord="fro") ** 2
                + myparams.lambda_s
                * sum(norm(qk.build_field(), ord="nuc") for qk in qtilde_frames)
                + myparams.mu * sum(norm((qk.build_field()@D).flatten(), ord=1)
                                    for qk in qtilde_frames)
                + myparams.lambda_E * norm(E.flatten(), ord=1)
            )
        else:
            objective = 0.5 * norm(res, ord="fro") ** 2
            + myparams.lambda_s * sum(norm(qk.build_field(), ord="nuc")
                                      for qk in qtilde_frames)
            + myparams.mu * sum(norm((qk.build_field()@D).flatten(), ord=1)
                                for qk in qtilde_frames)
        objective_list.append(objective)
        rel_decrease = np.abs((objective_list[-1] - objective_list[-2])) / np.abs(
            objective_list[-1]
        )
        rel_decrease_list.append(rel_decrease)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                "ranks_frame = ".format(current_it, myparams.maxit, rel_err, elapsed),
                *ranks
            )
        if (current_it > 5) and (rel_decrease < myparams.gtol):
            break

    if myparams.isError:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_BFBTV_v2(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
):
    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    if myparams.isError:
        E = np.zeros_like(snapshot_matrix)
    Nframes = len(transforms)
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0
    if np.size(nmodes_max) != Nframes:
        nmodes = list([nmodes_max]) * Nframes
    else:
        nmodes = nmodes_max
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))
    D = generate_discr_diff_mat(q.shape[1])
    spnorm_D = np.linalg.norm(D, 2)
    
    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    objective_0 = 0.5 * norm(q, ord="fro") ** 2
    objective_list = [objective_0]
    rel_decrease = 1
    rel_decrease_list = [1]
    rel_err_list = []
    ranks_hist = [[] for r in range(Nframes)]
    sum_elapsed = 0

    # Dual variables for updating Vt^k
    Z = []
    for k in range(Nframes):
        Z.append(np.zeros(vt.shape))

    N_U = myparams.tv_niter
    N_S = myparams.tv_niter
    N_V = myparams.tv_niter
    step_U = 0.1 / Nframes
    step_S = 0.1 / Nframes
    step_V_prim = 0.1 / Nframes
    step_V_dual = 0.99 / (step_V_prim*spnorm_D**2)
    
    while abs(rel_decrease) > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        ###################################
        #      2. Calculate residual      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        ranks = [0] * Nframes

        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            # Compute the derivative of the smooth term w.r.t. Q^k
            res_shifted = trafo.reverse(res)
            # Update Qtilde
            q_frame_field = q_frame.build_field()
            qtilde -= trafo.apply(q_frame.build_field())
            # Perform SVD on Q^k
            S = q_frame.modal_system["sigma"]
            U = q_frame.modal_system["U"]
            VT = q_frame.modal_system["VT"]
            # 1) Update in U^k (projected gradient descent)
            for subiter in range(N_U):
                # Gradient step
                G_U = res_shifted*S@VT.T
                U = U + step_U * G_U
                # Projection step
                U = U @ np.linalg.inv(sqrtm(U.T@U))

                # Update residual
                q_frame_field_old = q_frame.build_field()
                qtilde -= trafo.apply(q_frame_field_old)
                q_frame.modal_system = {
                    "U": U,
                    "sigma": S,
                    "VT": VT
                }
                q_frame_field = q_frame.build_field()
                qtilde += trafo.apply(q_frame.build_field())
                if myparams.isError:
                    res = q - qtilde - E
                else:
                    res = q - qtilde
                res_shifted = trafo.reverse(res)
            # 2) Update in s^k (proximal gradient descent)
            for subiter in range(N_S):
                G_S = khatri_rao(VT.T,U).T@res_shifted.flatten()
                #G_S = res_shifted.T@U@VT
                S = shrink(S + step_S * G_S, myparams.lambda_s*step_S)

                # Update residual
                q_frame_field_old = q_frame.build_field()
                qtilde -= trafo.apply(q_frame_field_old)
                q_frame.modal_system = {
                    "U": U,
                    "sigma": S,
                    "VT": VT
                }
                q_frame_field = q_frame.build_field()
                qtilde += trafo.apply(q_frame.build_field())
                if myparams.isError:
                    res = q - qtilde - E
                else:
                    res = q - qtilde
                res_shifted = trafo.reverse(res)
            # 3) Update in V^k (RFBPD)
            for subiter in range(N_V):
                # 3.a) Primal update
                G_V = res_shifted.T@U*S
                VT = VT + step_V_prim*G_V - step_V_prim*step_V_dual*Z[k]@D.T
                VT = VT @ np.linalg.inv(sqrtm(VT.T@VT))
                # 3.b) Dual update
                tmp = Z[k] + VT@D
                Z[k] = tmp - shrink(tmp, myparams.mu/step_V_dual)

                # Update residual
                q_frame_field_old = q_frame.build_field()
                qtilde -= trafo.apply(q_frame_field_old)
                q_frame.modal_system = {
                    "U": U,
                    "sigma": S,
                    "VT": VT
                }
                q_frame_field = q_frame.build_field()
                qtilde += trafo.apply(q_frame.build_field())
                if myparams.isError:
                    res = q - qtilde - E
                else:
                    res = q - qtilde
                res_shifted = trafo.reverse(res)
            # Reconstruct Q^k from its SVD factors
            q_frame.modal_system = {
                "U": U,
                "sigma": S,
                "VT": VT
            }
            q_frame_field = q_frame.build_field()
            
            S = q_frame.modal_system["sigma"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
            qtilde += trafo.apply(q_frame.build_field())
            if myparams.isError:
                res = q - qtilde - E
            else:
                res = q - qtilde
        if myparams.isError:
            E = shrink(E + stepsize * res, stepsize * myparams.lambda_E)
            objective = (
                0.5 * norm(res, ord="fro") ** 2
                + myparams.lambda_s * sum(norm(qk.build_field(), ord="nuc")
                                          for qk in qtilde_frames)
                + myparams.mu * sum(norm((D@qk.modal_system["VT"].T).flatten(),
                                         ord=1) for qk in qtilde_frames)
                + myparams.lambda_E * norm(E.flatten(), ord=1)
            )
        else:
            objective = 0.5 * norm(res, ord="fro") ** 2 + myparams.lambda_s * sum(
                norm(qk.build_field(), ord="nuc") for qk in qtilde_frames
            )
        objective_list.append(objective)
        rel_decrease = np.abs((objective_list[-1] - objective_list[-2])) / np.abs(
            objective_list[-1]
        )
        rel_decrease_list.append(rel_decrease)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                "ranks_frame = ".format(current_it, myparams.maxit, rel_err, elapsed),
                *ranks
            )
        if (current_it > 5) and (rel_decrease < myparams.gtol):
            break

    if myparams.isError:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_FB(
    snapshot_matrix,
    transforms,
    myparams,
    nmodes_max=None,
    method="BFB",
):
    """
    This function implements the Forward-Backward method (FB).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param nmodes: Number of modes to use in each frame
    :type nmodes: integer

    :param myparams: Parameters for the FB algorithm
    :type myparams: class:`sPOD_Param`

    :param method: Choice of the version of FB. Options are "BFB" (Block-coordinate
                   Forward Backward) and "JFB" (Joint Forward Backward).

    :return:
    :rtype: :class:`ReturnValue`
    """

    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )

    ###################################
    #        1. Initialization        #
    ###################################
    q = snapshot_matrix
    qtilde = np.zeros_like(q)
    if myparams.isError:
        E = np.zeros_like(snapshot_matrix)
    Nframes = len(transforms)
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        nmodes_max = np.min(np.shape(snapshot_matrix)) # use the smallest dimension beacuse after this singular values will be 0
    if np.size(nmodes_max) != Nframes:
        nmodes = list([nmodes_max]) * Nframes
    else:
        nmodes = nmodes_max
    qtilde_frames = [
        Frame(transfo, qtilde, Nmodes=nmodes[k]) for k, transfo in enumerate(transforms)
    ]
    norm_q = norm(reshape(q, -1))
    
    ###########################
    # Error of the truncated SVD
    r_ = np.sum(nmodes)
    (u, s, vt) = trunc_svd(q, nmodes_max=None, use_rSVD=myparams.use_rSVD)
    err_svd = np.linalg.norm(q - np.dot(u * s, vt), ord="fro") / norm_q
    print(
        "Relative error using a truncated SVD with {:d} modes:{:4.4e}".format(
            r_, err_svd
        )
    )
    ###########################

    current_it = 0
    objective_0 = 0.5 * norm(q, ord="fro") ** 2
    objective_list = [objective_0]
    rel_decrease = 1
    rel_decrease_list = [1]
    rel_err_list = []
    ranks_hist = [[] for r in range(Nframes)]
    sum_elapsed = 0
    while rel_decrease > myparams.eps and current_it < myparams.maxit:
        current_it += 1
        ###################################
        #      2. Calculate residual      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        norm_res = norm(reshape(res, -1))
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        if method == "JFB":
            qtilde = np.zeros_like(q)
        ranks = [0] * Nframes

        ###################################
        #      3. Update the frames       #
        ###################################
        t = time.perf_counter()
        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            res_shifted = trafo.reverse(res)
            q_frame_field = q_frame.build_field()
            if method == "BFB":
                qtilde -= trafo.apply(q_frame.build_field())
            stepsize = 1 / Nframes
            q_frame.set_orthonormal_system_svt(
                q_frame_field + stepsize * res_shifted, stepsize * myparams.lambda_s
            )
            if myparams.tv_niter > 0:
                q_frame.smoothen_time_amplitudes(
                    TV_iterations=myparams.tv_nit
                )
            S = q_frame.modal_system["sigma"]
            rank = np.sum(S > 0)
            ranks[k] = rank
            ranks_hist[k].append(rank)
            qtilde += trafo.apply(q_frame.build_field())
            if method == "BFB":
                if myparams.isError:
                    res = q - qtilde - E
                else:
                    res = q - qtilde
        if myparams.isError:
            E = shrink(E + stepsize * res, stepsize * myparams.lambda_E)
            objective = (
                0.5 * norm(res, ord="fro") ** 2
                + myparams.lambda_s
                * sum(norm(qk.build_field(), ord="nuc") for qk in qtilde_frames)
                + myparams.lambda_E * norm(E.flatten(), ord=1)
            )
        else:
            objective = 0.5 * norm(res, ord="fro") ** 2 + myparams.lambda_s * sum(
                norm(qk.build_field(), ord="nuc") for qk in qtilde_frames
            )
        objective_list.append(objective)
        rel_decrease = np.abs((objective_list[-1] - objective_list[-2])) / np.abs(
            objective_list[-1]
        )
        rel_decrease_list.append(rel_decrease)
        elapsed = time.perf_counter() - t
        sum_elapsed += elapsed
        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | t_cpu = {:2.2f}s | "
                "ranks_frame = ".format(current_it, myparams.maxit, rel_err, elapsed),
                *ranks
            )
        if (current_it > 5) and (rel_decrease < myparams.gtol):
            break

    if myparams.isError:
        print("CPU time in total: ", sum_elapsed)
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E)
    print("CPU time in total: ", sum_elapsed)
    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist)


def shifted_POD_ALM(snapshot_matrix, transforms, myparams, nmodes_max=None,
                    mu=None, qt_frames=None):
    """
    This function implements the Augmented Lagangian method (ALM).

    :param snapshot_matrix: Snapshot matrix with with dimensions :math:`M \times N`,
                            :math:`N` is the number of snapshots (i.e. time stamps)
                            and :math:`M` is the number of number of spatial samples
                            (i.e. the ODE dimension).
    :type snapshot_matrix: :class:`numpy.ndarray` (2-dimensional)

    :param transforms: List of transformations associated with the co-moving fields.
    :type transforms: list[Transform]

    :param myparams: Parameters for ALM algorithm
    :type myparams: class:`sPOD_Param`

    :param nmodes_max: Maximal number of modes allowed in each frame, default is
                       the number of snapshots :math:`N`.
                       Note: it is good to give a number large enough in order to
                       get the error down but smaller than :math:`N`.
                       This will increase the performance of the algorith.

    :param mu: Parameter of the augmented Lagrangian (i.e. weight of the
               quadratic term).
    :type mu: float

    :return:
    :rtype: :class:`ReturnValue`
    """
    assert (
        np.ndim(snapshot_matrix) == 2
    ), "Please give enter a snapshotmatrix with every snapshot in one column"
    if myparams.use_rSVD:
        warn(
            "Warning: Using rSVD to accelarate decomposition procedure may lead "
            "to different results."
        )
    ###################################
    #        1. Initialization        #
    ###################################
    qtilde = np.zeros_like(snapshot_matrix)
    E = np.zeros_like(snapshot_matrix)
    Nframes = len(transforms)
    
    # make a list of the number of maximal ranks in each frame
    if not np.all(nmodes_max):  # check if array is None, if so set nmodes_max onto N
        # use the smallest dimension beacuse after this singular values will be 0
        nmodes_max = np.min(np.shape(snapshot_matrix))
    if np.size(nmodes_max) != Nframes:
        nmodes_max = list([nmodes_max]) * Nframes
    else:
        nmodes_max = nmodes_max

    if qt_frames is None:
            qtilde_frames = [Frame(transfo, field=qtilde, Nmodes=nmodes_max[k])
                             for k, transfo in enumerate(transforms)]
    else:
            qtilde_frames = [Frame(transfo, field=qt_frames[k], Nmodes=nmodes_max[k])
                             for k, transfo in enumerate(transforms)]

    q = snapshot_matrix.copy()
    Y = np.zeros_like(snapshot_matrix)
    norm_q = norm(reshape(q, -1))
    it = 0
    mu_inv = 1 / mu
    rel_err = 1
    res_old = 0
    rel_err_list = []
    ranks_hist = [[] for r in range(Nframes)]
    sum_elapsed = 0.0
    elapsed_SVT = 0.0
    while rel_err > myparams.eps and it < myparams.maxit:
        it += 1  # counts the number of iterations in the loop
        ###################################
        #       2. Set qtilde to 0        #
        ###################################
        qtilde = np.zeros_like(q)
        ranks = [0] * Nframes
        ###################################
        #      3. Update the frames       #
        ###################################
        t = timer()

        for k, (trafo, q_frame) in enumerate(zip(transforms, qtilde_frames)):
            qtemp = 0
            for p, (trafo_p, frame_p) in enumerate(zip(transforms, qtilde_frames)):
                if p != k:
                    qtemp += trafo_p.apply(frame_p.build_field())
            qk = trafo.reverse(q - qtemp - E + mu_inv * Y)
            t_SVT_tmp = timer()
            [U, S, VT] = SVT(qk, mu_inv, nmodes_max[k], myparams.use_rSVD)
            elapsed_SVT += timer() - t_SVT_tmp
            rank = np.sum(S > 0)
            q_frame.modal_system = {
                "U": U[:, :rank],
                "sigma": S[:rank],
                "VT": VT[:rank, :],
            }
            ranks_hist[k].append(rank)
            ranks[k] = rank

            qtilde += trafo.apply(q_frame.build_field())
        ###################################
        #    4. Update the noise term     #
        ###################################
        if myparams.isError:
            E = shrink(q - qtilde + mu_inv * Y, myparams.lambda_E * mu_inv)
        ###################################
        #      5. Update multipliers      #
        ###################################
        if myparams.isError:
            res = q - qtilde - E
        else:
            res = q - qtilde
        Y = Y + mu * res

        ###################################
        #          6. Update mu           #
        ###################################
        dres = norm(res, ord="fro") - res_old
        res_old = norm(res, ord="fro")
        norm_dres = np.abs(dres)

        norm_res = norm(reshape(res, -1))
        rel_err_without_noise = norm(reshape(res + E, -1)) / norm_q
        rel_err = norm_res / norm_q
        rel_err_list.append(rel_err)
        elapsed = timer() - t
        sum_elapsed += elapsed

        if myparams.isVerbose:
            print(
                "Iter {:4d} / {:d} | Rel_err= {:4.4e} | norm(dres) = {:4.1e} | "
                "norm(Q-Qtilde)/norm(Q) = {:4.2e} norm(E)/norm(Q) = {:4.2e} | t_cpu = {:2.2e}s | "
                "ranks_frame = ".format(
                    it,
                    myparams.maxit,
                    rel_err,
                    mu * norm_dres / norm_q,
                    rel_err_without_noise,
                    norm(reshape(E, -1)) / norm_q,
                    elapsed,
                ),
                *ranks
            )

        if it > 5 and (np.abs(rel_err_list[-1] - rel_err_list[-4]) < myparams.gtol * abs(rel_err_list[-1]) or rel_err_list[-1] > rel_err_list[-4]):
            break

    qtilde = 0
    for p, (trafo_p, frame_p) in enumerate(zip(transforms, qtilde_frames)):
        qtilde += trafo_p.apply(frame_p.build_field())
        S = frame_p.modal_system["sigma"]
        frame_p.Nmodes = np.sum(S > 0)

    print("CPU time in total: ", sum_elapsed)
    if myparams.isError:
        return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist, E, tcpu_SVD=elapsed_SVT)

    return ReturnValue(qtilde_frames, qtilde, rel_err_list, ranks, ranks_hist,tcpu_SVD=elapsed_SVT)

def generate_discr_diff_mat(dim, order=1):
    r"""Generate the discrete differential matrix, i.e. the matrix with 1 on
    its diagonal and -1 on its first sub-diagonal.

    :param dim: Dimension of the matrix.
    :type dim: int

    :param order: Order of the differential (default value=1).
    :type order: int

    :return: The discrete gradient matrix.
    """
    L = np.eye(dim)

    D = np.zeros([dim, dim])
    i, j = np.indices(D.shape)
    D[i == j] = 1
    D[i == j + 1] = -1
    #D[0, dim - 1] = -1

    for i in range(order):
        L = np.matmul(L, D)

    return L

def force_constraint(qframes, transforms, q, Niter=1, alphas=None):
    """This function enforces the constraint 
    
    .. math::

        Q = sum_k T^k Q^k
    """
    norm_q = norm(reshape(q, -1))
    qtilde = np.zeros_like(q)
    if alphas == None:
        Nframes = len(transforms)
        alphas = [1 / Nframes] * Nframes

    for iter in range(Niter):
        qtilde = 0
        for k, (trafo, q_frame) in enumerate(zip(transforms, qframes)):
            qtilde += trafo.apply(q_frame.build_field())
            q_frame.Nmodes = -1

        res = q - qtilde
        qtilde = 0
        for k, (transfo, q_frame) in enumerate(zip(transforms, qframes)):
            res_shifted = transfo.reverse(res)
            q_frame_field = q_frame.build_field()
            q_frame.set_orthonormal_system(
                q_frame_field + res_shifted * alphas[k], use_rSVD=False
            )
            qtilde += transfo.apply(q_frame.build_field())

    res = q - qtilde
    norm_res = norm(reshape(res, -1))
    rel_err = norm_res / norm_q

    print("rel_err= {:4.4e}".format(rel_err))

    return ReturnValue(qframes, qtilde)


def give_interpolation_error(snapshot_data, transfo):
    """
    This function computes the interpolation error of the non-linear
    representation of the data.
    The error is "the best you can get" with the shifted POD.
    Therefore, it is the smallest possible error the shifted POD decomposition
    allows and can be used as the stopping criteria inside the sPOD algorithm.
    The relative error is computing using the Frobenius norm

    .. math::

        err = \| Q - T^(-1)[T[Q]] \|_F^2 / \| Q \|_F^2

    :param snapshot_data: Snapshot matrix.
    :type snapshot_data: :class:`numpy.ndarray` (2-dimensional)

    :param transfo:
    :type transfo: :class:`Transform`

    :return: Interpolation error.
    :rtype: float
    """
    from numpy import reshape

    Q = reshape(snapshot_data, [-1, snapshot_data.shape[-1]])
    rel_err = norm(Q - transfo.apply(transfo.reverse(Q)), ord="fro") / norm(
        Q, ord="fro"
    )
    return rel_err / 2


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

    # Load frames
    frame_list = []
    for k in range(Nframes):
        fname_frame = fname_base + "_%.2d" % k + ext
        print("frame %2d loaded: " % k, fname_frame)
        newframe = Frame(fname=fname_frame)
        frame_list.append(newframe)

    # Load sparse error matrix
    fname_error_matrix = fname_base + "_error_mat.npy"
    if load_ErrMat and os.path.isfile(fname_error_matrix):
        E = np.load(fname_error_matrix)
        return frame_list, E
    else:
        return frame_list
# ============================================================================ #


# ============================================================================ #
#                           CLASS of sPOD PARAMETERS                           #
# ============================================================================ #
@dataclass
class sPOD_Param:
    """
    Structure that stores the parameters for the sPOD algorithms.

    Attributes:
        gtol (float): If the relative approximation error decreases less than gtol, 
                      the algorithm stops
        eps (float): If the relative approximation error is below eps,
                     the algorithm stops
        maxit (int): Maximum number of iterations.
        isVerbose (bool): Should the algorithm print information while running?
        isError (bool): Should the algorithm use the error term?
        use_rSVD (bool): Set to True in order to use randomized version of the SVD.
        lambda_s (float): Regularization parameter for the nuclear norm of the
                          co-moving frames.
        lambda_E (float): Regularization parameter for the l1-norm of the error
                          term.
        mu (float): Regularization parameter for the Total variation term.
        tv_niter (int): Number of iterations for the Rescaled Forward-Backward
                        based Primal Dual algorithm.
    """

    gtol: float = 1e-5
    eps: float = 1e-16
    maxit: int = 10000
    isVerbose: bool = True
    isError: bool = False
    use_rSVD: bool = False
    lambda_s: float = 1e-2
    lambda_E: float = 1e-2
    mu: float = 1e-2
    tv_niter: int = -1
# ============================================================================ #
