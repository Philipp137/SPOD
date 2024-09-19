# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:19:19 2018

@author: Philipp Krah

This package provides all the infrastructure for the shifted propper orthogonal
decomposition (sPOD).
"""
# ============================================================================ #
#                              MODULES IMPORTATION                             #
# ============================================================================ #
import pickle

import numpy as np
import matplotlib.pyplot as plt
from numpy import reshape
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import norm, svd
from matplotlib.pyplot import semilogy, xlabel, ylabel
# ============================================================================ #


# ============================================================================ #
#                           CLASS of CO-MOVING FRAMES                          #
# ============================================================================ #
class Frame:
    """
    This class implements a frame whose definition is motivated by physics:
    All points are in a inertial system (a.k.a. a frame), if they can be
    transformed to the rest frame by a Galilean transformation.
    A frame is represented by an orthogonal system.

    :param Nmodes: Number of modes to use in this frame.
    :type Nmodes: int

    :param data_shape: List of four integers describing the dimensions of the
                       data in the three dimension of space and the dimension of
                       time respectively.
    :type data_shape: List[int]

    :param Ngrid: Dimension of the spatial discretization grid.
    :type Ngrid: int

    :param Ntime: Dimension of the time discretizatin grid.
    :type Ntime: int

    :param transfo: Associated transformation that transforms the co-moving
                    frame into a reference frame.
    :type transfo: :class:`Transform`

    :param dim: Dimension of the transformation.
    :type dim: int

    :param modal_system: Dictionary storing the orthonormal vectors of the SVD
                         in the frame.
    :type modal_system: dict[str, :class:`numpy.ndarray`]
    """

    def __init__(self, transform=None, field=None, Nmodes=None, fname=None):
        """
        Constructor.

        :param transform: Dimension of the spatial discretization grid.
        :type transform: :class:`Transform`, optional

        :param field: ???
        :type field: :class:`numpy.ndarray` (1-dimensional), optional

        :param Nmodes: Associated transformation that transforms the co-moving
                    frame into a reference frame.
        :type Nmodes: int, optional

        :param fname: Name of the file containing the data to load.
        :type fname: str, optional
        """
        if fname:
            self.load(fname)
            self.Nmodes = np.sum(self.modal_system["sigma"] > 0)
        else:
            data_shape = transform.data_shape
            self.data_shape = data_shape
            self.Ngrid = np.prod(data_shape[:3])
            self.Ntime = data_shape[3]
            self.transfo = transform
            self.dim = self.transfo.dim
            self.Nmodes = Nmodes
            self.modal_system = dict()
            # Transform the field to reference frame
            if not np.all(field == None):
                field = self.transfo.reverse(field)
                self.set_orthonormal_system(field)

    def save(self, fname):
        """
        Method that serializes and saves the current frame in a file using
        pickle.

        :param fname: Name of the file where the frame is saved.
        :type fname: str
        """
        with open(fname, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        """
        Method that deserializes and load the data from a file to build a frame.

        :param fname: Name of the file containing the data.
        :type fname: str
        """
        with open(fname, "rb") as f:
            frame_load = pickle.load(f)
        self.__init__(frame_load.transfo)
        self.modal_system = frame_load.modal_system

    def set_orthonormal_system_svt(self, field, gamma):
        """
        Method that sets the orthonormal vectors of the SVD in the
        corresponding frames using an SVT.

        :param field: ???
        :type field: :class:`numpy.ndarray` (2-dimensional)

        :param gamma: Name of the file containing the data.
        :type gamma: float
        """
        # Reshape to snapshot matrix
        X = reshape(field, [-1, self.Ntime])
        # Perform an SVT of the snapshot matrix
        [U, S, VT] = SVT(X, gamma)
        # Store the SVT of the snapshot matrix
        self.modal_system = {"U": U, "sigma": S, "VT": VT}

    def set_orthonormal_system(self, field, use_rSVD=False):
        """
        Method that sets the orthonormal vectors of the SVD in the
        corresponding frames using a truncated SVD.

        :param field: ???
        :type field: :class:`numpy.ndarray` (2-dimensional)

        :param use_rSVD: Boolean set to True in order to use randomized version
                         of the SVD.
        :type use_rSVD: bool, optional
        """
        # Reshape to snapshot matrix
        X = reshape(field, [-1, self.Ntime])
        # Perform a truncated SVD of the snapshot matrix to the specified number
        # of modes
        [U, S, VT] = trunc_svd(X, self.Nmodes, use_rSVD)
        # Store the snapshot matrix using only its reduced number of SVD modes
        self.modal_system = {"U": U, "sigma": S, "VT": VT}

    def smoothen_time_amplitudes(self, TV_iterations=100, clambda=1):
        """
        Method that enforces smoothness of the time amplitudes:
        min_{b} || \partial_t b(t)|| + \lambda || b(t) - a(t)||
        where a(t) is the input (nonsmooth field)
        and b(t) is the smoothed field.

        :param TV_iterations: Number of iterations for iterative algorithm that
                              computes the proximal operator of the TV
                              regularization.
        :type TV_iterations: int, optional

        :param clambda: Value of the regularization parameter.
        :type clambda: float, optional
        """
        from total_variation import solve_TVL1

        VT = self.modal_system["VT"]
        self.modal_system["VT"] = solve_TVL1(VT.T, clambda, iter_n=TV_iterations).T

    def build_field(self, rank=None):
        """
        Method that calculates the field from the SVD modes: X=U*S*VT.

        :param rank: Rank of the truncated SVD. By default, the method uses a
                     full SVD.
        :type rank: int, optional

        :return: The field corresponding to the modal system of the class.
        :rtype: :class:`numpy.ndarray` (2-dimensional)
        """
        # Modes from the singular value decomposition
        u = self.modal_system["U"]
        s = self.modal_system["sigma"]
        vh = self.modal_system["VT"]
        if rank:
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]

        return np.dot(u * s, vh)

    def plot_singular_values(self):
        """
        Method that plots the singular values of the modal system corresponding
        to the frame.
        """
        sigmas = self.modal_system["sigma"]
        semilogy(sigmas / sigmas[0], "r+")
        xlabel(r"$i$")
        ylabel(r"$\sigma_i/\sigma_0$")

    def concatenate(self, other):
        """
        Method that adds two frames in order to concatenate their modes.

        :param other: The frame to be added to the current one.
        :type other: Frame

        :return: The concatenation of the two frames.
        :rtype: Frame
        """
        # TODO make check if other and self can be added:
        # are they in the same frame? Are they from the same data etc.
        new = Frame(self.trafo, self.build_field(), self.Nmodes)
        # combine left singular vecotrs
        Uself = self.modal_system["U"]
        Uother = other.modal_system["U"]
        new.modal_system["U"] = np.concatenate([Uself, Uother], axis=1)

        # combine right singular vectors
        VTself = self.modal_system["VT"]
        VTother = other.modal_system["VT"]
        new.modal_system["VT"] = np.concatenate([VTself, VTother], axis=0)

        Sself = self.modal_system["sigma"]
        Sother = other.modal_system["sigma"]
        new.modal_system["sigma"] = np.concatenate([Sself, Sother])

        new.Nmodes += other.Nmodes

        return new

    def __add__(self, other):
        """
        Implementation of Python addition for objects Frame.

        :param other: The frame to be added to the current one.
        :type other: Frame

        :return: The sum of the two frames.
        :rtype: Frame
        """
        if isinstance(other, Frame):
            new_field = self.build_field() + other.build_field()
        elif np.shape(other) == self.data_shape:
            new_field = self.build_field() + other

        # Apply SVD and save modes
        self.set_orthonormal_system(new_field)

        return self
# ============================================================================ #


# ============================================================================ #
#                                 BUILD FRAMES                                 #
# ============================================================================ #
def build_all_frames(frames, transfos=None, ranks=None):
    """
    Build up the truncated data field from the result of
    the sPOD decomposition.

    :param frames: List of frames q_k , k = 1,...,K.
    :type frames: List[:class:`Frame`]

    :param transfos: List of transformations T^k.
    :type transfos: List[:class:`Frame`], optional

    :param ranks: List of ranks r_k (strictly positive integers).
    :type ranks: List[int], optional

    :return: q = sum_k T^k q^k where q^k is of rank r_k
    :rtype: :class:`numpy.ndarray` (2-dimensional)
    """
    if transfos is None:
        transfos = [frame.transfo for frame in frames]

    if ranks is not None:
        if type(ranks) == int:
            ranks = [ranks] * len(transfos)
    else:
        ranks = [frame.Nmodes for frame in frames]

    qtilde = 0
    for k, (transfo, frame) in enumerate(zip(transfos, frames)):
        qtilde += transfo.apply(frame.build_field(ranks[k]))

    return qtilde


def reconstruction_error(snapshotmatrix, frames, transfos=None, max_ranks=None):
    """
    :param snapshotmatrix: Snapshot matrix.
    :type snapshotmatrix: :class:`numpy.ndarray` (2-dimensional)

    :param frames: List of frames q_k , k = 1,...,K.
    :type frames: List[:class:`Frame`]

    :param transfos: List of transformations T^k.
    :type transfos: List[:class:`Frame`], optional

    :param max_ranks: List of ranks r_k (strictly positive integers).
    :type max_ranks: List[int], optional

    :return: Elemnt-wise error between the snapshot matrix and q = sum_k T^k q^k
    :rtype: :class:`numpy.ndarray` (2-dimensional)
    """
    import itertools

    if transfos is None:
        transfos = [frame.transfo for frame in frames]

    if max_ranks is not None:
        if type(max_ranks) == int:
            ranks = [max_ranks] * len(transfos)
    else:
        max_ranks = [frame.Nmodes for frame in frames]

    possible_ranks_list = [np.arange(max_rank + 1) for max_rank in max_ranks]
    possible_ranks_list = list(itertools.product(*possible_ranks_list))
    Nlist = len(possible_ranks_list)
    norm_q = norm(snapshotmatrix, ord="fro")
    max_dof = np.sum(np.asarray(max_ranks) + 1)
    error_matrix = 2 * np.ones([max_dof - 1, 3])
    for it, ranks in enumerate(possible_ranks_list):
        qtilde = np.zeros_like(snapshotmatrix)
        for k, (transfo, frame) in enumerate(zip(transfos, frames)):
            if ranks[k] > 0:
                qtilde += transfo.apply(frame.build_field(ranks[k]))
        rel_err = norm(qtilde - snapshotmatrix, ord="fro") / norm_q
        dof = np.sum(ranks)
        print(
            " iter = %d/%d ranks = " % (it + 1, Nlist)
            + " ".join(map(str, ranks))
            + " error = %1.1e" % rel_err
        )
        if rel_err < error_matrix[dof, -1]:
            error_matrix[dof, -1] = rel_err
            for ir, r in enumerate(ranks):
                error_matrix[dof, ir] = r

    return error_matrix
# ============================================================================ #


# ============================================================================ #
#                       DETERMINATION OF SHIFT VELOCITIES                      #
# ============================================================================ #
def shift_velocities(dx, dt, fields, n_velocities, v_min, v_max, v_step, n_modes):
    sigmas = np.zeros([int((v_max - v_min) / v_step), n_modes])
    v_shifts = np.linspace(v_min, v_max, int((v_max - v_min) / v_step))

    i = 0
    for v in v_shifts:
        example_frame = Frame(v, dx, dt, fields, n_modes)
        sigmas[i, :] = example_frame.modal_system["sigma"]
        i += 1

    # Plot singular value spectrum
    plt.plot(v_shifts, sigmas, "o")

    sigmas_temp = sigmas.copy()
    c_shifts = []

    for i in range(n_velocities):
        max_index = np.where(sigmas_temp == sigmas_temp.max())
        max_index_x = max_index[0]
        max_index_x = max_index_x[0]
        max_index_y = max_index[1]
        max_index_y = max_index_y[0]

        sigmas_temp[max_index_x, max_index_y] = 0

        c_shifts.append(v_shifts[max_index_x])

    return c_shifts
# ============================================================================ #


# ============================================================================ #
#                               PROXIMAL OPERATORS                             #
# ============================================================================ #
def shrink(x, gamma):
    r"""
    This function implements the proximal operator of the l1-norm of a vector
    :math:`x` living in an Hilbert space :math:`\mathcal{H}`.
    This operator is also known as the soft thresholding and it is uniquely
    defined as  the point :math:`y` in :math:`\mathcal{H}` such that

    .. math:: 

        \DeclareMathOperator*{\argmin}{arg\,min}
        y = \argmin_{z\in\mathcal{H}} \ell_{1}(z) + \frac{1}{2\gamma}\|z-x\|_{2}^{2} \, .

    :param x: point at which the proximal operator is computed.
    :type x: :class:`numpy.ndarray` of size (N,M)

    :param gamma: parameter of the operator (a strictly positive real number).
    :type gamma: float

    :return: The proximal operator of the l1-norm evaluated at the given point.
    """
    if not isinstance(gamma, float) or gamma < 0:
        raise TypeError("shrink() parameter <gamma> is not a positive float")

    return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)


def SVT(X, gamma, nmodes_max=None, use_rSVD=False):
    r"""
    This function implements the proximal operator of the nuclear norm of a
    matrix :math:`X`.
    This operator is also known as the Singular Value Thresholding (SVT) and it
    is uniquely defined as matrix :math:`Y` such that

    .. math::

        \DeclareMathOperator*{\argmin}{arg\,min}
        Y = \argmin_{Z\in\mathbb{R}^{N,M}} \|z\|_{*} + \frac{1}{2\gamma}\|Z-X\|_{2}^{2} \, .

    SVT can be performed on a truncated SVD of :math:`X` by setting the correct
    values of the parameters.

    :param X: matrix at which the proximal operator is applied.
    :type X: :class:`numpy.ndarray` of size (N,M)

    :param gamma: parameter of the operator (a strictly positive real number).
    :type gamma: float

    :param nmodes_max: Number of modes in the truncated SVD.
                       When it is set to None, the full SVD is performed.
    :type nmodes_max: int, optional

    :param use_rSVD: Boolean set to True in order to use randomized version of
                     the SVD.
    :type use_rSVD: bool, optional

    :return: The proximal operator of the nuclear norm evaluated at the given matrix.
    """
    u, s, vt = trunc_svd(X, nmodes_max, use_rSVD)
    s = shrink(s, gamma)
    return (u, s, vt)


def trunc_svd(X, nmodes_max=None, use_rSVD=False):
    """
    This function implements the truncated SVD of the input matrix :math:`X`.

    :param nmodes_max: Number of modes in the truncated SVD.
                       When it is set to None, the full SVD is performed.
    :type nmodes_max: int, optional

    :param use_rSVD: Boolean set to True in order to use randomized version of
                     the SVD.
    :type use_rSVD: bool, optional
    """
    if nmodes_max:
        if use_rSVD:
            u, s, vt = randomized_svd(X, n_components=nmodes_max)
        else:
            u, s, vt = svd(X, full_matrices=False)
            s = s[:nmodes_max]
            u = u[:, :nmodes_max]
            vt = vt[:nmodes_max, :]
    else:
        u, s, vt = svd(X, full_matrices=False)
    return (u, s, vt)
# ============================================================================ #


# ============================================================================ #
#                  UPDATE THE XTILDE FRAMES AND TRUNCATE MODES                 #
# ============================================================================ #
def update_and_reduce_modes(Xtilde_frames, alpha, X_coef_shift, Nmodes_reduce):
    """
    This function implements the 5. step of the SPOD algorithm (see Reiss2017)
    - calculate the new modes from the optimiced alpha combining
      Xtilde and R modes.
    - truncate the number of modes to the desired number of reduced modes
    - update the new Xtilde in the corresponding frames
    """
    for k, frame in enumerate(Xtilde_frames):
        Nmodes = frame.Nmodes
        alpha_k = alpha[k * Nmodes : (k + 1) * Nmodes]
        # Linear combination to get the new Xtilde
        Xnew_k = X_coef_shift[:, k * Nmodes : (k + 1) * Nmodes] @ alpha_k
        Xnew_k = reshape(Xnew_k, [-1, frame.Ntime])
        frame.Nmodes = Nmodes_reduce  # Reduce to the desired number of modes
        [U, S, VT] = trunc_svd(Xnew_k, Nmodes_reduce)
        frame.modal_system = {"U": U, "sigma": S, "VT": VT}
# ============================================================================ #
