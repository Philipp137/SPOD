import numpy as np
import scipy.sparse as sp


def derivative(N, h, coefficient, boundary="periodic"):
    """
    Compute the discrete derivative for periodic BC
    """
    dlow = -(np.size(coefficient) - 1) // 2
    dup = - dlow+1

    diagonals = []
    offsets = []
    for k in np.arange(dlow, dup):
        diagonals.append(coefficient[k - dlow] * np.ones(N - abs(k)))
        offsets.append(k)
        if k > 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(-N + k)
        if k < 0:
            diagonals.append(coefficient[k - dlow] * np.ones(abs(k)))
            offsets.append(N + k)

    return sp.diags(diagonals, offsets) / h


class finite_diffs:
    def __init__(self, Ngrid, dX):
        Ix = sp.eye(Ngrid[0])
        Iy = sp.eye(Ngrid[1])
        # stencilx = np.asarray( [-1/60,	3/20, 	-3/4, 	0, 	3/4, 	-3/20, 	1/60])
        stencil_x = np.asarray([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
        # stencil_x = np.asarray([-0.5,0,0.5])
        # stencil_xx = np.asarray([1/90,	-3/20, 	3/2, 	-49/18, 	3/2, 	-3/20, 	1/90])
        stencil_xx = np.asarray([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])
        self.Dx_mat = sp.kron(derivative(Ngrid[0], dX[0], stencil_x), Iy)
        self.Dy_mat = sp.kron(Ix, derivative(Ngrid[1], dX[1], stencil_x))
        self.Dxx_mat = sp.kron(derivative(Ngrid[0], dX[0] ** 2, stencil_xx), Iy)
        self.Dyy_mat = sp.kron(Ix, derivative(Ngrid[1], dX[1] ** 2, stencil_xx))


    def Dx(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q,-1)
        return np.reshape(self.Dx_mat @ q, input_shape)
    def Dxx(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dxx_mat @ q, input_shape)
    def Dy(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dy_mat @ q, input_shape)
    def Dyy(self, q):
        input_shape = np.shape(q)
        q = np.reshape(q, -1)
        return np.reshape(self.Dyy_mat @ q, input_shape)
    def rot(self, v1, v2):
        return self.Dx(v2) - self.Dy(v1)



def calculate_force(u_rel, chi, dX):
    """
    u_rel: relativ velocity u_fluid - u_solid
    chi: mask function multiplied by the inverse penalisation constant
    dX: array with lattice spacings
    """
    force = np.sum(np.sum(u_rel*chi)) * np.prod(dX)
    return force

def opt_goal_lift_drag(mu_vec, u_ROM_fun, give_mask, dX, uy_solid):
    """
       This routine defines the optimization goal of the kinematic optimization.
       We optimize drag or lift, depending if the input is ux (drag) or uy (lift).
    """
    u_tilde = u_ROM_fun(mu_vec)
    chi = give_mask(mu_vec)
    force = 0
    uys = uy_solid(mu_vec)
    for nt, us in enumerate(uys):
        u_rel = u_tilde[...,nt] - us
        force += calculate_force(u_rel, chi[...,nt], dX)
    print(force)
    return force

def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.flip(np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8))

def smoothstep(x,t,h):
    x0 = t - h
    x1 = t + h
    dist = np.where((x0<x) & (x<x1)  , 0.5*(1+np.cos((x-x0)*np.pi/(2*h))),np.zeros_like(x))
    dist = np.where(x<=x0, 1, dist)
    return dist


def build_mask(X, Y, L, dX, shifts,  Radius = 1):
    h = 1.5 * max(dX)  # definition of the smoothwidth of wabbit
    mask = np.asarray([smoothstep(np.sqrt((X - L[0] / 2 - delta[0] ) ** 2 + (Y - L[1] / 2 - delta[1]) ** 2), Radius, h) for delta in
                        shifts])
    mask = np.moveaxis(mask, 0, -1)

    return mask


def read_performance_file(file_list):
    if not isinstance(file_list, list):
        return np.loadtxt(file_list)

    perf_dat_list = []
    for file in file_list:
        perf_dat_list.append(np.loadtxt(file))

    return perf_dat_list