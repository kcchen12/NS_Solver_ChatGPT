#!/usr/bin/env python3
"""
Single-process friendly + MPI-capable 2D incompressible Navier_Stokes solver (fractional step / projection)
- If mpi4py is installed AND you run with mpirun, it runs in parallel.
- Otherwise it runs in single-process mode with plain `python script.py`.

Features:
- Cartesian domain, finite-volume-like conservative fluxes
- SSP-RK3 for momentum
- Immersed Boundary Method (Peskin delta) for stationary immersed solids
- Boundary conditions: inflow, far-field, convective outflow, no-slip wall
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


# -----------------------------
# MPI fallback (single-process)
# -----------------------------
class _FakeRequest:
    @staticmethod
    def Waitall(reqs):  # noqa: N802
        return


class _FakeComm:
    def Get_rank(self):  # noqa: N802
        return 0

    def Get_size(self):  # noqa: N802
        return 1

    def Irecv(self, *args, **kwargs):
        return _FakeRequest()

    def Isend(self, *args, **kwargs):
        return _FakeRequest()

    def allreduce(self, value, op=None):
        return value

    def gather(self, value, root=0):
        return [value]

    def Gatherv(self, sendbuf, recvbuf, root=0):
        # In single process, just copy if a recv is provided in the MPI-like signature.
        # Our gather_and_save handles rank==0 only, so this is rarely called.
        return


class _FakeMPI:
    COMM_WORLD = _FakeComm()
    PROC_NULL = -1
    DOUBLE = None
    MAX = None
    Request = _FakeRequest


try:
    from mpi4py import MPI  # type: ignore
except Exception:
    MPI = _FakeMPI()  # type: ignore


# -----------------------------
# MPI utilities (1D in x)
# -----------------------------
@dataclass
class Decomp1D:
    comm: Any
    rank: int
    size: int
    nx_global: int
    ny: int
    ng: int

    nx_local: int
    i0: int
    i1: int
    left: int
    right: int

    @staticmethod
    def build(comm, nx_global, ny, ng):
        rank = comm.Get_rank()
        size = comm.Get_size()

        # block decomposition
        counts = [nx_global // size] * size
        for r in range(nx_global % size):
            counts[r] += 1
        offsets = [0]
        for r in range(1, size):
            offsets.append(offsets[-1] + counts[r - 1])

        nx_local = counts[rank]
        i0 = offsets[rank]
        i1 = i0 + nx_local  # exclusive

        left = rank - 1 if rank > 0 else MPI.PROC_NULL
        right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

        return Decomp1D(comm, rank, size, nx_global, ny, ng,
                        nx_local, i0, i1, left, right)

    def halo_exchange_x(self, a):
        """
        Exchange halos in x for array a shaped (nx_local+2*ng, ny+2*ng).
        In single-process mode, this is a no-op.
        """
        if self.size == 1:
            return

        ng = self.ng
        send_left = a[ng:2*ng, :].copy()
        recv_right = np.empty_like(send_left)

        send_right = a[-2*ng:-ng, :].copy()
        recv_left = np.empty_like(send_right)

        req1 = self.comm.Irecv(recv_left, source=self.left, tag=11)
        req2 = self.comm.Irecv(recv_right, source=self.right, tag=12)
        req3 = self.comm.Isend(send_left, dest=self.left, tag=12)
        req4 = self.comm.Isend(send_right, dest=self.right, tag=11)
        MPI.Request.Waitall([req1, req2, req3, req4])

        if self.left != MPI.PROC_NULL:
            a[:ng, :] = recv_left
        if self.right != MPI.PROC_NULL:
            a[-ng:, :] = recv_right


# -----------------------------
# Peskin 4-pt delta kernel
# -----------------------------
def phi(r):
    r = abs(r)
    if r < 1.0:
        return 0.125 * (3 - 2*r + math.sqrt(1 + 4*r - 4*r*r))
    if r < 2.0:
        return 0.125 * (5 - 2*r - math.sqrt(-7 + 12*r - 4*r*r))
    return 0.0


phi_vec = np.vectorize(phi)


def delta_2d(x, y, h):
    return (1.0 / (h*h)) * phi_vec(x / h) * phi_vec(y / h)


# -----------------------------
# Boundary conditions
# -----------------------------
def apply_velocity_bcs(u, v, decomp: Decomp1D, U0, V0, dx, dy):
    """
    BCs:
      - Left boundary: inflow Dirichlet (u=U0, v=V0)
      - Top boundary: far-field Dirichlet (u=U0, v=V0)
      - Bottom boundary: no-slip wall (u=v=0)
      - Right boundary: convective outflow (implemented later; here use Neumann)
    """
    ng = decomp.ng

    # Bottom: no-slip (odd reflection gives zero at wall)
    u[:, :ng] = -u[:, ng:2*ng]
    v[:, :ng] = -v[:, ng:2*ng]

    # Top far-field: Dirichlet
    u[:, -ng:] = 2*U0 - u[:, -2*ng:-ng]
    v[:, -ng:] = 2*V0 - v[:, -2*ng:-ng]

    # Left inflow
    if decomp.left == MPI.PROC_NULL:
        u[:ng, :] = 2*U0 - u[ng:2*ng, :]
        v[:ng, :] = 2*V0 - v[ng:2*ng, :]

    # Right outflow (default: zero-gradient)
    if decomp.right == MPI.PROC_NULL:
        u[-ng:, :] = u[-2*ng:-ng, :]
        v[-ng:, :] = v[-2*ng:-ng, :]


def apply_phi_bc(phi, decomp: Decomp1D):
    """
    Pressure-correction Poisson BCs: homogeneous Neumann everywhere.
    """
    ng = decomp.ng

    # y-direction Neumann
    phi[:, :ng] = phi[:, ng:2*ng]
    phi[:, -ng:] = phi[:, -2*ng:-ng]

    # x-direction Neumann
    if decomp.left == MPI.PROC_NULL:
        phi[:ng, :] = phi[ng:2*ng, :]
    if decomp.right == MPI.PROC_NULL:
        phi[-ng:, :] = phi[-2*ng:-ng, :]


# -----------------------------
# Operators
# -----------------------------
def divergence(u, v, dx, dy, ng):
    dudx = (u[ng+1:-ng+1, ng:-ng] - u[ng-1:-ng-1, ng:-ng]) / (2*dx)
    dvdy = (v[ng:-ng, ng+1:-ng+1] - v[ng:-ng, ng-1:-ng-1]) / (2*dy)
    return dudx + dvdy


def laplacian(a, dx, dy, ng):
    axx = (a[ng+1:-ng+1, ng:-ng] - 2*a[ng:-ng, ng:-ng] +
           a[ng-1:-ng-1, ng:-ng]) / (dx*dx)
    ayy = (a[ng:-ng, ng+1:-ng+1] - 2*a[ng:-ng, ng:-ng] +
           a[ng:-ng, ng-1:-ng-1]) / (dy*dy)
    return axx + ayy


def advective_flux_rhs(u, v, dx, dy, ng):
    uc, vc = u, v

    # x-faces
    uL = uc[ng-1:-ng-1, ng:-ng]
    uC = uc[ng:-ng,     ng:-ng]
    uR = uc[ng+1:-ng+1, ng:-ng]

    vL = vc[ng-1:-ng-1, ng:-ng]
    vC = vc[ng:-ng,     ng:-ng]
    vR = vc[ng+1:-ng+1, ng:-ng]

    u_face_ip = 0.5 * (uC + uR)
    u_face_im = 0.5 * (uL + uC)

    uu_ip = np.where(u_face_ip >= 0.0, uC*uC, uR*uR)
    uu_im = np.where(u_face_im >= 0.0, uL*uL, uC*uC)

    uv_ip = np.where(u_face_ip >= 0.0, uC*vC, uR*vR)
    uv_im = np.where(u_face_im >= 0.0, uL*vL, uC*vC)

    d_uu_dx = (uu_ip - uu_im) / dx
    d_uv_dx = (uv_ip - uv_im) / dx

    # y-faces
    uB = uc[ng:-ng, ng-1:-ng-1]
    uC2 = uc[ng:-ng, ng:-ng]
    uT = uc[ng:-ng, ng+1:-ng+1]

    vB = vc[ng:-ng, ng-1:-ng-1]
    vC2 = vc[ng:-ng, ng:-ng]
    vT = vc[ng:-ng, ng+1:-ng+1]

    v_face_jp = 0.5 * (vC2 + vT)
    v_face_jm = 0.5 * (vB + vC2)

    vu_jp = np.where(v_face_jp >= 0.0, vC2*uC2, vT*uT)
    vu_jm = np.where(v_face_jm >= 0.0, vB*uB, vC2*uC2)
    d_vu_dy = (vu_jp - vu_jm) / dy

    vv_jp = np.where(v_face_jp >= 0.0, vC2*vC2, vT*vT)
    vv_jm = np.where(v_face_jm >= 0.0, vB*vB, vC2*vC2)
    d_vv_dy = (vv_jp - vv_jm) / dy

    conv_u = d_uu_dx + d_vu_dy
    conv_v = d_uv_dx + d_vv_dy
    return -conv_u, -conv_v


# -----------------------------
# IBM
# -----------------------------
def build_circle_markers(cx, cy, R, n_markers):
    theta = np.linspace(0, 2*np.pi, n_markers, endpoint=False)
    X = cx + R*np.cos(theta)
    Y = cy + R*np.sin(theta)
    return np.stack([X, Y], axis=1)


def ibm_interpolate_velocity(u, v, markers, i0_global, dx, dy, ng):
    h = dx
    U = np.zeros((markers.shape[0], 2), dtype=float)
    nx_tot, ny_tot = u.shape

    for m, (xm, ym) in enumerate(markers):
        ig = int(math.floor(xm / dx))
        jg = int(math.floor(ym / dy))

        u_sum = v_sum = w_sum = 0.0
        for ii in range(ig - 1, ig + 3):
            xl = (ii + 0.5) * dx
            i_loc = (ii - i0_global) + ng
            if i_loc < 0 or i_loc >= nx_tot:
                continue
            for jj in range(jg - 1, jg + 3):
                yl = (jj + 0.5) * dy
                j_loc = jj + ng
                if j_loc < 0 or j_loc >= ny_tot:
                    continue
                w = delta_2d(xm - xl, ym - yl, h) * dx * dy
                u_sum += u[i_loc, j_loc] * w
                v_sum += v[i_loc, j_loc] * w
                w_sum += w

        if w_sum > 0:
            U[m, 0] = u_sum / w_sum
            U[m, 1] = v_sum / w_sum
    return U


def ibm_spread_force(f_markers, markers, fx, fy, i0_global, dx, dy, ng):
    h = dx
    for (xm, ym), (Fx, Fy) in zip(markers, f_markers):
        ig = int(math.floor(xm / dx))
        jg = int(math.floor(ym / dy))

        for ii in range(ig - 1, ig + 3):
            xl = (ii + 0.5) * dx
            i_loc = (ii - i0_global) + ng
            if i_loc < 0 or i_loc >= fx.shape[0]:
                continue
            for jj in range(jg - 1, jg + 3):
                yl = (jj + 0.5) * dy
                j_loc = jj + ng
                if j_loc < 0 or j_loc >= fx.shape[1]:
                    continue
                w = delta_2d(xm - xl, ym - yl, h) * dx * dy
                fx[i_loc, j_loc] += Fx * w
                fy[i_loc, j_loc] += Fy * w


# -----------------------------
# Poisson solver (Jacobi)
# -----------------------------
def poisson_jacobi(phi, rhs, dx, dy, decomp: Decomp1D, iters=200, omega=0.8):
    ng = decomp.ng
    dx2 = dx * dx
    dy2 = dy * dy
    inv_denom = 1.0 / (2.0/dx2 + 2.0/dy2)

    phi_new = phi.copy()

    for _ in range(iters):
        decomp.halo_exchange_x(phi)
        apply_phi_bc(phi, decomp)

        pC = phi[ng:-ng, ng:-ng]
        pE = phi[ng+1:-ng+1, ng:-ng]
        pW = phi[ng-1:-ng-1, ng:-ng]
        pN = phi[ng:-ng, ng+1:-ng+1]
        pS = phi[ng:-ng, ng-1:-ng-1]

        phi_j = ((pE + pW) / dx2 + (pN + pS) / dy2 - rhs) * inv_denom
        phi_new[ng:-ng, ng:-ng] = (1 - omega) * pC + omega * phi_j

        # gauge fix
        if decomp.rank == 0:
            phi_new[ng, ng] = 0.0

        phi[:, :] = phi_new[:, :]

    return phi


# -----------------------------
# RK3 + Projection
# -----------------------------
def rk3_step(u, v, p, decomp: Decomp1D, params, markers):
    ng = decomp.ng
    dx, dy, dt, nu = params["dx"], params["dy"], params["dt"], params["nu"]
    U0, V0 = params["U0"], params["V0"]
    i0 = decomp.i0

    def rhs(u_in, v_in):
        decomp.halo_exchange_x(u_in)
        decomp.halo_exchange_x(v_in)
        apply_velocity_bcs(u_in, v_in, decomp, U0, V0, dx, dy)

        fx = np.zeros_like(u_in)
        fy = np.zeros_like(v_in)

        if markers.shape[0] > 0:
            U_lag = ibm_interpolate_velocity(
                u_in, v_in, markers, i0, dx, dy, ng)
            U_des = np.zeros_like(U_lag)
            F_lag = (U_des - U_lag) / dt
            ibm_spread_force(F_lag, markers, fx, fy, i0, dx, dy, ng)

        conv_u, conv_v = advective_flux_rhs(u_in, v_in, dx, dy, ng)
        diff_u = nu * laplacian(u_in, dx, dy, ng)
        diff_v = nu * laplacian(v_in, dx, dy, ng)

        Ru = conv_u + diff_u + fx[ng:-ng, ng:-ng]
        Rv = conv_v + diff_v + fy[ng:-ng, ng:-ng]
        return Ru, Rv

    u0 = u.copy()
    v0 = v.copy()

    # Stage 1
    Ru, Rv = rhs(u, v)
    u1 = u.copy()
    v1 = v.copy()
    u1[ng:-ng, ng:-ng] = u[ng:-ng, ng:-ng] + dt * Ru
    v1[ng:-ng, ng:-ng] = v[ng:-ng, ng:-ng] + dt * Rv

    # Stage 2
    Ru, Rv = rhs(u1, v1)
    u2 = u.copy()
    v2 = v.copy()
    u2[ng:-ng, ng:-ng] = 0.75*u0[ng:-ng, ng:-ng] + \
        0.25*(u1[ng:-ng, ng:-ng] + dt*Ru)
    v2[ng:-ng, ng:-ng] = 0.75*v0[ng:-ng, ng:-ng] + \
        0.25*(v1[ng:-ng, ng:-ng] + dt*Rv)

    # Stage 3
    Ru, Rv = rhs(u2, v2)
    u_hat = u.copy()
    v_hat = v.copy()
    u_hat[ng:-ng, ng:-ng] = (1/3)*u0[ng:-ng, ng:-ng] + \
        (2/3)*(u2[ng:-ng, ng:-ng] + dt*Ru)
    v_hat[ng:-ng, ng:-ng] = (1/3)*v0[ng:-ng, ng:-ng] + \
        (2/3)*(v2[ng:-ng, ng:-ng] + dt*Rv)

    # Convective outflow stabilization on global right boundary rank
    if decomp.right == MPI.PROC_NULL:
        Uc = max(U0, 1e-12)
        i_last = -ng-1
        i_prev = -ng-2
        u_hat[i_last, ng:-ng] = u0[i_last, ng:-ng] - dt * \
            Uc*(u0[i_last, ng:-ng] - u0[i_prev, ng:-ng]) / dx
        v_hat[i_last, ng:-ng] = v0[i_last, ng:-ng] - dt * \
            Uc*(v0[i_last, ng:-ng] - v0[i_prev, ng:-ng]) / dx

    # Projection
    decomp.halo_exchange_x(u_hat)
    decomp.halo_exchange_x(v_hat)
    apply_velocity_bcs(u_hat, v_hat, decomp, U0, V0, dx, dy)

    rhs_phi = (1.0/dt) * divergence(u_hat, v_hat, dx, dy, ng)

    phi = np.zeros_like(p)
    phi = poisson_jacobi(phi, rhs_phi, dx, dy, decomp,
                         iters=params["poisson_iters"], omega=params["poisson_omega"])

    decomp.halo_exchange_x(phi)
    apply_phi_bc(phi, decomp)

    dphidx = (phi[ng+1:-ng+1, ng:-ng] - phi[ng-1:-ng-1, ng:-ng]) / (2*dx)
    dphidy = (phi[ng:-ng, ng+1:-ng+1] - phi[ng:-ng, ng-1:-ng-1]) / (2*dy)

    u[ng:-ng, ng:-ng] = u_hat[ng:-ng, ng:-ng] - dt*dphidx
    v[ng:-ng, ng:-ng] = v_hat[ng:-ng, ng:-ng] - dt*dphidy
    p[ng:-ng, ng:-ng] = p[ng:-ng, ng:-ng] + phi[ng:-ng, ng:-ng]

    decomp.halo_exchange_x(u)
    decomp.halo_exchange_x(v)
    apply_velocity_bcs(u, v, decomp, U0, V0, dx, dy)

    return u, v, p


# -----------------------------
# Output
# -----------------------------
def save_single_or_gather(u, v, p, decomp: Decomp1D, params, step, outdir):
    """
    Single-process: save directly.
    MPI: gather on rank 0 then save.
    """
    ng = decomp.ng
    comm = decomp.comm
    rank = decomp.rank

    u_loc = u[ng:-ng, ng:-ng].copy()
    v_loc = v[ng:-ng, ng:-ng].copy()
    p_loc = p[ng:-ng, ng:-ng].copy()

    if decomp.size == 1:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"snapshot_{step:07d}.npz")
        np.savez_compressed(
            fname, u=u_loc, v=v_loc, p=p_loc,
            nx=params["nx"], ny=params["ny"],
            Lx=params["Lx"], Ly=params["Ly"],
            dt=params["dt"], nu=params["nu"], step=step
        )
        print(f"[single] wrote {fname}")
        return

    # MPI gather (original approach)
    nx_loc = u_loc.shape[0]
    ny = u_loc.shape[1]
    nx_counts = comm.gather(nx_loc, root=0)

    if rank == 0:
        nx_total = sum(nx_counts)
        u_g = np.empty((nx_total, ny), dtype=u_loc.dtype)
        v_g = np.empty((nx_total, ny), dtype=v_loc.dtype)
        p_g = np.empty((nx_total, ny), dtype=p_loc.dtype)
        displs = np.cumsum([0] + nx_counts[:-1]) * ny
        recvcounts = [c * ny for c in nx_counts]
    else:
        u_g = v_g = p_g = None
        displs = recvcounts = None

    comm.Gatherv(u_loc.ravel(), (u_g.ravel() if rank ==
                 0 else None, recvcounts, displs, MPI.DOUBLE), root=0)
    comm.Gatherv(v_loc.ravel(), (v_g.ravel() if rank ==
                 0 else None, recvcounts, displs, MPI.DOUBLE), root=0)
    comm.Gatherv(p_loc.ravel(), (p_g.ravel() if rank ==
                 0 else None, recvcounts, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.join(outdir, f"snapshot_{step:07d}.npz")
        np.savez_compressed(
            fname, u=u_g, v=v_g, p=p_g,
            nx=params["nx"], ny=params["ny"],
            Lx=params["Lx"], Ly=params["Ly"],
            dt=params["dt"], nu=params["nu"], step=step
        )
        print(f"[rank0] wrote {fname}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--Lx", type=float, default=2.0)
    parser.add_argument("--Ly", type=float, default=1.0)
    parser.add_argument("--U0", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out_every", type=int, default=200)
    parser.add_argument("--outdir", type=str, default="out")
    parser.add_argument("--poisson_iters", type=int, default=250)
    parser.add_argument("--poisson_omega", type=float, default=0.85)

    # IBM circle
    parser.add_argument("--ibm_circle", action="store_true", default=True)
    parser.add_argument("--cx", type=float, default=0.6)
    parser.add_argument("--cy", type=float, default=0.5)
    parser.add_argument("--R", type=float, default=0.12)
    parser.add_argument("--n_markers", type=int, default=250)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    ng = 2
    nx, ny = args.nx, args.ny
    dx = args.Lx / nx
    dy = args.Ly / ny

    decomp = Decomp1D.build(comm, nx, ny, ng)

    shape = (decomp.nx_local + 2*ng, ny + 2*ng)
    u = np.zeros(shape, dtype=float)
    v = np.zeros(shape, dtype=float)
    p = np.zeros(shape, dtype=float)

    # initial uniform flow
    u[:, :] = args.U0
    v[:, :] = 0.0
    apply_velocity_bcs(u, v, decomp, args.U0, 0.0, dx, dy)

    markers = build_circle_markers(
        args.cx, args.cy, args.R, args.n_markers) if args.ibm_circle else np.zeros((0, 2))

    params = dict(
        nx=nx, ny=ny, Lx=args.Lx, Ly=args.Ly,
        dx=dx, dy=dy, dt=args.dt, nu=args.nu,
        U0=args.U0, V0=0.0,
        poisson_iters=args.poisson_iters,
        poisson_omega=args.poisson_omega
    )

    if rank == 0:
        mode = "MPI" if size > 1 else "single-process"
        print(f"Mode: {mode} | nx={nx}, ny={ny}, dt={args.dt}, nu={args.nu}")
        print("BCs: inflow(left)=U0, far-field(top)=U0, wall(bottom)=no-slip, outflow(right)=convective-ish")
        if args.ibm_circle:
            print(
                f"IBM: circle at ({args.cx},{args.cy}), R={args.R}, markers={args.n_markers}")

    for step in range(1, args.steps + 1):
        u, v, p = rk3_step(u, v, p, decomp, params, markers)

        if step % args.out_every == 0 or step == 1:
            decomp.halo_exchange_x(u)
            decomp.halo_exchange_x(v)
            div_local = divergence(u, v, dx, dy, ng)
            l2_local = float(np.sqrt(np.mean(div_local**2)))
            l2 = comm.allreduce(l2_local, op=getattr(MPI, "MAX", None))
            if rank == 0:
                print(f"step {step:7d}  L2(div) ~ {l2:.3e}")

            save_single_or_gather(u, v, p, decomp, params, step, args.outdir)

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()
