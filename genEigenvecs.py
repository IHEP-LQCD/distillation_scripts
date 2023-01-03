# generate laplace eigen-vectors GPU or CPU
# based on PyQuda
# copyright @ xiangyu jiang

import functools
import os
from time import perf_counter
### if use cpu
# import numpy as np
# from scipy.sparse import linalg
import cupy as np
from cupyx.scipy.sparse import linalg

from pyquda import core, quda, mpi
from pyquda.utils import gauge_utils

Nc, Nd = 3, 4
Ne = 160
latt_size = [24, 24, 24, 192]
Lx, Ly, Lz, Lt = latt_size

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
mpi.init()


def _Amatmat(colvec, colmat, colmat_dag):
    colvec = colvec.reshape(Lz, Ly, Lx, Nc, -1)
    return (
        6 * colvec + (
            np.einsum("zyxab,zyxbc->zyxac", colmat[0], np.roll(colvec, -1, 2)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[1], np.roll(colvec, -1, 1)) +
            np.einsum("zyxab,zyxbc->zyxac", colmat[2], np.roll(colvec, -1, 0)) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[0], colvec), 1, 2) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[1], colvec), 1, 1) +
            np.roll(np.einsum("zyxab,zyxbc->zyxac", colmat_dag[2], colvec), 1, 0)
        )
    ).reshape(Lz * Ly * Lx * Nc, -1)


gauge = gauge_utils.readIldg("/dg_hpc/LQCD/DATA/light.b20.24_192.20220520/00.cfgs/s1_cfg_1050.lime")

quda.initQuda(mpi.gpuid)
core.smear(latt_size, gauge, 20, 0.12)
colmat_all = gauge.lexico().reshape(Nd, Lt, Lz, Ly, Lx, Nc, Nc)[:3]
quda.endQuda()

V = np.zeros((Lt, Lz * Ly * Lx * Nc, Ne), "<c16")
for t in range(Lt):
    s = perf_counter()
    colmat = np.asarray(colmat_all[:, t].copy())
    colmat_dag = colmat.transpose(0, 1, 2, 3, 5, 4).conj()
    Amatmat = functools.partial(_Amatmat, colmat=colmat, colmat_dag=colmat_dag)
    A = linalg.LinearOperator((Lz * Ly * Lx * Nc, Lz * Ly * Lx * Nc), matvec=Amatmat, matmat=Amatmat)
    # evals, evecs = primme.eigsh(A, Ne, tol=1e-7, which="SA", method="PRIMME_DEFAULT_MIN_TIME", aNorm=1.0)
    evals, evecs = linalg.eigsh(A, Ne, tol=1e-7, which="LA")
    print(12 - evals)
    V[t] = evecs
    print(FR"PYQUDA: {perf_counter()-s:.3f}sec for solving {Ne} eigen systems at t={t}.")

np.save("./s1_cfg_1050.evecs.npy", V.transpose(2, 0, 1))
