# generate perambulator multi-GPU
# dslash in double & save in single
# based on PyQuda
# copyright @ xiangyu jiang / chunjiang shi

from lattice import Dispatch
from pyquda.utils import gauge_utils, eigen_utils, source, layout
from pyquda.core import Nc, Ns
from pyquda import quda, core, enum_quda, mpi
from opt_einsum import contract
import cupyx as cpx
import cupy as cp
import numpy as np
from time import perf_counter
import os
import sys
os.environ["QUDA_RESOURCE_PATH"] = ".cache"


Ne = 160
latt_size = [24, 24, 24, 192]
### use 2 gpu for 24*192 latt.
grid_size = [1, 1, 1, 2]
### use 1 gpu for 16*128 latt.
# grid_size = [1, 1, 1, 1]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
mpi.init(grid_size)


def collectPeram(peram, root: int = 0, dtype="<c16"):
    Lx, Ly, Lz, Lt = latt_size
    Gx, Gy, Gz, Gt = grid_size
    sendbuf = peram.reshape((Lt * Ns * Ns * Ne * Ne)).astype(dtype)
    if mpi.rank == root:
        recvbuf = np.zeros((Gt * Gz * Gy * Gx, Lt * Ns * Ns * Ne * Ne), dtype)
    else:
        recvbuf = None
    if mpi.comm is not None:
        mpi.comm.Gatherv(sendbuf, recvbuf, root)
    else:
        recvbuf[0] = sendbuf
    if mpi.rank == root:
        ret = np.zeros((Lt * Gt, Ns, Ns, Ne, Ne), dtype)
        for i in range(Gx * Gy * Gz * Gt):
            gt = i % Gt
            gz = i // Gt % Gz
            gy = i // Gt // Gz % Gy
            gx = i // Gt // Gz // Gy
            ret[gt * Lt:(gt + 1) *
                Lt] = recvbuf[i].reshape((Lt, Ns, Ns, Ne, Ne))
        return ret
    else:
        return None

def calcPeram(key: str):
    gaugePath = F"/dg_hpc/LQCD/DATA/light.b20.24_192.20220520/00.cfgs/{key}.lime"
    evecPath = F"/dg_hpc/LQCD/DATA/light.b20.24_192.20220520/02.laplace_eigs/{key}.stout.n20.f0.12.laplace_eigs.3d.mod"
    savePath = F"/dg_hpc/LQCD/shichunjiang/DATA/light.b20.24_192.20220520/03.perambulator.single"
    print(f"Start calc: {key}")
    perambulator_tmp = cpx.empty_pinned((Lt, Ns, Ns), "<c8")
    perambulator_l = np.empty((Ne, Ne, Lt, Ns, Ns), "<c8")

    s = perf_counter()
    gauge = gauge_utils.readIldg(gaugePath)
    evecs = eigen_utils.readTimeSlice(evecPath, Ne)
    V_dag = cp.array(evecs.conj())
    SVnt = cp.empty((Ns, 2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<c8")
    print(
        FR"PYQUDA: {perf_counter()-s:.3f} sec for reading gauge configuration and eigen vectors.")

    ### set your dslsh params
    dslash_l = core.getDslash(latt_size, -0.074, 1e-11, 5000, 5.2, 0.9416346154, 0.6481490003,
                              1.1393286, multigrid=[[6, 6, 6, 4], [4, 4, 4, 4]], anti_periodic_t=True)
    dslash_l.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SILENT

    ### set if using BUCGSTAB / anti_periodic_t=False
    # dslash_l.invert_param.inv_type = enum_quda.QudaInverterType.QUDA_BICGSTAB_INVERTER
    # dslash_l.invert_param.solve_type = enum_quda.QudaSolveType.QUDA_DIRECT_PC_SOLVE

    ### stout smear
    core.smear(latt_size, gauge, 1, 0.241)
    dslash_l.loadGauge(gauge)
    print("END: smearing!")

    for ti in np.arange(Lt * Gt)[::1]:
        savePathAll = F"{savePath}/{key}.t{ti:03d}.peram.npy"
        if os.path.isfile(savePathAll):
            print(f"{savePathAll} exists!")
            continue
        s = perf_counter()
        for ni in range(Ne):
            # s0 = perf_counter()
            Vn = V_dag[ni].conj().astype("<c16")
            for si in range(Ns):
                b = source.source(latt_size, "colorvec", ti, si, 0, Vn)
                x = dslash_l.invert(b)
                SVnt[si] = x.data.reshape(
                    2, Lt, Lz, Ly, Lx // 2, Ns, Nc).astype("<c16")
            for nf in range(Ne):
                np.einsum("etzyxa,jetzyxia->tij",
                          V_dag[nf], SVnt, optimize=True).get(out=perambulator_tmp)
                perambulator_l[nf, ni] = perambulator_tmp
        ### save shape = (Lt, Ns, Ns, Ne, Ne)
        tosave = collectPeram(perambulator_l.transpose(
            2, 3, 4, 0, 1), dtype="<c8")
        if mpi.rank == 0:
            print(F"Save file: {savePathAll}")
            ### use .tofile(savePathAll) if saving binary data.
            np.save(savePathAll, np.roll(tosave, -ti, 0))
            print(
                FR"PYQUDA: {perf_counter()-s:.3f} sec for perambulator at rank={mpi.rank}, t={ti}.")
    dslash_l.destroy()
    quda.freeGaugeQuda()
    quda.freeCloverQuda()


if __name__ == "__main__":
    # key = sys.argv[1]
    quda.initQuda(mpi.gpuid)
    dispatcher = Dispatch("cfglist.txt",seed="wakuwaku")
    for cfg in dispatcher:
        print(cfg, end=" ")
        calcPeram(cfg)
    quda.endQuda()
