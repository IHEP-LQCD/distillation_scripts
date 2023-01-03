# generate distillation elementals GPU
# copyright @ xiangyu jiang

import os
import cupy
import numpy as np
import lattice
import lattice.dif_dict as dif_dict
import lattice.mom9_dict as mom_dict
from time import time

lattice.setBackend(cupy)

lattSize = [24, 24, 24, 192]

confs = lattice.GaugeFieldIldg(FR"/dg_hpc/LQCD/DATA/light.b20.24_192.20220520/00.cfgs/", ".lime")
eigs = lattice.EigenVectorTimeSlice(R"/dg_hpc/LQCD/DATA/light.b20.24_192.20220520/02.laplace_eigs/", ".stout.n20.f0.12.laplace_eigs.3d.mod")
difList = dif_dict.dictToList()[0:4]
momList = mom_dict.dictToList()
outPrefix = R"./test/"
outSuffix = R".mom9.npy"

elementals = lattice.ElementalGenerator(lattSize, confs, eigs, difList, momList)

res = np.zeros((192, len(difList), len(momList), 160, 160), "<c16")

dispatcher = lattice.Dispatch("cfglist.txt", "test")

for cfg in dispatcher:
    if os.path.isfile(f"{outPrefix}{cfg}{outSuffix}"):
        continue
    elem = elementals[cfg]
    print(cfg, end=" ")

    s = time()
    for t in range(192):
        # s0 = time()
        res[t] = elem(t).get()
        # print(f"timeslices = {t}, {time()-s0: .2f}s.")
    print(f"{time() - s:.2f}Sec", end=" ")
    print(f"{elem.sizeInByte / elem.timeInSec / 1024 ** 2:.2f}MB/s")
    np.save(f"{outPrefix}{cfg}{outSuffix}", res.transpose(1,2,0,3,4))
