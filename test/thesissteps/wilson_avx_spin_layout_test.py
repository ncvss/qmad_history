import torch
import numpy as np
import time
import copy
import socket
import os

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline


num_threads = torch.get_num_threads()
hostname = socket.gethostname()
print("running on host",hostname,"with",num_threads,"threads")


# split measurement into n_batch batches
# we alternate between operators and lattice dimensions
n_measurements = 200
n_batch = 5
assert n_measurements%n_batch == 0
n_batchlen = n_measurements//n_batch
n_warmup = 20

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions in",n_batch,"batches")
print("mass =", mass)

rng = g.random("thesis")

start_grid = [4,4,2,4]
# mehr als 32x32x32x32 ist nicht möglich, zu wenig Speicher führt zu Absturz
n_vols = 15
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [4*4*4*4*2**ii for ii in range(n_vols)]
names = ["mtsg","tmgs","mtsg_v","tmgs_v"]

results = {vv:{na:np.zeros(n_measurements) for na in names} for vv in vols}

# split data generation into batches that each iterate over all sites
# it does not work without that, pytorch does some strange things

for nb in range(0,n_measurements,n_batchlen):
    print("\ncurrent batch:",nb,flush=True)
    print("current grid layout: ")
    for L_incr in range(n_vols):
        cgrid = all_grids[L_incr]
        vol = cgrid[0]*cgrid[1]*cgrid[2]*cgrid[3]
        print(cgrid,end=" ",flush=True)

        # initialise the fields for this volume
        U_gpt = g.qcd.gauge.random(g.grid(cgrid, g.double), rng)
        grid = U_gpt[0].grid
        v_gpt = rng.cnormal(g.vspincolor(grid))

        U_mtsg = torch.tensor(compat.lattice_to_array(U_gpt))
        v_mtsg = torch.tensor(compat.lattice_to_array(v_gpt))
        v_tmgs = torch.permute(v_mtsg,(0,1,2,3,5,4)).contiguous()

        dw_mtsg = wilson.wilson_hop_mtsg(U_mtsg,mass)
        dw_tmgs = wilson.wilson_hop_tmgs(U_mtsg,mass)
        dw_ref = qcd_ml.qcd.dirac.dirac_wilson(U_mtsg,mass)

        for n in range(n_warmup):
            res_mtsg = dw_mtsg.tmsgMh(v_mtsg)
            res_tmgs = dw_tmgs.tmgsMh(v_tmgs)
            res_mtsg_v = dw_mtsg.avx_tmsgMhs(v_mtsg)
            res_tmgs_v = dw_tmgs.avx_tmgsMhs(v_tmgs)
            if n == 0 and nb == 0:
                res_ref = dw_ref(v_mtsg)
                res_tmgs_back = torch.permute(res_tmgs,(0,1,2,3,5,4))
                res_tmgs_v_back = torch.permute(res_tmgs_v,(0,1,2,3,5,4))
                print("computations equal:",[torch.allclose(res_ref,res_c) for res_c in [res_mtsg,res_mtsg_v,res_tmgs_back,res_tmgs_v_back]])

        for n in range(nb,nb+n_batchlen):
            start = time.perf_counter_ns()
            res_mtsg = dw_mtsg.tmsgMh(v_mtsg)
            stop = time.perf_counter_ns()
            results[vol]["mtsg"][n] = stop - start

            start = time.perf_counter_ns()
            res_tmgs = dw_tmgs.tmgsMh(v_tmgs)
            stop = time.perf_counter_ns()
            results[vol]["tmgs"][n] = stop - start

            start = time.perf_counter_ns()
            res_mtsg_v = dw_mtsg.avx_tmsgMhs(v_mtsg)
            stop = time.perf_counter_ns()
            results[vol]["mtsg_v"][n] = stop - start

            start = time.perf_counter_ns()
            res_tmgs_v = dw_tmgs.avx_tmgsMhs(v_tmgs)
            stop = time.perf_counter_ns()
            results[vol]["tmgs_v"][n] = stop - start


for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:n_measurements//5]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev",np.std(result_sort)/1000,"µs")
    
