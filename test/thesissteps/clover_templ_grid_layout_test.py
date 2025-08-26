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
print("csw =",csw)

rng = g.random("thes")

start_grid = [4,4,2,4]
# mehr als 32x32x32x32 ist auf meinem PC nicht möglich, zu wenig Speicher führt zu Absturz
n_vols = 7
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [4*4*4*4*2**ii for ii in range(n_vols)]
names = ["spin_vect","grid_vect"]

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
        vgfirst = v_mtsg[:,:,:,0:cgrid[3]//2]
        vgsecond = v_mtsg[:,:,:,cgrid[3]//2:cgrid[3]]
        v_mtsgt = torch.stack([vgfirst, vgsecond], dim=-1)

        dwc_spin = clover.wilson_clover_hop_mtsg_sigpre(U_mtsg,mass,csw)
        dwc_grid = clover.wilson_clover_hop_mtsgt2_sigpre(U_mtsg,mass,csw)
        dwc_ref = qcd_ml.qcd.dirac.dirac_wilson_clover(U_mtsg,mass,csw)

        for n in range(n_warmup):
            res_spin = dwc_spin.tmnsgMhs(v_mtsg)
            res_grid = dwc_grid.tmngsMht(v_mtsgt)
            if n == 0 and nb == 0:
                res_ref = dwc_ref(v_mtsg)
                res_grid_back = torch.cat([res_grid[:,:,:,:,:,:,0],res_grid[:,:,:,:,:,:,1]], dim=3)
                print("computations equal:",[torch.allclose(res_ref,res_c) for res_c in [res_spin,res_grid_back]])

        for n in range(nb,nb+n_batchlen):
            start = time.perf_counter_ns()
            res_spin = dwc_spin.tmnsgMhs(v_mtsg)
            stop = time.perf_counter_ns()
            results[vol]["spin_vect"][n] = stop - start

            start = time.perf_counter_ns()
            res_grid = dwc_grid.tmngsMht(v_mtsgt)
            stop = time.perf_counter_ns()
            results[vol]["grid_vect"][n] = stop - start



for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:n_measurements//5]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev:",np.std(result_sort)/1000,"µs")
    
