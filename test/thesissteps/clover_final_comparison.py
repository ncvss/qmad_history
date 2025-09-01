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
n_measurements = 20
n_batch = 1
assert n_measurements%n_batch == 0
n_batchlen = n_measurements//n_batch
n_warmup = 20

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions in",n_batch,"batches")
print("mass =", mass)

rng = g.random("th")

start_grid = [4,4,2,4]
# mehr als 32x32x32x32 ist auf meinem PC nicht möglich, zu wenig Speicher führt zu Absturz
n_vols = 15
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [4*4*4*4*2**ii for ii in range(n_vols)]
names = ["gpt","qcd_ml","qmad","qmad_gridl"]

# set max time for measure
max_time = 2.0e+9

results = {vv:{na:np.zeros(n_measurements) for na in names} for vv in vols}

# split data generation into batches that each iterate over all sites
# it does not work without that, pytorch does some strange things

for nb in range(0,n_measurements,n_batchlen):
    max_exceeded = {na:False for na in names}
    print("\ncurrent batch:",nb,flush=True)
    print("current grid layout: ")
    for L_incr in range(n_vols):
        cgrid = all_grids[L_incr]
        vol = cgrid[0]*cgrid[1]*cgrid[2]*cgrid[3]
        print(cgrid,end=" ",flush=True)
        print("time exceeded",max_time,":",max_exceeded)

        # initialise the fields for this volume
        U_gpt = g.qcd.gauge.random(g.grid(cgrid, g.double), rng)
        grid = U_gpt[0].grid
        v_gpt = rng.cnormal(g.vspincolor(grid))

        U_mtsg = torch.tensor(compat.lattice_to_array(U_gpt))
        v_mtsg = torch.tensor(compat.lattice_to_array(v_gpt))

        vgfirst = v_mtsg[:,:,:,0:cgrid[3]//2]
        vgsecond = v_mtsg[:,:,:,cgrid[3]//2:cgrid[3]]
        v_mtsgt = torch.stack([vgfirst, vgsecond], dim=-1)

        dwc_g = g.qcd.fermion.wilson_clover(U_gpt, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                  "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        if not max_exceeded["qcd_ml"]:
            dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U_mtsg,mass,csw)
        dwc_qmad = clover.wilson_clover_hop_mtsg_sigpre(U_mtsg,mass,csw)
        dwc_gridl = clover.wilson_clover_hop_mtsgt2_sigpre(U_mtsg,mass,csw)

        for n in range(n_warmup):
            res_g = dwc_g(v_gpt)
            if not max_exceeded["qcd_ml"]:
                res_py = dwc_py(v_mtsg)
            res_qmad = dwc_qmad.tmnsgMhs(v_mtsg)
            res_gridl = dwc_gridl.tmngsMht(v_mtsgt)
            if n == 0 and nb == 0:
                res_gpt_back = torch.tensor(compat.lattice_to_array(res_g))
                res_grid_back = torch.cat([res_gridl[:,:,:,:,:,:,0],res_gridl[:,:,:,:,:,:,1]], dim=3)
                print("computations equal:",[torch.allclose(res_gpt_back,res_ch) for res_ch in [res_qmad,res_grid_back]])

        for n in range(nb,nb+n_batchlen):
            start = time.perf_counter_ns()
            res_g = dwc_g(v_gpt)
            stop = time.perf_counter_ns()
            results[vol]["gpt"][n] = stop - start

            if not max_exceeded["qcd_ml"]:
                start = time.perf_counter_ns()
                res_py = dwc_py(v_mtsg)
                stop = time.perf_counter_ns()
                results[vol]["qcd_ml"][n] = stop - start
            else:
                results[vol]["qcd_ml"][n] = 1.0e+10

            start = time.perf_counter_ns()
            res_qmad = dwc_qmad.tmnsgMhs(v_mtsg)
            stop = time.perf_counter_ns()
            results[vol]["qmad"][n] = stop - start

            start = time.perf_counter_ns()
            res_gridl = dwc_gridl.tmngsMht(v_mtsgt)
            stop = time.perf_counter_ns()
            results[vol]["qmad_gridl"][n] = stop - start
        
        for na in names:
            if results[vol][na][nb+n_batchlen-1] > max_time:
                max_exceeded[na] = True
            


for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:n_measurements//5]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev:",np.std(result_sort)/1000,"µs")
    
