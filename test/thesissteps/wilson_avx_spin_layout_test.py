import torch
import numpy as np
import time

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline


num_threads = torch.get_num_threads()

# split measurement into batches
# we alternate between operators and lattice dimensions
# and make n_batch measurements for each
n_measurements = 200
n_warmup = 20

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions")
print("mass =", mass)

rng = g.random("thesis")

start_grid = [4,4,2,4]
n_vols = 12
vols = [4*4*4*4*2**ii for ii in range(n_vols)]
names = ["mtsg","tmgs"]

results = {vv:{na:np.zeros(n_measurements) for na in names} for vv in vols}

for L_incr in range(n_vols):
    start_grid[(L_incr+2)%4] *= 2
    vol = start_grid[0]*start_grid[1]*start_grid[2]*start_grid[3]
    print("---")
    print(vol,"sites")
    print("grid layout",start_grid)

    # initialise the fields for this volume
    U_gpt = g.qcd.gauge.random(g.grid(start_grid, g.double), rng)
    grid = U_gpt[0].grid
    v_gpt = rng.cnormal(g.vspincolor(grid))

    U_mtsg = torch.tensor(compat.lattice_to_array(U_gpt))
    v_mtsg = torch.tensor(compat.lattice_to_array(v_gpt))
    v_tmgs = torch.permute(v_mtsg,(0,1,2,3,5,4)).contiguous()

    dw_mtsg = wilson.wilson_hop_mtsg(U_mtsg,mass)
    dw_tmgs = wilson.wilson_hop_tmgs(U_mtsg,mass)

    for n in range(n_warmup):
        res_mtsg = dw_mtsg.avx_tmsgMhs(v_mtsg)
        res_tmgs = dw_tmgs.avx_tmgsMhs(v_tmgs)
        if n == 0:
            res_tmgs_back = torch.permute(res_tmgs,(0,1,2,3,5,4))
            print("computations equal:",torch.allclose(res_mtsg,res_tmgs_back))

    for n in range(n_measurements):
        start = time.perf_counter_ns()
        res_mtsg = dw_mtsg.avx_tmsgMhs(v_mtsg)
        stop = time.perf_counter_ns()
        results[vol]["mtsg"][n] = stop - start

        start = time.perf_counter_ns()
        res_tmgs = dw_tmgs.avx_tmgsMhs(v_tmgs)
        stop = time.perf_counter_ns()
        results[vol]["tmgs"][n] = stop - start
    
    for na in names:
        result_sort = np.sort(results[vol][na])[:n_measurements//5]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev",np.std(result_sort)/1000,"µs")
    
