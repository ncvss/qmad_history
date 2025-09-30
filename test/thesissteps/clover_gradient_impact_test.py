# test how much gradient computation impacts the performance

import torch
import numpy as np
import time
import copy
import socket

import gpt as g
import qcd_ml

from qmad_history import compat, clover, settings


print("settings:", settings.capab())

num_threads = torch.get_num_threads()
hostname = socket.gethostname()
print("running on host",hostname,"with",num_threads,"threads")


# split measurement into n_batch batches
# we alternate between operators and lattice dimensions
n_measurements = 200
n_batch = 10
assert n_measurements%n_batch == 0
n_batchlen = n_measurements//n_batch
n_warmup = 20

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions in",n_batch,"batches")
print("mass =", mass)
print("csw =",csw)

rng = g.random("these")

start_grid = [4,4,2,4]
n_vols = 15
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [4*4*4*4*2**ii for ii in range(n_vols)]
names = ["no_grad","with_grad"]

results = {vv:{na:np.zeros(n_measurements) for na in names} for vv in vols}

# split data generation into batches that each iterate over all sites
# required to further restrict the influence of background processes

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

        U = torch.tensor(compat.lattice_to_array(U_gpt))
        v = torch.tensor(compat.lattice_to_array(v_gpt))
        

        dwc = clover.wilson_clover_hop_mtsg_sigpre(U,mass,csw)
        dwc_ref = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)

        for n in range(n_warmup):
            vgrad = v.clone().detach().requires_grad_(True)
            res_no = dwc.tmnsgMhs(v)
            res_grad = dwc.tmnsgMhs(vgrad)
            # call gradient computation in each iteration
            loss_qmad = (res_grad * res_grad.conj()).real.sum()
            loss_qmad.backward()
            if n == 0 and nb == 0:
                vgrad2 = v.clone().detach().requires_grad_(True)
                res_ref = dwc_ref(vgrad2)
                loss_py = (res_ref * res_ref.conj()).real.sum()
                loss_py.backward()
                print("computations equal:",[torch.allclose(res_ref,res_ch) for res_ch in [res_no,res_grad]])
                print("gradients equal:",torch.allclose(vgrad.grad,vgrad2.grad))

        for n in range(nb,nb+n_batchlen):
            # reset in each iteration
            vgrad = v.clone().detach().requires_grad_(True)

            start = time.perf_counter_ns()
            res_no = dwc.tmnsgMhs(v)
            stop = time.perf_counter_ns()
            results[vol]["no_grad"][n] = stop - start

            start = time.perf_counter_ns()
            res_grad = dwc.tmnsgMhs(vgrad)
            stop = time.perf_counter_ns()
            results[vol]["with_grad"][n] = stop - start

            # call gradient computation in each iteration
            loss_qmad = (res_grad * res_grad.conj()).real.sum()
            loss_qmad.backward()



for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:n_measurements//5]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev",np.std(result_sort)/1000,"µs")
    
