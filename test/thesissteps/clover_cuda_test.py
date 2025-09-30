import torch
import numpy as np
import time
import copy

from qmad_history import clover, settings

print("settings:", settings.capab())

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

cuda0 = torch.device("cuda")
print("using device",cuda0)

# on the GPU this script was run on, it ran out of memory for 64,32,32,64
start_grid = [8,8,4,8]
start_vol = start_grid[0]*start_grid[1]*start_grid[2]*start_grid[3]
n_vols = 10
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [start_vol*2 *2**ii for ii in range(n_vols)]
names = ["fmunu","sigmaf",]

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
        # gpt is not installed, so we have to use random numbers as pseudo fields
        # does not change the performance of the computation
        U_cpu = torch.randn([4]+cgrid+[3,3], dtype=torch.cdouble)
        v_cpu = torch.randn(cgrid+[4,3], dtype=torch.cdouble)

        Ucu = U_cpu.to(cuda0)
        vcu = v_cpu.to(cuda0)

        dwc_f = clover.wilson_clover_hop_mtsg(Ucu, mass, csw)
        torch.cuda.synchronize()
        dwc_sf = clover.wilson_clover_hop_mtsg_sigpre(Ucu, mass, csw)


        for n in range(n_warmup):
            torch.cuda.synchronize()
            res_f = dwc_f.cu_tsg(vcu)
            torch.cuda.synchronize()
            res_sf = dwc_sf.cu_tsg_tn(vcu)
            torch.cuda.synchronize()
            # because these are pseudo fields, the results are not equal

        for n in range(nb,nb+n_batchlen):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_f = dwc_f.cu_tsg(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["fmunu"][n] = stop - start

            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_sf = dwc_sf.cu_tsg_tn(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["sigmaf"][n] = stop - start


for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:(n_measurements//5)]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev:",np.std(result_sort)/1000,"µs")
    
