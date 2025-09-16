import torch
import numpy as np
import time
import copy

from qmad_history import wilson


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
#print("csw =",csw)

cuda0 = torch.device("cuda:0")
print("using device",cuda0)

# this is on the gpu, so we need way larger lattices
start_grid = [8,8,4,8]
start_vol = start_grid[0]*start_grid[1]*start_grid[2]*start_grid[3]
n_vols = 12
all_grids = []
for i in range(n_vols):
    start_grid[(i+2)%4] *= 2
    all_grids.append(copy.copy(start_grid))

vols = [start_vol*2 *2**ii for ii in range(n_vols)]
names = ["tsg_kernel","tmsg_kernel","tmsgh_kernel","tsg_3d_kernel"]

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
        # gpt is not installed, so we have to use pseudo fields
        U_cpu = torch.randn([4]+cgrid+[3,3], dtype=torch.cdouble)
        v_cpu = torch.randn(cgrid+[4,3], dtype=torch.cdouble)

        Ucu = U_cpu.to(cuda0)
        vcu = v_cpu.to(cuda0)

        dw_cu = wilson.wilson_hop_mtsg(Ucu, mass)
        # dw_ref = qcd_ml.qcd.dirac.dirac_wilson(Ucu, mass)

        # calls = [dw_cu.cu_tsg,dw_cu.cu_Mtmsg,dw_cu.cu_Mtmsgh,dw_cu.cu_3d_tsg]

        for n in range(n_warmup):
            torch.cuda.synchronize()
            res_tsg = dw_cu.cu_tsg(vcu)
            torch.cuda.synchronize()
            res_tmsg = dw_cu.cu_Mtmsg(vcu)
            torch.cuda.synchronize()
            res_tmsgh = dw_cu.cu_Mtmsgh(vcu)
            torch.cuda.synchronize()
            res_3d_tsg = dw_cu.cuv2(vcu)
            torch.cuda.synchronize()
            if n == 0 and nb == 0:
                # res_ref = dw_ref(vcu)
                print("computations equal:",[torch.allclose(res_tsg,res_ch) for res_ch in [res_tmsg,res_tmsgh,res_3d_tsg]])

        for n in range(nb,nb+n_batchlen):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_tsg = dw_cu.cu_tsg(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["tsg_kernel"][n] = stop - start

            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_tmsg = dw_cu.cu_Mtmsg(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["tmsg_kernel"][n] = stop - start

            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_tmsgh = dw_cu.cu_Mtmsgh(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["tmsgh_kernel"][n] = stop - start

            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            res_3d_tsg = dw_cu.cuv2(vcu)
            torch.cuda.synchronize()
            stop = time.perf_counter_ns()
            results[vol]["tsg_3d_kernel"][n] = stop - start


for vol,cgrid in zip(vols,all_grids):
    print("---")
    print(vol,"sites")
    print("grid layout",cgrid)
    for na in names:
        result_sort = np.sort(results[vol][na])[:(n_measurements//5)]
        print(na,"time mean:",np.mean(result_sort)/1000,"µs")
        print(na+"_stdev:",np.std(result_sort)/1000,"µs")
    
