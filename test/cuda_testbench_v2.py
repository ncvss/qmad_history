# very simple benchmark test for cuda

import torch
import qmad_history.wilson
import time
import numpy as np

print("benchmark with multiple calls and warmup")
lat_dim = [32,32,32,64]
vol = lat_dim[0]*lat_dim[1]*lat_dim[2]*lat_dim[3]
cuda0 = torch.device("cuda:0")
mass = -0.5
n_rep = 200
n_warmup = 30
print("lattice:", lat_dim)
print("mass:",mass)
print("number of calls:",n_rep)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)

data_size_GiB = vol*(16*(4*3*3+4*3*2)+8*4)/(1024**3)
print("transferred data size in GiB", data_size_GiB)


#w_cpu = qmad_history.wilson.wilson_hop_mtsg(U, mass)
w_cu = qmad_history.wilson.wilson_hop_mtsg(Ucu, mass)
#w_full = qmad_history.wilson.wilson_full(Ucu, mass)

w_cu_versions = [
    w_cu.tmsgMh, w_cu.cuv2, w_cu.cuv4, w_cu.cuv6, w_cu.cuv9,
    ]
names = ["site","spin-colour","colour term spin-colour","mu-colour term spin-colour","colour term spin-colour hops split"]
cutimes = [np.zeros(n_rep) for _ in w_cu_versions]


for n in range(n_warmup):
    for w_cu_call in w_cu_versions:
        torch.cuda.synchronize()
        rescu = w_cu_call(vcu)
        torch.cuda.synchronize()

for n in range(n_rep):
    for i,w_cu_call in enumerate(w_cu_versions):
        torch.cuda.synchronize()
        cust = time.perf_counter_ns()
        rescu = w_cu_call(vcu)
        torch.cuda.synchronize()
        cuen = time.perf_counter_ns()
        cutimes[i][n] = (cuen-cust)

times_sorted = []
for t in cutimes:
    times_sorted.append(np.sort(t)[:(n_rep//5)])


print("cuda times and standard deviations (best 20%) in ms:")
for i,t in enumerate(times_sorted):
    print(names[i],"|",np.mean(t)/1000/1000,"|",np.std(t)/1000/1000)
