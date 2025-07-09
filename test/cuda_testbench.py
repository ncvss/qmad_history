# very simple benchmark test

import torch
import qmad_history.wilson
import time

print("do a very simple benchmark try")
lat_dim = [8,8,8,8]
cuda0 = torch.device("cuda:0")
mass = -0.5
n_reps = 2000
print("lattice:", lat_dim)
print("mass:",mass)
print("number of calls:",n_reps)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)


w_cpu = qmad_history.wilson.wilson_hop_mtsg(U, mass)
w_cu = qmad_history.wilson.wilson_hop_mtsg(Ucu, mass)
w_full = qmad_history.wilson.wilson_full(Ucu, mass)

w_cu_versions = [w_cu.tmsgMh, w_cu.cuv2, w_cu.cuv3, w_cu.cuv4, w_cu.cuv5, w_cu.cuv6, w_cu.cuv7, w_cu.cuv8, w_cu.cuv9,
                 w_full.cuv10, w_full.cuv11]
cutimes = []
correctnesses = []

for i in range(n_reps):
    res = w_cpu.templ_tmsgMhs(v)

cpust = time.perf_counter_ns()
for i in range(n_reps):
    res = w_cpu.templ_tmsgMhs(v)
cpuen = time.perf_counter_ns()

for w_cu_call in w_cu_versions:
    torch.cuda.synchronize()
    cust = time.perf_counter_ns()
    for i in range(n_reps):
        rescu = w_cu_call(vcu)
    torch.cuda.synchronize()
    cuen = time.perf_counter_ns()
    cutimes.append(cuen-cust)
    rescu_b = rescu.cpu()
    correctnesses.append(torch.allclose(res,rescu_b))

corr2 = w_full.cuv10(vcu)
corr2b = corr2.cpu()
print("dummy computation coherent:", torch.allclose(corr2b,rescu_b))

print("cpu and cuda computations equal:", list(enumerate(correctnesses)))
print("cpu (avx) time per call in us:",(cpuen-cpust)/1000/n_reps)
print("cuda times per call in us:")
for i,t in enumerate(cutimes):
    print("v"+str(i)+":",t/1000/n_reps)
