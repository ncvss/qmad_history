# very simple benchmark test

import torch
import qmad_history.wilson
import time

print("do a very simple benchmark try")
lat_dim = [16,16,8,16]
cuda0 = torch.device("cuda:0")
mass = -0.5
n_reps = 10000
print("lattice:", lat_dim)
print("mass:",mass)
print("number of calls:",n_reps)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)


w_cpu = qmad_history.wilson.wilson_hop_mtsg(U, mass)
w_cu = qmad_history.wilson.wilson_hop_mtsg(Ucu, mass)

cpust = time.perf_counter_ns()
for i in range(n_reps):
    res = w_cpu.templ_tmsgMhs(v)
cpuen = time.perf_counter_ns()

torch.cuda.synchronize()
cust = time.perf_counter_ns()
for i in range(n_reps):
    rescu = w_cu.tmsgMh(vcu)
torch.cuda.synchronize()
cuen = time.perf_counter_ns()

torch.cuda.synchronize()
cust2 = time.perf_counter_ns()
for i in range(n_reps):
    rescu2 = w_cu.cuv2(vcu)
torch.cuda.synchronize()
cuen2 = time.perf_counter_ns()

rescu_back = rescu.cpu()
rescu2_back = rescu2.cpu()

print("cpu and cuda computations equal:", torch.allclose(res,rescu_back) and torch.allclose(res,rescu2_back))
print("cpu (avx) time per call in us:",(cpuen-cpust)/1000/n_reps)
print("cuda time per call in us:",(cuen-cust)/1000/n_reps)
print("cuda v2 time per call in us:",(cuen2-cust2)/1000/n_reps)
