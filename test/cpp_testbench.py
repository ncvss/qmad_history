# very simple benchmark test

import torch
import qmad_history.wilson
import time

print("perform a very simple benchmark test")
lat_dim = [16,16,8,16]
cuda0 = torch.device("cuda:0")
mass = -0.5
n_reps = 2000
print("lattice:", lat_dim)
print("mass:",mass)
print("number of calls:",n_reps)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)



w_cpu = qmad_history.wilson.wilson_hop_mtsg(U, mass)

for i in range(n_reps):
    res = w_cpu.templ_tmsgMhs(v)

cpust = time.perf_counter_ns()
for i in range(n_reps):
    res = w_cpu.templ_tmsgMhs(v)
cpuen = time.perf_counter_ns()


print("cpu (avx) time per call in us:",(cpuen-cpust)/1000/n_reps)

