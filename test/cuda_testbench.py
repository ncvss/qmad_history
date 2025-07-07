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

w_cu_versions = [w_cu.tmsgMh, w_cu.cuv2, w_cu.cuv3, w_cu.cuv4, w_cu.cuv5, w_cu.cuv6, w_cu.cuv7, w_cu.cuv8, w_cu.cuv9, w_full]
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

# torch.cuda.synchronize()
# cust = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu = w_cu.tmsgMh(vcu)
# torch.cuda.synchronize()
# cuen = time.perf_counter_ns()

# rescu_back = rescu.cpu()

# torch.cuda.synchronize()
# cust2 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu2 = w_cu.cuv2(vcu)
# torch.cuda.synchronize()
# cuen2 = time.perf_counter_ns()

# rescu2_back = rescu2.cpu()

# torch.cuda.synchronize()
# cust3 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu3 = w_cu.cuv3(vcu)
# torch.cuda.synchronize()
# cuen3 = time.perf_counter_ns()

# rescu3_back = rescu3.cpu()

# torch.cuda.synchronize()
# cust4 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu4 = w_cu.cuv4(vcu)
# torch.cuda.synchronize()
# cuen4 = time.perf_counter_ns()

# rescu4_back = rescu4.cpu()

# torch.cuda.synchronize()
# cust5 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu5 = w_cu.cuv5(vcu)
# torch.cuda.synchronize()
# cuen5 = time.perf_counter_ns()

# rescu5_back = rescu5.cpu()

# torch.cuda.synchronize()
# cust6 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu6 = w_cu.cuv6(vcu)
# torch.cuda.synchronize()
# cuen6 = time.perf_counter_ns()

# rescu6_back = rescu6.cpu()

# torch.cuda.synchronize()
# cust7 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu7 = w_cu.cuv7(vcu)
# torch.cuda.synchronize()
# cuen7 = time.perf_counter_ns()

# rescu7_back = rescu7.cpu()

# torch.cuda.synchronize()
# cust8 = time.perf_counter_ns()
# for i in range(n_reps):
#     rescu8 = w_cu.cuv8(vcu)
# torch.cuda.synchronize()
# cuen8 = time.perf_counter_ns()

# rescu8_back = rescu8.cpu()


print("cpu and cuda computations equal:", list(enumerate(correctnesses)))
    #   torch.allclose(res,rescu_back), torch.allclose(res,rescu2_back), torch.allclose(res,rescu3_back),
    #   torch.allclose(res,rescu4_back), torch.allclose(res,rescu5_back), torch.allclose(res,rescu6_back),
    #   torch.allclose(res,rescu7_back), torch.allclose(res,rescu8_back), )
print("cpu (avx) time per call in us:",(cpuen-cpust)/1000/n_reps)
print("cuda times per call in us:")
for i,t in enumerate(cutimes):
    print("v"+str(i)+":",t/1000/n_reps)
# print("cuda time per call in us:",(cuen-cust)/1000/n_reps)
# print("cuda v2 time per call in us:",(cuen2-cust2)/1000/n_reps)
# print("cuda v3 time per call in us:",(cuen3-cust3)/1000/n_reps)
# print("cuda v4 time per call in us:",(cuen4-cust4)/1000/n_reps)
# print("cuda v5 time per call in us:",(cuen5-cust5)/1000/n_reps)
# print("cuda v6 time per call in us:",(cuen6-cust6)/1000/n_reps)
# print("cuda v7 time per call in us:",(cuen7-cust7)/1000/n_reps)
# print("cuda v8 time per call in us:",(cuen8-cust8)/1000/n_reps)
