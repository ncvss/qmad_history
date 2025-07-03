# test if operator works on a cuda machine
# and if the output is then correct

import torch
import qmad_history.wilson

lat_dim = [8,8,8,16]
cuda0 = torch.device("cuda:0")
mass = -0.5

print("test if wilson dirac operator works on cuda the same as on cpu")
print("lattice:", lat_dim)
print("mass:",mass)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)

w_cpu = qmad_history.wilson.wilson_hop_mtsg(U, mass)
w_cu = qmad_history.wilson.wilson_hop_mtsg(Ucu, mass)

res = w_cpu.tmsgMh(v)

w_cu_versions = [w_cu.tmsgMh, w_cu.cuv2, w_cu.cuv3, w_cu.cuv4, w_cu.cuv5, w_cu.cuv6, w_cu.cuv7, w_cu.cuv8, w_cu.cuv9]

rescus = []

for w_cu_call in w_cu_versions:
    rescu = w_cu_call(vcu)
    rescu_b = rescu.cpu()
    rescus.append(rescu_b)


print("cpu and cuda computations equal:", [torch.allclose(res, rescu_b) for rescu_b in rescus])

differsites = (torch.abs(res-rescus[-1])<0.01)
print("number of sites that are the same:",torch.sum(differsites))

# for x in range(8):
#     for y in range(8):
#         for z in range(8):
#             print("all correct at x,y,z=",x,y,z,":",torch.all(differsites[x,y,z]))

# test how the tensor looks like
swaps = 0
diffflat = torch.flatten(differsites)
for i in range(torch.numel(diffflat)-1):
    if diffflat[i] != diffflat[i+1]:
        swaps += 1
        # print("swap at",i)

print("correctness swaps at the following number of spots:",swaps)
