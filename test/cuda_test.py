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
rescu = w_cu.tmsgMh(vcu)
rescuv2 = w_cu.cuv2(vcu)
rescuv3 = w_cu.cuv3(vcu)
rescuv4 = w_cu.cuv4(vcu)

rescu_back = rescu.cpu()
rescuv2_b = rescuv2.cpu()
rescuv3_b = rescuv3.cpu()
rescuv4_b = rescuv4.cpu()

print("cpu and cuda computations equal:",
      torch.allclose(res,rescu_back), torch.allclose(res,rescuv2_b), torch.allclose(res,rescuv3_b), torch.allclose(res,rescuv4_b))

differsites = (torch.abs(res-rescuv4_b)>0.01)
print("number of sites that differ:",torch.sum(differsites))
print("tensors:")
print(res[0,1,2,4])
print(rescuv4_b[0,1,2,4])
print(res[4,1,2,4])
print(rescuv4_b[4,1,2,4])
print(res[7,1,3,3])
print(rescuv4_b[7,1,3,3])
print(res[5,5,5,4])
print(rescuv4_b[5,5,5,4])

print("tensor of differing sites:")
print(differsites[0,1,2])
