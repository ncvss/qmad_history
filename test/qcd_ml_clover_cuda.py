# use qcd_ml on the gpu

import torch
import qcd_ml
from qmad_history import clover
import time
import numpy as np

cuda0 = torch.device("cuda")
print("using device",cuda0)
mass = -0.55
csw = 1.0

print("test if wilson clover works on gpu")
print("mass:",mass)
print("csw:",csw)


lat_dim_1 = [32,32,32,64]
print("lattice 1:", lat_dim_1)

U = torch.empty([4]+lat_dim_1+[3,3], dtype=torch.cdouble)
U[:,:,:,:,:] = torch.eye(3, dtype=torch.cdouble)

v = torch.randn(lat_dim_1+[4,3], dtype=torch.cdouble)


U2 = torch.tensor(np.load("./test/1500.config.npy"), dtype=torch.cdouble)
lat_dim_2 = list(U2.shape[1:5])
print("lattice 2:", lat_dim_2)

# v2 = torch.zeros(lat_dim_2+[4,3], dtype=torch.cdouble)
# v2[0,1,0,0,1,2] = 1
v2 = torch.randn(lat_dim_2+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)
U2cu = U2.to(cuda0)
v2cu = v2.to(cuda0)

torch.cuda.synchronize()
dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(Ucu, mass, csw)
torch.cuda.synchronize()
dwc_sf = clover.wilson_clover_hop_mtsg_sigpre(Ucu, mass, csw)
torch.cuda.synchronize()
dwc_f = clover.wilson_clover_hop_mtsg(Ucu, mass, csw)
torch.cuda.synchronize()

dwc2 = qcd_ml.qcd.dirac.dirac_wilson_clover(U2cu, mass, csw)
torch.cuda.synchronize()
dwc2_sf = clover.wilson_clover_hop_mtsg_sigpre(U2cu, mass, csw)
torch.cuda.synchronize()
dwc2_f = clover.wilson_clover_hop_mtsg(U2cu, mass, csw)
torch.cuda.synchronize()

# F = dwc2_f.field_strength.cpu()
# print("field strength:")
# print(F[0,1,0,0])

time1 = time.perf_counter_ns()
res_py = dwc(vcu)
torch.cuda.synchronize()
time2 = time.perf_counter_ns()
print("python time (lattice 1):",time2-time1)
torch.cuda.synchronize()
res_sf = dwc_sf.cu_tsg_tn(vcu)
torch.cuda.synchronize()
res_f = dwc_f.cu_tsg(vcu)
torch.cuda.synchronize()

res2_py = dwc2(v2cu)
torch.cuda.synchronize()
for _ in range(1):
    res2_f = dwc2_f.cu_tsg(v2cu)
    torch.cuda.synchronize()
    res2_sf = dwc2_sf.cu_tsg_tn(v2cu)
    torch.cuda.synchronize()

res_py_cpu = res_py.cpu()
res_sf_cpu = res_sf.cpu()
res_f_cpu = res_f.cpu()
res2_py_cpu = res2_py.cpu()
res2_sf_cpu = res2_sf.cpu()
res2_f_cpu = res2_f.cpu()
torch.cuda.synchronize()

# print("result order: sigmaF, Fmunu")
# print("config 1500 field:")
# print("single fermion variable (0,1,0,0):")
# print(res2_py_cpu[0,1,0,0])
# print(res2_sf_cpu[0,1,0,0])
# print(res2_f_cpu[0,1,0,0])
# torch.cuda.synchronize()

# print("single fermion variable (0,1,1,0):")
# print(res2_py_cpu[0,1,1,0])
# print(res2_sf_cpu[0,1,1,0])
# print(res2_f_cpu[0,1,1,0])
# torch.cuda.synchronize()

print("result differences:")
print("unit gauge:", torch.sum(torch.abs(res_py_cpu-res_sf_cpu)), torch.sum(torch.abs(res_py_cpu-res_f_cpu)))
print("config 1500:", torch.sum(torch.abs(res2_py_cpu-res2_sf_cpu)), torch.sum(torch.abs(res2_py_cpu-res2_f_cpu)))

# result for only Fmunu:
# single fermion variable (0,1,0,0):
# tensor([[-9.3548e-02-0.0060j, -1.5900e-01+0.1705j, -5.5511e-17-0.2132j],
#         [ 5.5511e-17+0.0049j,  2.2157e-01+0.0910j,  1.4058e-01-0.0116j],
#         [-2.2157e-01+0.0910j,  0.0000e+00+0.0327j, -7.7165e-02+0.0612j],
#         [-1.4058e-01-0.0116j,  7.7165e-02+0.0612j,  0.0000e+00-0.0007j]],
#        dtype=torch.complex128)
# single fermion variable (0,1,1,0):
# tensor([[-5.5511e-17+0.0227j, -1.0984e-01-0.0128j, -1.5808e-01-0.1925j],
#         [ 1.0984e-01-0.0128j,  0.0000e+00+0.0981j,  1.0474e-01-0.1262j],
#         [ 1.5808e-01-0.1925j, -1.0474e-01-0.1262j, -1.1102e-16-0.1191j],
#         [-5.5511e-17-0.0509j,  1.5496e-01+0.2021j,  1.1527e-01-0.1896j]],
#        dtype=torch.complex128)
