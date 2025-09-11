# use qcd_ml on the gpu

import torch
import qcd_ml
from qmad_history import clover
import time
import numpy as np

cuda0 = torch.device("cuda")
print("using device",cuda0)
mass = -0.5
csw = 1.0

print("test if wilson clover works on gpu")
print("mass:",mass)

U = torch.tensor(np.load("./test/1500.config.npy"), dtype=torch.cdouble)
lat_dim_1 = list(U.shape[1:5])
lat_dim_2 = [16,16,16,16]
print("lattice:", lat_dim_1, "and", lat_dim_2)

v = torch.randn(lat_dim_1+[4,3], dtype=torch.cdouble)

U2 = torch.empty([4]+lat_dim_2+[3,3], dtype=torch.cdouble)
U2[:,:,:,:,:] = torch.eye(3, dtype=torch.cdouble)
v2 = torch.zeros(lat_dim_2+[4,3], dtype=torch.cdouble)
v2[0,1,0,0,1,2] = 1
v2[0,1,0,0,3,0] = 1
v2[0,1,0,0,0,1] = 1

Ucu = U.to(cuda0)
vcu = v.to(cuda0)
U2cu = U2.to(cuda0)
v2cu = v2.to(cuda0)

# torch.cuda.synchronize()
# dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(Ucu, mass, csw)
# torch.cuda.synchronize()

# dwc_sf = clover.wilson_clover_hop_mtsg_sigpre(Ucu, mass, csw)
# torch.cuda.synchronize()

# dwc_f = clover.wilson_clover_hop_mtsg(Ucu, mass, csw)
# torch.cuda.synchronize()


dwc2 = qcd_ml.qcd.dirac.dirac_wilson_clover(U2cu, mass, csw)
torch.cuda.synchronize()

dwc2_sf = clover.wilson_clover_hop_mtsg_sigpre(U2cu, mass, csw)
torch.cuda.synchronize()

dwc2_f = clover.wilson_clover_hop_mtsg(U2cu, mass, csw)
torch.cuda.synchronize()

# res_py = dwc(vcu)
# torch.cuda.synchronize()
# res_sf = dwc_sf.cu_tsg_tn(vcu)
# torch.cuda.synchronize()
# res_f = dwc_f.cu_tsg(vcu)
# torch.cuda.synchronize()

res2_py = dwc2(v2cu)
torch.cuda.synchronize()
res2_f = dwc2_f.cu_tsg(v2cu)
torch.cuda.synchronize()
res2_sf = dwc2_sf.cu_tsg_tn(v2cu)
torch.cuda.synchronize()

# res_py_cpu = res_py.cpu()
# res_sf_cpu = res_sf.cpu()
# res_f_cpu = res_f.cpu()
res2_py_cpu = res2_py.cpu()
res2_sf_cpu = res2_sf.cpu()
res2_f_cpu = res2_f.cpu()
torch.cuda.synchronize()

print("result order: Fmunu, sigmaF")
print("unit gauge field:")
print("single fermion variable (0,1,0,0):")
print(res2_py_cpu[0,1,0,0])
print(res2_sf_cpu[0,1,0,0])
print(res2_f_cpu[0,1,0,0])
torch.cuda.synchronize()

print("single fermion variable (0,1,1,0):")
print(res2_py_cpu[0,1,1,0])
print(res2_sf_cpu[0,1,1,0])
print(res2_f_cpu[0,1,1,0])
torch.cuda.synchronize()

print("result differences:")
# print("config 1500:", torch.sum(torch.abs(res_py_cpu-res_sf_cpu)), torch.sum(torch.abs(res_py_cpu-res_f_cpu)))
print("unit gauge:", torch.sum(torch.abs(res2_py_cpu-res2_sf_cpu)), torch.sum(torch.abs(res2_py_cpu-res2_f_cpu)))

