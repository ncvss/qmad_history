# use qcd_ml on the gpu

import torch
import qcd_ml
from qmad_history import clover
import time

lat_dim = [16,16,16,16]
cuda0 = torch.device("cuda:0")
mass = -0.5
csw = 1.0

print("test if wilson clover works on gpu")
print("lattice:", lat_dim)
print("mass:",mass)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)

torch.cuda.synchronize()
dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(Ucu, mass, csw)
torch.cuda.synchronize()
#time.sleep(5.1)

torch.cuda.synchronize()
dwc_sf = clover.wilson_clover_hop_mtsg_sigpre(Ucu, mass, csw)
torch.cuda.synchronize()
#time.sleep(5.3)

dwc_f = clover.wilson_clover_hop_mtsg(Ucu, mass, csw)

torch.cuda.synchronize()
res = dwc(vcu)
torch.cuda.synchronize()
#time.sleep(5.4)
torch.cuda.synchronize()
res3 = dwc_sf.cu_tsg_tn(vcu)
torch.cuda.synchronize()
#time.sleep(5.6)

res4 = dwc_f.cu_tsg(vcu)

torch.cuda.synchronize()
res1cpu = res.cpu()
res3cpu = res3.cpu()
res4cpu = res4.cpu()
torch.cuda.synchronize()
print("single fermion variable (0,1,2,3):")
print(res1cpu[0,1,2,3])
print(res3cpu[0,1,2,3])
print(res4cpu[0,1,2,3])
torch.cuda.synchronize()
#time.sleep(5.7)
torch.cuda.synchronize()
print("result differences:", torch.sum(torch.abs(res1cpu-res3cpu)), torch.sum(torch.abs(res1cpu-res4cpu)))

