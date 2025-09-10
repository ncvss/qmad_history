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
U2 = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)

Ucu = U.to(cuda0)
vcu = v.to(cuda0)
U2cu = U2.to(cuda0)

dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(Ucu, mass, csw)
time.sleep(5)

dwc2 = qcd_ml.qcd.dirac.dirac_wilson_clover(U2cu, mass, csw)
time.sleep(6)

dwc_sf = clover.wilson_clover_hop_mtsg_sigpre(Ucu, mass, csw)
time.sleep(7)

res = dwc(vcu)
res2 = dwc2(vcu)
res3 = dwc_sf.cu_tsg_tn(vcu)

print("result difference:", torch.sum(torch.abs(res-res2)), torch.sum(torch.abs(res-res3)))

