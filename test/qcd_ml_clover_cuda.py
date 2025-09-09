# use qcd_ml on the gpu

import torch
import qcd_ml

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

dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(Ucu, mass, csw)

res = dwc(vcu)

