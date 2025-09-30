import torch
import qcd_ml
from qmad_history import clover

lat_dim = [16,16,16,16]
mass = -0.5
csw = 1.0

print("test if wilson clover works on gpu")
print("lattice:", lat_dim)
print("mass:",mass)

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)
dwc2 = clover.wilson_clover_hop_mtsg(U, mass, csw)

dwcv = dwc(v)
dwcv2 = dwc2.debug_cuda(v)

print(torch.allclose(dwcv,dwcv2))
# result is the same, even though these are not actual gauge fields
