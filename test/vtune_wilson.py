# import sys

# print(sys.executable)

import torch
import qmad_history.wilson

lat_dim = [8,8,8,16]
U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)
mass = -0.5
dw = qmad_history.wilson.wilson_hop_mtsg(U, mass)

for i in range(100):
    dwv = dw.templ_tmgsMhs(v)

