import torch
from qmad_history import wilson

lat = [8,8,8,8]

U = torch.randn([4]+lat+[3,3], dtype=torch.cdouble)
v = torch.randn(lat+[4,3], dtype=torch.cdouble)
U = torch.transpose(U, 5, 6)

dw = wilson.wilson_hop_mtsg(U, -0.5)
res = dw.tmgsMh(v)
