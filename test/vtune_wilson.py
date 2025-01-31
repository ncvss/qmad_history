# import sys

# print(sys.executable)

import torch
import qmad_history.wilson
import time

#start = time.perf_counter()
lat_dim = [8,8,8,16]
U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)
mass = -0.5
dw = qmad_history.wilson.wilson_hop_mtsg(U, mass)
#stop = time.perf_counter()
#print("init time:",stop-start,"\nnow sleep")

time.sleep(7)

for i in range(1):
    dwv = dw.templ_tmgsMhs(v)

