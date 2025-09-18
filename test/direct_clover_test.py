import torch
import socket
import numpy as np
import time

import gpt as g
import qcd_ml

from qmad_history import compat, clover, settings

print(settings.capab())

print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f'Machine has {num_threads} threads')

lat_dim = [8,8,8,16]
print("lattice_dimensions =",lat_dim)

rng = g.random("qmad")
U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_g[0].grid
v_g = rng.cnormal(g.vspincolor(grid))

mass = -0.5
print("mass_parameter =",mass)
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0
print("csw =", csw)

U = torch.tensor(compat.lattice_to_array(U_g))
# v = torch.tensor(compat.lattice_to_array(v_g))

v = torch.zeros(lat_dim+[4,3], dtype=torch.cdouble)
v[0,2,3,4,0,1] = 1

dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)
dwc_hd = clover.wilson_clover_hop(U, mass, csw)

dwvref = dwc_py(v)

print("sites [0,2,3,4] and [0,3,3,4]:")
print(dwvref[0,2,3,4])
print(dwvref[0,3,3,4])

for c in dwc_hd.all_calls():
    cv = c(v)
    print(torch.allclose(dwvref, cv))
    print(cv[0,2,3,4])
    print(cv[0,3,3,4])
