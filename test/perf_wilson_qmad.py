import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore

from qmad_history import compat, wilson, clover

n_measurements = 5000

lat_dim = [8,8,8,16]

rng = g.random("run")
U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_g[0].grid
v_g = rng.cnormal(g.vspincolor(grid))
dst_g = g.lattice(v_g)

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

U = torch.tensor(compat.lattice_to_array(U_g))
v = torch.tensor(compat.lattice_to_array(v_g))

dw_ho = wilson.wilson_hop_mtsg(U, mass)

for _ in range(n_measurements):
    vres = dw_ho.templ_tmgsMhs(v)

