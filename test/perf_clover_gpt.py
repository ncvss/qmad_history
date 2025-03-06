import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore

from qmad_history import compat, wilson, clover

n_measurements = 10000

lat_dim = [16,16,16,32]

print("repetitions=", n_measurements)
print("grid=",lat_dim)
print("number of threads=",torch.get_num_threads())

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

dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

for _ in range(n_measurements):
    dst_g = dwc_g(v_g)
