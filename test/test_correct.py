import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore
import qcd_ml

from qmad_history import compat, wilson, clover

print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f'Machine has {num_threads} threads')

lat_dim = [8,8,8,16]
print("lattice_dimensions =",lat_dim)

rng = g.random("alltests")
U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_g[0].grid
v_g = rng.cnormal(g.vspincolor(grid))
dst_g = g.lattice(v_g)

mass = -0.5
print("mass_parameter =",mass)
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0
print("csw =", csw)

U = torch.tensor(compat.lattice_to_array(U_g))
v = torch.tensor(compat.lattice_to_array(v_g))
vn = torch.transpose(v, 4, 5).contiguous()


dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )
dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

dw_py = qcd_ml.qcd.dirac.dirac_wilson(U, mass)
dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)


dw_d = wilson.wilson_direct(U, mass)
dw_eo = wilson.wilson_eo(U, mass)
dw_ho = wilson.wilson_hop_mtsg(U, mass)
dw_hn = wilson.wilson_hop_tmgs(U, mass)

ve = v[dw_eo.emask]
vo = v[dw_eo.omask]

dwv_py = dw_py(v)

check_correct = []

for dw in [dw_d, dw_ho]:
    check_correct.append(str(dw))
    for order, c in dw.all_calls().items():
        check_correct.append((order,torch.allclose(c(v),dwv_py)))

check_correct.append(str(dw_hn))
for order, c in dw_hn.all_calls().items():
    check_correct.append((order,torch.allclose(c(vn).transpose(4,5),dwv_py)))

check_correct.append(str(dw_eo))
for order, c in dw_eo.all_calls().items():
    dwv_eo = c(ve, vo)
    dwv_eo_back = torch.zeros_like(dwv_py)
    dwv_eo_back[dw_eo.emask] = dwv_eo[0]
    dwv_eo_back[dw_eo.omask] = dwv_eo[1]
    check_correct.append((order,torch.allclose(dwv_py,dwv_eo_back)))


for cc in check_correct:
    print(cc)


