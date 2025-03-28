# I suppose the memory problems come from class initialisation
# I test that hypothesis here
# initialisation times in seconds: 4, 13, 13, 13, 16, 16

import torch
import socket
import numpy as np
import time

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

rng = g.random("voltest")

lat_dim = [32,16,16,32]
vol = 32**2 * 16**2

operator_names = ["gpt", "qcd_ml", "qmad_dir", "qmad_vect", "qmad_vect_templ", "qmad_grid", "qmad_grid_clover"]

# initialise the fields for this volume
U_gpt = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_gpt[0].grid
v_gpt = rng.cnormal(g.vspincolor(grid))
dst_g = g.lattice(v_gpt)

U_qml = torch.tensor(compat.lattice_to_array(U_gpt))
v_qml = torch.tensor(compat.lattice_to_array(v_gpt))

vgfirst = v_qml[:,:,:,0:lat_dim[3]//2]
vgsecond = v_qml[:,:,:,lat_dim[3]//2:lat_dim[3]]
v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)

# initialise the operators
start = time.perf_counter()
print("start init", flush=True)

dwc_gpt = g.qcd.fermion.wilson_clover(U_gpt, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )
gpttime = time.perf_counter()
print("gpt:", gpttime-start, flush=True)

dwc_qcd_ml = qcd_ml.qcd.dirac.dirac_wilson_clover(U_qml, mass, csw)
qmltime = time.perf_counter()
print("qcd_ml:", qmltime-gpttime, flush=True)

dwc_qmad_dir = clover.wilson_clover_fpre(U_qml, mass, csw)
x = time.perf_counter()
print("qmad direct:", x-qmltime, flush=True)

dwc_qmad_vect = clover.wilson_clover_hop_mtsg(U_qml, mass, csw)
# this is also the templ operator
y = time.perf_counter()
print("qmad vectorised:", y-x, flush=True)

dwc_qmad_grid = clover.wilson_clover_hop_mtsgt2_sigpre(U_qml, mass, csw)
zz = time.perf_counter()
print("qmad with grid layout:", zz-y, flush=True)

dwc_qmad_grid_clover = clover.wilson_clover_hop_mtsg_sigpre(U_qml, mass, csw)
ttt = time.perf_counter()
print("qmad with grid clover computation:", ttt-zz, flush=True)

op_inputs = [v_gpt, v_qml, v_qml, v_qml, v_qml, v_grid2, v_qml]
op_calls = [dwc_gpt, dwc_qcd_ml, dwc_qmad_dir.xtmnghs, dwc_qmad_vect.avx_tmgsMhns,
            dwc_qmad_vect.templ_tmgsMhns, dwc_qmad_grid.tmngsMht, dwc_qmad_grid_clover.tmngsMhs]

print("init complete", flush=True)

for op_id in range(len(operator_names)):
    any_result = op_calls[op_id](op_inputs[op_id])

base_data = 2*v_qml.element_size()*v_qml.nelement() + U_qml.element_size()*U_qml.nelement()
fstr_qmad = dwc_qmad_dir.field_strength
fstr_qmad_data = len(fstr_qmad)*fstr_qmad[0].element_size()*fstr_qmad[0].nelement()
fstr_grid = dwc_qmad_grid.field_strength_sigma
fstr_grid_data = fstr_grid.element_size()*fstr_grid.nelement()
hops_qmad = dwc_qmad_vect.hop_inds
hops_qmad_data = hops_qmad.element_size()*hops_qmad.nelement()
hops_grid = dwc_qmad_grid.hop_inds
hops_grid_data = hops_grid.element_size()*hops_grid.nelement()

print("data in B:")
print("gpt", base_data + fstr_grid_data)
print("qcd_ml", base_data)
print("qmad_dir", base_data + fstr_qmad_data)
print("qmad_vect", base_data + fstr_qmad_data + hops_qmad_data)
print("qmad_vect_templ", base_data + fstr_qmad_data + hops_qmad_data)
print("qmad_grid", base_data + fstr_grid_data + hops_grid_data)
print("qmad_grid_clover", base_data + fstr_grid_data + hops_grid_data)
