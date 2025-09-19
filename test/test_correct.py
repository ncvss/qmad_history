import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline

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
dw_heo = wilson.wilson_hop_eo(U, mass)
dw_ho = wilson.wilson_hop_mtsg(U, mass)
dw_hn = wilson.wilson_hop_tmgs(U, mass)

dw_roof = wilson_roofline.wilson_hop_mtsg_roofline(U, mass, lat_dim)

dw_grid = wilson.wilson_hop_mtsgt(U, mass)
dw_grid2 = wilson.wilson_hop_mtsgt2(U, mass)

ve = v[dw_eo.emask]
vo = v[dw_eo.omask]

vge = v[:,:,:,0:lat_dim[3]:2]
vgo = v[:,:,:,1:lat_dim[3]:2]
v_grid = torch.stack([vge, vgo], dim=-1)
vgfirst = v[:,:,:,0:lat_dim[3]//2]
vgsecond = v[:,:,:,lat_dim[3]//2:lat_dim[3]]
v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)

v_grid_back = torch.zeros_like(v)
v_grid_back[:,:,:,0:lat_dim[3]:2] = v_grid[:,:,:,:,:,:,0]
v_grid_back[:,:,:,1:lat_dim[3]:2] = v_grid[:,:,:,:,:,:,1]
v_grid2_back = torch.cat([v_grid2[:,:,:,:,:,:,0],v_grid2[:,:,:,:,:,:,1]], dim=3)
print("conversion to grid layout 1 worked: ", torch.allclose(v,v_grid_back))
print("conversion to grid layout 2 worked: ", torch.allclose(v,v_grid2_back))


dwv_py = dw_py(v)

check_correct = ["\nwilson"]

for dw in [dw_d, dw_ho]:
    check_correct.append(str(dw))
    for order, c in zip(dw.all_call_names(),dw.all_calls()):
        check_correct.append((order,torch.allclose(c(v),dwv_py)))

check_correct.append(str(dw_hn))
for order, c in zip(dw_hn.all_call_names(),dw_hn.all_calls()):
    check_correct.append((order,torch.allclose(c(vn).transpose(4,5),dwv_py)))

for dw in [dw_eo, dw_heo]:
    check_correct.append(str(dw))
    for order, c in zip(dw.all_call_names(),dw.all_calls()):
        dwv_eo = c([ve,vo])
        dwv_eo_back = torch.zeros_like(dwv_py)
        dwv_eo_back[dw_eo.emask] = dwv_eo[0]
        dwv_eo_back[dw_eo.omask] = dwv_eo[1]
        check_correct.append((order,torch.allclose(dwv_py,dwv_eo_back)))

check_correct.append(str(dw_roof))
for order, c in zip(dw_roof.all_call_names(),dw_roof.all_calls()):
    dwv_roof = c(v).reshape(lat_dim+[4,3])
    check_correct.append((order, torch.allclose(dwv_roof,dwv_py)))

check_correct.append(str(dw_grid))
for order, c in zip(dw_grid.all_call_names(),dw_grid.all_calls()):
    dwv_grid = c(v_grid)
    dwv_grid_back = torch.zeros_like(dwv_py)
    dwv_grid_back[:,:,:,0:lat_dim[3]:2] = dwv_grid[:,:,:,:,:,:,0]
    dwv_grid_back[:,:,:,1:lat_dim[3]:2] = dwv_grid[:,:,:,:,:,:,1]
    check_correct.append((order,torch.allclose(dwv_py,dwv_grid_back)))

check_correct.append(str(dw_grid2))
for order, c in zip(dw_grid2.all_call_names(),dw_grid2.all_calls()):
    dwv_grid2 = c(v_grid2)
    dwv_grid2_back = torch.cat([dwv_grid2[:,:,:,:,:,:,0],dwv_grid2[:,:,:,:,:,:,1]], dim=3)
    check_correct.append((order,torch.allclose(dwv_py,dwv_grid2_back)))


check_correct.append("\nwilson clover")

dwcv_py = dwc_py(v)

dwc_d = clover.wilson_clover_direct_false(U, mass, csw)
dwc_hd = clover.wilson_clover_hop(U, mass, csw)
dwc_f = clover.wilson_clover_fpre(U, mass, csw)
dwc_ho = clover.wilson_clover_hop_mtsg(U, mass, csw)
dwc_hn = clover.wilson_clover_hop_tmgs(U, mass, csw)
dwc_s = clover.wilson_clover_sigpre(U, mass, csw)

dwc_grid = clover.wilson_clover_hop_mtsgt_sigpre(U, mass, csw)

dwc_gridcl = clover.wilson_clover_hop_mtsg_sigpre(U, mass, csw)

dwc_grid2 = clover.wilson_clover_hop_mtsgt2_sigpre(U, mass, csw)

for dw in [dwc_d, dwc_hd, dwc_f, dwc_ho, dwc_gridcl]:
    check_correct.append(str(dw))
    for order, c in zip(dw.all_call_names(),dw.all_calls()):
        check_correct.append((order,torch.allclose(c(v),dwcv_py)))

check_correct.append(str(dwc_hn))
for order, c in zip(dwc_hn.all_call_names(),dwc_hn.all_calls()):
    check_correct.append((order,torch.allclose(c(vn).transpose(4,5),dwcv_py)))

check_correct.append(str(dwc_grid))
for order, c in zip(dwc_grid.all_call_names(),dwc_grid.all_calls()):
    dwcv_grid = c(v_grid)
    dwcv_grid_back = torch.zeros_like(dwcv_py)
    dwcv_grid_back[:,:,:,0:lat_dim[3]:2] = dwcv_grid[:,:,:,:,:,:,0]
    dwcv_grid_back[:,:,:,1:lat_dim[3]:2] = dwcv_grid[:,:,:,:,:,:,1]
    check_correct.append((order,torch.allclose(dwcv_py,dwcv_grid_back)))

check_correct.append(str(dwc_grid2))
for order, c in zip(dwc_grid2.all_call_names(),dwc_grid2.all_calls()):
    dwcv_grid2 = c(v_grid2)
    dwcv_grid2_back = torch.cat([dwcv_grid2[:,:,:,:,:,:,0],dwcv_grid2[:,:,:,:,:,:,1]], dim=3)
    check_correct.append((order,torch.allclose(dwcv_py,dwcv_grid2_back)))

# print("my clover:")
# print(dwcv_grid_back[0,1,0,2]-dwv_grid_back[0,1,0,2])
# print(dwcv_grid_back[0,1,0,3]-dwv_grid_back[0,1,0,3])
# print("qcd_ml:")
# print(dwcv_py[0,1,0,2]-dwv_py[0,1,0,2])
# print(dwcv_py[0,1,0,3]-dwv_py[0,1,0,3])
# aktuell: nur der erste Eintrag des 6-Blocks ist korrekt
# Grund: die Indizes waren off by 1, jetzt stimmt es

for cc in check_correct:
    if type(cc) is tuple:
        print(f"{cc[0]:20} {str(cc[1]):6}")
    else:
        print("---")
        print(cc)


# the following only works with avx
if settings.capab("vectorise"):

    # test correctness of boundary conditions
    print()
    boundconds = [[1,1,1,-1],[-1,1,-1,1]]
    for bound in boundconds:
        print("check with boundary conditions", bound)

        dw_g_b = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                                    "isAnisotropic":False,"boundary_phases":bound,}, )
        resgb = dw_g_b(v_g)
        resgb_torch = torch.tensor(compat.lattice_to_array(resgb))

        dw_h_b = wilson.wilson_hop_mtsg(U, mass, bound)
        resbqm = dw_h_b.templbound_tmsgMhs(v)
        resUb = dw_h_b.templUbound_tmsgMhs(v)

        print(str(dw_h_b)+".templbound_tmsgMhs:", torch.allclose(resgb_torch,resbqm))
        print(str(dw_h_b)+".templUbound_tmsgMhs:", torch.allclose(resgb_torch,resUb))

        dwc_g_b = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                    "isAnisotropic":False,"boundary_phases":bound,}, )
        resgbc = dwc_g_b(v_g)
        resgbc_torch = torch.tensor(compat.lattice_to_array(resgbc))

        dwc_b = clover.wilson_clover_hop_mtsg_sigpre(U, mass, csw, bound)
        rescUb = dwc_b.Ubound_tmngsMhs(v)
        print(str(dwc_b)+".Ubound_tmngsMhs:", torch.allclose(rescUb,resgbc_torch))



    # test for correctness of gradients
    print("\ngradient check:")

    vgr = torch.randn(lat_dim+[4,3], dtype=torch.cdouble, requires_grad=True)
    vgr2 = vgr.clone().detach().requires_grad_(True)

    dwgr_qml = qcd_ml.qcd.dirac.dirac_wilson(U, mass)
    res_qml = dwgr_qml(vgr)
    loss_qml = (res_qml * res_qml.conj()).real.sum()
    loss_qml.backward()

    dwgr_qmad = wilson.wilson_hop_mtsg(U, mass)
    res_qmad = dwgr_qmad.templ_tmgsMhs(vgr2)
    loss_qmad = (res_qmad * res_qmad.conj()).real.sum()
    loss_qmad.backward()

    print(str(dwgr_qmad))
    print("templ_tmgsMhs:", torch.allclose(vgr.grad, vgr2.grad))


    vcgr = torch.randn(lat_dim+[4,3], dtype=torch.cdouble, requires_grad=True)
    vcgr2 = vcgr.clone().detach().requires_grad_(True)

    dwcgr_qml = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)
    res_qml = dwcgr_qml(vcgr)
    loss_qml = (res_qml * res_qml.conj()).real.sum()
    loss_qml.backward()

    dwcgr_qmad = clover.wilson_clover_hop_mtsg_sigpre(U, mass, csw)
    res_qmad = dwcgr_qmad.tmngsMhs(vcgr2)
    loss_qmad = (res_qmad * res_qmad.conj()).real.sum()
    loss_qmad.backward()

    print(str(dwcgr_qmad))
    print("tmngsMhs:", torch.allclose(vcgr.grad, vcgr2.grad))

