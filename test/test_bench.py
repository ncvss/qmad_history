import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline

print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f'Machine has {num_threads} threads')

print(settings.capab)

n_measurements = 500
n_warmup = 20
print("n_measurements =",n_measurements)

lat_dim = [16,16,16,32]
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

#dw_py = qcd_ml.qcd.dirac.dirac_wilson(U, mass)


dw_d = wilson.wilson_direct(U, mass)
dw_eo = wilson.wilson_eo(U, mass)
dw_ho = wilson.wilson_hop_mtsg(U, mass)
dw_hn = wilson.wilson_hop_tmgs(U, mass)

dw_roof = wilson_roofline.wilson_hop_mtsg_roofline(U, mass, lat_dim)

dw_grid = wilson.wilson_hop_mtsgt(U, mass)
dw_grid2 = wilson.wilson_hop_mtsgt2(U, mass)

ve = v[dw_eo.emask]
vo = v[dw_eo.omask]
veo = [ve, vo]

vge = v[:,:,:,0:lat_dim[3]:2]
vgo = v[:,:,:,1:lat_dim[3]:2]
v_grid = torch.stack([vge, vgo], dim=-1)
vgfirst = v[:,:,:,0:lat_dim[3]//2]
vgsecond = v[:,:,:,lat_dim[3]//2:lat_dim[3]]
v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)

algo_name = ["gpt"]
funcs = [dw_g]
opsetup_name = ["gpt"]

vs = {"gpt": v_g, str(dw_d): v, str(dw_ho): v, str(dw_hn): vn,
      str(dw_eo): veo, str(dw_roof): v, str(dw_grid): v_grid, str(dw_grid2): v_grid2}

for dw in [dw_d, dw_eo, dw_ho, dw_hn, dw_roof, dw_grid, dw_grid2]:
    algo_name += [str(dw)+"."+x for x in dw.all_call_names()]
    opsetup_name += [str(dw)] * len(dw.all_call_names())
    funcs += dw.all_calls()

results = {x:np.zeros(n_measurements) for x in algo_name}

for i in range(n_warmup):
    for j in range(len(algo_name)):
        vres = funcs[j](vs[opsetup_name[j]])

for i in range(n_measurements):
    for j in range(len(algo_name)):
        start = time.perf_counter_ns()
        vres = funcs[j](vs[opsetup_name[j]])
        stop = time.perf_counter_ns()
        results[algo_name[j]][i] = stop - start

results_sorted = dict()
for x in results:
    results_sorted[x] = np.sort(results[x])[:(n_measurements//5)]

print(f"\n{"Dirac Wilson":35}: {"time in us":>15} {"std in us":>15}")

for x,y in results_sorted.items():
    print(f"{x:35}: {np.mean(y)/1000:>15.3f} {np.std(y)/1000:>15.3f}")
        

dwc_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

dwc_d = clover.wilson_clover_direct_false(U, mass, csw)
dwc_f = clover.wilson_clover_fpre(U, mass, csw)
dwc_ho = clover.wilson_clover_hop_mtsg(U, mass, csw)
dwc_hn = clover.wilson_clover_hop_tmgs(U, mass, csw)
dwc_s = clover.wilson_clover_sigpre(U, mass, csw)

dwc_grid = clover.wilson_clover_hop_mtsgt_sigpre(U, mass, csw)

dwc_gridcl = clover.wilson_clover_hop_mtsg_sigpre(U, mass, csw)

dwc_grid2 = clover.wilson_clover_hop_mtsgt2_sigpre(U, mass, csw)

algo_name_c = ["gpt"]
funcsc = [dwc_g]
opsetup_name_c = ["gpt"]

vsc = {"gpt": v_g, str(dwc_d): v, str(dwc_ho): v, str(dwc_hn): vn,
       str(dwc_s): v, str(dwc_f): v, str(dwc_grid): v_grid, str(dwc_gridcl): v,  str(dwc_grid2): v_grid2}

for dw in [dwc_d, dwc_f, dwc_ho, dwc_hn, dwc_s, dwc_grid, dwc_gridcl, dwc_grid2]:
    algo_name_c += [str(dw)+"."+x for x in dw.all_call_names()]
    opsetup_name_c += [str(dw)] * len(dw.all_call_names())
    funcsc += dw.all_calls()

# print(len(algo_name_c))
# print(len(funcsc))
# for ff in funcsc:
#     print(ff)
resultsc = {x:np.zeros(n_measurements) for x in algo_name_c}

for i in range(n_warmup):
    for j in range(len(algo_name_c)):
        vresc = funcsc[j](vsc[opsetup_name_c[j]])

for i in range(n_measurements):
    for j in range(len(algo_name_c)):
        start = time.perf_counter_ns()
        vresc = funcsc[j](vsc[opsetup_name_c[j]])
        stop = time.perf_counter_ns()
        resultsc[algo_name_c[j]][i] = stop - start

resultsc_sorted = dict()
for x in resultsc:
    resultsc_sorted[x] = np.sort(resultsc[x])[:(n_measurements//5)]

print(f"\n{'Dirac Wilson Clover':35}: {'time in us':>15} {'std in us':>15}")

for x,y in resultsc_sorted.items():
    print(f"{x:35}: {np.mean(y)/1000:>15.3f} {np.std(y)/1000:>15.3f}")

