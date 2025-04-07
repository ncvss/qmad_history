import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline

print(settings.capab)

print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f'Machine has {num_threads} threads')

n_measurements = 200
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

dw_py = qcd_ml.qcd.dirac.dirac_wilson(U, mass)
dw_ho = wilson.wilson_hop_mtsg(U, mass)

algo_name = ["no-op", "qcd_ml", "dw_hop_mtsg.templ_tmgsMhs"]
funcs = [lambda vec: vec, dw_py, dw_ho.templ_tmgsMhs]
results = {x:np.zeros(n_measurements) for x in algo_name}

def lossfunc(vec):
    return (vec * vec.conj()).real.sum()

print("loss(vec) = (vec * vec.conj()).real.sum()")

for i in range(n_warmup):
    for j in range(len(algo_name)):
        vgrad = v.clone().detach().requires_grad_(True)
        res = funcs[j](vgrad)
        loss = lossfunc(res)
        loss.backward()

for i in range(n_measurements):
    for j in range(len(algo_name)):
        vgrad = v.clone().detach().requires_grad_(True)
        res = funcs[j](vgrad)
        loss = lossfunc(res)

        start = time.perf_counter_ns()
        loss.backward()
        stop = time.perf_counter_ns()
        results[algo_name[j]][i] = stop - start

results_sorted = dict()
for x in results:
    results_sorted[x] = np.sort(results[x])[:(n_measurements//5)]

print(f"\n{'Dirac Wilson':35}: {'time in us':>15} {'std in us':>15}")

for x,y in results_sorted.items():
    print(f"{x:35}: {np.mean(y)/1000:>15.3f} {np.std(y)/1000:>15.3f}")



dwc_py = qcd_ml.qcd.dirac.dirac_wilson_clover(U, mass, csw)
dwc_ho = clover.wilson_clover_hop_mtsg_sigpre(U, mass, csw)

algo_name_c = ["no-op", "qcd_ml", "dwc_hop_mtsg_sigpre.tmngsMhs"]
funcs_c = [lambda vec: vec, dwc_py, dwc_ho.tmngsMhs]
results_c = {x:np.zeros(n_measurements) for x in algo_name_c}

for i in range(n_warmup):
    for j in range(len(algo_name_c)):
        vgrad = v.clone().detach().requires_grad_(True)
        res = funcs_c[j](vgrad)
        loss = lossfunc(res)
        loss.backward()

for i in range(n_measurements):
    for j in range(len(algo_name)):
        vgrad = v.clone().detach().requires_grad_(True)
        res = funcs_c[j](vgrad)
        loss = lossfunc(res)

        start = time.perf_counter_ns()
        loss.backward()
        stop = time.perf_counter_ns()
        results_c[algo_name_c[j]][i] = stop - start

results_sorted_c = dict()
for x in results_c:
    results_sorted_c[x] = np.sort(results_c[x])[:(n_measurements//5)]

print(f"\n{'Dirac Wilson Clover':35}: {'time in us':>15} {'std in us':>15}")

for x,y in results_sorted_c.items():
    print(f"{x:35}: {np.mean(y)/1000:>15.3f} {np.std(y)/1000:>15.3f}")


