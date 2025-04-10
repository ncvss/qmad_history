import torch
import socket
import numpy as np
import time

import gpt as g # type: ignore

from qmad_history import compat, settings, wilson_roofline


# change the amount of data used in the dirac wilson computation
# the input U and v have a different grid than what the dirac wilson expects
# the computation is exactly the same,
# but the grid from which the data is taken is smaller or larger

rng = g.random("alltests")

print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f"Machine has {num_threads} threads")

print(settings.capab)

# make measurements in batches, so that we need less initialisations
n_measurements = 300
n_batch = 20
assert n_measurements % n_batch == 0
n_warmup = 10
print("n_measurements =",n_measurements)

mass = -0.5
print("mass_parameter =",mass)
kappa = 1.0/2.0/(mass + 4.0)

base_lat_dim = [16,16,16,16]
print("base_lattice_dimensions =",base_lat_dim)


#shrinks = list(range(-2,3))
test_lat_dims = [[16,16,16,8],[16,16,16,12],[16,16,16,16],[16,16,16,24],[16,16,16,32],[16,16,16,48]]
#test_lat_dims = [[8,8,8,16],[8,8,8,8]]
test_vols = [ld[0]*ld[1]*ld[2]*ld[3] for ld in test_lat_dims]
lat_dict = {test_vols[i]:test_lat_dims[i] for i in range(len(test_lat_dims))}
print("input_lattice_dimensions =", test_lat_dims)
print("input_vols =", test_vols)

results = {vo:np.zeros(n_measurements) for vo in test_vols}
datasizes = dict()

for i in range(0,n_measurements,n_batch):
    print(i, end=" ", flush=True)
    for vol, lat_dim in lat_dict.items():

        
        U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_g[0].grid
        v_g = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_g)

        U = torch.tensor(compat.lattice_to_array(U_g))
        v = torch.tensor(compat.lattice_to_array(v_g))

        dw = wilson_roofline.wilson_hop_mtsg_roofline(U, mass, base_lat_dim)
        
        for j in range(n_warmup):
            vres = dw.templ_tmgsMhs(v)

        for j in range(i,i+n_batch):
            start = time.perf_counter_ns()
            vres = dw.templ_tmgsMhs(v)
            stop = time.perf_counter_ns()
            results[vol][j] = stop - start
        
        # print(lat_dim)
        # print(dw.hop_inds[0])
        # print(dw.hop_inds[1])

        if i == 0:
            datasizes[vol] = (v.element_size() * v.nelement() + U.element_size() * U.nelement()
                            + dw.hop_inds.element_size() * dw.hop_inds.nelement()
                            + vres.element_size() * vres.nelement())
            # print(dw.hop_inds[0])
            # print(dw.hop_inds[8])
            # print(dw.hop_inds[17])
            # print(dw.hop_inds[8*16+1])

results_sorted = dict()
for vo in results:
    results_sorted[vo] = np.sort(results[vo])[:(n_measurements//5)]


print(f"\n {'input size in B':>15}  {'time in us':>15}  {'std in us':>15}  {'grid dimensions'}  ")

for x in sorted(test_vols):
    y = results_sorted[x]
    print(f"[{datasizes[x]:>15}, {np.mean(y)/1000:>15.3f}, {np.std(y)/1000:>15.3f}, {lat_dict[x]}],")

