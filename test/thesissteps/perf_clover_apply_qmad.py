# just apply the operator a few times for perf
# often enough so that the initialisation does not contribute

import torch
import socket

import gpt as g

from qmad_history import compat, clover


num_threads = torch.get_num_threads()
hostname = socket.gethostname()
print("running on host",hostname,"with",num_threads,"threads")


# split measurement into n_batch batches
# we alternate between operators and lattice dimensions
n_measurements = 1000

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions")
print("mass =", mass)
print("csw =",csw)

rng = g.random("thy")

grid0 = [32,32,32,64]
vol = 2**21

print("grid layout:",grid0)

U_gpt = g.qcd.gauge.random(g.grid(grid0, g.double), rng)
grid = U_gpt[0].grid
v_gpt = rng.cnormal(g.vspincolor(grid))

U_mtsg = torch.tensor(compat.lattice_to_array(U_gpt))
v_mtsg = torch.tensor(compat.lattice_to_array(v_gpt))

dwc_qmad = clover.wilson_clover_hop_mtsg_sigpre(U_mtsg,mass,csw)

for _ in range(n_measurements):
    res = dwc_qmad.tmnsgMhs(v_mtsg)

print("qmad perf complete")
