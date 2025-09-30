# apply the clover from GPT a number of times
# then measure the statistics with perf
# large number of repetitions to reduce impact of initialisation

import torch
import socket

import gpt as g


num_threads = torch.get_num_threads()
hostname = socket.gethostname()
print("running on host",hostname,"with",num_threads,"threads")


n_measurements = 1000

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print(n_measurements,"repetitions")
print("mass =", mass)
print("csw =",csw)

rng = g.random("thee")

grid0 = [32,32,32,64]
vol = grid0[0]*grid0[1]*grid0[2]*grid0[3]

print("grid layout:",grid0)

U_gpt = g.qcd.gauge.random(g.grid(grid0, g.double), rng)
grid = U_gpt[0].grid
v_gpt = rng.cnormal(g.vspincolor(grid))

dwc_g = g.qcd.fermion.wilson_clover(U_gpt, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                  "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

for _ in range(n_measurements):
    res_g = dwc_g(v_gpt)

print("gpt perf complete")
