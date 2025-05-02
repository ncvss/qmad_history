import numpy as np
import torch
from qcd_ml.qcd.dirac import dirac_wilson_clover, dirac_wilson
from qcd_ml.util.solver import GMRES
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
import gpt as g
from qcd_ml.compat.gpt import lattice2ndarray, ndarray2lattice

import qmad_history.clover
from qmad_history.compat import lattice_to_array
import qmad_history
import time

lat_dim = [16,8,8,16]
grid = g.grid(lat_dim, g.double)
psi = g.vspincolor(grid)
rng = g.random("sermg")
rng.cnormal(psi)

U = g.qcd.gauge.random(grid, rng)
Ut = torch.tensor(lattice_to_array(U))
psi_torch = torch.tensor(lattice_to_array(psi))

def torch_innerprod(x,y):
    return (x.conj() * y).sum()

reps = 10000

start = time.perf_counter_ns()
for i in range(reps):
    val = np.abs(g.inner_product(psi,psi)) ** 0.5
stop = time.perf_counter_ns()
print(f"gpt innerprod time:   {stop - start:15}")

start = time.perf_counter_ns()
for i in range(reps):
    val = np.abs(torch_innerprod(psi_torch,psi_torch)) ** 0.5
stop = time.perf_counter_ns()
print(f"torch innerprod time: {stop-start:15}")

# Ergebnis: Beide Methoden sind annähernd gleich schnell


start = time.perf_counter_ns()
for i in range(reps):
    div = psi/20.29
stop = time.perf_counter_ns()
print(f"gpt divide time:   {stop-start:15}")

start = time.perf_counter_ns()
for i in range(reps):
    div = psi_torch/20.29
stop = time.perf_counter_ns()
print(f"torch divide time: {stop-start:15}")

# Ergebnis: torch braucht 100 mal so lange
