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


# create random gauge config
lat_dim = [8,8,8,16]
grid = g.grid(lat_dim, g.double)
psi = g.vspincolor(grid)
rng = g.random("sermg")
rng.cnormal(psi)

U = g.qcd.gauge.random(grid, rng)
Ut = torch.tensor(lattice_to_array(U))
psi_torch = torch.tensor(lattice_to_array(psi))

masses = [0.55, 0.4, 0.3, 0.2, 0.1, -0.1, -0.25]

print("lattice dimensions:", lat_dim)
maxit = 1200
innit = 100
preci = 1e-4
print("maximum iterations:", maxit)
print("inner iterations:", innit)
print("precision for residuum:", preci)
ver = False

print("times in ms:")

for en,mass in enumerate(masses):
    w_gpt = g.qcd.fermion.wilson_clover(U, {"mass": mass,
        "csw_r": 1.0,
        "csw_t": 1.0,
        "xi_0": 1.0,
        "nu": 1.0,
        "isAnisotropic": False,
        "boundary_phases": [1,1,1,1]})

    w_torch = dirac_wilson_clover(Ut, mass, 1.0)
    #w = lambda x: torch.tensor(lattice2ndarray(w_gpt(ndarray2lattice(x.numpy(), U[0].grid, g.vspincolor))))
    w_qmad = qmad_history.clover.wilson_clover_hop_mtsg_sigpre(Ut, mass, 1.0)
    w_call = w_qmad.tmngsMhs

    p1 = g.copy(psi)
    p2 = torch.clone(psi_torch)
    p3 = torch.clone(psi_torch)

    # tried this for warmup, but that wasnt the reason it was so slow
    #p0 = g.copy(psi)
    #x0, ret0 = GMRES(w_gpt, p0, p0, maxiter=maxit, eps=preci, inner_iter=innit, innerproduct=lambda x,y: g.inner_product(x,y), do_timing=True, verbose=ver)

    x_gpt, ret = GMRES(w_gpt, p1, p1, maxiter=maxit, eps=preci, inner_iter=innit, innerproduct=lambda x,y: g.inner_product(x,y), do_timing=True, verbose=ver)
    x_torch, ret_torch = GMRES(w_torch, p2, p2, maxiter=maxit, eps=preci, inner_iter=innit, do_timing=True, verbose=ver)
    x_qmad, ret_qmad = GMRES(w_call, p3, p3, maxiter=maxit, eps=preci, inner_iter=innit, do_timing=True, verbose=ver)

    if en == 0:
        line1 = f"| mass   | code  "
        for ty in ret["timings"]:
            line1 += f" | {ty:13}"
        line1 += f" | converged | break |"
        print(line1)
        print("-"*103)

    gptline = f"| {mass:6} | gpt   "
    for tim in ret["timings"].values():
        gptline += f" | {tim/1000000:>13.3f}"
    gptline += f" | {ret["converged"]:9} | {ret["breakdown"]:5} |"

    qmlline = "|        | qcd_ml"
    for tim in ret_torch["timings"].values():
        qmlline += f" | {tim/1000000:>13.3f}"
    qmlline += f" | {ret_torch["converged"]:9} | {ret_torch["breakdown"]:5} |"

    qmadline = "|        | qmad  "
    for tim in ret_qmad["timings"].values():
        qmadline += f" | {tim/1000000:>13.3f}"
    qmadline += f" | {ret_qmad["converged"]:9} | {ret_qmad["breakdown"]:5} |"

    print(gptline)
    print(qmlline)
    print(qmadline)
    print("-"*103)

