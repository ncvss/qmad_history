import torch
import qmad_history
#import gpt as g
from qmad_history import wilson

lat_dim = [2,2,2,2]

mass = -0.5
U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim +[4,3], dtype=torch.cdouble)

boundary_phases = [-1,1,1,1]


#dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
#                                            "isAnisotropic":False,"boundary_phases":boundary_phases,}, )

dw = wilson.wilson_hop_mtsg(U, mass, boundary_phases=boundary_phases)

print(dw.hop_phases)