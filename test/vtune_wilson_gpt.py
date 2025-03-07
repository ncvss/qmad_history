import time
#import sys

#print(sys.path)
#sys.path.append('/home/nicovoss/Documents/GitHub/compile.gpt/centers/UR/hpd/build/gpt/lib/cgpt/build')
#sys.path.append('/home/nicovoss/Documents/GitHub/compile.gpt/centers/UR/hpd/build/gpt/lib')

import gpt as g

lat_dim = [8,8,8,16]

rng = g.random("simplerun")
U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_g[0].grid
v_g = rng.cnormal(g.vspincolor(grid))
dst_g = g.lattice(v_g)

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)


dw_g = g.qcd.fermion.wilson_clover(U_g, {"kappa":kappa,"csw_r":0.0,"csw_t":0.0,"xi_0":1,"nu":1,
                                            "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

#time.sleep(7)

for i in range(1):
    dst_g = dw_g(v_g)

#print(dst_g[0,0,0,0])
