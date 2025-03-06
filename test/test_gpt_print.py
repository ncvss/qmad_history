import gpt as g # type: ignore


lat_dim = [4,4,4,4]


rng = g.random("run")
U_g = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
grid = U_g[0].grid
v_g = rng.cnormal(g.vspincolor(grid))
dst_g = g.lattice(v_g)

print(g((v_g + g.vspincolor(g.grid([4,4,4,4], g.single)))))