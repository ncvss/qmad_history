import torch 
import qmad_history.wilson
import numpy as np
import time

# test the initialisation of the full matrix wilson operator

lat_dim = [8,8,8,8]
vol = lat_dim[0]*lat_dim[1]*lat_dim[2]*lat_dim[3]
mass = -0.5

U = torch.randn([4]+lat_dim+[3,3], dtype=torch.cdouble)
v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)

init_st = time.perf_counter_ns()
w_full = qmad_history.wilson.wilson_full(U, mass)
init_en = time.perf_counter_ns()

old_st = time.perf_counter_ns()
# compare with direct computation
grid = lat_dim
strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
npind = np.indices(grid, sparse=False)
indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

hop_inds = []
for coord in range(4):
    # index after a negative step in coord direction
    minus_hop_ind = torch.clone(indices)
    minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
    # index after a positive step in coord direction
    plus_hop_ind = torch.clone(indices)
    plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
    # compute flattened index by dot product with strides
    hop_inds.append(torch.matmul(minus_hop_ind, strides))
    hop_inds.append(torch.matmul(plus_hop_ind, strides))
hop_inds = torch.stack(hop_inds, dim=1).contiguous()

gamx = [[3,2,1,0],[3,2,1,0],[2,3,0,1],[2,3,0,1]]
sparse_addr_old = torch.empty([vol,4,3,49],dtype=torch.int32)
for t in range(vol):
    for s in range(4):
        for g in range(3):
            sparse_addr_old[t,s,g,0] = t*12+s*3+g
            for mu in range(8):
                for gi in range(3):
                    sparse_addr_old[t,s,g,1+mu*6+gi] = hop_inds[t,mu]*12+s*3+gi
                    sparse_addr_old[t,s,g,1+mu*6+gi+3] = hop_inds[t,mu]*12+gamx[mu//2][s]*3+gi
old_en = time.perf_counter_ns()

print(w_full.sparse_addr[0,0,0])
print(sparse_addr_old[0,0,0])
print("--------")
print(w_full.sparse_addr[0,1,0])
print(sparse_addr_old[0,1,0])
print("both versions of computation equal:", torch.allclose(w_full.sparse_addr,sparse_addr_old))
print("new init time in s:", (init_en-init_st)/1000/1000/1000)
print("old init time in s:", (old_en-old_st)/1000/1000/1000)
