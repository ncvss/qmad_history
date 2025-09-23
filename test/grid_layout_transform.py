import torch
import socket
import numpy as np
import time


print()
num_threads = torch.get_num_threads()
print("running on host", socket.gethostname())
print(f'Machine has {num_threads} threads')

n_measurements = 300
n_warmup = 20
print("n_measurements =", n_measurements)
print("n_warmup =", n_warmup)

lat_dim = [16,8,8,16]
print("lattice_dimensions =",lat_dim)

v = torch.randn(lat_dim+[4,3], dtype=torch.cdouble)


results = np.full(n_measurements, 1.0e12)
results2 = np.full(n_measurements, 1.0e12)

for i in range(n_warmup):
    vgfirst = v[:,:,:,0:lat_dim[3]//2]
    vgsecond = v[:,:,:,lat_dim[3]//2:lat_dim[3]]
    v_grid = torch.stack([vgfirst, vgsecond], dim=-1).contiguous()
    v_grid_back = torch.cat([v_grid[:,:,:,:,:,:,0],v_grid[:,:,:,:,:,:,1]], dim=3).contiguous()

for i in range(n_measurements):
    start = time.perf_counter_ns()
    vgfirst = v[:,:,:,0:lat_dim[3]//2]
    vgsecond = v[:,:,:,lat_dim[3]//2:lat_dim[3]]
    v_grid = torch.stack([vgfirst, vgsecond], dim=-1).contiguous()
    stop = time.perf_counter_ns()
    results[i] = stop - start
    
    start = time.perf_counter_ns()
    v_grid_back = torch.cat([v_grid[:,:,:,:,:,:,0],v_grid[:,:,:,:,:,:,1]], dim=3).contiguous()
    stop = time.perf_counter_ns()
    results2[i] = stop - start

results_sorted = np.sort(results)[:(n_measurements//5)]
results2_sorted = np.sort(results2)[:(n_measurements//5)]

print(f"\n{'measurements':35}: {'time':>15} {'std':>15}")

print(f"{"transform to grid layout":35}: {np.mean(results_sorted)/1000:>15.3f} {np.std(results_sorted)/1000:>15.3f}")
print(f"{"transform back":35}: {np.mean(results2_sorted)/1000:>15.3f} {np.std(results2_sorted)/1000:>15.3f}")
