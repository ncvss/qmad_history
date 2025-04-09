import torch
import socket
import numpy as np
import time
import os

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline


num_threads = torch.get_num_threads()
hostname = socket.gethostname()
omp_places = os.environ.get('OMP_PLACES')

# split measurement into batches
# we alternate between operators and lattice dimensions
# and make n_batch measurements for each
n_measurements = 500
n_batch = 20
n_warmup = 10
assert n_measurements%n_batch == 0

#grid_dims = [[4,4,4,4],[8,4,4,8],[8,8,8,8],[8,8,8,16],[16,16,8,16],[16,16,16,16],[16,16,16,32],[32,16,16,32]]
#grid_dims = [[4,4,4,4],[8,4,4,8],[8,8,8,8],[8,8,8,16],[16,16,8,16],[16,16,16,16]]
grid_dims = [[4,4,4,4],[4,4,4,8],[8,4,4,8],[8,8,4,8],[8,8,8,8],[8,8,8,16],[16,8,8,16],[16,16,8,16],[16,16,16,16],[16,16,16,32],[32,16,16,32]]
grid_vols = [l[0]*l[1]*l[2]*l[3] for l in grid_dims]
print(grid_vols)
# important to transform to list, as an iterator is only usable once
grid_stats = list(zip(grid_vols,grid_dims))

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

rng = g.random("voltest")

# the following operators are measured
operator_names = ["gpt", "qcd_ml", "qmad_dir", "qmad_vect", "qmad_vect_templ", "qmad_grid", "qmad_grid_clover"]

results = {na:{vol:np.zeros(n_measurements) for vol in grid_vols} for na in operator_names}

# data size transferred in byte
datasizes = {na:dict() for na in operator_names}

print("status of data generation:")

for n in range(0,n_measurements,n_batch):
    print(n, "vol:", end=" ", flush=True)
    for vol,lat_dim in grid_stats:

        print(vol, end=" ", flush=True)

        # initialise the fields for this volume
        U_gpt = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_gpt[0].grid
        v_gpt = rng.cnormal(g.vspincolor(grid))
        #dst_g = g.lattice(v_gpt)

        U_qml = torch.tensor(compat.lattice_to_array(U_gpt))
        v_qml = torch.tensor(compat.lattice_to_array(v_gpt))

        vgfirst = v_qml[:,:,:,0:lat_dim[3]//2]
        vgsecond = v_qml[:,:,:,lat_dim[3]//2:lat_dim[3]]
        v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)
        
        # initialise the operators and measure one at a time

        dwc = g.qcd.fermion.wilson_clover(U_gpt, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                      "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )
        v_any = v_gpt

        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["gpt"][vol][i] = stop - start


        dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(U_qml, mass, csw)
        v_any = v_qml
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qcd_ml"][vol][i] = stop - start


        dwc0 = clover.wilson_clover_fpre(U_qml, mass, csw)
        dwc = dwc0.xtmnghs
        v_any = v_qml
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qmad_dir"][vol][i] = stop - start
        
        if n == 0:
            fstr_qmad = dwc0.field_strength
            fstr_qmad_data = len(fstr_qmad)*fstr_qmad[0].element_size()*fstr_qmad[0].nelement()


        dwc0 = clover.wilson_clover_hop_mtsg(U_qml, mass, csw)
        dwc = dwc0.avx_tmgsMhns
        v_any = v_qml
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qmad_vect"][vol][i] = stop - start
        
        if n == 0:
            hops_qmad = dwc0.hop_inds
            hops_qmad_data = hops_qmad.element_size()*hops_qmad.nelement()
    

        # this is also the templ operator
        dwc = dwc0.templ_tmgsMhns
        v_any = v_qml
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qmad_vect_templ"][vol][i] = stop - start


        dwc0 = clover.wilson_clover_hop_mtsgt2_sigpre(U_qml, mass, csw)
        dwc = dwc0.tmngsMht
        v_any = v_grid2
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qmad_grid"][vol][i] = stop - start

        if n == 0:
            fstr_grid = dwc0.field_strength_sigma
            fstr_grid_data = fstr_grid.element_size()*fstr_grid.nelement()
            hops_grid = dwc0.hop_inds
            hops_grid_data = hops_grid.element_size()*hops_grid.nelement()


        dwc0 = clover.wilson_clover_hop_mtsg_sigpre(U_qml, mass, csw)
        dwc = dwc0.tmngsMhs
        v_any = v_qml
        
        for i in range(n_warmup):
            any_result = dwc(v_any)

        for i in range(n,n+n_batch):
            start = time.perf_counter_ns()
            any_result = dwc(v_any)
            stop = time.perf_counter_ns()
            results["qmad_grid_clover"][vol][i] = stop - start
        

        # set data sizes once
        if n == 0:
            base_data = 2*v_qml.element_size()*v_qml.nelement() + U_qml.element_size()*U_qml.nelement()
            
            datasizes["gpt"][vol] = base_data + fstr_grid_data
            datasizes["qcd_ml"][vol] = base_data
            datasizes["qmad_dir"][vol] = base_data + fstr_qmad_data
            datasizes["qmad_vect"][vol] = base_data + fstr_qmad_data + hops_qmad_data
            datasizes["qmad_vect_templ"][vol] = base_data + fstr_qmad_data + hops_qmad_data
            datasizes["qmad_grid"][vol] = base_data + fstr_grid_data + hops_grid_data
            datasizes["qmad_grid_clover"][vol] = base_data + fstr_grid_data + hops_qmad_data
        
    print()

print()

with open(f"./test/testresults/dataset_number.txt", "r") as datn:
    dataset_str_old = datn.read()

dataset_num = int(dataset_str_old)+1
dataset = f"{dataset_num:03}"

with open(f"./test/testresults/dataset_number.txt", "w") as datn:
    datn.write(dataset)


with open(f"./test/testresults/rawdata_{dataset}.txt", "w") as output_results:
    output_results.write(str(results))


with open(f"./test/testresults/plotdata_{dataset}.txt", "w") as plot_out:
    plot_out.write(f"host = '{hostname}'\nthreadnumber = {num_threads}\n")
    plot_out.write(f"omp_places = '{omp_places}'\n")
    plot_out.write(f"n_measurements = {n_measurements}\nn_batch = {n_batch}\n")
    plot_out.write(f"mass = {mass}\ncsw = {csw}\n")
    plot_out.write(f"grid_volumes = {grid_vols}\ngrid_dims = {grid_dims}\n\n")

    meanstr = "means = ["
    meanstdstr = "meanstdevs = ["
    thrptstr = "thrpts = ["

    for op in operator_names:
        plot_out.write(f"# {op}\n")
        means_us = []
        stdevs_us = []
        bests_us = []
        thrpts_mean_GiBps = []
        thrpts_peak_GiBps = []
        for vol,ldim in grid_stats:
            results_sorted = np.sort(results[op][vol])[:(n_measurements // 5)]
            mean_us = np.mean(results_sorted)/1000
            stdev_us = np.std(results_sorted)/1000
            best_us = results_sorted[0]/1000
            data_B = datasizes[op][vol]
            thrpt_mean_GiBps = data_B/(1024**3)/(mean_us/(1000**2))
            thrpt_peak_GiBps = data_B/(1024**3)/(best_us/(1000**2))
            means_us.append(mean_us)
            stdevs_us.append(stdev_us)
            bests_us.append(best_us)
            thrpts_mean_GiBps.append(thrpt_mean_GiBps)
            thrpts_peak_GiBps.append(thrpt_peak_GiBps)
        
        plot_out.write(f"{op}_runtime_mean_in_us = {means_us}\n")
        plot_out.write(f"{op}_runtime_stdevs_in_us = {stdevs_us}\n")
        plot_out.write(f"{op}_best_runtime_in_us = {bests_us}\n")
        plot_out.write(f"{op}_throughput_means_in_GiB_per_s = {thrpts_mean_GiBps}\n")
        plot_out.write(f"{op}_best_throughput_in_GiB_per_s = {thrpts_peak_GiBps}\n")

        meanstr += f"{op}_runtime_mean_in_us, "
        meanstdstr += f"{op}_runtime_stdevs_in_us, "
        thrptstr += f"{op}_throughput_means_in_GiB_per_s, "
    
    plot_out.write("\n# sets of all measurements for python execution\n")
    plot_out.write(meanstr+"]\n")
    plot_out.write(meanstdstr+"]\n")
    plot_out.write(thrptstr+"]\n")
        
        



