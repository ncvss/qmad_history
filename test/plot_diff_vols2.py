import torch
import socket
import numpy as np
import time

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline


num_threads = torch.get_num_threads()
hostname = socket.gethostname()

# split measurement into batches
# we alternate between operators and lattice dimensions
# and make n_batch measurements for each
n_measurements = 500
n_batch = 20
n_warmup = 5

grid_dims = [[8,4,4,8],[8,8,8,8],[8,8,8,16],[16,16,8,16],[16,16,16,16],[16,16,16,32],[32,16,16,32]]
grid_vols = [l[0]*l[1]*l[2]*l[3] for l in grid_dims]
# important to transform to list, as an iterator is only usable once
grid_stats = list(zip(grid_vols,grid_dims))

mass = -0.5
kappa = 1.0/2.0/(mass + 4.0)
csw = 1.0

print()

rng = g.random("voltest")

# the following operators are measured
operator_names = ["gpt", "qcd_ml", "qmad_dir", "qmad_vect", "qmad_vect_templ", "qmad_grid", "qmad_grid_clover"]

results = {na:{vol:np.zeros(n_measurements) for vol in grid_vols} for na in operator_names}

# data size in byte
datasizes = {na:dict() for na in operator_names}

for n in range(0,n_measurements,n_batch):
    print(n, end=" ", flush=True)
    for vol,lat_dim in grid_stats:

        # initialise the fields for this volume
        U_gpt = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
        grid = U_gpt[0].grid
        v_gpt = rng.cnormal(g.vspincolor(grid))
        dst_g = g.lattice(v_gpt)

        U_qml = torch.tensor(compat.lattice_to_array(U_gpt))
        v_qml = torch.tensor(compat.lattice_to_array(v_gpt))

        vgfirst = v_qml[:,:,:,0:lat_dim[3]//2]
        vgsecond = v_qml[:,:,:,lat_dim[3]//2:lat_dim[3]]
        v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)
        
        # initialise the operators
        dwc_gpt = g.qcd.fermion.wilson_clover(U_gpt, {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
                                                      "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

        dwc_qcd_ml = qcd_ml.qcd.dirac.dirac_wilson_clover(U_qml, mass, csw)

        dwc_qmad_dir = clover.wilson_clover_fpre(U_qml, mass, csw)

        dwc_qmad_vect = clover.wilson_clover_hop_mtsg(U_qml, mass, csw)
        # this is also the templ operator

        dwc_qmad_grid = clover.wilson_clover_hop_mtsgt2_sigpre(U_qml, mass, csw)

        dwc_qmad_grid_clover = clover.wilson_clover_hop_mtsg_sigpre(U_qml, mass, csw)

        op_inputs = [v_gpt, v_qml, v_qml, v_qml, v_qml, v_grid2, v_qml]
        op_calls = [dwc_gpt, dwc_qcd_ml, dwc_qmad_dir.xtmnghs, dwc_qmad_vect.avx_tmgsMhns,
                    dwc_qmad_vect.templ_tmgsMhns, dwc_qmad_grid.tmngsMht, dwc_qmad_grid_clover.tmngsMhs]
        
        for i in range(n,n+n_batch):
            for op_id in range(len(operator_names)):
                any_result = op_calls[op_id](op_inputs[op_id])
        
        for i in range(n,n+n_batch):
            for op_id in range(len(operator_names)):
                start = time.perf_counter_ns()
                any_result = op_calls[op_id](op_inputs[op_id])
                stop = time.perf_counter_ns()
                results[operator_names[op_id]][vol][i] = stop - start

        # set data sizes once
        if n == 0:
            base_data = 2*v_qml.element_size()*v_qml.nelement() + U_qml.element_size()*U_qml.nelement()
            
            fstr_qmad = dwc_qmad_dir.field_strength
            fstr_grid = dwc_qmad_grid.fs
            hops_qmad = dwc_qmad_vect.hop_inds
            hops_grid = dwc_qmad_grid.hop_inds

            datasizes["gpt"][vol] = base_data + fstr_grid.element_size()*fstr_grid.nelement()
            datasizes["qcd_ml"][vol] = base_data
            datasizes["qmad_dir"][vol] = base_data + fstr_qmad.element_size()*fstr_qmad.nelement()
            datasizes["qmad_vect"][vol] = base_data + fstr_qmad.element_size()*fstr_qmad.nelement() + hops_qmad.element_size()*hops_qmad.nelement()
            datasizes["qmad_vect_templ"][vol] = datasizes["qmad_vect"][vol]
            datasizes["qmad_grid"][vol] = base_data + fstr_grid.element_size()*fstr_grid.nelement() + hops_grid.element_size()*hops_grid.nelement()
            datasizes["qmad_grid_clover"][vol] = datasizes["qmad_grid"][vol]


with open(f"./test/testresults/rawdata.txt", "w") as output_results:
    output_results.write(results)

#results_sorted = {na:dict() for na in operator_names}
with open(f"./test/testresults/plotdata.txt", "w") as plot_out:
    plot_out.write(f"host = '{hostname}'\nthreadnumber = {num_threads}\n")
    plot_out.write(f"mass = {mass}\ncsw = {csw}\n")
    plot_out.write(f"grid_volumes = {grid_vols}\ngrid_dims = {grid_dims}\n\n")

    for op in operator_names:
        plot_out.write(f"{op}\n")
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
        
        plot_out.write(f"runtime_mean_in_us = {means_us}\nruntime_stdevs_in_us = {stdevs_us}\n")
        plot_out.write(f"best_runtime_in_us = {bests_us}\n")
        plot_out.write(f"throughput_means_in_GiB_per_s = {thrpts_mean_GiBps}\n")
        plot_out.write(f"best_throughput_in_GiB_per_s = {thrpts_peak_GiBps}\n")
        
        



