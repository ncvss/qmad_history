import torch
import socket
import numpy as np
import time

import gpt as g
import qcd_ml

from qmad_history import compat, wilson, clover, settings, wilson_roofline


# I tried to run this script, and it crashed my computer because it needed too much ram
# I need to find another way to eliminate correlations between measurements

# the idea behind this script was to alternate between all operators for all volumes
# so that backgroud processes have an equal probability to affect them

# print()
# num_threads = torch.get_num_threads()
# hostname = socket.gethostname()
# #print("running on host", socket.gethostname())
# #print(f'Machine has {num_threads} threads')

# #print(settings.capab)

# n_measurements = 500
# n_warmup = 20
# #print("n_measurements =",n_measurements)

# grid_dims = [[8,8,4,8],[8,8,8,16],[16,16,8,16],[16,16,16,32],[32,16,16,32]]
# grid_vols = [l[0]*l[1]*l[2]*l[3] for l in grid_dims]
# grid_stats = zip(grid_vols,grid_dims)

# # lat_dim = [16,16,16,32]
# # print("lattice_dimensions =",lat_dim)

# mass = -0.5
# #print("mass_parameter =",mass)
# kappa = 1.0/2.0/(mass + 4.0)
# csw = 1.0
# #print("csw =", csw)

# rng = g.random("voltest")

# operator_names = ["gpt", "qcd_ml", "qmad_dir", "qmad_vect", "qmad_vect_templ", "qmad_grid", "qmad_grid_clover"]

# # one U and v for each data type, memory layout and grid dimension
# U_dict = {na:dict() for na in operator_names}
# v_dict = {na:dict() for na in operator_names}

# for vol,lat_dim in grid_stats:
#     U_gpt = g.qcd.gauge.random(g.grid(lat_dim, g.double), rng)
#     grid = U_gpt[0].grid
#     v_gpt = rng.cnormal(g.vspincolor(grid))
#     dst_g = g.lattice(v_gpt)
#     U_dict["gpt"][vol] = U_gpt
#     v_dict["gpt"][vol] = v_gpt

#     U_qml = torch.tensor(compat.lattice_to_array(U_gpt))
#     v_qml = torch.tensor(compat.lattice_to_array(v_gpt))
#     U_dict["qcd_ml"][vol] = U_qml
#     v_dict["qcd_ml"][vol] = v_qml
#     U_dict["qmad_dir"][vol] = U_qml
#     v_dict["qmad_dir"][vol] = v_qml
#     U_dict["qmad_vect"][vol] = U_qml
#     v_dict["qmad_vect"][vol] = v_qml
#     U_dict["qmad_vect_templ"][vol] = U_qml
#     v_dict["qmad_vect_templ"][vol] = v_qml
#     U_dict["qmad_grid_clover"][vol] = U_qml
#     v_dict["qmad_grid_clover"][vol] = v_qml

#     vgfirst = v_qml[:,:,:,0:lat_dim[3]//2]
#     vgsecond = v_qml[:,:,:,lat_dim[3]//2:lat_dim[3]]
#     v_grid2 = torch.stack([vgfirst, vgsecond], dim=-1)
#     U_dict["qmad_grid"][vol] = U_qml
#     v_dict["qmad_grid"][vol] = v_grid2




# # for now only wilson clover
# operator_dict = {na:dict() for na in operator_names}

# for vol in grid_vols:
#     operator_dict["gpt"][vol] = g.qcd.fermion.wilson_clover(U_dict["gpt"][vol], {"kappa":kappa,"csw_r":csw,"csw_t":csw,"xi_0":1,"nu":1,
#                                                             "isAnisotropic":False,"boundary_phases":[1.0,1.0,1.0,1.0],}, )

#     operator_dict["qcd_ml"][vol] = qcd_ml.qcd.dirac.dirac_wilson_clover(U_dict["qcd_ml"][vol], mass, csw)

#     operator_dict["qmad_dir"][vol] = clover.wilson_clover_fpre(U_dict["qmad_dir"][vol], mass, csw)

#     operator_dict["qmad_vect"][vol] = clover.wilson_clover_hop_mtsg(U_dict["qmad_vect"][vol], mass, csw)
#     operator_dict["qmad_vect_templ"][vol] = operator_dict["qmad_vect"][vol]

#     operator_dict["qmad_grid"][vol] = clover.wilson_clover_hop_mtsgt2_sigpre(U_dict["qmad_grid"][vol], mass, csw)

#     operator_dict["qmad_grid_clover"][vol] = clover.wilson_clover_hop_mtsg_sigpre(U_dict["qmad_grid_clover"][vol], mass, csw)


# call_dict = {na:dict() for na in operator_names}

# for vol in grid_vols:
#     call_dict["gpt"][vol] = operator_dict["gpt"][vol]
#     call_dict["qcd_ml"][vol] = operator_dict["qcd_ml"][vol]
#     call_dict["qmad_dir"][vol] = operator_dict["qmad_dir"][vol].xtmnghs
#     call_dict["qmad_vect"][vol] = operator_dict["qmad_vect"][vol].avx_tmgsMhns
#     call_dict["qmad_vect_templ"][vol] = operator_dict["qmad_vect_templ"][vol].templ_tmgsMhns
#     call_dict["qmad_grid"][vol] = operator_dict["qmad_grid"][vol].grid_tmngsMht
#     call_dict["qmad_grid_clover"][vol] = operator_dict["qmad_grid_clover"][vol].tmngsMhs

# results = {na:{vol:np.zeros(n_measurements) for vol in grid_vols} for na in operator_names}

# # data size in byte
# datasizes = {na:dict() for na in operator_names}

# for vol in grid_vols:
#     base_data = (
#         2*v_dict["qcd_ml"][vol].element_size()*v_dict["qcd_ml"][vol].nelement()
#         + U_dict["qcd_ml"][vol].element_size()*U_dict["qcd_ml"][vol].nelement()
#     )
#     fstr_qmad = operator_dict["qmad_dir"][vol].field_strength
#     fstr_grid = operator_dict["qmad_grid"][vol].fs
#     hops_qmad = operator_dict["qmad_vect"][vol].hop_inds
#     hops_grid = operator_dict["qmad_grid"][vol].hop_inds

#     datasizes["gpt"][vol] = base_data + fstr_grid.element_size()*fstr_grid.nelement()
#     datasizes["qcd_ml"][vol] = base_data
#     datasizes["qmad_dir"][vol] = base_data + fstr_qmad.element_size()*fstr_qmad.nelement()
#     datasizes["qmad_vect"][vol] = base_data + fstr_qmad.element_size()*fstr_qmad.nelement() + hops_qmad.element_size()*hops_qmad.nelement()
#     datasizes["qmad_vect_templ"][vol] = datasizes["qmad_vect"][vol]
#     datasizes["qmad_grid"][vol] = base_data + fstr_grid.element_size()*fstr_grid.nelement() + hops_grid.element_size()*hops_grid.nelement()
#     datasizes["qmad_grid_clover"][vol] = datasizes["qmad_grid"][vol]
    

# for n in range(n_warmup):
#     for vol in grid_vols:
#         for op in operator_names:
#             print(op)
#             any_result = call_dict[op][vol](v_dict[op][vol])

# print("measurement started")
# for n in range(n_measurements):
#     print(n, end=" ", flush=True)
#     for vol in grid_vols:
#         for op in operator_names:
#             start = time.perf_counter_ns()
#             any_result = call_dict[op][vol](v_dict[op][vol])
#             stop = time.perf_counter_ns()
#             results[op][vol][n] = stop - start

# with open(f"./test/testresults/rawdata.txt", "w") as output_results:
#     output_results.write(results)


# #results_sorted = {na:dict() for na in operator_names}
# with open(f"./test/testresults/plotdata.txt", "w") as plot_out:
#     plot_out.write(f"host = '{hostname}'\nthreadnumber = {num_threads}\n")
#     plot_out.write(f"mass = {mass}\ncsw = {csw}\n")
#     plot_out.write(f"grid_volumes = {grid_vols}\ngrid_dims = {grid_dims}\n\n")

#     for op in operator_names:
#         plot_out.write(f"{op}\n")
#         means_us = []
#         stdevs_us = []
#         bests_us = []
#         thrpts_mean_GiBps = []
#         thrpts_peak_GiBps = []
#         for vol,ldim in grid_stats:
#             results_sorted = np.sort(results[op][vol])[:(n_measurements // 5)]
#             mean_us = np.mean(results_sorted)/1000
#             stdev_us = np.std(results_sorted)/1000
#             best_us = results_sorted[0]/1000
#             data_B = datasizes[op][vol]
#             thrpt_mean_GiBps = data_B/(1024**3)/(mean_us/(1000**2))
#             thrpt_peak_GiBps = data_B/(1024**3)/(best_us/(1000**2))
#             means_us.append(mean_us)
#             stdevs_us.append(stdev_us)
#             bests_us.append(best_us)
#             thrpts_mean_GiBps.append(thrpt_mean_GiBps)
#             thrpts_peak_GiBps.append(thrpt_peak_GiBps)
        
#         plot_out.write(f"runtime_mean_in_us = {means_us}\nruntime_stdevs_in_us = {stdevs_us}\n")
#         plot_out.write(f"best_runtime_in_us = {bests_us}\n")
#         plot_out.write(f"throughput_means_in_GiB_per_s = {thrpts_mean_GiBps}\n")
#         plot_out.write(f"best_throughput_in_GiB_per_s = {thrpts_peak_GiBps}\n")
        
        

