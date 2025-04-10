import matplotlib.pyplot as plt
import numpy as np

base_lattice_dim = [16, 16, 16, 16]
base_lattice_str = f"{base_lattice_dim[0]}x{base_lattice_dim[1]}x{base_lattice_dim[2]}x{base_lattice_dim[3]}"
basevol = base_lattice_dim[0]*base_lattice_dim[1]*base_lattice_dim[2]*base_lattice_dim[3]

wilson_FLOP = basevol*5496
# bandwidth of hpd: 19.6 GB/s per core * 4 cores (according to Dr. Bandwidth)
hpd_bandwidth_B_p_s = 19.6 * 10**9 * 4
# peak FLOP/s: 2 doubles in register * 4 operations * 4 cores * 2.4 GHz
hpd_4core_FLOP_p_s = 2 * 4 * 4 * 2.4 * 10**9
# as the above value does not agree with experiment, maybe we use execution units from 8 cores?
hpd_8core_FLOP_p_s = 2 * 4 * 8 * 2.4 * 10**9

# 16x8x8x16
# data_8thr_sockets = [
#     [       10027008,        1012.230,           1.774, [16, 8, 8, 8]],
#     [       13172736,        1015.779,           2.485, [16, 8, 8, 12]],
#     [       16318464,        1023.446,           2.199, [16, 8, 8, 16]],
#     [       22609920,        1036.637,           2.362, [16, 8, 8, 24]],
#     [       28901376,        1045.869,           2.262, [16, 8, 8, 32]],
#     [       41484288,        1057.629,          14.544, [16, 8, 8, 48]],
# ]

# data_8thr_threads = [
#     [       10027008,        1017.359,           1.066, [16, 8, 8, 8]],
#     [       13172736,        1020.857,           1.016, [16, 8, 8, 12]],
#     [       16318464,        1028.539,           0.939, [16, 8, 8, 16]],
#     [       22609920,        1050.299,           2.221, [16, 8, 8, 24]],
#     [       28901376,        1121.790,           3.720, [16, 8, 8, 32]],
#     [       41484288,        1500.529,          10.945, [16, 8, 8, 48]],
# ]


# 16x16x16x16
data_8thr_sockets = [
    [       40108032,        3984.344,          57.330, [16, 16, 16, 8]],
    [       52690944,        4051.156,          50.337, [16, 16, 16, 12]],
    [       65273856,        4287.037,          31.062, [16, 16, 16, 16]],
    [       90439680,        4450.687,          25.367, [16, 16, 16, 24]],
    [      115605504,        5238.331,          36.409, [16, 16, 16, 32]],
    [      165937152,        6934.767,          27.885, [16, 16, 16, 48]],
]

data_8thr_threads = [
    [       40108032,        4111.664,           7.821, [16, 16, 16, 8]],
    [       52690944,        4179.769,          43.423, [16, 16, 16, 12]],
    [       65273856,        4318.423,          32.492, [16, 16, 16, 16]],
    [       90439680,        5219.628,          31.825, [16, 16, 16, 24]],
    [      115605504,        6427.413,          73.932, [16, 16, 16, 32]],
    [      165937152,        8159.567,          55.022, [16, 16, 16, 48]],
]


x = [wilson_FLOP/dat[0] for dat in data_8thr_threads]

y_threads = [wilson_FLOP/(dat[1]/1000**2) for dat in data_8thr_threads]
y_sockets = [wilson_FLOP/(dat[1]/1000**2) for dat in data_8thr_sockets]

actual_wilson_data = 0
for dat in data_8thr_threads:
    if dat[3][0]*dat[3][1]*dat[3][2]*dat[3][3] == basevol:
        actual_wilson_data = dat[0]


plt.figure()

plt.title(f"Roofline plot of a pseudo Wilson Dirac operator that allows varying input\ndata size, {base_lattice_str} grid, varying OMP_PLACES, on hpd-node-002")


plt.plot(x,y_threads, label=f"Wilson perf. for OMP_PLACES=threads")
plt.plot(x,y_sockets, label=f"Wilson perf. for OMP_PLACES=sockets")

plt.plot(x,[hpd_4core_FLOP_p_s for _ in x], label="peak perf. w/ 8 thr. on 4 cores")
plt.plot(x,[hpd_8core_FLOP_p_s for _ in x], label="peak perf. w/ 8 thr. on 8 cores")
plt.plot(x,[hpd_bandwidth_B_p_s*xx for xx in x],label="bandwidth limited peak perf. on 4 cores")

plt.axvline(x=wilson_FLOP/actual_wilson_data,label=f"Intensity of the correct operator",color="red",linestyle='--')

plt.legend()
plt.xlabel("Intensity in FLOP/byte")
plt.ylabel("Performance in FLOP/second")

plt.xscale("log")
plt.yscale("log")

plt.grid(which="both")
# plt.yticks(range(0,110,10))

plt.savefig(f"./test/testresults/wilson_roofline_hpd_8thr_{base_lattice_str}.pdf")

