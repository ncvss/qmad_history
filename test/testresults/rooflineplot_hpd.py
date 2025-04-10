import matplotlib.pyplot as plt
import numpy as np

base_lattice_dimensions = [16, 8, 8, 16]
basevol = 16*8*8*16

wilson_FLOP = basevol*5496
# bandwidth of hpd: 19.6 GB/s per core * 4 cores (according to Dr. Bandwidth)
hpd_bandwidth_B_p_s = 19.6 * 10**9 * 4
# peak FLOP/s: 2 doubles in register * 4 operations * 4 cores * 2.4 GHz
hpd_4core_FLOP_p_s = 2 * 4 * 4 * 2.4 * 10**9
# as the above value does not agree with experiment, maybe we use execution units from 8 cores?
hpd_8core_FLOP_p_s = 2 * 4 * 8 * 2.4 * 10**9


data_8thr = [
    [       10027008,        1047.660,          49.088, [16, 8, 8, 8]],
    [       13172736,        1073.598,         113.635, [16, 8, 8, 12]],
    [       16318464,        1087.410,         153.164, [16, 8, 8, 16]],
    [       22609920,        1056.158,           6.446, [16, 8, 8, 24]],
    [       28901376,        1122.017,          40.018, [16, 8, 8, 32]],
    [       41484288,        1512.197,          92.072, [16, 8, 8, 48]],

]

# {
#     1:[
#         [3440640    ,    2649.940  ,         2.716 , [8, 8, 8, 4]],
#         [5013504    ,    2705.936   ,        4.622 , [8, 8, 8, 8]],
#         [6586368    ,    2798.551   ,        5.553  ,[8, 8, 8, 12]],
#         [8159232     ,   2916.289      ,    12.388  ,[8, 8, 8, 16]],
#         [11304960    ,    3139.590    ,       8.647 , [8, 8, 8, 24]],
#         [14450688    ,    3279.491    ,       6.519,  [8, 8, 8, 32]],
#         [20742144    ,    4198.875    ,      13.946,  [8, 8, 8, 48] ],
#     ],
#     2:[
#         [3440640  ,      1331.650      ,     1.011 , [8, 8, 8, 4]],
#         [5013504   ,     1361.989      ,     2.430 , [8, 8, 8, 8]],
#         [6586368    ,    1423.532      ,     3.186 , [8, 8, 8, 12]],
#        [ 8159232    ,    1489.055     ,      6.434 , [8, 8, 8, 16]],
#        [11304960     ,   1625.998      ,     8.377  ,[8, 8, 8, 24]],
#        [14450688     ,   1763.965      ,     5.314  ,[8, 8, 8, 32]],
#        [20742144     ,   2373.417       ,    7.301 , [8, 8, 8, 48]],
#     ],
#     4:[
#         [3440640       ,  670.814     ,      1.111 , [8, 8, 8, 4]],
#         [5013504     ,    692.589     ,      2.525 , [8, 8, 8, 8]],
#         [6586368   ,      733.391     ,      3.756 , [8, 8, 8, 12]],
#         [8159232    ,     790.678     ,      3.423 , [8, 8, 8, 16]],
#        [11304960    ,     891.027     ,      5.461 , [8, 8, 8, 24]],
#        [14450688    ,    1096.042    ,       7.580  ,[8, 8, 8, 32]],
#        [20742144    ,    1625.482     ,      5.995 , [8, 8, 8, 48]],
#     ]
# }

x = [wilson_FLOP/dat[0] for dat in data_8thr]

y = [wilson_FLOP/(dat[1]/1000**2) for dat in data_8thr]

actual_wilson_data = 0
for dat in data_8thr:
    if dat[3][0]*dat[3][1]*dat[3][2]*dat[3][3] == basevol:
        actual_wilson_data = dat[0]


plt.figure()

plt.title("Roofline-style plot of a pseudo Dirac Wilson operator\nthat allows varying input data size, on hpd")


plt.plot(x,y, label=f"Wilson Dirac")

plt.plot(x,[hpd_4core_FLOP_p_s for _ in x], label="peak performance 8 thr. on 4 cores")
plt.plot(x,[hpd_8core_FLOP_p_s for _ in x], label="peak performance 8 thr. on 8 cores")
plt.plot(x,[hpd_bandwidth_B_p_s*xx for xx in x],label="bandwidth limited peak performance")

plt.axvline(x=wilson_FLOP/actual_wilson_data,label="Intensity of the correct operator",color="red",linestyle='--')

plt.legend()
plt.xlabel("Intensity in FLOP/byte")
plt.ylabel("Performance in FLOP/second")

plt.xscale("log")
plt.yscale("log")

plt.grid(which="both")
# plt.yticks(range(0,110,10))

plt.savefig("./test/testresults/wilson_roofline_hpd_8thr.pdf")

