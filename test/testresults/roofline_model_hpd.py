import matplotlib.pyplot as plt
import numpy as np

# nur die beiden Roofline-Geraden

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 13



wilson_FLOP_per_byte = 5496/((4*3*3+4*3*2)*16+8*4)
# bandwidth of hpd: 19.6 GB/s per core * 4 cores (according to Dr. Bandwidth)
hpd_bandwidth_B_p_s = 19.6 * 10**9 * 4
# peak FLOP/s: 2 doubles in register * 4 operations * 4 cores * 2.4 GHz
hpd_4core_FLOP_p_s = 2 * 4 * 4 * 2.4 * 10**9
# as the above value does not agree with experiment, maybe we use execution units from 8 cores?
hpd_8core_FLOP_p_s = 2 * 4 * 8 * 2.4 * 10**9

# bandwidth: 25.6 GB/s
haswell_bandwidth_B_p_s = 25.6*10**9
# peak FLOP/s: 4 doubles in register * 4 operations * 4 cores * 3.3 GHz
haswell_FLOP_p_s_avx = 4*4*4*3.3*10**9



# x = list(range(2,15))
# x2 = [0.2,0.5]+list(range(1,9))

x = [0.4,0.6]+list(range(1,17))
x2 = x

plt.figure(figsize=(7,4.8))

#plt.title(f"Roofline model for 4 cores of node 002 of HPD")

plt.plot(x2,[hpd_4core_FLOP_p_s for _ in x2], label="peak perf. on 4 cores")
#plt.plot(x,[hpd_8core_FLOP_p_s for _ in x], label="peak perf. w/ 8 thr. on 8 cores")
plt.plot(x2,[hpd_bandwidth_B_p_s*xx for xx in x2],label="bandwidth limited peak perf. on 4 cores")

plt.axvline(x=wilson_FLOP_per_byte,label=f"Intensity of the Wilson Dirac operator",color="red",linestyle='--')


plt.legend()
plt.xlabel("Intensity in FLOP/byte")
plt.ylabel("Performance in FLOP/second")

plt.xscale("log")
plt.yscale("log")
plt.ylim(7*10**9,1.5*10**12)

plt.grid()
# plt.yticks(range(0,110,10))

plt.savefig(f"./test/testresults/hpd2_base_roofline3.pdf")

# roofline for my pc

plt.figure(figsize=(7,4.8))

plt.plot(x,[haswell_FLOP_p_s_avx for _ in x], label="peak perf.")
plt.plot(x,[haswell_bandwidth_B_p_s*xx for xx in x],label="bandwidth limited peak perf.")

plt.axvline(x=wilson_FLOP_per_byte,label=f"Intensity of the Wilson Dirac operator",color="red",linestyle='--')


plt.legend()
plt.xlabel("Intensity in FLOP/byte")
plt.ylabel("Performance in FLOP/second")

plt.xscale("log")
plt.yscale("log")

plt.grid()
# plt.yticks(range(0,110,10))

plt.savefig(f"./test/testresults/pc1_base_roofline3.pdf")

