import matplotlib.pyplot as plt
import math
import numpy as np

names = ["GPT", "qcd_ml", "naive C++ code", "AVX vectorised code", "AVX code with templates", "AVX code on Grid data layout",
         "AVX code with optimised Clover term"]

# read from the raw data file

with open(f"./test/testresults/plotdata_008.txt", "r") as datn:
    plotdata_str = datn.read()

exec(plotdata_str)


plt.figure()

for ys, yerrs, na in zip(means, meanstdevs, names):
    if na != "qcd_ml":
        plt.errorbar(grid_volumes,ys,yerr=yerrs,capsize=2.0,label=na)

plt.legend()
plt.title(f"Runtime of different implementations of the Wilson clover operator\nfor different lattice sizes with {threadnumber} threads on {host}")
plt.xlabel("number of grid points")
plt.ylabel("runtime in µs")
plt.grid()
plt.xscale("log")
plt.yscale("log")

xlabels = ["$2^{"+str(int(math.log2(x)))+"}$" for x in grid_volumes]
plt.xticks(grid_volumes, xlabels)

plt.savefig(f"./test/testresults/clover_voltime_{host[0:3]}_{threadnumber}thr_var.pdf")


plt.figure()

for ys, na in zip(thrpts, names):
    if na != "qcd_ml":
        plt.plot(grid_volumes, ys, label=na)

plt.legend()
plt.title(f"Throughput of different optimisation levels of the Wilson clover operator\nfor different lattice sizes  with {threadnumber} threads on {host}")
plt.xlabel("number of grid points")
plt.ylabel("throughput in GiB/s")
plt.grid()
plt.xscale("log")

plt.xticks(grid_volumes,xlabels)

plt.savefig(f"./test/testresults/clover_volthroughput_{host[0:3]}_{threadnumber}thr_var.pdf")

