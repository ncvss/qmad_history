# make a histogram to show how much the performance varies

import matplotlib.pyplot as plt
from numpy import array, mean

with open(f"./test/testresults/rawdata_007.txt", "r") as datn:
    rawdata_str = datn.read()

rawdata = eval(rawdata_str)

gsize = 16**3 * 32


codes = ["gpt", "qcd_ml", "qmad_grid", "qmad_grid_clover"]
names = ["GPT", "qcd_ml", "AVX code on Grid data layout",
         "AVX code with optimised Clover term"]

plt.figure()
plt.hist([rawdata[na][gsize]/mean(rawdata[na][gsize]) for na in codes],
            bins=24,
            label=names)
plt.title(f"Histogram of the time measurements of the Wilson clover operator\non a 16x16x16x32 grid on hpd with 8 threads")
plt.xlabel("time normalized by mean")
plt.ylabel("count")
plt.legend()
plt.grid()
plt.savefig(f"./test/testresults/clover_time_hist_hpd_1632.pdf")
