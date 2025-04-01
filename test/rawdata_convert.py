from numpy import array
import numpy as np

with open(f"./test/testresults/rawdata_003.txt", "r") as datn:
    rawdata_str = datn.read()

rawdata = eval(rawdata_str)

for op in rawdata:
    print(op)
    for vol, datarr in rawdata[op].items():
        print(vol, np.mean(datarr))

# this method works to get the data out of text files
