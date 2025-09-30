import torch
import socket
import numpy as np
import time

from qmad_history import wilson_roofline

# visualize the hop accesses from the roofline wilson in 2 dimensions

base_lat_dim = [4,4,4,4]
in_lat_dims = [[4,4,4,4],[4,4,4,8],[4,4,6,8],[4,4,4,12],[4,4,4,16]]

for in_lat_dim in in_lat_dims:
    print(in_lat_dim[2:],"grid addresses:")
    for x in range(in_lat_dim[2]*in_lat_dim[3]):
        print(f" {x:>2}", end="\n" if (x+1)%in_lat_dim[3] == 0 else "")
    print()

    mass = -0.5
    U = torch.randn([4] + in_lat_dim + [3, 3], dtype=torch.cdouble)
    v = torch.randn(in_lat_dim + [4, 3], dtype=torch.cdouble)

    dw = wilson_roofline.wilson_hop_mtsg_roofline(U, mass, base_lat_dim)

    last_2_dim_hops = dw.hop_inds.reshape(base_lat_dim+[9])[0,0,:,:,4:9]


    print("hops from each point of a",base_lat_dim[2:],"grid onto a",in_lat_dim[2:],"grid\n")

    for x in last_2_dim_hops:
        row1 = f""
        row2 = f""
        row3 = f""
        for y in x:
            row1 += f"    {int(y[0]):>2}     "
            row2 += f" {int(y[2]):>2} {int(y[4]):>2} {int(y[3]):>2}  "
            row3 += f"    {int(y[1]):>2}     "
        print(row1)
        print(row2)
        print(row3)
        print()

    print("missing sites: ", end="")
    for x in range(in_lat_dim[2]*in_lat_dim[3]):
        if not torch.any(x == last_2_dim_hops):
            print(x, end=" ")
    print("\n")
