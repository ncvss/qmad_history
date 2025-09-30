import torch
import numpy as np

from .settings import capab

class wilson_hop_mtsg_roofline:
    """
    Wilson Dirac operator that creates a lookup table for the hops.
    The memory layout is U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    
    U and v may be of a different size, so that there is less data transfered and thus the flop/byte
    can be varied, which allows us to fit to the roofline model.
    (The computation and thus the amount of flop is always the same, so the flop/s
    can be computed by measuring the time.)

    base_grid is the space-time of the result tensor (which is output in a flattened shape).
    """
    def __init__(self, U, mass_parameter, base_grid):
        self.U = U
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.mass_parameter = mass_parameter

        var_grid = list(U.shape[1:5])
        strides = torch.tensor([var_grid[1]*var_grid[2]*var_grid[3],
                                var_grid[2]*var_grid[3],
                                var_grid[3],
                                1],
                               dtype=torch.int32)
        
        npind = np.indices(base_grid, sparse=False)
        base_indices = torch.tensor(npind).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)
        
        # via the hop lookup table, for every site in the base grid, sites in U and v are accessed
        # we map every base grid point to a U grid point by shrinking or expanding to the right size
        # for this, the indices are multiplied by the ratio, and rounded down
        grid_ratio = torch.tensor([var_grid[i]/base_grid[i] for i in range(4)])
        indices = torch.floor(base_indices*grid_ratio).to(torch.int32)

        # the hops are computed on the U grid in the same way as before
        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+var_grid[coord], var_grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, var_grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))
        
        # this function also needs a no-hop address
        hop_inds.append(torch.matmul(indices,strides))

        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()

    def __str__(self):
        return "dw_roofline_hop_mtsg"

    def templ_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_roof_templ_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                                 self.mass_parameter)
    
    def all_calls(self):
        return [self.templ_tmgsMhs] if capab("vectorise") else []
    def all_call_names(self):
        return ["templ_tmgsMhs"] if capab("vectorise") else []
