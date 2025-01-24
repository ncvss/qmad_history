import torch
import numpy as np

from .settings import capab

# import os

# this_dir = os.path.dirname(os.path.curdir)
# settings_file = os.path.join(this_dir, "qmad_history", "settings.txt")

# capab = dict()

# # take the settings from the text file
# with open(settings_file, "r") as sfile:
#     data = sfile.readlines()
#     for li in data:
#         wo = li.split()
#         capab[wo[0]] = bool(wo[1])

class wilson_direct:
    """
    Dirac Wilson operator with gauge config U, without precomputations,
    memory layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    Also includes versions that split space-time into blocks.
    """
    def __init__(self, U, mass_parameter):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter

    def __str__(self):
        return "dw_dir_mxtsg"

    def xtsgMhm(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtsgMhm(self.U, v, self.mass_parameter)
    def xtMmghs(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtMmghs(self.U, v, self.mass_parameter)
    def xtMmdghs(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtMmdghs(self.U, v, self.mass_parameter)
    def block_dbxtsghm(self, v, blocksize=4):
        return torch.ops.qmad_history.dw_block_mxtsg_dbxtsghm(self.U, v, self.mass_parameter, blocksize)
    def block_bxtsghm(self, v, blocksize=4):
        return torch.ops.qmad_history.dw_block_mxtsg_bxtsghm(self.U, v, self.mass_parameter, blocksize)
    
    def all_calls(self):
        return [self.xtsgMhm, self.xtMmghs, self.xtMmdghs, self.block_dbxtsghm, self.block_bxtsghm]
    def all_call_names(self):
        return ["xtsgMhm", "xtMmghs", "xtMmdghs", "block_dbxtsghm", "block_bxtsghm"]



class wilson_eo:
    """
    Dirac Wilson operator with gauge config U on even-odd checkerboard.
    """
    def __init__(self, U, mass_parameter):
        # the dimensions have to have even sizes for the algorithm to work
        dims = list(U.shape)[1:5]
        for d in dims:
            if d%2 != 0:
                raise Exception("Grid has to have even number of points in each dimension")
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        
        # choose the even and odd sites in the gauge fields
        emask = torch.tensor([[[[(x+y+z+t)%2 == 0 for t in range(dims[3])] for z in range(dims[2])]
                            for y in range(dims[1])] for x in range(dims[0])], dtype=torch.bool)
        omask = torch.logical_not(emask)

        eodim = dims[:]
        eodim[-1] //= 2
        self.Ue = U[:,emask]
        self.Uo = U[:,omask]
        self.mass_parameter = mass_parameter
        self.eodim = eodim
        self.emask = emask
        self.omask = omask
    
    def __str__(self):
        return "dw_eo_pmtsg"

    def pxtMmghs(self, vboth):
        return torch.ops.qmad_history.dw_eo_pmtsg_pxtMmghs(self.Ue, self.Uo, vboth[0], vboth[1], self.mass_parameter, self.eodim)

    def all_calls(self):
        return [self.pxtMmghs]
    def all_call_names(self):
        return ["pxtMmghs"]


class wilson_hop_mtsg:
    """
    Dirac Wilson operator that creates a lookup table for the hops.
    The axes are U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    """
    def __init__(self, U, mass_parameter):
        self.U = U
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.mass_parameter = mass_parameter

        grid = U.shape[1:5]
        strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
        npind = np.indices(grid, sparse=False)
        indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))
        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()

    def __str__(self):
        return "dw_hop_mtsg"

    def tMmgsh(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_tMmgsh(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def tMgshm(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_tMgshm(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def tmgsMh(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_tmgsMh(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def avx_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_avx_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                          self.mass_parameter)
    def templ_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_templ_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                            self.mass_parameter)
    
    def all_calls(self):
        return [self.tMmgsh, self.tMgshm, self.tmgsMh] + ([self.avx_tmgsMhs, self.templ_tmgsMhs] if capab["vectorise"] else [])
    def all_call_names(self):
        return ["tMmgsh", "tMgshm", "tmgsMh"] + (["avx_tmgsMhs", "templ_tmgsMhs"] if capab["vectorise"] else [])


class wilson_hop_tmgs:
    """
    Dirac Wilson operator that creates a lookup table for the hops.
    The axes are U[x,y,z,t,mu,g,h] and v[x,y,z,t,h,s].
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter):
        U = torch.permute(U, (1,2,3,4,0,5,6)).contiguous()
        assert tuple(U.shape[4:7]) == (4,3,3,)
        self.U = U
        self.mass_parameter = mass_parameter

        grid = U.shape[0:4]
        strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
        npind = np.indices(grid, sparse=False)
        indices = torch.tensor(npind, dtype=torch.int32).permute((1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        hop_inds = []
        for coord in range(4):
            # index after a negative step in coord direction
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+grid[coord], grid[coord])
            # index after a positive step in coord direction
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, grid[coord])
            # compute flattened index by dot product with strides
            hop_inds.append(torch.matmul(minus_hop_ind, strides))
            hop_inds.append(torch.matmul(plus_hop_ind, strides))
        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous()

    def __str__(self):
        return "dw_hop_tmgs"
    
    def tMmghs(self, v):
        return torch.ops.qmad_history.dw_hop_tmgs_tMmghs(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def tMmgsh(self, v):
        return torch.ops.qmad_history.dw_hop_tmgs_tMmgsh(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def tMmgshu(self, v):
        return torch.ops.qmad_history.dw_hop_tmgs_tMmgshu(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def tmgsMh(self, v):
        return torch.ops.qmad_history.dw_hop_tmgs_tmgsMh(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def avx_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_avx_tmgs_tmgsMhs(self.U, v, self.hop_inds,
                                                          self.mass_parameter)
    
    def all_calls(self):
        return [self.tMmghs, self.tMmgsh, self.tMmgshu, self.tmgsMh] + ([self.avx_tmgsMhs] if capab["vectorise"] else [])
    def all_call_names(self):
        return ["tMmghs", "tMmgsh", "tMmgshu", "tmgsMh"] + (["avx_tmgsMhs"] if capab["vectorise"] else [])


