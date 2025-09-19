import torch
import numpy as np

from .settings import capab
from .util import get_hop_indices

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
    def tempdir_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_tempdir_mtsg_tmgsMhs(self.U, v, self.mass_parameter)
    
    def all_calls(self):
        return [self.xtsgMhm, self.xtMmghs, self.xtMmdghs, self.block_dbxtsghm, self.block_bxtsghm] + ([self.tempdir_tmgsMhs] if capab("vectorise") else [])
    def all_call_names(self):
        return ["xtsgMhm", "xtMmghs", "xtMmdghs", "block_dbxtsghm", "block_bxtsghm"] + (["tempdir_tmgsMhs"] if capab("vectorise") else [])



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
    The normal boundary phases are 1 for all directions, but can be set to 1 or -1.
    Most calls currently do not take different boundary phases
    """
    def __init__(self, U: torch.Tensor, mass_parameter, boundary_phases=[1,1,1,1]):
        self.U = U
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.mass_parameter = mass_parameter

        op_device = U.device
        # device of U must be the device of all other tensors

        grid = U.shape[1:5]
        self.hop_inds = get_hop_indices(grid).to(op_device)

        # for every grid point, the phase for a hop in all 8 directions (negative and positive)
        hop_phases = torch.ones(list(grid)+[8], dtype=torch.int8)
        # for the sites at the lower boundary [0], a hop in negative direction has the phase
        # for the sites at the upper boundary [-1], a hop in positive direction has the phase
        for edge in range(2):
            hop_phases[-edge,:,:,:,0+edge] = boundary_phases[0]
            hop_phases[:,-edge,:,:,2+edge] = boundary_phases[1]
            hop_phases[:,:,-edge,:,4+edge] = boundary_phases[2]
            hop_phases[:,:,:,-edge,6+edge] = boundary_phases[3]
        self.hop_phases = hop_phases.to(op_device)

        # different implementation of the phases: multiply the gauge field at the boundary with the phase
        if all([phf == 1 for phf in boundary_phases]):
            self.phase_U = U
        else:
            phase_U = U.clone()
            phase_U[0,-1] *= boundary_phases[0]
            phase_U[1,:,-1] *= boundary_phases[1]
            phase_U[2,:,:,-1] *= boundary_phases[2]
            phase_U[3,:,:,:,-1] *= boundary_phases[3]
            self.phase_U = phase_U
        
        self.axes_even = all([ax%2 == 0 for ax in grid])
        """Denotes if all space-time axes have even length, for blocked operator."""
        

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
    def tmsgMh(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_tmsgMh.default(self.U, v, self.hop_inds,
                                                         self.mass_parameter)
    def block_btmsgMh(self, v):
        return torch.ops.qmad_history.dw_hop_block_mtsg_btmsgMh(self.U, v, self.hop_inds,
                                                                self.mass_parameter)
    def avx_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_avx_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                          self.mass_parameter)
    def avx_tmsgMhs(self, v):
        return torch.ops.qmad_history.dw_avx_mtsg_tmsgMhs(self.U, v, self.hop_inds,
                                                          self.mass_parameter)
    def templ_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_templ_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                            self.mass_parameter)
    def tempipe_tmgsMhs(self, v):
        return torch.ops.qmad_history.dw_tempipe_mtsg_tmgsMhs(self.U, v, self.hop_inds,
                                                            self.mass_parameter)
    def templ_tmsgMhs(self, v):
        return torch.ops.qmad_history.dw_templ_mtsg_tmsgMhs(self.U, v, self.hop_inds,
                                                            self.mass_parameter)
    def templbound_tmsgMhs(self, v):
        return torch.ops.qmad_history.dw_templbound_mtsg_tmsgMhs(self.U, v, self.hop_inds, self.hop_phases,
                                                            self.mass_parameter)
    def templUbound_tmsgMhs(self, v):
        return torch.ops.qmad_history.dw_templ_mtsg_tmsgMhs(self.phase_U, v, self.hop_inds,
                                                            self.mass_parameter)
    def cuv2(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv2.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv3(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv3.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv4(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv4.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv5(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv5.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv6(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv6.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv7(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv7.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv8(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv8.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cuv9(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cuv9.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cu_tsg(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cu_tsg.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cu_Mtmsg(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cu_Mtmsg.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cu_Mtmsgh(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cu_Mtmsgh.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    def cu_3d_tsg(self, v):
        return torch.ops.qmad_history.dw_hop_mtsg_cu_3d_tsg.default(self.U, v, self.hop_inds,
                                                               self.mass_parameter)
    
    def all_calls(self):
        return [self.tMmgsh, self.tMgshm, self.tmgsMh, self.tmsgMh] + (
            ([self.block_btmsgMh] if self.axes_even else [])
            + ([self.avx_tmgsMhs, self.avx_tmsgMhs, self.templ_tmgsMhs, self.tempipe_tmgsMhs, self.templ_tmsgMhs, self.templbound_tmsgMhs, self.templUbound_tmsgMhs] if capab("vectorise") else [])
            + ([self.cuv2,] if torch.cuda.is_available() else [])
            )
    def all_call_names(self):
        return ["tMmgsh", "tMgshm", "tmgsMh", "tmsgMh"] + (
            (["block_btmsgMh"] if self.axes_even else [])
            + (["avx_tmgsMhs", "avx_tmsgMhs", "templ_tmgsMhs", "tempipe_tmgsMhs", "templ_tmsgMhs", "templbound_tmsgMhs", "templUbound_tmsgMhs"] if capab("vectorise") else [])
            + (["cuv2",] if torch.cuda.is_available() else [])
            )


class wilson_hop_eo:
    """
    Dirac Wilson operator with gauge config U on even-odd checkerboard that precomputes the hop addresses.
    The input U still has the axes U[mu,x,y,z,t,g,h].
    The call takes one even and one odd U and v each, which have the t axis split in half.
    The input v is a list of even and odd v.
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
        self.eodim = eodim
        """Dimensions of the even and odd lattices."""
        eovol = eodim[0]*eodim[1]*eodim[2]*eodim[3]
        self.emask = emask
        """Boolean tensor to select all even sites, with the t axis being halved compared to the base grid."""
        self.omask = omask
        """Boolean tensor to select all odd sites, with the t axis being halved compared to the base grid."""
        self.Ue = U[:,emask]
        self.Uo = U[:,omask]
        assert self.Ue.shape == (4,eovol,3,3) and self.Uo.shape == (4,eovol,3,3)
        self.mass_parameter = mass_parameter

        # create hops for sites on even and odd grid; their neighbours are on the opposite grid
        # this has to be on CPU, integer arithmetic is not implemented on GPU
        strides = torch.tensor([eodim[1]*eodim[2]*eodim[3], eodim[2]*eodim[3], eodim[3], 1], dtype=torch.int32)
        indices = torch.tensor(np.indices(eodim, sparse=False), dtype=torch.int32)
        indices = torch.permute(indices, (1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

        # +1 step in t direction: if x+y+z and current grid have same parity, it has the same address
        # -1 step in t direction: if x+y+z and current grid have opposite parity, it has the same address
        # the other steps are the same
        # we compute a tensor that contains the correct t step for each site and parity
        eostep = torch.empty([2,eovol], dtype=torch.int32)
        xyz_select = torch.tensor([1,1,1,0], dtype=torch.int32)
        eostep[0] = torch.remainder(torch.matmul(indices, xyz_select), 2)
        eostep[1] = torch.remainder(torch.matmul(indices, xyz_select)+1, 2)
        assert torch.all(torch.logical_or(eostep == 0, eostep == 1))

        hop_inds = [[],[]]
        for evod in range(2):
            # xyz hops are the same
            for coord in range(3):
                # index after a negative step in coord direction
                minus_hop_ind = torch.clone(indices)
                minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]-1+eodim[coord], eodim[coord])
                # index after a positive step in coord direction
                plus_hop_ind = torch.clone(indices)
                plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+1, eodim[coord])
                # compute flattened index by dot product with strides
                hop_inds[evod].append(torch.matmul(minus_hop_ind, strides))
                hop_inds[evod].append(torch.matmul(plus_hop_ind, strides))

            # t hops
            coord = 3
            minus_hop_ind = torch.clone(indices)
            minus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+eostep[evod]-1+eodim[coord], eodim[coord])
            plus_hop_ind = torch.clone(indices)
            plus_hop_ind[:,coord] = torch.remainder(indices[:,coord]+eostep[evod], eodim[coord])
            # compute flattened index by dot product with strides
            hop_inds[evod].append(torch.matmul(minus_hop_ind, strides))
            hop_inds[evod].append(torch.matmul(plus_hop_ind, strides))
        
        self.hop_inds_e = torch.stack(hop_inds[0], dim=1).contiguous()
        self.hop_inds_o = torch.stack(hop_inds[1], dim=1).contiguous()
        assert torch.all(self.hop_inds_e < eovol) and torch.all(self.hop_inds_e >= 0)
        assert torch.all(self.hop_inds_o < eovol) and torch.all(self.hop_inds_o >= 0)
        assert self.hop_inds_e.shape == (eovol, 8) and self.hop_inds_o.shape == (eovol, 8)

        
    
    def __str__(self):
        return "dw_hop_eo_pmtsg"
    
    def ptMmsgh(self, v):
        return torch.ops.qmad_history.dw_eo_hop_pmtsg_ptMmsgh(self.Ue, self.Uo, v[0], v[1],
                                                              self.hop_inds_e, self.hop_inds_o,
                                                              self.mass_parameter)
    def pMtmsgh(self, v):
        return torch.ops.qmad_history.dw_eo_hop_pmtsg_pMtmsgh(self.Ue, self.Uo, v[0], v[1],
                                                              self.hop_inds_e, self.hop_inds_o,
                                                              self.mass_parameter)
    
    def all_calls(self):
        return [self.pMtmsgh, self.ptMmsgh]
    def all_call_names(self):
        return ["pMtmsgh", "ptMmsgh"]


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
        self.hop_inds = get_hop_indices(grid)

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
        return [self.tMmghs, self.tMmgsh, self.tMmgshu, self.tmgsMh] + ([self.avx_tmgsMhs] if capab("vectorise") else [])
    def all_call_names(self):
        return ["tMmghs", "tMmgsh", "tMmgshu", "tmgsMh"] + (["avx_tmgsMhs"] if capab("vectorise") else [])


class wilson_hop_mtsgt:
    """
    Dirac Wilson operator that creates a lookup table for the hops.
    The axes are U[mu,x,y,z,t1,g,h,t2] and v[x,y,z,t1,s,h,t2].
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        # t axis needs to be divisible by register length
        tlen = U.shape[4]
        assert tlen%2 == 0
        self.mass_parameter = mass_parameter

        # transform U to have neighboring t sites as last index
        Ueven = U[:,:,:,:,0:tlen:2]
        Uodd = U[:,:,:,:,1:tlen:2]
        self.U = torch.stack([Ueven, Uodd], dim=-1)

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]//2]
        self.hop_inds = get_hop_indices(grid)

    def __str__(self):
        return "dw_hop_mtsgt"
    
    def templ_tmgsMht(self, v):
        return torch.ops.qmad_history.dw_templ_mtsgt_tmgsMht(self.U, v, self.hop_inds,
                                                             self.mass_parameter)
    
    def all_calls(self):
        return [] + ([self.templ_tmgsMht] if capab("vectorise") else [])
    def all_call_names(self):
        return [] + (["templ_tmgsMht"] if capab("vectorise") else [])


class wilson_hop_mtsgt2:
    """
    Dirac Wilson operator that creates a lookup table for the hops.
    The axes are U[mu,x,y,z,t2,g,h,t1] and v[x,y,z,t2,s,h,t1].
    The grid gets split in half, with a register having 1 site from each block.
    t1 is the block number, t2 is the site.
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        # t axis needs to be divisible by register length
        tlen = U.shape[4]
        assert tlen%2 == 0
        self.mass_parameter = mass_parameter

        # transform U to have t sites from each half as last index
        Ufirst = U[:,:,:,:,0:tlen//2]
        Usecond = U[:,:,:,:,tlen//2:tlen]
        self.U = torch.stack([Ufirst, Usecond], dim=-1)

        # # add halo (additional array entries after the boundary for hops in t direction)
        # halo_grid = list(Usimd.shape)
        # halo_grid[3] += 2
        # self.U = torch.tensor(halo_grid, dtype=torch.cdouble)
        # self.U[:,:,:,1:-1] = Usimd
        # self.U[:,:,:,0] = Usimd[:,:,:,-1]
        # self.U[:,:,:,-1] = Usimd[:,:,:,0]

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]//2]
        self.hop_inds = get_hop_indices(grid)

    def __str__(self):
        return "dw_hop_mtsgt2"
    
    def grid_tmgsMht(self, v):
        return torch.ops.qmad_history.dw_grid_mtsgt2_tmgsMht(self.U, v, self.hop_inds,
                                                             self.mass_parameter)
    
    def all_calls(self):
        return [] + ([self.grid_tmgsMht] if capab("vectorise") else [])
    def all_call_names(self):
        return [] + (["grid_tmgsMht"] if capab("vectorise") else [])


class wilson_full:
    """
    Wilson Dirac operator that computes the full matrix in space-time-spinor-colour space.
    Currently, it just does a dummy computation that uses the wrong numbers.
    """
    def __init__(self, U: torch.Tensor, mass_parameter, boundary_phases=[1,1,1,1]):
        U
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.mass_parameter = mass_parameter

        op_device = U.device
        # device of U must be the device of all other tensors

        grid = U.shape[1:5]
        vol = grid[0]*grid[1]*grid[2]*grid[3]
        hop_inds = get_hop_indices(grid)
        assert tuple(hop_inds.shape) == (vol,8,)

        self.dummy_dw = torch.randn([vol,4,3,49],dtype=torch.cdouble,device=op_device)

        # computation taken directly (very slow)
        # gamx = [[3,2,1,0],[3,2,1,0],[2,3,0,1],[2,3,0,1]]
        # self.sparse_addr = torch.empty([vol,4,3,49],dtype=torch.int32,device=op_device)
        # for t in range(vol):
        #     for s in range(4):
        #         for g in range(3):
        #             self.sparse_addr[t,s,g,0] = t*12+s*3+g
        #             for mu in range(8):
        #                 for gi in range(3):
        #                     self.sparse_addr[t,s,g,1+mu*6+gi] = hop_inds[t,mu]*12+s*3+gi
        #                     self.sparse_addr[t,s,g,1+mu*6+gi+3] = hop_inds[t,mu]*12+gamx[mu//2][s]*3+gi
        # assert torch.all(self.sparse_addr>=0) and torch.all(self.sparse_addr<vol*4*3)

        # the sparse address tensor will have the indices t,s,g,contributions
        # the contributions will be: first the mass term, then hops in order mu,dir,gamma/no gamma,gi
        # first the mass term, which is simply the address of each site at that same site
        mass_ind_np = np.indices([vol,4,3],sparse=False)
        mass_ind = torch.tensor(mass_ind_np, dtype=torch.int32).permute((1,2,3,0,))
        tsg_strides = torch.tensor([12,3,1], dtype=torch.int32)
        mass_ind = torch.matmul(mass_ind,tsg_strides)
        mass_ind = torch.reshape(mass_ind, [vol,4,3,1])
        assert tuple(mass_ind.shape) == (vol,4,3,1,)
        # now the hop term contributions
        # hop_inds already is in order t,mu,dir (mu,dir are one axis)
        # so we first take the correct hop address by broadcasting hop_inds to the sparse shape
        # multiplied by 12, as hop_inds takes sites but we take spin-colour components
        sparse2 = torch.broadcast_to(hop_inds,[4,3,2,3,vol,8])*12
        # gamx is brought to the mu,dir,s shape
        gamx_tensor = torch.tensor([[3,2,1,0]]*4+[[2,3,0,1]]*4,dtype=torch.int32)
        s_tensor = torch.tensor([0,1,2,3],dtype=torch.int32)
        # permute sparse2 so we can add gamx
        sparse2 = torch.permute(sparse2, (2,4,1,3,5,0))
        # multiplied by 3, as gamx takes spin components but we take spin-colour components
        sparse2[0] += s_tensor*3
        sparse2[1] += gamx_tensor*3
        # now reshape to add gi term
        sparse2 = torch.permute(sparse2, (1,5,2,4,0,3))
        gi_tensor = torch.tensor([0,1,2],dtype=torch.int32)
        sparse2 += gi_tensor
        assert tuple(sparse2.shape) == (vol,4,3,8,2,3)
        sparse2 = torch.reshape(sparse2, [vol,4,3,48])
        sparse_addr_2 = torch.cat([mass_ind, sparse2],dim=3).contiguous()
        self.sparse_addr = sparse_addr_2.to(op_device)
         
        
    def cuv10(self, v):
        return torch.ops.qmad_history.dw_full_cuv10.default(self.dummy_dw, v, self.sparse_addr)
    def cuv11(self, v):
        return torch.ops.qmad_history.dw_full_cuv11.default(self.dummy_dw, v, self.sparse_addr)



