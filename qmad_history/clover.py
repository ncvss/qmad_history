import torch

from .settings import capab
from .util import get_hop_indices, get_clover_path_matrices, _PathBufferTemp, _sigma, _triag_mask_1, _triag_mask_2



class wilson_clover_direct_false:
    """
    Wilson clover Dirac operator with gauge config U, without precomputations,
    memory layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    The result is wrong because the computation is incomplete, it is here for performance.
    """
    def __init__(self, U, mass_parameter, csw):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

    def __str__(self):
        return "dwc_dir_mxtsg_false"
    
    def fcall(self, v):
        # alternate name: xtMmghijkls
        return torch.ops.qmad_history.dwc_dir_mxtsg_false(self.U, v, self.mass_parameter, self.csw)
    
    def all_calls(self):
        return [self.fcall]
    def all_call_names(self):
        return ["xtMmghijkls"]


class wilson_clover_hop:
    """
    Wilson clover Dirac operator that creates a lookup table for the hops, and makes no precomputation.
    The axes are U[mu,x,y,z,t,g,gi], v[x,y,z,t,s,gi].
    """
    def __init__(self, U, mass_parameter, csw):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U

        self.mass_parameter = mass_parameter
        self.csw = csw

        grid = U.shape[1:5]
        self.hop_inds = get_hop_indices(grid)
    
    def __str__(self):
        return "dwc_hop_mtsg"
    
    def tmsgMh_dir(self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_tmsgMh_dir(self.U, v, self.hop_inds,
                                                              self.mass_parameter, self.csw)
    def tmsgMh_dir_rearr(self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_tmsgMh_dir_rearr(self.U, v, self.hop_inds,
                                                                    self.mass_parameter, self.csw)
    
    def all_calls(self):
        return [self.tmsgMh_dir, self.tmsgMh_dir_rearr]
    def all_call_names(self):
        return ["tmsgMh_dir", "tmsgMh_dir_rearr"]



class wilson_clover_fpre:
    """
    Wilson clover Dirac operator with gauge config U that precomputes field strength matrices,
    with memory layout U[mu,x,y,z,t,g,h], v[x,y,z,t,s,h] and F[munu][x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter, csw):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Qmunu = get_clover_path_matrices(U)
        
        # only a flat list, as it needs to be accessed by C++
        self.field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                self.field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        assert len(self.field_strength) == 6
        for F in self.field_strength:
            assert tuple(F.shape[4:6]) == (3,3)
    
    def __str__(self):
        return "dwc_fpre_mntsg"
        
    def xtsghmn (self, v):
        return torch.ops.qmad_history.dwc_fpre_mntsg_xtsghmn(self.U, v, self.field_strength,
                                                             self.mass_parameter, self.csw)
    def xtmghsn (self, v):
        return torch.ops.qmad_history.dwc_fpre_mntsg_xtmghsn(self.U, v, self.field_strength,
                                                             self.mass_parameter, self.csw)
    def xtmnghs (self, v):
        return torch.ops.qmad_history.dwc_fpre_mntsg_xtmnghs(self.U, v, self.field_strength,
                                                             self.mass_parameter, self.csw)
    def xtmdnghs (self, v):
        return torch.ops.qmad_history.dwc_fpre_mntsg_xtmdnghs(self.U, v, self.field_strength,
                                                             self.mass_parameter, self.csw)
    
    def all_calls(self):
        return [self.xtsghmn, self.xtmghsn, self.xtmnghs, self.xtmdnghs]
    def all_call_names(self):
        return ["xtsghmn", "xtmghsn", "xtmnghs", "xtmdnghs"]


class wilson_clover_sigpre:
    """
    Wilson clover Dirac operator with gauge config U, layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    
    field_strength * sigma * v is precomputed by computing the tensor product of
    field_strength * sigma.
    """
    def __init__(self, U, mass_parameter, csw):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Qmunu = get_clover_path_matrices(U)

        dim = list(U.shape[1:5])
        self.dim = dim
        # tensor product of the sigma matrix and field strength tensor
        field_strength_sigma = torch.zeros(dim+[4,3,4,3], dtype=torch.cdouble)
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                Fmunu = (Qmunu[mu][nu] - Qmunu[nu][mu]) / 8
                Fsigma = torch.einsum('xyztgh,sr->xyztsgrh',Fmunu,_sigma[mu][nu])
                field_strength_sigma += 2*Fsigma
        
        self.field_strength_sigma = field_strength_sigma.contiguous().reshape(dim+[12,12])
    
    def __str__(self):
        return "dwc_sigpre_mxtsg"
        
    def xtMmdghs (self, v):
        return (torch.ops.qmad_history.dw_dir_mxtsg_xtMmdghs(self.U, v, self.mass_parameter)
                - self.csw/4 * torch.matmul(self.field_strength_sigma,v.reshape(self.dim+[12,1])).reshape(self.dim+[4,3]))

    def all_calls(self):
        return [self.xtMmdghs]
    def all_call_names(self):
        return ["xtMmdghs"]


class wilson_clover_hop_mtsg:
    """
    Wilson clover Dirac operator that creates a lookup table for the hops and uses AVX instructions.
    The axes are U[mu,x,y,z,t,g,gi], v[x,y,z,t,s,gi] and F[x,y,z,t,munu,g,gi].
    """
    def __init__(self, U, mass_parameter, csw):
        self.U = U
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4

        self.mass_parameter = mass_parameter
        self.csw = csw

        op_device = U.device
        # device of U must be the device of all other tensors

        grid = U.shape[1:5]
        self.hop_inds = get_hop_indices(grid).to(op_device)
        
        Qmunu = get_clover_path_matrices(U)
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4).to(op_device)
        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)
    
    def __str__(self):
        return "dwc_hop_mtsg_fpre"
    
    def tmsgMhn(self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_tmsgMhn_fpre(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    def avx_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_avx_mtsg_tmgsMhns(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    def avx_tmsgMhns(self, v):
        return torch.ops.qmad_history.dwc_avx_mtsg_tmsgMhns(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    def templ_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_templ_mtsg_tmgsMhns(self.U, v, self.field_strength,
                                                              self.hop_inds, self.mass_parameter,
                                                              self.csw)
    def templ_tmsgMhns(self, v):
        return torch.ops.qmad_history.dwc_templ_mtsg_tmsgMhns(self.U, v, self.field_strength,
                                                              self.hop_inds, self.mass_parameter,
                                                              self.csw)
    def cu_tsg(self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_cu_tsg_fpre(self.U, v, self.field_strength,
                                                              self.hop_inds, self.mass_parameter,
                                                              self.csw)
    def debug_cuda(self, v):
        return torch.ops.qmad_history.dwc_debug_cuda_fpre(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    
    def all_calls(self):
        return [self.tmsgMhn, self.debug_cuda] + ([self.avx_tmgsMhns, self.avx_tmsgMhns, self.templ_tmgsMhns, self.templ_tmsgMhns] if capab("vectorise") else [])
    def all_call_names(self):
        return ["tmsgMhn", "debug_cuda"] + (["avx_tmgsMhns", "avx_tmsgMhns", "templ_tmgsMhns", "templ_tmsgMhns"] if capab("vectorise") else [])



class wilson_clover_hop_tmgs:
    """
    Wilson clover Dirac operator that creates a lookup table for the hops and uses AVX instructions.
    The axes are U[x,y,z,t,mu,g,gi], v[x,y,z,t,gi,s] and F[x,y,z,t,munu,g,gi].
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter, csw):
        self.U = torch.permute(U, (1,2,3,4,0,5,6)).contiguous()
        assert tuple(self.U.shape[4:7]) == (4,3,3,)

        self.mass_parameter = mass_parameter
        self.csw = csw

        grid = U.shape[1:5]
        self.hop_inds = get_hop_indices(grid)

        Qmunu = get_clover_path_matrices(U)
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4)
        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)

    def __str__(self):
        return "dwc_hop_tmgs"

    def avx_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_avx_tmgs_tmgsMhns(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    
    def all_calls(self):
        return [self.avx_tmgsMhns] if capab("vectorise") else []
    def all_call_names(self):
        return ["avx_tmgsMhns"] if capab("vectorise") else []


class wilson_clover_hop_mtsgt_sigpre:
    """
    Wilson clover Dirac operator with gauge config U, layout U[mu,x,y,z,t1,g,h,t2] and v[x,y,z,t1,s,h,t2].
    field_strength * sigma * v is precomputed by computing the tensor product of field_strength * sigma,
    and only the upper triangle of two 6x6 blocks is passed for the field strength.
    The hops are precomputed.
    """
    def __init__(self, U, mass_parameter, csw):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        # t axis needs to be divisible by register length
        tlen = U.shape[4]
        assert tlen%2 == 0
        self.mass_parameter = mass_parameter
        self.csw = csw

        # transform U to have neighboring t sites as last index
        Ueven = U[:,:,:,:,0:tlen:2]
        Uodd = U[:,:,:,:,1:tlen:2]
        self.U = torch.stack([Ueven, Uodd], dim=-1)

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]//2]
        self.hop_inds = get_hop_indices(grid)
        

        Qmunu = get_clover_path_matrices(U)

        dim = list(U.shape[1:5])
        self.dim = dim
        # tensor product of the sigma matrix and field strength tensor
        field_strength_sigma = torch.zeros(dim+[4,3,4,3], dtype=torch.cdouble)
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                Fmunu = (Qmunu[mu][nu] - Qmunu[nu][mu]) / 8
                Fsigma = torch.einsum('xyztgh,sr->xyztsgrh',Fmunu,_sigma[mu][nu])
                # csw gets absorbed into the matrices
                field_strength_sigma += 2*(-self.csw/4)*Fsigma
        
        field_strength_sigma = field_strength_sigma.contiguous().reshape(dim+[12,12])

        # transform to grid layout (t register as last index)
        fs_even_t = field_strength_sigma[:,:,:,0:dim[3]:2]
        fs_odd_t = field_strength_sigma[:,:,:,1:dim[3]:2]
        field_strength_sigma = torch.stack([fs_even_t, fs_odd_t], dim=-1)

        # until here, the computation works as expected
        # this is hermitian and has two 6x6 blocks (diagonal has numerical artifacts)
        # print(fs_sigma_grid[0,0,2,2,0:6])

        self.field_strength_sigma = torch.stack([field_strength_sigma[:,:,:,:,_triag_mask_1],
                                                 field_strength_sigma[:,:,:,:,_triag_mask_2]],dim=4)
        assert tuple(self.field_strength_sigma.shape[4:7]) == (2,21,2,)

        # this is also the expected 
        # print(self.fs[0,0,2,2,0])

    
    def __str__(self):
        return "dwc_sigpre_hop_mtsgt"
        
    def tmngsMht (self, v):
        return torch.ops.qmad_history.dwc_templ_mtsgt_tmngsMht(self.U, v, self.field_strength_sigma,
                                                               self.hop_inds, self.mass_parameter, self.csw)
        #        - self.csw/4 * torch.matmul(self.field_strength_sigma,v.reshape(self.dim+[12,1])).reshape(self.dim+[4,3]))

    def all_calls(self):
        return [] + ([self.tmngsMht] if capab("vectorise") else [])
    def all_call_names(self):
        return [] + (["tmngsMht"] if capab("vectorise") else [])


class wilson_clover_hop_mtsg_sigpre:
    """
    Wilson clover Dirac operator with gauge config U, layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    field_strength * sigma * v is precomputed by computing the tensor product of field_strength * sigma,
    and only the upper triangle of two 6x6 blocks is passed for the field strength.
    The hops are precomputed.
    """
    def __init__(self, U, mass_parameter, csw, boundary_phases=[1,1,1,1]):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.mass_parameter = mass_parameter
        self.csw = csw
        self.U = U

        op_device = U.device
        # device of U must be the device of all other tensors
        dev_sigma = _sigma.to(op_device)
        dev_triag_mask_1 = _triag_mask_1.to(op_device)
        dev_triag_mask_2 = _triag_mask_2.to(op_device)

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]]
        self.hop_inds = get_hop_indices(grid).to(op_device)
        

        Qmunu = get_clover_path_matrices(U)

        dim = list(U.shape[1:5])
        self.dim = dim
        # tensor product of the sigma matrix and field strength tensor
        field_strength_sigma = torch.zeros(dim+[4,3,4,3], dtype=torch.cdouble, device=op_device)
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                Fmunu = (Qmunu[mu][nu] - Qmunu[nu][mu]) / 8
                Fsigma = torch.einsum('xyztgh,sr->xyztsgrh',Fmunu,dev_sigma[mu][nu])
                # csw gets absorbed into the matrices
                field_strength_sigma += 2*(-self.csw/4)*Fsigma
        
        field_strength_sigma = field_strength_sigma.contiguous().reshape(dim+[12,12])

        # this should be hermitian and have two 6x6 blocks (diagonal has numerical artifacts)
        # print(field_strength_sigma[0,0,2,2,0:6])

        self.field_strength_sigma = torch.stack([field_strength_sigma[:,:,:,:,dev_triag_mask_1],
                                                 field_strength_sigma[:,:,:,:,dev_triag_mask_2]],dim=-1)
        assert tuple(self.field_strength_sigma.shape[4:6]) == (21,2,)

        # print(self.fs[0,0,2,2])

        # implementation of the phases: multiply the gauge field at the boundary with the phase
        if all([phf == 1 for phf in boundary_phases]):
            self.phase_U = U
        else:
            phase_U = U.clone()
            phase_U[0,-1] *= boundary_phases[0]
            phase_U[1,:,-1] *= boundary_phases[1]
            phase_U[2,:,:,-1] *= boundary_phases[2]
            phase_U[3,:,:,:,-1] *= boundary_phases[3]
            self.phase_U = phase_U

    def __str__(self):
        return "dwc_sigpre_hop_mtsg"
    

    def tmnsgMh (self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_tmnsgMh_sigpre(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)
    def avx_tmnsgMhs (self, v):
        return torch.ops.qmad_history.dwc_avx_mtsg_tmnsgMhs_sigpre(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)
    def tmngsMhs (self, v):
        return torch.ops.qmad_history.dwc_grid_mtsg_tmngsMhs(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)
        #        - self.csw/4 * torch.matmul(self.field_strength_sigma,v.reshape(self.dim+[12,1])).reshape(self.dim+[4,3]))
    def tmnsgMhs (self, v):
        return torch.ops.qmad_history.dwc_grid_mtsg_tmnsgMhs(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)
    def Ubound_tmngsMhs (self, v):
        return torch.ops.qmad_history.dwc_grid_mtsg_tmngsMhs(self.phase_U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)
    def cu_tsg_tn (self, v):
        return torch.ops.qmad_history.dwc_hop_mtsg_cu_tsg_sigpre(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)

    def all_calls(self):
        return [self.tmnsgMh] + ([self.avx_tmnsgMhs, self.tmngsMhs, self.tmnsgMhs, self.Ubound_tmngsMhs] if capab("vectorise") else [])
    def all_call_names(self):
        return ["tmnsgMh"] + (["avx_tmnsgMhs", "tmngsMhs", "tmnsgMhs", "Ubound_tmngsMhs"] if capab("vectorise") else [])


class wilson_clover_hop_mtsgt2_sigpre:
    """
    Dirac Wilson operator that creates a lookup table for the hops.
    The axes are U[mu,x,y,z,t2,g,h,t1] and v[x,y,z,t2,s,h,t1].
    The grid gets split in half, with a register having 1 site from each block.
    t1 is the block number, t2 is the site.
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    field_strength * sigma * v is precomputed by computing the tensor product of field_strength * sigma,
    and only the upper triangle of two 6x6 blocks is passed for the field strength.
    """
    def __init__(self, U, mass_parameter, csw):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        # t axis needs to be divisible by register length
        tlen = U.shape[4]
        assert tlen%2 == 0
        self.mass_parameter = mass_parameter
        self.csw = csw

        # transform U to have t sites from each half as last index
        Ufirst = U[:,:,:,:,0:tlen//2]
        Usecond = U[:,:,:,:,tlen//2:tlen]
        self.U = torch.stack([Ufirst, Usecond], dim=-1)

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]//2]
        self.hop_inds = get_hop_indices(grid)


        Qmunu = get_clover_path_matrices(U)

        dim = list(U.shape[1:5])
        self.dim = dim
        # tensor product of the sigma matrix and field strength tensor
        field_strength_sigma = torch.zeros(dim+[4,3,4,3], dtype=torch.cdouble)
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                Fmunu = (Qmunu[mu][nu] - Qmunu[nu][mu]) / 8
                Fsigma = torch.einsum('xyztgh,sr->xyztsgrh',Fmunu,_sigma[mu][nu])
                # csw gets absorbed into the matrices
                field_strength_sigma += 2*(-self.csw/4)*Fsigma
        
        field_strength_sigma = field_strength_sigma.contiguous().reshape(dim+[12,12])

        # transform to grid layout (t register as last index, 2 furthest points are in a register)
        fs_even_t = field_strength_sigma[:,:,:,0:tlen//2]
        fs_odd_t = field_strength_sigma[:,:,:,tlen//2:tlen]
        field_strength_sigma = torch.stack([fs_even_t, fs_odd_t], dim=-1)

        self.field_strength_sigma = torch.stack([field_strength_sigma[:,:,:,:,_triag_mask_1],
                                                 field_strength_sigma[:,:,:,:,_triag_mask_2]],dim=4)
        assert tuple(self.field_strength_sigma.shape[4:7]) == (2,21,2,)
    
    def __str__(self):
        return "dwc_sigpre_hop_mtsgt2"
        
    def tmngsMht (self, v):
        return torch.ops.qmad_history.dwc_grid_mtsgt2_tmngsMht(self.U, v, self.field_strength_sigma,
                                                               self.hop_inds, self.mass_parameter)
        #        - self.csw/4 * torch.matmul(self.field_strength_sigma,v.reshape(self.dim+[12,1])).reshape(self.dim+[4,3]))

    def all_calls(self):
        return [] + ([self.tmngsMht] if capab("vectorise") else [])
    def all_call_names(self):
        return [] + (["tmngsMht"] if capab("vectorise") else [])

