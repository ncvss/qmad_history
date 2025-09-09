import torch
import numpy as np

from .settings import capab


# path buffer only for intermediate computations
# derived from the version in qcd_ml
class _PathBufferTemp:
    def __init__(self, U, path):
        self.path = path

        self.accumulated_U = torch.zeros_like(U[0])
        self.accumulated_U[:,:,:,:] = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.cdouble)

        for mu, nhops in self.path:
            if nhops < 0:
                direction = -1
                nhops *= -1
            else:
                direction = 1

            for _ in range(nhops):
                if direction == -1:
                    U = torch.roll(U, 1, mu + 1) # mu + 1 because U is (mu, x, y, z, t)
                    self.accumulated_U = torch.matmul(U[mu], self.accumulated_U)
                else:
                    self.accumulated_U = torch.matmul(U[mu].adjoint(), self.accumulated_U)
                    U = torch.roll(U, -1, mu + 1)




# gamma and sigma matrices for intermediate computations

_gamma = [torch.tensor([[0,0,0,1j]
                ,[0,0,1j,0]
                ,[0,-1j,0,0]
                ,[-1j,0,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,0,-1]
                ,[0,0,1,0]
                ,[0,1,0,0]
                ,[-1,0,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,1j,0]
                ,[0,0,0,-1j]
                ,[-1j,0,0,0]
                ,[0,1j,0,0]], dtype=torch.cdouble)
    , torch.tensor([[0,0,1,0]
                ,[0,0,0,1]
                ,[1,0,0,0]
                ,[0,1,0,0]], dtype=torch.cdouble)
    ]

_sigma = torch.stack([torch.stack([(torch.matmul(_gamma[mu], _gamma[nu]) 
                        - torch.matmul(_gamma[nu], _gamma[mu])) / 2.0
                        for nu in range(4)], dim=0) for mu in range(4)], dim=0)

# masks to choose upper and lower triangle
_triag_mask_1 = torch.tensor([[(sw < 6 and sh < 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)
_triag_mask_2 = torch.tensor([[(sw >= 6 and sh >= 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)

class wilson_clover_direct_false:
    """
    Dirac Wilson Clover operator with gauge config U, without precomputations,
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


class wilson_clover_fpre:
    """
    Dirac Wilson Clover operator with gauge config U, precomputes field strength matrices,
    memory layout U[mu,x,y,z,t,g,h], v[x,y,z,t,s,h] and F[munu][x,y,z,t,g,h]
    """
    def __init__(self, U, mass_parameter, csw):
        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
        
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
    Dirac Wilson Clover operator with gauge config U, layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    field_strength * sigma * v is precomputed by computing the tensor product of
    field_strength * sigma.
    """
    def __init__(self, U, mass_parameter, csw):

        assert tuple(U.shape[5:7]) == (3,3,)
        assert U.shape[0] == 4
        self.U = U
        self.mass_parameter = mass_parameter
        self.csw = csw

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U

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
    Dirac Wilson Clover operator that creates a lookup table for the hops and uses AVX instructions.
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
        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous().to(op_device)

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4).to(op_device)
        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)
    
    def __str__(self):
        return "dwc_hop_mtsg"
    
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
        return torch.ops.qmad_history.dwc_hop_mtsg_cu_tsg_fpre.default(self.U, v, self.field_strength,
                                                              self.hop_inds, self.mass_parameter,
                                                              self.csw)
    
    def all_calls(self):
        return [self.tmsgMhn] + ([self.avx_tmgsMhns, self.avx_tmsgMhns, self.templ_tmgsMhns, self.templ_tmsgMhns] if capab["vectorise"] else [])
    def all_call_names(self):
        return ["tmsgMhn"] + (["avx_tmgsMhns", "avx_tmsgMhns", "templ_tmgsMhns", "templ_tmsgMhns"] if capab["vectorise"] else [])



class wilson_clover_hop_tmgs:
    """
    Dirac Wilson Clover operator that creates a lookup table for the hops and uses AVX instructions.
    The axes are U[x,y,z,t,mu,g,gi], v[x,y,z,t,gi,s] and F[x,y,z,t,munu,g,gi].
    For simplicity, and because the path buffer requires it, the input U is still U[mu,x,y,z,t,g,h].
    """
    def __init__(self, U, mass_parameter, csw):
        self.U = torch.permute(U, (1,2,3,4,0,5,6)).contiguous()
        assert tuple(self.U.shape[4:7]) == (4,3,3,)

        self.mass_parameter = mass_parameter
        self.csw = csw

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

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
        
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
        return [self.avx_tmgsMhns] if capab["vectorise"] else []
    def all_call_names(self):
        return ["avx_tmgsMhns"] if capab["vectorise"] else []


class wilson_clover_hop_mtsgt_sigpre:
    """
    Dirac Wilson Clover operator with gauge config U, layout U[mu,x,y,z,t1,g,h,t2] and v[x,y,z,t1,s,h,t2].
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
        

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U

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
        return [] + ([self.tmngsMht] if capab["vectorise"] else [])
    def all_call_names(self):
        return [] + (["tmngsMht"] if capab["vectorise"] else [])


class wilson_clover_hop_mtsg_sigpre:
    """
    Dirac Wilson Clover operator with gauge config U, layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
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

        grid = [U.shape[1], U.shape[2], U.shape[3], U.shape[4]]
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
        self.hop_inds = torch.stack(hop_inds, dim=1).contiguous().to(op_device)
        

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U

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

        self.field_strength_sigma = torch.stack([field_strength_sigma[:,:,:,:,_triag_mask_1],
                                                 field_strength_sigma[:,:,:,:,_triag_mask_2]],dim=-1)
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
        return torch.ops.qmad_history.dwc_hop_mtsg_cu_tsg_sigpre.default(self.U, v, self.field_strength_sigma,
                                                             self.hop_inds, self.mass_parameter)

    def all_calls(self):
        return [self.tmnsgMh] + ([self.avx_tmnsgMhs, self.tmngsMhs, self.tmnsgMhs, self.Ubound_tmngsMhs] if capab["vectorise"] else [])
    def all_call_names(self):
        return ["tmnsgMh"] + (["avx_tmnsgMhs", "tmngsMhs", "tmnsgMhs", "Ubound_tmngsMhs"] if capab["vectorise"] else [])


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

        Hp = lambda mu, lst: lst + [(mu, 1)]
        Hm = lambda mu, lst: lst + [(mu, -1)]
        
        plaquette_paths = [[[
                Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
                , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
                , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
                , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
                ] for nu in range(4)] for mu in range(4)]

        #plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

        # Every path from the clover terms has equal starting and ending points.
        # This means the transport keeps the position of the vector field unchanged
        # and only multiplies it with a matrix independent of the vector field.
        # That matrix can thus be precomputed.
        Qmunu = [[torch.zeros_like(U[0]) for nu in range(4)] for mu in range(4)]
        for mu in range(4):
            for nu in range(4):
                # the terms for mu = nu cancel out in the final expression, so we do not compute them
                if mu != nu:
                    for ii in range(4):
                        clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                        Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U

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
        return [] + ([self.tmngsMht] if capab["vectorise"] else [])
    def all_call_names(self):
        return [] + (["tmngsMht"] if capab["vectorise"] else [])

