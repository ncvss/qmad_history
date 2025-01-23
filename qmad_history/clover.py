import torch
import numpy as np


# path buffer only for intermediate computations
# derived from the version in qcd_ml
class _PathBufferTemp:
    def __init__(self, U, path):
        if isinstance(U, list):
            # required by torch.roll below.
            U = torch.stack(U)
        self.path = path

        self.accumulated_U = torch.zeros_like(U[0])
        self.accumulated_U[:,:,:,:] = torch.complex(
                torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.double)
                , torch.zeros(3, 3, dtype=torch.double)
                )

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
    
    def __call__(self, v):
        # alternate name: xtMmghijkls
        return torch.ops.qmad_history.dwc_dir_mxtsg_false(self.U, v, self.mass_parameter, self.csw)


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

        plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
                        Qmunu[mu][nu] += plaquette_path_buffers[mu][nu][ii].accumulated_U
        
        # only a flat list, as it needs to be accessed by C++
        self.field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                self.field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        assert len(self.field_strength) == 6
        for F in self.field_strength:
            assert tuple(F.shape[4:6]) == (3,3)
        
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


class wilson_clover_sigpre:
    """
    Dirac Wilson Clover operator with gauge config U, layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    field_strength * sigma * v is precomputed by computing the tensor product of
    field_strength * sigma.
    """
    def __init__(self, U, mass_parameter, csw):
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

        _sigma = [[(torch.matmul(_gamma[mu], _gamma[nu]) 
                    - torch.matmul(_gamma[nu], _gamma[mu])) / 2
                    for nu in range(4)] for mu in range(4)]

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

        plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
                        Qmunu[mu][nu] += plaquette_path_buffers[mu][nu][ii].accumulated_U

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
        
    def xtMmdghs (self, v):
        return (torch.ops.qmad_history.dw_dir_mxtsg_xtMmdghs(self.U, v, self.mass_parameter)
                - self.csw/4 * torch.matmul(self.field_strength_sigma,v.reshape(self.dim+[12,1])).reshape(self.dim+[4,3]))


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

        plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
                        Qmunu[mu][nu] += plaquette_path_buffers[mu][nu][ii].accumulated_U
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4)
        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)

    def avx_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_avx_mtsg_tmgsMhns(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)
    def templ_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_templ_mtsg_tmgsMhns(self.U, v, self.field_strength,
                                                              self.hop_inds, self.mass_parameter,
                                                              self.csw)

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

        plaquette_path_buffers = [[[_PathBufferTemp(U, pi) for pi in pnu] for pnu in pmu] for pmu in plaquette_paths]

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
                        Qmunu[mu][nu] += plaquette_path_buffers[mu][nu][ii].accumulated_U
        
        field_strength = []
        # the field strength is antisymmetric, so we only need to compute nu < mu
        for mu in range(4):
            for nu in range(mu):
                field_strength.append((Qmunu[mu][nu] - Qmunu[nu][mu]) / 8)
        
        self.field_strength = torch.stack(field_strength, dim=4)
        assert tuple(self.field_strength.shape[4:7]) == (6,3,3,)

    def avx_tmgsMhns(self, v):
        return torch.ops.qmad_history.dwc_avx_tmgs_tmgsMhns(self.U, v, self.field_strength,
                                                            self.hop_inds, self.mass_parameter,
                                                            self.csw)

