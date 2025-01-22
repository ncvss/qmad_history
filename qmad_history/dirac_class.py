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


class wilson_direct:
    """
    Dirac Wilson operator with gauge config U, without precomputations,
    memory layout U[mu,x,y,z,t,g,h] and v[x,y,z,t,s,h].
    Also includes versions that split space-time into blocks.
    """
    def __init__(self, U, mass_parameter):
        self.U = U
        self.mass_parameter = mass_parameter

    def xtsgMhm(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtsgMhm(self.U, v, self.mass_parameter)
    def xtMmghs(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtMmghs(self.U, v, self.mass_parameter)
    def xtMmdghs(self, v):
        return torch.ops.qmad_history.dw_dir_mxtsg_xtMmdghs(self.U, v, self.mass_parameter)
    def block_dbxtsghm(self, v, blocksize):
        return torch.ops.qmad_history.dw_block_mxtsg_dbxtsghm(self.U, v, self.mass_parameter, blocksize)
    def block_bxtsghm(self, v, blocksize):
        return torch.ops.qmad_history.dw_block_mxtsg_bxtsghm(self.U, v, self.mass_parameter, blocksize)
    


class wilson_clover_direct:
    """
    Dirac Wilson Clover operator with gauge config U, precomputes field strength matrices,
    memory layout U[mu,x,y,z,t,g,h], v[x,y,z,t,s,h] and F[munu][x,y,z,t,g,h]
    """
    def __init__(self, U, mass_parameter, csw):
        if isinstance(U, list):
            U = torch.stack(U)
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

    def pxtMmghs(self, ve, vo):
        return torch.ops.qmad_history.dw_eo_pmtsg_pxtMmghs(self.Ue, self.Uo, ve, vo, self.mass_parameter, self.eodim)




class dirac_wilson_avx:
    """
    Dirac Wilson operator that creates a lookup table for the hops and uses AVX instructions.
    The axes are U[mu,x,y,z,t,g,gi] and v[x,y,z,t,s,g].
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

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dw_call_256d_om_template(self.U, v, self.hop_inds,
                                                                     self.mass_parameter)






class dirac_wilson_clover_avx:
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

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.dwc_call_256d_om_template(self.U, v, self.field_strength,
                                                                      self.hop_inds, self.mass_parameter,
                                                                      self.csw)


class domain_wall_dirac:
    """
    Domain Wall Dirac operator in Shamir formulation with gauge config U.
    """
    def __init__(self, U, mass_parameter, m5_parameter):
        self.U = U
        self.mass_parameter = mass_parameter
        self.m5_parameter = m5_parameter

    def __call__(self, v):
        return torch.ops.qcd_ml_accel_dirac.domain_wall_dirac_call(self.U, v, self.mass_parameter, self.m5_parameter)

