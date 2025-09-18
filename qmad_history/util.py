import torch
import numpy as np


class _PathBufferTemp:
    """
    Path buffer only for intermediate computations in the inititalisation of the clover.
    Derived from the version in qcd_ml.
    """

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
"""
Gamma matrices in chiral Euclidean representation, for intermediate computations.
"""

_sigma = torch.stack([torch.stack([(torch.matmul(_gamma[mu], _gamma[nu]) 
                        - torch.matmul(_gamma[nu], _gamma[mu])) / 2.0
                        for nu in range(4)], dim=0) for mu in range(4)], dim=0)
"""
Sigma matrices for intermediate computations.
"""


_triag_mask_1 = torch.tensor([[(sw < 6 and sh < 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)
"""
Mask boolean tensor to choose the upper triangle of the upper 6x6 block of a 12x12 matrix.
"""
_triag_mask_2 = torch.tensor([[(sw >= 6 and sh >= 6 and sh <= sw) for sw in range(12)] for sh in range(12)],
                            dtype=torch.bool)
"""
Mask boolean tensor to choose the upper triangle of the lower 6x6 block of a 12x12 matrix.
"""


def get_hop_indices(grid):
    """
    For each site on a 4d grid, computes the flattened addresses of all 8 next neighbour sites.
    """

    # this has to be on CPU, integer arithmetic is not implemented on GPU
    strides = torch.tensor([grid[1]*grid[2]*grid[3], grid[2]*grid[3], grid[3], 1], dtype=torch.int32)
    indices = torch.tensor(np.indices(grid, sparse=False), dtype=torch.int32)
    indices = torch.permute(indices, (1,2,3,4,0,)).flatten(start_dim=0, end_dim=3)

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
    
    hop_inds = torch.stack(hop_inds, dim=1).contiguous()
    return hop_inds


def get_clover_path_matrices(U: torch.Tensor):
    """
    For a given gauge configuration, returns a list Qmunu[mu][nu] that contains the sum of the gauge matrices
    when transporting along four plaquette paths that form a clover shape
    oriented in directions mu,nu (except for mu == nu).
    """

    assert tuple(U.shape[5:7]) == (3,3,)
    assert U.shape[0] == 4

    Hp = lambda mu, lst: lst + [(mu, 1)]
    Hm = lambda mu, lst: lst + [(mu, -1)]
    
    plaquette_paths = [[[
            Hm(mu, Hm(nu, Hp(mu, Hp(nu, []))))
            , Hm(nu, Hp(mu, Hp(nu, Hm(mu, []))))
            , Hp(nu, Hm(mu, Hm(nu, Hp(mu, []))))
            , Hp(mu, Hp(nu, Hm(mu, Hm(nu, []))))
            ] for nu in range(4)] for mu in range(4)]

    # Every path from the clover terms has equal starting and ending points.
    # This means the transport keeps the position of the vector field unchanged
    # and only multiplies it with a matrix independent of the vector field.
    # That matrix can thus be precomputed.
    Qmunu = [[(torch.zeros_like(U[0]) if nu != mu else 0) for nu in range(4)] for mu in range(4)]
    for mu in range(4):
        for nu in range(4):
            # the terms for mu = nu cancel out in the final expression, so we do not compute them
            if mu != nu:
                for ii in range(4):
                    clover_leaf_buffer = _PathBufferTemp(U, plaquette_paths[mu][nu][ii])
                    Qmunu[mu][nu] += clover_leaf_buffer.accumulated_U
    
    return Qmunu

