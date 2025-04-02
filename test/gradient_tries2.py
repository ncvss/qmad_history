import torch
import qcd_ml

import gpt as g

from qmad_history import clover, compat


print("\nNow: an actual Wilson operator computation")
v = torch.randn([8,8,8,16,4,3], dtype=torch.cdouble, requires_grad=True)
v2 = v.clone().detach().requires_grad_(True)
rng = g.random("gradient")
U_g = g.qcd.gauge.random(g.grid([8,8,8,16], g.double), rng)
U = torch.tensor(compat.lattice_to_array(U_g))

dwc = qcd_ml.qcd.dirac.dirac_wilson_clover(U, -0.5, 1.0)
res = dwc(v)

# offset_f = torch.ones(v.shape, dtype=torch.cdouble, requires_grad=True)
# print("offset 0,0,0,0:\n", offset_f[0,0,0,0])
# resplus = res + offset_f

loss_f = (res * res.conj()).real.sum()

loss_f.backward()
print("gradient shape:", v.grad.shape)
print("is contiguous:", v.grad.is_contiguous())
# output: [8,8,8,16,4,3] as expected
# also it is contiguous

# as the sum is symmetric, the gradient of this should be the same
# as the partial derivative wrt v up to this point
# print("offset grad:\n",offset_f.grad[0,0,0,0])


print(v.grad[0,1,2,3])
# print(dw(offset_f.grad)[0,1,2,3])


dwq = clover.wilson_clover_hop_mtsg_sigpre(U, -0.5, 1.0)

resq = dwq.tmngsMhs(v2)

loss2 = (resq * resq.conj()).real.sum()

loss2.backward()

print("my comp.:\n", v2.grad[0,1,2,3])

print(torch.allclose(v.grad, v2.grad))
# this returns True! The gradients are equal!
