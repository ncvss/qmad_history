import torch

x = torch.tensor([1,2,3,4,5], dtype=torch.float)  # input tensor
print("input:\n", x)
y = torch.zeros(3)  # expected output
prefactor = torch.randn(3, 5, requires_grad=True)
offset = torch.randn(3, requires_grad=True)
print("prefactor:\n",prefactor)
print("offset:\n", offset)
z = torch.matmul(prefactor, x)+offset
print("result:\n", z)
rescale = torch.tensor([1,2,11], dtype=torch.float, requires_grad=True)
print("rescale:\n", rescale)
z = rescale*z
print("result:\n", z)
loss = torch.sum(z-y)
print("loss:\n", loss)

loss.backward()
print("gradient of prefactor:\n", prefactor.grad)
print("gradient of offset:\n", offset.grad)
print("gradient of rescale:\n", rescale.grad)


print("\nnow try complex numbers:")

xc = torch.tensor([1+1j, 2+1j, 3+1j], dtype=torch.cdouble)
print("input:\n", xc)
yc = torch.zeros(3, dtype=torch.cdouble)
prefactorc = torch.randn([3,3], dtype=torch.cdouble, requires_grad=True)
print("prefactor:\n",prefactorc)
zc = torch.matmul(prefactorc, xc)
print("result:\n", zc)
lossc = torch.sum(torch.real(zc-yc) + torch.imag(zc-yc))
print("loss:\n", lossc)

lossc.backward()
print("gradient of prefactor:\n", prefactorc.grad)


print("\ncomplex scalars")

scalar_c = torch.tensor([1.1+0.7j], dtype=torch.cdouble)
print("first scalar:", scalar_c)
factor_c = torch.tensor([3.5+2.3j], dtype=torch.cdouble, requires_grad=True)
print("second scalar (which has derivative):", factor_c)
pc = scalar_c*factor_c
print("product:", pc)
ysc = torch.zeros(1, dtype=torch.cdouble)
losss = torch.real(pc-ysc)+torch.imag(pc-ysc)
print("scalar loss:", losss)

losss.backward()
print("complex scalar gradient:", factor_c.grad)
