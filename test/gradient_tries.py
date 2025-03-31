import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
prefactor = torch.randn(5, 3, requires_grad=True)
offset = torch.randn(3, requires_grad=True)
print("prefactor:\n",prefactor)
print("offset:\n", offset)
z = torch.matmul(x, prefactor)+offset
print("result:\n", z)
loss = torch.sum(z-y)
print("loss:\n", loss)

loss.backward()
print(prefactor.grad)
print(offset.grad)
