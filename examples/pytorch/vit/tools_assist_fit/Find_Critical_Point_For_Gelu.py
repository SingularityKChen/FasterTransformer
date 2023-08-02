'''
    The purpose of this code is 
        to explore the variation trend (critical point) of the GELU function in PyTorch, 
        aiding in the selection of fitting strategies.
'''
import torch
from torch.nn.functional import gelu
from torch.optim import SGD

# To initialize a negative/positive number as input.
# You can explore the patterns in the positive number range 
# and the negative number range by varying the range of the initialized input data.
x = torch.tensor([-4.0], requires_grad=True)

# Using the SGD optimizer.
optimizer = SGD([x], lr=0.001)

# Performing gradient descent.
for step in range(100000):
    optimizer.zero_grad()
    output = gelu(x)
    output.backward()
    optimizer.step()

# Print the critical points.
print(f"The lowest point on the negative side of the GELU function is at x = {x.item()}")
print(f"The corresponding GELU function value is {gelu(x).item()}")
