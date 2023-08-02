'''
    In this section, 
        we have reproduced the i-gelu approximation method proposed in 'I-BERT: Integer-only BERT Quantization.' 
        We plot the results and observe the fitting performance.

    Ref: Kim S, Gholami A, Yao Z, et al. I-bert: Integer-only bert quantization[C]//International conference on machine learning. PMLR, 2021: 5506-5518.
'''
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def l(x):
    a = -0.2888
    b = -1.769
    sgn = torch.sign(x)
    clip = torch.clamp(torch.abs(x), min=None, max=-b)
    l_x = sgn * (a * (clip + b)**2 + 1)
    return l_x

def i_gelu(x):
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=x.device))
    return x * 0.5 * (1 + l(x / sqrt_2))

x = torch.linspace(-10, 10, 10000)  # x values in the range [-10, 10]
y_i_gelu = i_gelu(x)  # i-GELU values
y_gelu = F.gelu(x)  # original GELU values

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_i_gelu.numpy(), label='i-GELU')
plt.plot(x.numpy(), y_gelu.numpy(), label='GELU')
plt.title('i-GELU vs GELU')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('i-gelu.png')
