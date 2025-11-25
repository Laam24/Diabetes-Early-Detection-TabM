import torch
import pandas as pd
import numpy as np

print("--- System Check ---")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Check if PyTorch works
x = torch.tensor([1, 2, 3])
print(f"PyTorch Tensor Test: {x}")

print("--- Setup Successful! You are ready to research. ---")