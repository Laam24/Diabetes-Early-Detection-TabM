import torch
from model import TabM_Regressor

# Assume we have 10 input features (Lags, Insulin, etc.)
model = TabM_Regressor(input_dim=10, num_models=4)
print("Model Architecture:")
print(model)

# Create Fake Data (5 patients, 10 features)
fake_data = torch.randn(5, 10)

# Ask model to predict
output = model(fake_data)

print(f"\nInput shape: {fake_data.shape}")
print(f"Output shape: {output.shape}")
print("Test Successful. The Brain is alive.")