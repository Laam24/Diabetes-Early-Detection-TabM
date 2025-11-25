import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchEnsembleLayer(nn.Module):
    """
    Corrected Layer: Does NOT repeat input. 
    It assumes input is ALREADY expanded to (Batch * Num_Models, Features).
    """
    def __init__(self, in_features, out_features, num_models=4):
        super().__init__()
        self.num_models = num_models
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Shared Weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 2. Personalities (Alpha) & Votes (Gamma)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        
        # 3. Bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.alpha, mean=1.0, std=0.1)
        nn.init.normal_(self.gamma, mean=1.0, std=0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape is (Batch_Size * Num_Models, In_Features)
        # We need to find the original batch size to align alpha correctly
        total_rows = x.size(0)
        real_batch_size = total_rows // self.num_models
        
        # 1. Align Alpha (Personalities)
        # Alpha is (4, In). We want (Batch*4, In) arranged like [m1, m2, m3, m4, m1, m2...]
        # .repeat(real_batch_size, 1) creates exactly that pattern
        alpha_mask = self.alpha.repeat(real_batch_size, 1)
        
        # Apply Alpha
        x_scaled = x * alpha_mask
        
        # 2. Shared Weight Matrix
        result = F.linear(x_scaled, self.weight)
        
        # 3. Align Gamma (Votes)
        gamma_mask = self.gamma.repeat(real_batch_size, 1)
        
        # Apply Gamma
        result = result * gamma_mask
        
        # 4. Add Bias
        result = result + self.bias
        
        return result

class TabM_Regressor(nn.Module):
    def __init__(self, input_dim, num_models=4):
        super().__init__()
        self.num_models = num_models
        
        # Layers
        self.layer1 = BatchEnsembleLayer(input_dim, 64, num_models)
        self.relu = nn.ReLU()
        self.layer2 = BatchEnsembleLayer(64, 32, num_models)
        
        # Output Head
        # Takes the combined knowledge of all models (32 features * 4 models = 128)
        self.output_head = nn.Linear(32 * num_models, 1)

    def forward(self, x):
        # x Input Shape: (Batch_Size, Features) -> e.g. (32, 10)
        batch_size = x.size(0)
        
        # --- STEP 1: EXPAND ONCE ---
        # We repeat each sample 'num_models' times interleave
        # [P1, P2] -> [P1, P1, P1, P1, P2, P2, P2, P2]
        x = x.repeat_interleave(self.num_models, dim=0)
        
        # --- STEP 2: PROCESS ---
        x = self.layer1(x) # Output: (Batch*4, 64)
        x = self.relu(x)
        
        x = self.layer2(x) # Output: (Batch*4, 32)
        x = self.relu(x)
        
        # --- STEP 3: RESHAPE (AGGREGATE) ---
        # We have (128, 32). We want (32, 128).
        # We group the 4 model outputs for each patient side-by-side
        x = x.view(batch_size, -1)
        
        # --- STEP 4: FINAL PREDICTION ---
        x = self.output_head(x) # Output: (32, 1)
        
        return x