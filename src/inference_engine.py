import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. MODEL ARCHITECTURE (From Research Notebook)
# ==========================================

class BatchEnsembleLayer(nn.Module):
    def __init__(self, in_features, out_features, num_models=4):
        super().__init__()
        self.num_models = num_models
        
        # Shared Weight
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Ensemble Personalities (Alpha) and Votes (Gamma)
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_features))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.alpha, mean=1.0, std=0.1)
        nn.init.normal_(self.gamma, mean=1.0, std=0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x shape: (Batch * Num_Models, In_Features)
        total_rows = x.size(0)
        real_batch_size = total_rows // self.num_models
        
        # 1. Align Alpha
        alpha_mask = self.alpha.repeat(real_batch_size, 1)
        x_scaled = x * alpha_mask
        
        # 2. Shared Weight
        result = F.linear(x_scaled, self.weight)
        
        # 3. Align Gamma
        gamma_mask = self.gamma.repeat(real_batch_size, 1)
        result = result * gamma_mask
        
        # 4. Bias
        return result + self.bias

class TabM_Regressor(nn.Module):
    def __init__(self, input_dim, num_models=4):
        super().__init__()
        self.num_models = num_models
        
        # Architecture: Input -> 64 -> 32 -> Output
        self.layer1 = BatchEnsembleLayer(input_dim, 64, num_models)
        self.relu = nn.ReLU()
        self.layer2 = BatchEnsembleLayer(64, 32, num_models)
        
        # Output Head: Combines all models (32 features * 4 models)
        self.output_head = nn.Linear(32 * num_models, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Step 1: Expand Input (Repeat data for each ensemble member)
        x = x.repeat_interleave(self.num_models, dim=0)
        
        # Step 2: Process Layers
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        
        # Step 3: Reshape (Bring expert opinions side-by-side)
        x = x.view(batch_size, -1)
        
        # Step 4: Final Decision
        return self.output_head(x)

# # ... (Keep Imports and Model Classes exactly the same as before) ...

# ==========================================
# 2. METRICS (Updated to include Precision)
# ==========================================

def calculate_dual_metrics(y_true_reg, y_pred_reg):
    """
    Evaluates Hypoglycemia (Low) and Hyperglycemia (High) detection.
    """
    # Medical Thresholds
    HYPO_THRESH = 70
    HYPER_THRESH = 180
    
    # Convert tensors to numpy
    if isinstance(y_true_reg, torch.Tensor):
        y_true_reg = y_true_reg.cpu().numpy()
    if isinstance(y_pred_reg, torch.Tensor):
        y_pred_reg = y_pred_reg.cpu().numpy()
        
    # --- BINARIZE INTO 3 CLASSES ---
    def categorize(glucose_array):
        cats = np.zeros_like(glucose_array, dtype=int) # Default 0 (Normal)
        cats[glucose_array < HYPO_THRESH] = 1          # 1 = Hypo
        cats[glucose_array > HYPER_THRESH] = 2         # 2 = Hyper
        return cats
    
    y_true_class = categorize(y_true_reg)
    y_pred_class = categorize(y_pred_reg)
    
    # --- CALCULATE METRICS ---
    acc = accuracy_score(y_true_class, y_pred_class)
    
    # labels=[1, 2] ensures we specifically look at Hypo and Hyper
    prec = precision_score(y_true_class, y_pred_class, labels=[1, 2], average=None, zero_division=0)
    rec = recall_score(y_true_class, y_pred_class, labels=[1, 2], average=None, zero_division=0)
    f1 = f1_score(y_true_class, y_pred_class, labels=[1, 2], average=None, zero_division=0)
    
    results = {
        "Accuracy": acc,
        
        "Hypo_Precision": prec[0], # <--- ADDED
        "Hypo_Recall": rec[0],
        "Hypo_F1": f1[0],
        
        "Hyper_Precision": prec[1], # <--- ADDED
        "Hyper_Recall": rec[1],
        "Hyper_F1": f1[1]
    }
    return results

# ... (Keep Utility Functions load_model and predict_single_value the same) ...

# ==========================================
# 3. UTILITY FUNCTIONS (For Web App)
# ==========================================

def load_model(model_path, device='cpu'):
    """Loads the FP16 model safely."""
    # Input dim is 4 based on your sliding window [t, t-15, t-30, t-45]
    model = TabM_Regressor(input_dim=4, num_models=4)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Convert to FP16 (Optimization)
    model.half()
    model.to(device)
    model.eval()
    return model

def predict_single_value(model, inputs_list, device='cpu'):
    """
    Takes a list of 4 numbers, runs the model, returns float.
    inputs_list: [Glucose_Now, Glucose_15m_Ago, Glucose_30m_Ago, Glucose_45m_Ago]
    """
    # 1. Convert to Tensor
    input_tensor = torch.tensor([inputs_list], dtype=torch.float32)
    
    # 2. Convert to Half Precision (FP16) to match model
    input_tensor = input_tensor.half().to(device)
    
    # 3. Predict
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # 4. Return as python float
    return prediction.item()