import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): Path to the processed csv file.
        """
        # 1. Load the Data
        self.data = pd.read_csv(csv_path)
        
        # 2. Separate Features (Input) and Target (Output)
        # Inputs: Everything except 'Target_Glucose' and 'Patient_ID'
        # Target: 'Target_Glucose'
        
        # We drop Patient_ID because the model can't do math on a string like "1001"
        # We drop Target_Glucose because that is the answer we want to predict
        feature_cols = [c for c in self.data.columns if c not in ['Target_Glucose', 'Patient_ID', 'Timestamp']]
        
        self.X = self.data[feature_cols].values.astype(np.float32)
        self.y = self.data['Target_Glucose'].values.astype(np.float32)
        
        print(f"Dataset Loaded. Input Shape: {self.X.shape}, Target Shape: {self.y.shape}")

    def __len__(self):
        # Returns the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Returns one sample at index 'idx'
        # Convert numpy array to PyTorch Tensor (the format GPU/CPU understands)
        input_tensor = torch.tensor(self.X[idx])
        target_tensor = torch.tensor(self.y[idx])
        
        return input_tensor, target_tensor