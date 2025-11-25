import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, random_split

# Import your modules
from dataset import DiabetesDataset
from model import TabM_Regressor

# --- SETTINGS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(script_dir, '..', 'data_processed', 'Shanghai_Processed.csv')
ORIGINAL_MODEL_PATH = os.path.join(script_dir, '..', 'models', 'tabm_float32.pth')
QUANTIZED_MODEL_PATH = os.path.join(script_dir, '..', 'models', 'tabm_fp16.pth')

BATCH_SIZE = 32

def get_file_size(file_path):
    """Returns file size in Kilobytes (KB)"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / 1024

def evaluate_model(model, dataloader, device, is_fp16=False):
    """Runs the model on test data and calculates RMSE"""
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)
            
            # CRITICAL: If model is FP16, input data must be FP16 too
            if is_fp16:
                inputs = inputs.half()
            
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            running_loss += loss.item()
            
    avg_loss = running_loss / len(dataloader)
    rmse = np.sqrt(avg_loss)
    return rmse

def run_quantization():
    print("--- STARTING QUANTIZATION RESEARCH ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. PREPARE DATA
    # We need the same test set as before to make a fair comparison
    full_dataset = DiabetesDataset(PROCESSED_DATA_PATH)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    # Note: We use a fixed seed so the split is the same as training (optional but good practice)
    generator = torch.Generator().manual_seed(42) 
    _, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. LOAD ORIGINAL MODEL (Float32)
    # We need to know input dim
    sample_input, _ = full_dataset[0]
    input_dim = sample_input.shape[0]
    
    model_fp32 = TabM_Regressor(input_dim=input_dim, num_models=4).to(device)
    model_fp32.load_state_dict(torch.load(ORIGINAL_MODEL_PATH, map_location=device))
    
    print("\n[1] Evaluating Original Model (Float32)...")
    rmse_fp32 = evaluate_model(model_fp32, test_loader, device, is_fp16=False)
    size_fp32 = get_file_size(ORIGINAL_MODEL_PATH)
    print(f"    -> Accuracy (RMSE): {rmse_fp32:.4f} mg/dL")
    print(f"    -> Disk Size:       {size_fp32:.2f} KB")

    # 3. APPLY QUANTIZATION (Convert to Float16)
    print("\n[2] Quantizing Model to Float16 (Half Precision)...")
    model_fp16 = model_fp32.half() # This is the magic line
    
    # 4. SAVE QUANTIZED MODEL
    torch.save(model_fp16.state_dict(), QUANTIZED_MODEL_PATH)
    size_fp16 = get_file_size(QUANTIZED_MODEL_PATH)
    
    # 5. EVALUATE QUANTIZED MODEL
    print("[3] Evaluating Quantized Model (Float16)...")
    # Note: We assume CPU supports half precision (most do), otherwise this might be slow
    rmse_fp16 = evaluate_model(model_fp16, test_loader, device, is_fp16=True)
    
    print(f"    -> Accuracy (RMSE): {rmse_fp16:.4f} mg/dL")
    print(f"    -> Disk Size:       {size_fp16:.2f} KB")

    # 6. RESEARCH CONCLUSION
    print("\n--- FINAL RESEARCH REPORT ---")
    reduction = (1 - (size_fp16 / size_fp32)) * 100
    acc_loss = rmse_fp16 - rmse_fp32
    
    print(f"Model Size Reduction: {reduction:.2f}% (Smaller is better)")
    print(f"Accuracy Drop:        {acc_loss:.4f} mg/dL (Closer to 0 is better)")
    
    if reduction > 40 and abs(acc_loss) < 0.5:
        print("\nRESULT: SUCCESS. The model is lightweight and accurate.")
    else:
        print("\nRESULT: MIXED. Check metrics.")

if __name__ == "__main__":
    run_quantization()