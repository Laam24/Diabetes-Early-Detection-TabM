import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

# Import your custom modules
from dataset import DiabetesDataset
from model import TabM_Regressor

# --- SETTINGS ---
# Get the folder where THIS script (train.py) is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script, not the terminal
PROCESSED_DATA_PATH = os.path.join(script_dir, '..', 'data_processed', 'Shanghai_Processed.csv')
MODEL_SAVE_PATH = os.path.join(script_dir, '..', 'models', 'tabm_float32.pth')

# Print to verify (Optional)
print(f"Looking for data at: {PROCESSED_DATA_PATH}")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20  # How many times we look at the entire dataset

def train_model():
    # 1. SETUP DEVICE (Use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. PREPARE DATA
    # Load the dataset
    full_dataset = DiabetesDataset(PROCESSED_DATA_PATH)
    
    # Calculate sizes for Train (80%) and Test (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Randomly split the data
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create DataLoaders (The conveyer belts that feed data to the model)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data Loaded. Train Samples: {train_size}, Test Samples: {test_size}")

    # 3. INITIALIZE MODEL
    # We need to know how many features (columns) are in the input to set up the model
    # We grab one sample to check dimensions
    sample_input, _ = full_dataset[0]
    input_dim = sample_input.shape[0]
    
    model = TabM_Regressor(input_dim=input_dim, num_models=4).to(device)
    
    # 4. DEFINE LOSS & OPTIMIZER
    criterion = nn.MSELoss()  # Mean Squared Error (Standard for Regression)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam is a smart optimizer

    # 5. TRAINING LOOP
    print("\n--- Starting Training ---")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # A. The Guess
            predictions = model(inputs)
            
            # Resize targets to match predictions (Batch, 1)
            targets = targets.view(-1, 1)
            
            # B. The Grade (Loss)
            loss = criterion(predictions, targets)
            
            # C. The Correction (Backprop)
            optimizer.zero_grad() # Clear old gradients
            loss.backward()       # Calculate new gradients
            optimizer.step()      # Update weights
            
            running_loss += loss.item()
            
        # Calculate average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        
        # 6. VALIDATION LOOP (Testing)
        model.eval() # Set model to evaluation mode (no learning here)
        test_loss = 0.0
        with torch.no_grad(): # Don't calculate gradients (saves memory)
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                test_loss += loss.item()
                
        avg_test_loss = test_loss / len(test_loader)
        
        # Print Progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        # Save the model if it's the best one so far
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            # Ensure 'models' folder exists
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
    print("\n--- Training Complete ---")
    print(f"Best Model Saved to: {MODEL_SAVE_PATH}")
    print(f"Final Test MSE: {best_loss:.4f}")
    
    # Calculate RMSE (Root Mean Squared Error) - Easier to understand
    # If MSE is 400, RMSE is 20 (meaning we are off by 20 mg/dL on average)
    print(f"Average Error (RMSE): {np.sqrt(best_loss):.2f} mg/dL")

if __name__ == "__main__":
    train_model()