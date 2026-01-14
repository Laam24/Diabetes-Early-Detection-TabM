import pandas as pd
import numpy as np
import torch
import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import our custom research logic
from inference_engine import load_model, predict_single_value, calculate_dual_metrics

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pointing to the FP16 model (Optimized)
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'tabm_fp16_nb.pth')

# --- APP SETUP ---
app = FastAPI(title="Diabetes Early Warning System")

# Allow the browser (HTML) to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
DEVICE = "cpu" # Safer for web server

@app.on_event("startup")
async def startup_event():
    """Load the model once when server starts."""
    global model
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = load_model(MODEL_PATH, device=DEVICE)
        print("✅ Model loaded successfully (FP16 Mode)")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# --- DATA MODELS ---
class ManualInput(BaseModel):
    # Expects a list of 4 numbers [t, t-15, t-30, t-45]
    readings: List[float] 

# --- HELPER: PREPROCESSING PIPELINE ---
def process_uploaded_file(file_content, filename):
    """
    Replicates the Research Notebook pipeline:
    1. Load Excel/CSV
    2. Interpolate Gaps
    3. Resample to 15 mins
    4. Create Lag Features
    """
    # 1. Load Logic
    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_content))
    else:
        df = pd.read_excel(io.BytesIO(file_content))
        
    # 2. Standardize
    if 'Date' in df.columns: df.rename(columns={'Date': 'Timestamp'}, inplace=True)
    
    # Find Glucose Column
    cgm_col = [c for c in df.columns if 'CGM' in c]
    if not cgm_col: raise ValueError("No 'CGM' column found in file.")
    cgm_col = cgm_col[0]
    
    # 3. Clean & Interpolate
    df['Glucose'] = pd.to_numeric(df[cgm_col], errors='coerce')
    df['Glucose'] = df['Glucose'].interpolate(method='linear')
    
    # 4. Resample (15 min)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.set_index('Timestamp')
    df_res = df[['Glucose']].resample('15T').mean()
    df_res['Glucose'] = df_res['Glucose'].interpolate(method='linear')
    
    # 5. Create Sliding Windows (Lags 1, 2, 3)
    # Input required: [t, t-15, t-30, t-45] -> In code: [Glucose, Lag1, Lag2, Lag3]
    # NOTE: In our training logic, "Glucose" is t=0. 
    # So we need features: Glucose (t), Lag1 (t-15), Lag2 (t-30), Lag3 (t-45)
    
    df_res['Lag_1'] = df_res['Glucose'].shift(1)
    df_res['Lag_2'] = df_res['Glucose'].shift(2)
    df_res['Lag_3'] = df_res['Glucose'].shift(3)
    
    # Target (t+15 future)
    df_res['Target_Glucose'] = df_res['Glucose'].shift(-1)
    
    df_res.dropna(inplace=True)
    return df_res

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "System Operational", "model": "Q-TabM (FP16)"}

@app.post("/predict_manual")
def predict_manual(data: ManualInput):
    """
    Receives [100, 95, 90, 85] -> Returns Prediction
    """
    if len(data.readings) != 4:
        raise HTTPException(status_code=400, detail="Model requires exactly 4 history points.")
    
    # Run Inference
    pred = predict_single_value(model, data.readings, device=DEVICE)
    
    # Determine Risk
    status = "Normal"
    color = "green"
    if pred < 70:
        status = "HYPOGLYCEMIA RISK (Low)"
        color = "red"
    elif pred > 180:
        status = "HYPERGLYCEMIA RISK (High)"
        color = "orange"
        
    return {
        "prediction": round(pred, 2),
        "status": status,
        "alert_color": color
    }

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Receives File -> Runs Pipeline -> Returns Graph Data & Metrics
    """
    try:
        content = await file.read()
        
        # 1. Run Preprocessing Pipeline
        df_proc = process_uploaded_file(content, file.filename)
        
        # 2. Prepare Tensor
        # Columns: Glucose, Lag_1, Lag_2, Lag_3
        features = df_proc[['Glucose', 'Lag_1', 'Lag_2', 'Lag_3']].values
        targets = df_proc['Target_Glucose'].values
        
        # 3. Run Inference Batch
        tensor_in = torch.tensor(features, dtype=torch.float32).half().to(DEVICE)
        tensor_target = torch.tensor(targets, dtype=torch.float32).half().to(DEVICE)
        
        with torch.no_grad():
            preds = model(tensor_in).view(-1).float().cpu().numpy()
            targets = tensor_target.float().cpu().numpy() # Convert back for metrics
            
        # ... inside predict_batch function ...

        # 4. Calculate Metrics
        metrics = calculate_dual_metrics(torch.tensor(targets), torch.tensor(preds))
        
        # 5. Format for Graph (JSON) - UPDATED WITH ALL METRICS
        limit = 200
        response_data = {
            "timestamps": df_proc.index.strftime('%H:%M').tolist()[:limit],
            "actuals": targets[:limit].tolist(),
            "predictions": preds[:limit].tolist(),
            "metrics": {
                "RMSE": round(float(np.sqrt(np.mean((targets - preds)**2))), 2),
                "Accuracy": f"{metrics['Accuracy']*100:.1f}%",
                
                "Hypo Precision": f"{metrics['Hypo_Precision']*100:.1f}%",
                "Hypo Recall": f"{metrics['Hypo_Recall']*100:.1f}%",
                "Hypo F1": f"{metrics['Hypo_F1']*100:.1f}%",
                
                "Hyper Precision": f"{metrics['Hyper_Precision']*100:.1f}%",
                "Hyper Recall": f"{metrics['Hyper_Recall']*100:.1f}%",
                "Hyper F1": f"{metrics['Hyper_F1']*100:.1f}%"
            }
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))