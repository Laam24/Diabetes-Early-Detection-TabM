# Early Detection of Hypoglycemia using Quantized TabM

## 📌 Research Abstract
This project implements a lightweight Deep Learning framework for the early detection of diabetic attacks (Hypoglycemia and Hyperglycemia) using Continuous Glucose Monitoring (CGM) data. 

We utilize a **Tabular Multi-prediction (TabM)** architecture tailored for time-series forecasting. To ensure feasibility for resource-constrained wearable devices (Edge AI), the model is optimized using **FP16 Quantization**, achieving a **38% reduction in memory footprint** with zero loss in predictive accuracy.

## 📊 Key Results
The model predicts glucose levels **15 minutes into the future**.

| Metric | Baseline (FP32) | Optimized (FP16) | Impact |
| :--- | :--- | :--- | :--- |
| **RMSE** (Error) | 7.87 mg/dL | 7.93 mg/dL | Negligible Change |
| **Hypo Recall** (Safety) | **96.0%** | **96.0%** | **0% Loss (Perfect Retention)** |
| **Hyper Recall** (Detection) | 94.2% | 94.0% | -0.2% Drop |
| **Model Size** | 16.57 KB | 10.27 KB | **38.0% Reduction** |

## 📂 Project Structure
- `src/`: Source code for data processing, model definition, training, and quantization.
- `notebooks/`: Full research pipeline including EDA, LOSO Validation, and Visualization.
- `models/`: Saved trained models (.pth).

## ⚙️ Methodology
1.  **Data:** Shanghai T1DM Dataset (Minimally Invasive CGM).
2.  **Preprocessing:** Sliding Window approach (Lag Features) with Linear Interpolation.
3.  **Model:** TabM Regressor with Batch Ensemble layers.
4.  **Validation:** Leave-One-Subject-Out (LOSO) to ensure patient generalization.
5.  **Optimization:** Post-Training Quantization (FP16).

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Download the **Shanghai T1DM Dataset** and place it in `data_raw/Shanghai_T1DM/`.
3.  Run the full pipeline notebook: `notebooks/Full_Research_Project.ipynb`