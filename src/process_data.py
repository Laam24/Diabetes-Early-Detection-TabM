import pandas as pd
import numpy as np
import glob
import os

# --- ROBUST CONFIGURATION ---
# 1. Get the exact location of this script (src/process_data.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Construct paths relative to the script
# Go up one level ('..') from src to the Project Root, then down to data_raw
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data_raw', 'Shanghai_T1DM')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data_processed', 'Shanghai_Processed.csv')
INPUT_WINDOW = 3     # Look back 3 steps (3 * 15min = 45mins history)
PREDICT_STEPS = 1    # Predict 1 step ahead (15mins future)

def load_and_process_files():
    """
    Loops through all Excel/CSV files, cleans them, 
    and merges them into one Master Dataset.
    """
    
    # 1. Get list of all files
    # We look for both .csv and .xlsx files
    all_files = glob.glob(f"{RAW_DATA_PATH}/*.csv") + glob.glob(f"{RAW_DATA_PATH}/*.xlsx")
    print(f"--> Found {len(all_files)} files in {RAW_DATA_PATH}")

    master_list = [] # We will store each patient's clean dataframe here

    for filepath in all_files:
        try:
            # 2. Extract Patient ID
            # Filename looks like: "1001_0_20210730.xlsx"
            # We split by '\' or '/' depending on Windows/Mac to get the name
            filename = os.path.basename(filepath)
            patient_id = filename.split('_')[0] # Gets "1001"

            # 3. Load the File
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)

            # 4. Standardize Column Names
            # We rename 'Date' to 'Timestamp' for consistency
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'Timestamp'}, inplace=True)
            
            # Find the Glucose Column (It might be 'CGM (mg / dl)' or just 'CGM')
            # We look for any column containing "CGM"
            cgm_col = [c for c in df.columns if 'CGM' in c]
            if not cgm_col:
                print(f"Skipping {filename} - No CGM column found.")
                continue
            cgm_col = cgm_col[0] # Take the first match

            # 5. Clean Glucose Data
            # Convert to numbers, turning "data not available" into NaN
            df['Glucose'] = pd.to_numeric(df[cgm_col], errors='coerce')
            
            # Linear Interpolation: Draw lines through the gaps
            df['Glucose'] = df['Glucose'].interpolate(method='linear')

            # 6. Resample Time (The most crucial step)
            # We force the data to be exactly on the 00, 15, 30, 45 minute marks
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df.set_index('Timestamp')
            
            # '15T' means 15 Minutes. We take the mean if multiple values exist in that window.
            df_resampled = df[['Glucose']].resample('15T').mean()
            
            # Interpolate again because resampling might create new gaps
            df_resampled['Glucose'] = df_resampled['Glucose'].interpolate(method='linear')

            # 7. Create Sliding Windows (The Features)
            # We create columns for t-1, t-2, t-3
            for i in range(1, INPUT_WINDOW + 1):
                df_resampled[f'Glucose_Lag_{i}'] = df_resampled['Glucose'].shift(i)
            
            # Create the Target (t+1)
            df_resampled['Target_Glucose'] = df_resampled['Glucose'].shift(-PREDICT_STEPS)

            # 8. Final Cleanup
            # The shifting created NaNs at the start and end. Drop them.
            df_resampled.dropna(inplace=True)
            
            # Add Patient ID column so the model knows who this is
            df_resampled['Patient_ID'] = patient_id
            
            # Add to our master list
            master_list.append(df_resampled)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # 9. Save to CSV
    if master_list:
        final_df = pd.concat(master_list)
        final_df.to_csv(OUTPUT_PATH)
        print(f"\nSUCCESS! Processed data saved to: {OUTPUT_PATH}")
        print(f"Total Samples: {len(final_df)}")
        print(f"Features Created: {[c for c in final_df.columns if 'Lag' in c]}")
    else:
        print("\nFAILURE: No data was processed.")

if __name__ == "__main__":
    load_and_process_files()