import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import os, glob

surgeries = ['Suturing', 'Needle_Passing', 'Knot_Tying']
folds = ['Intermediate/C_OUT/', 'Expert/D_OUT/', 'Expert/E_OUT/', 'Intermediate/F_OUT/', 'Expert/ALL_EXPERTS/', 'Intermediate/ALL_INTERMEDIATES/']
dataset = 'data/jigsaws-data/'
results = 'LOUO/'

def extract_windows(x_data, y_data, window_size=24, stride=12):
    x_windows = []
    y_windows = []

    for seq, label in zip(x_data, y_data):
        for start in range(0, len(seq)-window_size+1, stride):
            x_windows.append(seq[start:start+window_size])
            y_windows.append(label)
    
    return np.array(x_windows), list(y_windows)

def calculate_gan_metrics(real_data, synth_data):
    """
    Accepts windowed numpy arrays for real and synthetic data, 
    calculates metrics, and returns them as a dictionary.
    """
    # Flatten the temporal dimension to compare overall feature distributions
    # Shape becomes [N * 24, 76]
    real_flat = real_data.reshape(-1, real_data.shape[-1])
    synth_flat = synth_data.reshape(-1, synth_data.shape[-1])
    
    wasserstein_scores = []
    jsd_scores = []
    
    # Calculate distance for every single kinematic feature (all 76 dimensions)
    for i in range(real_flat.shape[1]):
        real_feature = real_flat[:, i]
        synth_feature = synth_flat[:, i]
        
        # --- Wasserstein Distance ---
        w_dist = wasserstein_distance(real_feature, synth_feature)
        wasserstein_scores.append(w_dist)
        
        # --- Jensen-Shannon Divergence ---
        bins = np.histogram_bin_edges(np.concatenate([real_feature, synth_feature]), bins=50)
        p, _ = np.histogram(real_feature, bins=bins, density=True)
        q, _ = np.histogram(synth_feature, bins=bins, density=True)
        
        # Add tiny epsilon to avoid division by zero
        p = p + 1e-10
        q = q + 1e-10
        
        jsd = jensenshannon(p, q)
        jsd_scores.append(jsd)

    return {
        'Mean Wasserstein': np.mean(wasserstein_scores),
        'Mean JSD': np.mean(jsd_scores)
    }

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================

all_metrics = []

for surgery in surgeries:
    print(f"\nEvaluating GAN outputs for {surgery}...")
    base_data_path = os.path.join(dataset, surgery, 'splits')
    base_result_path = os.path.join(results, surgery)
    
    for fold in folds:
        parts = fold.strip('/').split('/')
        skill = parts[0]
        fold_name = parts[1]
        
        # 1. Load Synthetic Data (.npy)
        synth_path_search = os.path.join(base_result_path, fold_name, '*.npy')
        synth_files = glob.glob(synth_path_search)
        
        if not synth_files:
            print(f"  -> No synthetic data found for {fold_name}. Skipping.")
            continue
            
        synth_data = np.load(synth_files[0])

        # 2. Locate Real Data (.txt files)            
        real_files = glob.glob(os.path.join(base_data_path, fold, '*.txt'))

        if not real_files:
            print(f"  -> No real txt files found for {fold_name}. Skipping.")
            continue
            
        # 3. Read and Window the Real Data
        x_data_real = []
        for file in real_files:
            x_data_real.append(np.loadtxt(file))
            
        # We pass a dummy y array just to satisfy the extract_windows function signature
        dummy_y = [0] * len(x_data_real)
        real_windows, _ = extract_windows(x_data_real, dummy_y)
        
        if len(real_windows) == 0:
            print(f"  -> No windows extracted for {fold_name}. Skipping.")
            continue
            
        # 4. Calculate Metrics
        metrics = calculate_gan_metrics(real_windows, synth_data)
        
        # 5. Append to Master List
        metrics['Surgery'] = surgery
        metrics['Fold'] = fold_name
        all_metrics.append(metrics)

# ==========================================
# EXPORT TO CSV
# ==========================================

if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    
    # Reorder columns for readability
    cols = ['Surgery', 'Fold', 'Mean Wasserstein', 'Mean JSD']
    df_metrics = df_metrics[cols]
    
    print("\n### Final GAN Evaluation Metrics ###")
    print(df_metrics.to_markdown(index=False))
    
    df_metrics.to_csv("GAN_Evaluation_Metrics.csv", index=False)
    print("\nSuccessfully saved to GAN_Evaluation_Metrics.csv")
else:
    print("\nNo metrics were calculated. Check your file paths.")