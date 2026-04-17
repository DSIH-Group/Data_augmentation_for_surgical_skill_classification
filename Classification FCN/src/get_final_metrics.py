import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os, glob
from collections import Counter

dataset = '../jigsaws-data/'
# change folds and surgeries as required (ablation testing/normal testing, as ablation only includes knot tying)
surgeries = ['Knot_Tying']
folds = ['Novice/B/', 'Intermediate/C/', 'Expert/D/', 'Expert/E/', 'Intermediate/F/', 'Novice/G/', 'Novice/H/', 'Novice/I/']
ablation_variation = ['classification-windowed-synthetic-exp-int', 'classification-windowed-augmented-expert', 'classification-windowed-augmented-all']
LABEL_MAPPING = {'Expert': 0, 'Intermediate': 1, 'Novice': 2}

def extract_windows(x_data, y_data, window_size = 24, stride = 12):
    x_windows = []
    y_windows = []

    for seq, label in zip(x_data, y_data):
        for start in range(0, len(seq)-window_size+1, stride):
            x_windows.append(seq[start:start+window_size])
            y_windows.append(label)
    
    return np.array(x_windows), list(y_windows)

def split_jigsaws_inputs(x_data):
    """
    Slices the (N, 24, 76) tensor into the 20 inputs expected by the CNN.
    Each of the 4 manipulators gets split into 5 feature groups:
    Pos(3), RotMat(9), LinVel(3), RotVel(3), Gripper(1).
    """
    inputs = []
    lengths = [3, 9, 3, 3, 1]
    
    for manipulator_idx in range(4):
        base_idx = manipulator_idx * 19
        current_idx = base_idx
        for length in lengths:
            # Slice out the specific features for all batches and timesteps
            inputs.append(x_data[:, :, current_idx : current_idx + length])
            current_idx += length
            
    return inputs

for surgery in surgeries:
    surgery_results = []
    base_path = f'{dataset}/{surgery}/splits'
    for variation in ablation_variation:
        i = 1
        for fold in folds:
            model_path = f'../results/{surgery}/{variation}/jigsaws-data/Experimental_setup/{surgery}/unBalanced/GestureClassification/UserOut/{i}_Out/itr_1/architecture__fcn/reg__1e-05/lr__0.001/filters__8/kernel_size__3/amsgrad__0/model_best.keras'
            
            # Locate the text files
            path = os.path.join(base_path, fold, '*.txt')
            files = glob.glob(path)
            
            # We will store ONE vote per actual surgery trial
            y_true_trials = []
            y_pred_trials = []
            
            # Grab the correct integer mapping for the current fold
            skill = fold.split('/')[0]
            label_int = LABEL_MAPPING[skill]
            
            # 1. Load the best saved model
            model = tf.keras.models.load_model(model_path)
            
            # 2. Process ONE trial (file) at a time
            for file in files:
                seq = np.loadtxt(file) 
                
                # Extract windows for JUST this specific trial
                # (extract_windows expects lists, so we wrap seq and label_int in brackets)
                x_test_windows, _ = extract_windows([seq], [label_int])
                
                if len(x_test_windows) > 0:
                    # --- A. Slice the inputs for the CNN ---
                    formatted_trial_data = split_jigsaws_inputs(np.array(x_test_windows))
                    
                    # --- B. Get predictions for all windows in this single trial ---
                    p = model.predict(formatted_trial_data, batch_size=512, verbose=0)
                    
                    # --- C. YOUR MAJORITY VOTING LOGIC ---
                    votes = np.argmax(p, axis=1).tolist()
                    majority_vote = Counter(votes).most_common(1)[0][0]
                    
                    # --- D. Record the single trial-level prediction ---
                    y_pred_trials.append(majority_vote)
                    y_true_trials.append(label_int)
                    
            # 3. Generate the metrics based on the trial-level votes
            if len(y_pred_trials) > 0:
                report = classification_report(y_true_trials, y_pred_trials, output_dict=True, zero_division=0)
                
                metrics = {
                    'Accuracy': report['accuracy'],
                    'Macro Precision': report['macro avg']['precision'],
                    'Macro Recall': report['macro avg']['recall'],
                    'Macro F1-Score': report['macro avg']['f1-score'],
                    'Micro Precision': report['micro avg']['precision'] if 'micro avg' in report else report['accuracy'],
                    'Micro Recall': report['micro avg']['recall'] if 'micro avg' in report else report['accuracy'],
                    'Micro F1-Score': report['micro avg']['f1-score'] if 'micro avg' in report else report['accuracy']
                }
                
                metrics['Fold'] = f"{i}_Out ({skill})"
                surgery_results.append(metrics)
            else:
                print(f"No valid trials extracted for {fold}.")
                
            i += 1

    # 4. After the fold loop finishes, convert the master list into a Pandas DataFrame
    if surgery_results:
        df_surgery = pd.DataFrame(surgery_results)
        
        # Reorder columns to ensure 'Fold' is the very first column on the left
        cols = ['Fold'] + [col for col in df_surgery.columns if col != 'Fold']
        df_surgery = df_surgery[cols]
        
        print(f"\n### {surgery} Master Table ###")
        print(df_surgery.to_markdown(index=False))
        
        # 5. Save the final table to a CSV
        save_name = f"Ablation_Metrics_{surgery}.csv"
        df_surgery.to_csv(save_name, index=False)
        print(f"Saved to {save_name}")