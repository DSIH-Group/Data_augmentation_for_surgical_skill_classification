"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation, jigsaws_data_loading
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def main (args):
  """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
  ## Data loading
  if args.data_name in ['stock', 'energy']:
    ori_data = real_data_loading(args.data_name, args.seq_len)
  elif args.data_name == 'jigsaws':
    ori_data = jigsaws_data_loading(args.seq_len, args.skill, args.type, args.data_path)
  elif args.data_name == 'sine':
    # Set number of samples and its dimensions
    no, dim = 10000, 5
    ori_data = sine_data_generation(no, args.seq_len, dim)
    
  print(args.data_name + ' dataset is ready.')
    
  ## Synthetic data generation by TimeGAN
  # Set newtork parameters
  parameters = dict()  
  parameters['module'] = args.module
  parameters['hidden_dim'] = args.hidden_dim
  parameters['num_layer'] = args.num_layer
  parameters['iterations'] = args.iteration
  parameters['batch_size'] = args.batch_size
      
  generated_data = timegan(ori_data, parameters, save_path=args.model_save_path, load_path=args.model_load_path)   
  print('Finish Synthetic Data Generation')

  if args.data_name == 'jigsaws':
    ## SAVE THE GENERATED DATA
    print("Saving generated JIGSAWS data to file...")
    try:
        np.save(os.path.join(args.results_save_path, 'synthetic_jigsaws_data.npy'), generated_data)
    except Exception as e:
        print(f"Error saving data: {e}")
  
  ## Performance metrics   
  # Output initialization
  metric_results = dict()
  
  if args.metric_iteration > 0:
      print("Starting Metrics Evaluation (This takes time)...")
      
      # 1. Discriminative Score
      discriminative_score = list()
      for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)
      metric_results['discriminative'] = np.mean(discriminative_score)
      print('Discrim score done')
          
      # 2. Predictive score
      predictive_score = list()
      for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)   
      metric_results['predictive'] = np.mean(predictive_score)    
      print('Pred score done')
      
      print(metric_results)
  else:
      print("Skipping Metrics to save time (Set --metric_iteration > 0 to enable)")

  # 3. Visualization
  # (Your visualization_metrics.py is already optimized, so this is safe now!)
  visualization(ori_data, generated_data, 'pca', result_dir=args.results_save_path)
  visualization(ori_data, generated_data, 'tsne', result_dir=args.results_save_path)
  
  return ori_data, generated_data, metric_results


if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['sine','stock','energy', 'jigsaws'],
      default='stock',
      type=str)
  parser.add_argument(
      '--seq_len',
      help='sequence length',
      default=24,
      type=int)
  parser.add_argument(
      '--module',
      choices=['gru','lstm','lstmLN'],
      default='gru',
      type=str)
  parser.add_argument(
      '--hidden_dim',
      help='hidden state dimensions (should be optimized)',
      default=24,
      type=int)
  parser.add_argument(
      '--num_layer',
      help='number of layers (should be optimized)',
      default=3,
      type=int)
  parser.add_argument(
      '--iteration',
      help='Training iterations (should be optimized)',
      default=50000,
      type=int)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch (should be optimized)',
      default=128,
      type=int)
  parser.add_argument(
      '--metric_iteration',
      help='iterations of the metric computation',
      default=0,
      type=int)
  parser.add_argument('--model_save_path', type=str, default=None, help='File path to save the model')
  parser.add_argument('--model_load_path', type=str, default=None, help='File path to load a saved model')
  parser.add_argument(
      '--skill',
      choices=['Novice', 'Intermediate', 'Expert'],
      default='Novice',
      type=str)
  parser.add_argument(
      '--type',
      choices=['Knot_Tying', 'Suturing', 'Needle_Passing'],
      default='Knot_Tying',
      type=str)
  parser.add_argument(
     '--data_path',
     type=str,
     default=None,
     help='Path to data files'
  )
  parser.add_argument('--results_save_path', type=str, default=None, help='Dir path to save the results')
  
  args = parser.parse_args() 
  
  # Calls main function  
  ori_data, generated_data, metrics = main(args)