#from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import torch
import json
import argparse
import optuna
#import mlflow
import time
import ast
import os
from utils import preprocess, load_config, generate_data, run_clf, calculate_dcr
from paths import (ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR,
                   CONFIGS_DIR, RESULTS_DIR, FOLD_INDEXES_DIR)

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def load_fold_indexes(dataset_name, fold):
    with open(FOLD_INDEXES_DIR / f'{dataset_name}_fold_indexes.json', 'r') as f:
        fold_indexes = json.load(f)
    return fold_indexes[f'fold_{fold}']

def adjust_syn_data(df_train, df_syn):
    cols = df_train.columns
    dtypes = df_train.dtypes

    values = np.zeros_like(df_train.values)
    df_syn_ = pd.DataFrame(values, columns=cols)

    df_syn_[df_syn.columns] = df_syn.values
    df_syn_ = df_syn_.astype(dtypes)
    return df_syn_[cols]

if __name__ == '__main__':

    SAVED_MODELS_DIR = MODELS_DIR
    CONFIG_FILE_PATH = CONFIGS_DIR / 'datasets_config.json'

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='adult', help="String with the dataset name")
    parser.add_argument("--model_name", type=str, default='all', help="String with the dataset name")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (0-4)")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    fold = args.fold

    dataset_path = f'{ORIGINAL_DATA_DIR}/{dataset_name}.csv'
    output_dir = f'{SYNTHETIC_DATA_DIR}/{dataset_name}'
    model_dir = f'{SAVED_MODELS_DIR}/{dataset_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if model_name == 'all':
        #models = ['tvae', 'ctgan', 'arf', 'realtabformer', 'ctabgan']
        models = ['arf', 'realtabformer']
    else:
        models = [model_name]
    
    for model_name in models:
    
        output_path = f'{output_dir}/{dataset_name}_{model_name}_fold_{fold}_best.csv'
        model_path = f'{model_dir}/{dataset_name}_{model_name}_fold_{fold}_best.pkl'

        config = load_config(CONFIG_FILE_PATH, dataset_name)
        target_col = config['target_col']
        target_value = config['target_value']
        sensitive_col = config['sensitive_col']
        sensitive_value = config['sensitive_value']
        sensitive_value_type = config['sensitive_value_type']

        df_params = pd.read_csv(RESULTS_DIR / f'result_trials_{dataset_name}_{model_name}_{fold}.csv')
        df_params.drop(df_params.filter(regex="Unname"),axis=1, inplace=True)
        best_params = df_params[df_params.columns[:list(df_params.columns).index('Accuracy')]].to_dict(orient='index')[df_params['MCC'].argmax()]
        if model_name == 'ddpm':
            best_params['model_params'] = ast.literal_eval(best_params['model_params'])
        print(best_params)
        df = pd.read_csv(dataset_path)

        # Replace the train_test_split with fold index loading
        fold_indexes = load_fold_indexes(dataset_name, fold)
        
        df_train = df.loc[fold_indexes['train']]
        df_val = df.loc[fold_indexes['val']]
        df_test = df.loc[fold_indexes['test']]

        t0 = time.time()
        df_syn = generate_data(df_train, dataset_name, model_name, best_params)
        t1 = time.time()
        
        df_syn.to_csv(output_path)

