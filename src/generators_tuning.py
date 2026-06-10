#from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import torch
import json
import argparse
import optuna
import time
import os
from utils import preprocess, load_config, generate_data, run_clf, calculate_dcr, calculate_mmd
from optuna.samplers import TPESampler
from paths import (ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR,
                   CONFIGS_DIR, RESULTS_DIR, FOLD_INDEXES_DIR)
import boto3

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def adjust_syn_data(df_train, df_syn):
    cols = df_train.columns
    dtypes = df_train.dtypes

    values = np.zeros_like(df_train.values)
    df_syn_ = pd.DataFrame(values, columns=cols)

    df_syn_[df_syn.columns] = df_syn.values
    df_syn_ = df_syn_.astype(dtypes)
    return df_syn_[cols]

def load_fold_indexes(dataset_name, fold):
    with open(FOLD_INDEXES_DIR / f'{dataset_name}_fold_indexes.json', 'r') as f:
        fold_indexes = json.load(f)
    return fold_indexes[f'fold_{fold}']

def objective(trial):
    hpt_config = load_config(CONFIG_FILE_PATH_HPT, model_name)
    
    suggested_params = {}
    for param_name, param_config in hpt_config.items():
        param_type = param_config['type']
        if param_type == 'int':
            if 'step' in param_config:
                suggested_value = trial.suggest_int(param_name, param_config['range'][0], param_config['range'][1], step=param_config['step'])
            else:
                suggested_value = trial.suggest_int(param_name, param_config['range'][0], param_config['range'][1])
        elif param_type == 'loguniform':
            suggested_value = trial.suggest_loguniform(param_name, param_config['range'][0], param_config['range'][1])
        elif param_type == 'int_loguniform':
            suggested_value = int(trial.suggest_loguniform(param_name, param_config['range'][0], param_config['range'][1]))
        elif param_type == 'categorical':
            suggested_value = trial.suggest_categorical(param_name, param_config['values'])
        elif param_type == 'float':
            if 'step' in param_config:
                suggested_value = trial.suggest_float(param_name, param_config['range'][0], param_config['range'][1], step=param_config['step'])
            else:
                suggested_value = trial.suggest_float(param_name, param_config['range'][0], param_config['range'][1])
        else:
            raise("Invalid parameter type!")
        suggested_params[param_name] = suggested_value

    if model_name == 'ddpm':
        suggested_params['model_params'] = {'n_layers_hidden':suggested_params['n_layers_hidden'], 'n_units_hidden':suggested_params['n_units_hidden'], 'dropout':suggested_params['dropout']}
        suggested_params.pop('n_layers_hidden')
        suggested_params.pop('n_units_hidden')
        suggested_params.pop('dropout')

    # Replace the train_test_split with fold index loading
    fold_indexes = load_fold_indexes(dataset_name, fold)
    
    df_train = df.loc[fold_indexes['train']]
    df_val = df.loc[fold_indexes['val']]
    df_test = df.loc[fold_indexes['test']]

    t0 = time.time()
    df_syn = generate_data(df_train, dataset_name, model_name, suggested_params)
    t1 = time.time()
    
    # Store synthetic data in trial user attributes
    trial.set_user_attr('synthetic_data', df_syn)
    
    result = run_clf(df_syn, df_val, target_col, target_value, sensitive_col, sensitive_value, sensitive_value_type, clf_name='lgbm')

    df_train_, df_val_ = preprocess(df_train, df_val)
    df_syn_, _ = preprocess(df_syn, df_val)

    df_syn_ = adjust_syn_data(df_train_, df_syn_)

    dcr_syn = calculate_dcr(df_syn_, df_train_)
    result['dcr_mean'] = float(dcr_syn.mean())
    result['dcr_q05'] = np.quantile(dcr_syn, 0.05)

    result['mmd_syn_real'] = calculate_mmd(df_syn_, df_train_)

    result['running_time'] = t1 - t0
    #dist_ori = calculate_distance(df_train_, df_train_)

    if model_name == 'ddpm':
        suggested_params['model_params'] = str(suggested_params['model_params'])
        
    suggested_params.update(result)

    #results = pd.concat([suggested_params, result], axis=1)
    results = pd.DataFrame(suggested_params, index=[0])
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = str(RESULTS_DIR / f'result_trials_{dataset_name}_{model_name}_{fold}.csv')
    if not os.path.isfile(results_path):
       results.to_csv(results_path)
    else: 
       results.to_csv(results_path, mode='a', header=False)

    if (AWS_ACCESS_KEY_ID is not None) and (AWS_SECRET_ACCESS_KEY is not None):
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3 = session.resource('s3')
        s3.meta.client.upload_file(Filename=results_path, Bucket='mscvolume', Key=results_path.split('/')[-1])

    return result['MCC'] #.values[0]

if __name__ == '__main__':

    SAVED_MODELS_DIR = MODELS_DIR
    CONFIG_FILE_PATH = CONFIGS_DIR / 'datasets_config.json'

    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

    parser = argparse.ArgumentParser()

    #parser.add_argument("--dataset_path", type=str, default='../data/original_data/adult.csv', help="Path to file with the data")
    parser.add_argument("--dataset_name", type=str, default='adult', help="String with the dataset name")
    parser.add_argument("--model_name", type=str, default='ctgan', help="String with the dataset name")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (0-4)")
    parser.add_argument("--n_trials", type=int, default=10, help="Optuna hpt optimization trials")

    args = parser.parse_args()
    #dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    model_name = args.model_name
    fold= args.fold
    n_trials = args.n_trials 

    CONFIG_FILE_PATH_HPT = CONFIGS_DIR / f'generators_hpt_config_{dataset_name}.json'

    dataset_path = f'{ORIGINAL_DATA_DIR}/{dataset_name}.csv'
    output_dir = f'{SYNTHETIC_DATA_DIR}/{dataset_name}'
    model_dir = f'{SAVED_MODELS_DIR}/{dataset_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    #models = ['tvae', 'ctgan','ctabgan', 'ddpm', 'realtabformer'] 
    #for model_name in models:
    
    output_path = f'{output_dir}/{dataset_name}_{model_name}.csv'
    model_path = f'{model_dir}/{dataset_name}_{model_name}.pkl'

    config = load_config(CONFIG_FILE_PATH, dataset_name)
    target_col = config['target_col']
    target_value = config['target_value']
    sensitive_col = config['sensitive_col']
    sensitive_value = config['sensitive_value']
    sensitive_value_type = config['sensitive_value_type']

    hpt_config = load_config(CONFIG_FILE_PATH_HPT, model_name)
    
    df = pd.read_csv(dataset_path)

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    
    # Get the synthetic data from the best trial
    best_trial = study.best_trial
    best_synthetic_data = best_trial.user_attrs['synthetic_data']
    
    # Save the best synthetic data
    synthetic_output_path = f'{output_dir}/{dataset_name}_{model_name}_fold_{fold}_best.csv'
    best_synthetic_data.to_csv(synthetic_output_path, index=False)
    
    # Upload to S3 if credentials are available
    if (AWS_ACCESS_KEY_ID is not None) and (AWS_SECRET_ACCESS_KEY is not None):
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3 = session.resource('s3')
        s3.meta.client.upload_file(
            Filename=synthetic_output_path, 
            Bucket='mscvolume', 
            Key=synthetic_output_path.split('/')[-1]
        )

    print(f"Best hyperparameters: {best_params}")
    print(f"Best MCC: {best_value}")
    print(f"Best synthetic data saved to: {synthetic_output_path}")

    with open(CONFIGS_DIR / f"{dataset_name}_{model_name}_best_params.json", "w") as f:
        json.dump(best_params, f)