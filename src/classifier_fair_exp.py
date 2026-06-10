from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import json
import argparse
import os
from utils import preprocess, load_config, generate_data, run_clf, calculate_dcr

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference,\
                            false_positive_rate_difference, true_positive_rate_difference,\
                            false_negative_rate_difference
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, QuantileTransformer
from synthcity.plugins import Plugins
#from data_generator import DataGenerator
from lightgbm import LGBMClassifier

from paths import (ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR, CONFIGS_DIR,
                   RESULTS_DIR, FAIR_EXP_DIR, FOLD_INDEXES_DIR)

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient

seed = 0
random.seed(seed)
np.random.seed(seed)

def adjust_syn_data(df_train, df_syn):
    cols = df_train.columns
    dtypes = df_train.dtypes

    values = np.zeros_like(df_train.values)
    df_syn_ = pd.DataFrame(values, columns=cols)

    df_syn_[df_syn.columns] = df_syn.values
    df_syn_ = df_syn_.astype(dtypes)
    return df_syn_[cols]

def run_clf_fair_method(df_train, df_test, target_col, target_value, sensitive_col, sensitive_value, sensitive_value_type, method='threshold_opt', constraint='equalized_odds', clf_name='lgbm', seed=0):
    clf_dict = {'lgbm': LGBMClassifier(random_state=seed)}
    
    X_train, y_train = df_train.drop(target_col, axis=1), [1 if i == target_value else 0 for i in df_train[target_col]]
    X_test, y_test = df_test.drop(target_col, axis=1), [1 if i == target_value else 0 for i in df_test[target_col]]

    if sensitive_value_type == 'Protected':
        S_train = (X_train[sensitive_col] == sensitive_value).astype(int).values
        S_test = (X_test[sensitive_col] == sensitive_value).astype(int).values
    elif sensitive_value_type == 'Privileged':
        S_train = (X_train[sensitive_col] != sensitive_value).astype(int).values
        S_test = (X_test[sensitive_col] != sensitive_value).astype(int).values


    X_train, X_test = preprocess(X_train, X_test)

    clf = clf_dict[clf_name]

    if method ==  'lgbm':
        # No unfairness mitigation
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

    elif method == 'threshold_opt':
        fairness_model = ThresholdOptimizer(
            estimator=clf,
            constraints=constraint,
            predict_method="predict_proba",
            prefit=False,
        )
        fairness_model.fit(X_train, y_train, sensitive_features=S_train)
        y_pred = fairness_model.predict(X_test, sensitive_features=S_test)
    
    elif method == 'exp_grad':
        if constraint == 'demographic_parity':
            constraints = DemographicParity()
        elif constraint == 'equalized_odds':
            constraints = EqualizedOdds()
        fairness_model = ExponentiatedGradient(
            estimator=clf,
            constraints=constraints,
            sample_weight_name="sample_weight",
        )
        fairness_model.fit(X_train, y_train, sensitive_features=S_train)
        y_pred = fairness_model.predict(X_test)

    else:
        raise ValueError("Invalid method. Choose 'lgbm', 'threshold_opt', or 'exp_grad'.")

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    dem_disp = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    pred_eq = false_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    eq_opp = true_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    eq_odds = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    
    return pd.DataFrame({
            'Accuracy': accuracy,
            'F1-Score': f1,
            'MCC': mcc,
            'Demographic Disparity': dem_disp,
            'Equalized Odds': eq_odds,
            'Equality of Opportunity': eq_opp,
            'Predictive Equality': pred_eq
            }, index=[0]).round(2)

def load_fold_indexes(dataset_name, fold):
    with open(FOLD_INDEXES_DIR / f'{dataset_name}_fold_indexes.json', 'r') as f:
        fold_indexes = json.load(f)
    return fold_indexes[f'fold_{fold}']

if __name__ == '__main__':

    SAVED_MODELS_DIR = MODELS_DIR
    CONFIG_FILE_PATH = CONFIGS_DIR / 'datasets_config.json'

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default='adult', help="String with the dataset name")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (0-4)")
    parser.add_argument("--constraint", type=str, default='FNR', help="method constraint")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    fold = args.fold
    constraint = args.constraint

    dataset_path = f'{ORIGINAL_DATA_DIR}/{dataset_name}.csv'
    output_dir = f'{SYNTHETIC_DATA_DIR}/{dataset_name}'
    model_dir = f'{SAVED_MODELS_DIR}/{dataset_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    syn_files = os.listdir(output_dir)
    models = [file.split('_')[-4] for file in syn_files]
    print(models)

    methods = ['lgbm', 'threshold_opt', 'exp_grad']
    all_results = []

    df = pd.read_csv(dataset_path)
    fold_indexes = load_fold_indexes(dataset_name, fold)
    
    df_train = df.loc[fold_indexes['train']]
    df_val = df.loc[fold_indexes['val']]
    df_test = df.loc[fold_indexes['test']]

    for method in methods:
        for model_name in models:
            output_path = f'{output_dir}/{dataset_name}_{model_name}.csv'
            model_path = f'{model_dir}/{dataset_name}_{model_name}.pkl'

            config = load_config(CONFIG_FILE_PATH, dataset_name)
            target_col = config['target_col']
            target_value = config['target_value']
            sensitive_col = config['sensitive_col']
            sensitive_value = config['sensitive_value']
            sensitive_value_type = config['sensitive_value_type']

            df_params = pd.read_csv(RESULTS_DIR / f'result_trials_{dataset_name}_{model_name}_{fold}.csv')
            df_params.drop(df_params.filter(regex="Unname"),axis=1, inplace=True)
            best_params = df_params[df_params.columns[:list(df_params.columns).index('Accuracy')]].to_dict(orient='index')[df_params['MCC'].argmax()]

            df_syn_path = os.path.join(output_dir, f'{dataset_name}_{model_name}_fold_{fold}_best.csv')
            df_syn = pd.read_csv(df_syn_path, index_col=0)

            result = run_clf_fair_method(df_syn, df_test, target_col, target_value, sensitive_col, sensitive_value, \
                                         sensitive_value_type=sensitive_value_type, method=method, constraint=constraint, seed=fold)

            result_dict = result.squeeze().to_dict()
            result_dict.update({
                'model': model_name,
                'method': method,
                'constraint': constraint,
                'fold': fold
            })
            result_dict.update(best_params)
            all_results.append(result_dict)

        # Add results for real data
        result_real = run_clf_fair_method(df_train, df_test, target_col, target_value, sensitive_col, sensitive_value, \
                                          sensitive_value_type=sensitive_value_type, method=method, constraint=constraint, seed=fold)
        result_real_dict = result_real.squeeze().to_dict()
        result_real_dict.update({
            'model': 'Real',
            'method': method,
            'constraint': constraint,
            'fold': fold
        })
        all_results.append(result_real_dict)

    # Save all results to a single CSV file
    results_df = pd.DataFrame(all_results)
    os.makedirs(FAIR_EXP_DIR, exist_ok=True)
    output_file = FAIR_EXP_DIR / f'fair_exp_results_{dataset_name}_fold_{fold}_{constraint}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")