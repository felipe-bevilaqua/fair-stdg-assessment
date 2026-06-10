from fairlearn.datasets import fetch_bank_marketing
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from io import BytesIO
import pandas as pd
import numpy as np
import random
import requests
import os
import json

seed = 0
random.seed(seed)
np.random.seed(seed)

datasets_dict = {
                'adult':{
                    'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                    'url_test': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
                    'cols':['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
                            },
                'compas':{
                    'url':'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv',
                    'cols':['sex', 'age', 'age_cat', 'race',
                            'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                            'priors_count', 'c_charge_degree', 'two_year_recid']
                },
                'german':{
                        'url':'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                        'cols':['checking-account', 'duration', 'credit-history', 'purpose',
                                'credit-amount', 'savings-account', 'employment-since',
                                'installment-rate', 'personal-status-and-sex', 'other-debtors',
                                'residence-since', 'property', 'age', 'other-installment', 'housing',
                                'existing-credits', 'job', 'number-people-provide-maintenance-for',
                                'telephone', 'foreign-worker', 'class-label'],
                        'feature_mapping':{
                            'A11': '< 0 DM',
                            'A12': '0 <= ... < 200 DM',
                            'A13': '... >= 200 DM / salary assignments for at least 1 year',
                            'A14': 'no checking account',
                            'A30': 'no credits taken/ all credits paid back duly',
                            'A31': 'all credits at this bank paid back duly',
                            'A32': 'existing credits paid back duly till now',
                            'A33': 'delay in paying off in the past',
                            'A34': 'critical account/ other credits existing (not at this bank)',
                            'A40': 'car (new)',
                            'A41': 'car (used)',
                            'A42': 'furniture/equipment',
                            'A43': 'radio/television',
                            'A44': 'domestic appliances',
                            'A45': 'repairs',
                            'A46': 'education',
                            'A48': 'retraining',
                            'A49': 'business',
                            'A410': 'others',
                            'A61': '... < 100 DM',
                            'A62': '100 <= ... < 500 DM',
                            'A63': '500 <= ... < 1000 DM',
                            'A64': '.. >= 1000 DM',
                            'A65': 'unknown/ no savings account',
                            'A71': 'unemployed',
                            'A72': '... < 1 year',
                            'A73': '1 <= ... < 4 years',
                            'A74': '4 <= ... < 7 years',
                            'A75': '.. >= 7 years',
                            'A91': 'male : divorced/separated',
                            'A92': 'female : divorced/separated/married',
                            'A93': 'male : single',
                            'A94': 'male : married/widowed',
                            'A95': 'female : single',
                            'A101': 'none',
                            'A102': 'co-applicant',
                            'A103': 'guarantor',
                            'A121': 'real estate',
                            'A122': 'if not A121 : building society savings agreement/ life insurance',
                            'A123': 'if not A121/A122 : car or other, not in attribute 6',
                            'A124': 'unknown / no property',
                            'A141': 'bank',
                            'A142': 'stores',
                            'A143': 'none',
                            'A151': 'rent',
                            'A152': 'own',
                            'A153': 'for free',
                            'A171': 'unemployed/ unskilled - non-resident',
                            'A172': 'unskilled - resident',
                            'A173': 'skilled employee / official',
                            'A174': 'management/ self-employed/ highly qualified employee/ officer',
                            'A191': 'none',
                            'A192': 'yes, registered under the customers name',
                            'A201': 'yes',
                            'A202': 'no'
                            }
                                },
}

def create_folders(folders_list):
    """Create folders if they don't exist."""
    for folder in folders_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_response(url):
    return(requests.get(url).content)

def download_data(name):
    if name == 'adult':
        response = get_response(datasets_dict['adult']['url'])
        df = pd.read_csv(BytesIO(response),sep=',', names = datasets_dict[name]['cols'])

    elif name == 'adult_test':
        response = get_response(datasets_dict['adult']['url_test'])
        df = pd.read_csv(BytesIO(response),sep=',', names = datasets_dict['adult']['cols'], skiprows=1)

    elif name == 'german':
        response = get_response(datasets_dict['german']['url'])
        df = pd.read_csv(BytesIO(response), sep=' ', names = datasets_dict['german']['cols'])
        for col in df.columns:
            if df[col].dtype == object:  
                df[col] = df[col].map(datasets_dict[name]['feature_mapping'])
        df['sex'] = [values[0] for values in df['personal-status-and-sex'].str.split(' : ')]
        df[ 'personal-status'] = [values[1] for values in df['personal-status-and-sex'].str.split(' : ')]
        df = df.drop('personal-status-and-sex', axis=1)    

    elif name == 'compas':
        response = get_response(datasets_dict['compas']['url'])
        df = pd.read_csv(BytesIO(response), usecols= datasets_dict['compas']['cols'])

    elif name == 'bank_marketing':
        df = pd.concat(fetch_bank_marketing(as_frame=True, return_X_y=True), axis=1)
        df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                'housing', 'loan', 'contact', 'day', 'month', 'duration',
                'campaign', 'pdays', 'previous', 'poutcome','y']
        df['age'] = [1 if (age >= 25) and (age < 60) else 0 for age in df['age'] ]

    return(df)

def split_data(path, output_folder, n_folds, config_path):
    df = pd.read_csv(path)
    dataset_name = path.split('.csv')[0].split('/')[-1]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    config = load_config(config_path, dataset_name)
    target_col = config['target_col']
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    fold_indexes = {}
 
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df[target_col])):
        train_val_df = df.iloc[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_df.index, 
            test_size=0.25,  # 0.25 of 0.8 (train_val) is 0.2 of total
            stratify=train_val_df[target_col],
            random_state=0
        )
        
        fold_indexes[f'fold_{fold}'] = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist()
        }
    
    output_file = os.path.join(output_folder, f'{dataset_name}_fold_indexes.json')
    with open(output_file, 'w') as f:
        json.dump(fold_indexes, f)

def load_config(config_path, dataset_name):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config[dataset_name]

if __name__ == '__main__':
    from paths import DATA_DIR, ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR, FOLD_INDEXES_DIR, CONFIGS_DIR

    folders = [DATA_DIR, ORIGINAL_DATA_DIR, SYNTHETIC_DATA_DIR]
    create_folders(folders)

    dataset_names = ['adult', 'german', 'compas', 'bank_marketing'] #'adult_test',
    config_path = CONFIGS_DIR / 'datasets_config.json'

    for name in dataset_names:
        print(name)
        df = download_data(name)

        if name == 'adult':
            df_ = download_data('adult_test')
            df_['income'] = df_['income'].map({' >50K.': ' >50K', ' <=50K.': ' <=50K'})
            df = pd.concat([df, df_], axis=0)

        output_csv = ORIGINAL_DATA_DIR / f'{name}.csv'
        df.to_csv(output_csv, index=False)

        split_data(
            path=str(output_csv),
            output_folder=str(FOLD_INDEXES_DIR),
            n_folds=5,
            config_path=config_path
        )