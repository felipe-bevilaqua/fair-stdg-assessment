import os
import pandas as pd
import numpy as np
import random
import torch
from synthcity.utils.callbacks import EarlyStopping
from synthcity.utils.serialization import save_to_file, load_from_file
from synthcity.plugins import Plugins
from realtabformer import REaLTabFormer
from be_great import GReaT
import argparse
import pickle
import sys
from paths import CTABGAN_DIR
sys.path.append(str(CTABGAN_DIR))
from model.ctabgan import CTABGAN

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class DataGenerator:
    def __init__(self, model_name, dataset_name, model_args={}):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.model_args = model_args 
        self.model = self.get_model()
        self.dataset_config = self.get_dataset_config()
        print(self.model_args)

    def get_model(self):
        early_stop =  EarlyStopping(patience=100, min_epochs=100)
        model_mapping = {
            'arf': lambda: Plugins().get("arf", **self.model_args),
            "tvae": lambda: Plugins().get("tvae", **self.model_args),
            "ctgan": lambda: Plugins().get("ctgan", **self.model_args),
            "ddpm": lambda: Plugins().get("ddpm", callbacks=[early_stop], **self.model_args),
            "great": lambda: GReaT(llm='distilgpt2', batch_size=32, epochs=50, **self.model_args),
            "realtabformer": lambda: REaLTabFormer(model_type="tabular", gradient_accumulation_steps=4, logging_steps=100, **self.model_args),
            "ctabgan": lambda: CTABGAN(**self.model_args),
        }
        return model_mapping[self.model_name]()

    def get_dataset_config(self):
        dataset_config = {
            "adult": {
                "categorical_columns": ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'],
                "log_columns": [],
                "mixed_columns": {'capital-loss': [0.0], 'capital-gain': [0.0]},
                "general_columns": ["age"],
                "non_categorical_columns": [],
                "integer_columns": ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
                "problem_type": {"Classification": 'income'}
            },
            "compas": {
                "categorical_columns": ['sex', 'age_cat', 'race', 'c_charge_degree'],
                "log_columns": [],
                "mixed_columns": {},
                "general_columns": [],
                "non_categorical_columns": [],
                "integer_columns": ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'two_year_recid'],
                "problem_type": {"Classification": 'two_year_recid'}
            },
            "german": {
                "categorical_columns": ['checking-account', 'credit-history', 'purpose', 'savings-account', 'employment-since', 'other-debtors', 'property', 'other-installment', 'housing', 'job', 'telephone', 'foreign-worker', 'sex', 'personal-status'],
                "log_columns": [],
                "mixed_columns": {},
                "general_columns": [],
                "non_categorical_columns": [],
                "integer_columns": ['duration', 'credit-amount', 'installment-rate', 'residence-since', 'age', 'existing-credits', 'number-people-provide-maintenance-for', 'class-label'],
                "problem_type": {"Classification": 'class-label'}
            },
            "bank_marketing": {
                "categorical_columns": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'age',  'day'],
                "log_columns": [],
                "mixed_columns": {},
                "general_columns": [],
                "non_categorical_columns": [],
                "integer_columns": ['balance', 'duration', 'campaign', 'pdays', 'previous', 'y'],
                "problem_type": {"Classification": 'y'}
                },
            "toy": {}
        }
        return dataset_config[self.dataset_name]

    def fit(self, df):
        if self.model_name == 'ctabgan':
            self.model.fit(df, test_ratio=0.1, **self.dataset_config)
        else:
            self.model.fit(df)

    def save_model(self, path):
        if self.model_name in ["tvae", "ctgan", "ddpm", "arf"]:
            save_to_file(path, self.model)
        elif self.model_name in ["be_great", "realtabformer"]:
            self.model.save(path)
        elif self.model_name in ["ctabgan"]:
            with open(path, "wb") as f:
                pickle.dump(self, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def sample(self, n_samples):
        if self.model_name in ["tvae", "ctgan", "ddpm", "arf"]:
            synthetic_data = self.model.generate(n_samples).data
        elif self.model_name in ["be_great", "realtabformer"]:
            synthetic_data = self.model.sample(n_samples)
        elif self.model_name in ["ctabgan"]:
            synthetic_data = self.model.generate_samples(n_samples)
        return synthetic_data