from data_generator import DataGenerator
import os
import json
import pandas as pd
import numpy as np
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import torch
#import ptitprince as pt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference,\
                            false_positive_rate_difference, true_positive_rate_difference,\
                            false_negative_rate_difference
from torch.utils.data import DataLoader
#import gower
#import plotly.express as px
from lightgbm import LGBMClassifier
import shap

import torch
import torch.nn as nn
import torch.nn.functional as F

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(X_train, X_test):
    num_cols = X_train.select_dtypes('number').columns
    cat_cols = X_train.select_dtypes('object').columns
    X_train_cat, X_test_cat = X_train[cat_cols], X_test[cat_cols]
    X_train, X_test = X_train[num_cols], X_test[num_cols]

    X_train_list = []
    X_test_list = []
    if len(num_cols) > 0:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=num_cols)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=num_cols)
        X_train_list.append(X_train)
        X_test_list.append(X_test)
    if len(cat_cols) > 0:
        X_train_cat = X_train_cat.astype(str).fillna('NONE')
        X_test_cat = X_test_cat.astype(str).fillna('NONE')
        
        binary_cols = [col for col in cat_cols if X_train_cat[col].nunique() <= 2]
        non_binary_cols = [col for col in cat_cols if X_train_cat[col].nunique() > 2]

        if len(non_binary_cols)>0:
            X_train_cat_non_binary = pd.get_dummies(X_train_cat[non_binary_cols])
            X_test_cat_non_binary = pd.get_dummies(X_test_cat[non_binary_cols])

            missing_cols = set(X_train_cat_non_binary.columns) - set(X_test_cat_non_binary.columns)
            for col in missing_cols:
                X_test_cat_non_binary[col] = 0
            X_test_cat_non_binary = X_test_cat_non_binary[X_train_cat_non_binary.columns]

            X_train_list.append(X_train_cat_non_binary)
            X_test_list.append(X_test_cat_non_binary)

        if len(binary_cols)>0:
            #X_train_cat = pd.concat([X_train_cat[binary_cols], X_train_cat_non_binary], axis=1)
            #X_test_cat = pd.concat([X_test_cat[binary_cols], X_test_cat_non_binary], axis=1)
            for col in binary_cols:
                X_train_cat[col] = pd.factorize(X_train_cat[col], sort=True)[0]
                X_test_cat[col] = pd.factorize(X_test_cat[col], sort=True)[0]
            X_train_list.append(X_train_cat[binary_cols])
            X_test_list.append(X_test_cat[binary_cols])
            
    X_train_list = [df.reset_index(drop=True) for df in X_train_list]
    X_test_list = [df.reset_index(drop=True) for df in X_test_list]
        
    X_train = pd.concat(X_train_list, axis=1) #
    X_test = pd.concat(X_test_list, axis=1)    
    
    X_train.columns = X_train.columns.str.replace(',', ' ')
    X_train.columns = X_train.columns.str.replace(':', ' ')
    X_test.columns = X_test.columns.str.replace(',', ' ')
    X_test.columns = X_test.columns.str.replace(':', ' ')
    
    return X_train, X_test



def load_config(CONFIG_FILE_PATH, name):
    with open(CONFIG_FILE_PATH, 'r') as file:
        configs = json.load(file)
    return configs[name]

def generate_data(df, dataset_name, model_name, generator_params={}, output_path=None, model_path = None):
    model = DataGenerator(model_name=model_name, dataset_name=dataset_name, model_args=generator_params)
    model.fit(df)
    df_syn = model.sample(df.shape[0])
    if model_path is not None:
        model.save_model(model_path)
    if output_path is not None:
        df_syn.to_csv(output_path)
    return(df_syn)

def run_clf(df_train, df_test, target_col, target_value, sensitive_col, sensitive_value, sensitive_value_type,  clf_name='lgbm', seed=0):
    clf_dict = {'lgbm': LGBMClassifier(random_state=seed),
                'log_reg': LogisticRegression()}
    
    X_train, y_train = df_train.drop(target_col, axis = 1), [1 if i == target_value else 0 for i in df_train[target_col]]
    X_test, y_test = df_test.drop(target_col, axis = 1), [1 if i == target_value else 0 for i in df_test[target_col]]

    #1S_train = (X_train[sensitive_col] == sensitive_value).astype(int).values
    if sensitive_value_type == 'Protected':
        S_test = (X_test[sensitive_col] == sensitive_value).astype(int).values
    elif sensitive_value_type == 'Privileged':
        S_test = (X_test[sensitive_col] != sensitive_value).astype(int).values

    X_train, X_test = preprocess(X_train, X_test)
    #X_train_syn_, X_test_syn_ = preprocess(X_train_syn, X_test) # X_test_syn preprocessed according to X_train_syn
    
    clf = clf_dict[clf_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob =  clf.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test,  y_pred_prob)
    dem_disp = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    pred_eq = false_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    eq_opp = true_positive_rate_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    eq_odds = equalized_odds_difference(y_true=y_test, y_pred=y_pred, sensitive_features=S_test)
    
    return {
            'Accuracy': accuracy,
            'F1-Score': f1,
            'MCC': mcc,
            'AUC': auc,
            'Demographic Disparity': dem_disp,
            'Equalized Odds': eq_odds,
            'Equality of Opportunity': eq_opp,
            'Predictive Equality': pred_eq
            }

def calculate_dcr(df1, df2, device='cpu'):
    t1 = torch.tensor(df1.astype(float).values).to(device)
    t2 = torch.tensor(df2.astype(float).values).to(device)
    dist=torch.cdist(t1, t2)
    return dist.min(axis=0)[0]

class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2).to(device)
        self.bandwidth = bandwidth #.to(device)

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):
    # https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
    def __init__(self, kernel=RBF().to(device)):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()

        return XX - 2 * XY + YY
    

def calculate_mmd(df1, df2, mmd_kernels=10, batch_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t1 = torch.tensor(df1.astype(float).values).to(device)
    t2 = torch.tensor(df2.astype(float).values).to(device)

    real_data = DataLoader(t1, batch_size=batch_size)
    syn_data = DataLoader(t2, batch_size=batch_size)

    mmd = MMDLoss(kernel=RBF(n_kernels=mmd_kernels, mul_factor=2.0, bandwidth=None).to(device))

    mmd_list = []
    for real, syn in zip(real_data, syn_data):
        mmd_value = mmd(real, syn)
        mmd_list.append(mmd_value.item())

    return sum(mmd_list) / len(mmd_list)

def plot_pca_tsne(df, df_syn, name, output_path=None):
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)
    X_pca = pca1.fit_transform(df)
    X_pca_syn = pca2.fit_transform(df_syn)

    # Perform t-SNE on both datasets
    tsne1 = TSNE(n_components=2, random_state=42)
    tsne2 = TSNE(n_components=2, random_state=42)
    X_tsne = tsne1.fit_transform(df)
    X_tsne_syn = tsne2.fit_transform(df_syn)

    # Create a subplot for PCA and t-SNE visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA visualization
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], ax=axes[0], label='Original Data', color='blue', alpha=0.5)
    sns.scatterplot(x=X_pca_syn[:, 0], y=X_pca_syn[:, 1], ax=axes[0], label='Synthetic Data', color='red', alpha=0.5)
    axes[0].set_title("PCA visualization")
    axes[0].set_xlabel("First Principal Component")
    axes[0].set_ylabel("Second Principal Component")
    axes[0].legend()

    # t-SNE visualization
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], ax=axes[1], label='Original Data', color='blue', alpha=0.5)
    sns.scatterplot(x=X_tsne_syn[:, 0], y=X_tsne_syn[:, 1], ax=axes[1], label='Synthetic Data', color='red', alpha=0.5)
    axes[1].set_title("t-SNE visualization")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].legend()

    fig.suptitle(name)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)

def plot_variables_dist(df, df_syn, name, output_path=None):
    n_cols = len(df.columns)
    fig, axes = plt.subplots(3, int(n_cols/3)+1, figsize=(30, 18))
    axes = axes.flatten()

    # Create a new DataFrame to concatenate both original and synthetic data
    combined_df = df.copy()
    df_syn_aux = df_syn.copy()
    combined_df['Data Type'] = 'Original Data'
    df_syn_aux['Data Type'] = 'Synthetic Data'
    combined_df = pd.concat([combined_df, df_syn_aux], ignore_index=True)

    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    num_cols = [col for col in df.columns if col not in cat_cols]
    for i, column in enumerate(df.columns):
        #print(i, column)
        ax = axes[i]

        # Check if the column data type is numeric
        if column in num_cols:
            # Plot the CDF for the numerical columns
            sorted_data = np.sort(df[column])
            sorted_syn_data = np.sort(df_syn[column])
            #y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            #y_syn_values = np.arange(1, len(sorted_syn_data) + 1) / len(sorted_syn_data)


            #ax.plot(sorted_data, y_values, color='blue', label='Original Data', alpha=0.5)
            #Ax.plot(sorted_syn_data, y_syn_values, color='red', label='Synthetic Data', alpha=0.5)
            ort = "h"; pal = ['blue', 'red']; sigma = .2
            #ax = pt.RainCloud(x = 'Data Type', y = column, data = combined_df, palette = pal, bw = sigma, width_viol = .6, ax = ax, orient = ort, move = .2)
            #sns.ecdfplot(y = column, data = combined_df, hue='Data Type', palette=['blue', 'red'], ax=ax)
            sns.kdeplot(x = column, data = combined_df, hue='Data Type', palette=['blue', 'red'], ax=ax)
        else:
            # Get unique categories and counts for each data type
            original_counts = df[column].value_counts().sort_index()
            synthetic_counts = df_syn[column].value_counts().sort_index()

            # Make sure both datasets have the same number of unique categories
            all_categories = original_counts.index.union(synthetic_counts.index)
            original_counts = original_counts.reindex(all_categories, fill_value=0)
            synthetic_counts = synthetic_counts.reindex(all_categories, fill_value=0)

            # Plot bars for each data type side-by-side
            bar_width = 0.35
            x_positions = np.arange(len(original_counts))

            ax.bar(x_positions - bar_width / 2, original_counts, width=bar_width, color='blue', label='Original Data')
            ax.bar(x_positions + bar_width / 2, synthetic_counts, width=bar_width, color='red', label='Synthetic Data')

            # Set x-axis tick labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(original_counts.index, rotation=90, ha='center')
            ax.legend()

        # Add a legend
        ax.set_title(column)
        fig.tight_layout()
        fig.suptitle(name)

        if output_path is not None:
            fig.savefig(output_path)
