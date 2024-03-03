"""
Ablation study for comparing Syn-Slm versus CatBoost (traditional) on synthetic data
"""

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import norm
from sklearn.preprocessing import QuantileTransformer

import matplotlib.pyplot as plt
import seaborn as sns


import pickle
from tqdm import tqdm
import joblib

import argparse

import os


from utils.utils_data import TrueSampler
from utils.utils_model import MLPDiffusionContinuous
from utils.ddpm import MyDDPM, training_loop, generate_imputation


REPO_DIR = os.environ.get("REPO_DIR")

import sys
sys.path.insert(0, os.path.join(REPO_DIR, "tab-ddpm/utils"))

from utils_syn import catboost_pred_model, test_rmse


seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser(description="Ablation study for comparing Syn-Slm versus CatBoost (traditional) on synthetic tabular data")
parser.add_argument("--sigma", type = float, default = 0.2)
parser.add_argument("--device", type = str, default = "cuda:0")
args = parser.parse_args()

##################### Model Training #####################

device = args.device
sigma = args.sigma
n_samples = 3000

print(f"Configuration: sigma = {sigma} on device {device}")

# Generate simulated data #
true_sampler = TrueSampler(sigma=sigma)
X, y = true_sampler.sample(n_samples)
yx_train_unnorm = np.concatenate([y[:, None], X], axis=1)


# quantile transformation: ddpm is trained on the quantile-transformed data
qt_train = QuantileTransformer(output_distribution="normal", random_state=seed)
qt_train.fit(yx_train_unnorm)
yx_train_norm = qt_train.transform(yx_train_unnorm)
yx_train_norm = torch.tensor(yx_train_norm, dtype=torch.float32)


# Training #

d_in = yx_train_norm.shape[1]
d_time = 128
hidden_dims = [512, 256, 256, 256, 256, 128]
n_steps = 1000
n_epochs = 1000
lr = 1e-3

noise_pred_network = MLPDiffusionContinuous(
    d_in=d_in, hidden_dims=hidden_dims, dim_t=d_time
)
tabular_ddpm = MyDDPM(network=noise_pred_network, n_steps=n_steps, device=device)
optimizer = optim.Adam(tabular_ddpm.parameters(), lr=lr)

training_loop(
    yx_train_norm,
    tabular_ddpm,
    n_epochs,
    optimizer,
    store_path=f"./ckpt/tabular_ddpm_{sigma}.pt",
)


##################### Syn-Slm #####################

# Get the test data and the corresponding quantile transformer #

## Generate test data
n_test = 500
true_sampler = TrueSampler(sigma=sigma)
X, y = true_sampler.sample(n_test)
yx_test_unnorm = np.concatenate([y[:, None], X], axis=1)

## Quantile transformation
qt_test = QuantileTransformer(output_distribution="normal", random_state=seed)
qt_test.fit(yx_test_unnorm)
yx_test_norm = qt_test.transform(yx_test_unnorm)
yx_test_norm = torch.tensor(yx_test_norm, dtype=torch.float32)


# Conditional generation for predicting y #

n_rep = 20  # number of repetitions to get SE of RMSE
n_test = yx_test_norm.shape[0]
D = 100  # Monte Carlo size
D_batch = 100  # non-linear growth, so split into batches

batch_sizes = [D_batch] * (D // D_batch) + ([D % D_batch] if D % D_batch != 0 else [])


rmse_list = []

for i in tqdm(range(n_rep)):
    y_dict = {
        "y_true": yx_test_unnorm[:, 0],
        "y_pred_list": [],
    }

    for i, batch_size in enumerate(batch_sizes):
        print(f"Processing batch {i} with batch size {batch_size}...")

        # stacking all together for fast processing
        input_norm = np.vstack([yx_test_norm] * batch_size)

        input_norm = torch.tensor(input_norm, dtype=torch.float32)
        input_mask = torch.ones(input_norm.shape, dtype=torch.float32)
        input_mask[:, 0] = 0

        output_norm = generate_imputation(
            tabular_ddpm, input_norm, input_mask, resampling_steps=10
        )
        output_unnorm = qt_test.inverse_transform(output_norm.cpu().detach().numpy())

        y_dict["y_pred_list"].extend(
            [temp[:, 0] for temp in np.split(output_unnorm, batch_size, axis=0)]
        )

    # Calculate the RMSE
    y_true = y_dict["y_true"]
    y_pred_list = y_dict["y_pred_list"]
    y_pred_array = np.stack(y_pred_list, axis=0)  # (D, n_test)

    y_pred_synthetic = y_pred_array.mean(axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred_synthetic) ** 2))

    rmse_list.append(rmse)
    pickle.dump(rmse_list, open(f"./result/rmse_list_{sigma}.pkl", "wb"))


##################### CatBoost (traditional) #####################

# Simulate transfer learning and fine-tuning: Use a subset of samples for training the traditional predictive model
n_traditional = 1000
yx_train_unnorm_cb = yx_train_unnorm[:n_traditional]

columns_names = ["y", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]
df_train = pd.DataFrame(yx_train_unnorm_cb)
df_test = pd.DataFrame(yx_test_unnorm)
df_train.columns = df_test.columns = columns_names

n, n_test = len(yx_train_unnorm_cb), len(yx_test_unnorm)
r_model, r_val = 0.8, 0.2
n_model, n_val = int(n * r_model), int(n * r_val)

df_model = df_train.iloc[:n_model, :]
df_val = df_train.iloc[n_model:, :]


# train predictive model with early stopping
model_fit = catboost_pred_model(
    df_model, df_val, num_features_list=columns_names[1:], verbose=False
)


# Evaluate the result on the test set
rmse = test_rmse(model_fit, df_test, columns_names[1:])


##################### Results #####################
print(f"Configuration: sigma = {sigma}")
print(f"RMSE with Syn-Slm: {np.mean(rmse_list)} +/- {np.std(rmse_list)}")
print("RMSE on the test set for traditional approach:", rmse)
