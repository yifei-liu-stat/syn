SEED = 2023
device = "cuda:0"


import os

REPO_DIR = os.environ.get("REPO_DIR")
TDDPM_DIR = os.path.join(REPO_DIR, "tab-ddpm")


import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform


import os
from tqdm import tqdm
import json
import pickle

import matplotlib.pyplot as plt

from sample import TrueSampler

import sys
sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

from utils_tabddpm import (
    train_tabddpm,
    generate_sample,
)



######### Get original data and pre-training data ready #########

# Configurations
sigma = 0.2
n_pretrain = 5000  # pre-training size
keyword = f"reg_{n_pretrain}"
synthetic_sample_dir = f"./ckpt/{keyword}/"

## Raw data split
n_train = 500  # raw training size
n_val = 200  # validation size
n_test = 1000  # test or evaluation size



true_sampler = TrueSampler(sigma=sigma)

np.random.seed(SEED)

X_pretrain, y_pretrain = true_sampler.sample(n_pretrain)
X_train, y_train = true_sampler.sample(n_train)
X_val, y_val = true_sampler.sample(n_val)
X_test, y_test = true_sampler.sample(n_test)


# Save the data in format suggested by the TDDPM repo
raw_data_dir = os.path.join(TDDPM_DIR, f"data/reg_raw")
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

    print(f"Saving raw data to {raw_data_dir} ...")

    np.save(os.path.join(raw_data_dir, "X_num_train.npy"), X_train)
    np.save(os.path.join(raw_data_dir, "y_train.npy"), y_train)

    np.save(os.path.join(raw_data_dir, "X_num_val.npy"), X_val)
    np.save(os.path.join(raw_data_dir, "y_val.npy"), y_val)

    np.save(os.path.join(raw_data_dir, "X_num_test.npy"), X_test)
    np.save(os.path.join(raw_data_dir, "y_test.npy"), y_test)

    info_dict = {
        "task_type": "regression",
        "name": "reg_raw",
        "id": "reg_raw",
        "train_size": n_train,
        "val_size": n_val,
        "test_size": n_test,
        "n_num_features": X_test.shape[1],
    }
    print(f"Saving raw dataset meta information to {raw_data_dir} ...")
    json.dump(info_dict, open(os.path.join(raw_data_dir, "info.json"), "w"))
else:
    print(
        f"Raw data information already exists in {raw_data_dir}, use existing validation and test set."
    )

    # use the same validation and test set in the pre-training configuration
    X_val = np.load(os.path.join(raw_data_dir, "X_num_val.npy"))
    y_val = np.load(os.path.join(raw_data_dir, "y_val.npy"))
    X_test = np.load(os.path.join(raw_data_dir, "X_num_test.npy"))
    y_test = np.load(os.path.join(raw_data_dir, "y_test.npy"))


pretrain_data_dir = os.path.join(TDDPM_DIR, f"data/reg_{n_pretrain}")
if not os.path.exists(pretrain_data_dir):
    os.makedirs(pretrain_data_dir)

    print(f"Saving pre-training data to {pretrain_data_dir} ...")

    np.save(os.path.join(pretrain_data_dir, "X_num_train.npy"), X_pretrain)
    np.save(os.path.join(pretrain_data_dir, "y_train.npy"), y_pretrain)

    np.save(os.path.join(pretrain_data_dir, "X_num_val.npy"), X_val)
    np.save(os.path.join(pretrain_data_dir, "y_val.npy"), y_val)

    np.save(os.path.join(pretrain_data_dir, "X_num_test.npy"), X_test)
    np.save(os.path.join(pretrain_data_dir, "y_test.npy"), y_test)

    info_dict = {
        "task_type": "regression",
        "name": f"reg_{n_pretrain}",
        "id": f"reg_{n_pretrain}",
        "train_size": n_pretrain,
        "val_size": n_val,
        "test_size": n_test,
        "n_num_features": X_test.shape[1],
    }
    print(f"Saving pre-training dataset meta information to {pretrain_data_dir} ...")
    json.dump(info_dict, open(os.path.join(pretrain_data_dir, "info.json"), "w"))
else:
    print(f"Pre-training data information already exists in {pretrain_data_dir}")



######### Pre-train the model #########

# train_tabddpm(
#     pipeline_config_path="./ckpt/base_config.toml",
#     real_data_dir=os.path.join(TDDPM_DIR, f"data/{keyword}"),
#     steps=50000,
#     temp_parent_dir=synthetic_sample_dir,
#     device=device,
# )

# print(f"Pre-training finished. Pre-trained model saved to {synthetic_sample_dir}")

########## Generate synthetic samples and check performance #########

generate_sample(
    pipeline_config_path=f"./ckpt/{keyword}/config.toml",
    ckpt_path=f"./ckpt/{keyword}/model.pt",
    num_samples=10000,
    batch_size=10000,
    temp_parent_dir=synthetic_sample_dir,
)

print(f"Synthetic samples generated and saved to {synthetic_sample_dir}")


########## Fine-tune the model on raw data ##########
ckpt_dir = f"./ckpt/{keyword}"

# raw data split: 500 + 200 + 1000
train_tabddpm(
    pipeline_config_path=os.path.join(ckpt_dir, "config.toml"),
    real_data_dir=raw_data_dir,
    ckpt_path=os.path.join(ckpt_dir, "model.pt"),
    pipeline_dict_path=os.path.join(ckpt_dir, "pipeline_dict.joblib"),
    steps=1000,
    lr=3e-6,
    temp_parent_dir=f"./ckpt/{keyword}_finetuned",
    device=device,
)

print(f"Fine-tuning finished. Fine-tuend model saved to {ckpt_dir}_finetuned")