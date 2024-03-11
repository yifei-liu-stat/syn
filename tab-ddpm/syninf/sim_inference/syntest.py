"""
Perform Syn-Test on simulated regression dataset using the existing splits and fine-tuned models.
"""

import os

REPO_DIR = os.environ.get("REPO_DIR")
TDDPM_DIR = os.path.join(REPO_DIR, "tab-ddpm")

import torch


import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform
from scipy.stats import norm

from pytorch_lightning import seed_everything

import time
from copy import deepcopy
import shutil
from tqdm import tqdm
import json
import pickle

import seaborn as sns
import matplotlib.pyplot as plt


from sample import TrueSampler


import sys

sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

from utils_tabddpm import (
    train_tabddpm,
    generate_sample,
)

from utils_syn import (
    concat_data,
    catboost_pred_model,
    blackbox_test_stat,
    combine_Hommel,
)


class PathConfig:
    tddpm_dir: str = TDDPM_DIR
    syninf_dir: str = os.path.join(TDDPM_DIR, "syninf")


pathconfig = PathConfig()


class ExpConfig:
    n_train: int = 1000
    n_inf: int = 200
    sigma: float = 0.2
    keyword: str = "inf_10000"

    null_features_list: list = ["num_7"]
    rho: float = 10
    D: int = 1000

    alpha: float = 0.05
    epsilon: float = 0.01

    cache_dir: str = "./temp/"
    cache_dir_spare: str = "./temp2/"

    cuda_id: int = 0
    seed: int = 2023

    """
    - n_train, n_inf: sizes of raw training sample and inference sample
    - keyword: indicates pre-training size
    - null_features_list: features to be tested
    - rho: synthetic-to-raw ratio
    - D: MC size for Syn-Test
    - cache_dir, cache_dir_spare: temporary directories for saving intermediate results. e.g. one for twin_1 and another one for twin_2
    """

    def __init__(self):
        self.ckpt_dir = f"./ckpt/{self.keyword}"
        self.raw_data_dir = os.path.join(pathconfig.tddpm_dir, f"data/inf_raw")

        suffix = "_".join(self.null_features_list)
        self.result_dict_save_path = f"./results/result_dict_{suffix}.pkl"
        self.raw_result_dict_save_path = f"./results/raw_result_dict_{suffix}.pkl"


expconfig = ExpConfig()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


seed_everything(expconfig.seed)
torch.cuda.set_device(expconfig.cuda_id)






############ Raw data and pre-training data ############

sigma = 0.2
n_pretrain = 10000  # pre-training size
keyword = f"inf_{n_pretrain}"
synthetic_sample_dir = f"./ckpt/{keyword}/"

n_train = 1000  # raw training size
n_val = 200  # validation size
n_test = 1000  # test or evaluation size



true_sampler = TrueSampler(sigma=sigma, null_feature=True)


X_pretrain, y_pretrain = true_sampler.sample(n_pretrain)
X_train, y_train = true_sampler.sample(n_train)
X_val, y_val = true_sampler.sample(n_val)
X_test, y_test = true_sampler.sample(n_test)

# Raw data
raw_data_dir = os.path.join(TDDPM_DIR, f"data/inf_raw")

os.makedirs(raw_data_dir, exist_ok=True)

print(f"Saving raw data to {raw_data_dir} ...")

np.save(os.path.join(raw_data_dir, "X_num_train.npy"), X_train)
np.save(os.path.join(raw_data_dir, "y_train.npy"), y_train)

np.save(os.path.join(raw_data_dir, "X_num_val.npy"), X_val)
np.save(os.path.join(raw_data_dir, "y_val.npy"), y_val)

np.save(os.path.join(raw_data_dir, "X_num_test.npy"), X_test)
np.save(os.path.join(raw_data_dir, "y_test.npy"), y_test)

info_dict = {
    "task_type": "regression",
    "name": "inf_raw",
    "id": "inf_raw",
    "train_size": n_train,
    "val_size": n_val,
    "test_size": n_test,
    "n_num_features": X_test.shape[1],
}
print(f"Saving raw dataset meta information to {raw_data_dir} ...")
json.dump(info_dict, open(os.path.join(raw_data_dir, "info.json"), "w"))


# Pre-training data
pretrain_data_dir = os.path.join(TDDPM_DIR, f"data/inf_{n_pretrain}")

os.makedirs(pretrain_data_dir, exist_ok=True)

print(f"Saving pre-training data to {pretrain_data_dir} ...")

np.save(os.path.join(pretrain_data_dir, "X_num_train.npy"), X_pretrain)
np.save(os.path.join(pretrain_data_dir, "y_train.npy"), y_pretrain)

np.save(os.path.join(pretrain_data_dir, "X_num_val.npy"), X_val)
np.save(os.path.join(pretrain_data_dir, "y_val.npy"), y_val)

np.save(os.path.join(pretrain_data_dir, "X_num_test.npy"), X_test)
np.save(os.path.join(pretrain_data_dir, "y_test.npy"), y_test)

info_dict = {
    "task_type": "regression",
    "name": f"inf_{n_pretrain}",
    "id": f"inf_{n_pretrain}",
    "train_size": n_pretrain,
    "val_size": n_val,
    "test_size": n_test,
    "n_num_features": X_test.shape[1],
}
print(f"Saving pre-training dataset meta information to {pretrain_data_dir} ...")
json.dump(info_dict, open(os.path.join(pretrain_data_dir, "info.json"), "w"))


############ Twin folders for Syn-Test ############


raw_data_dir = os.path.join(TDDPM_DIR, f"data/inf_raw")

info_dict = json.load(open(os.path.join(raw_data_dir, "info.json"), "r"))
X_train = np.load(os.path.join(raw_data_dir, "X_num_train.npy"))
y_train = np.load(os.path.join(raw_data_dir, "y_train.npy"))

m = int(X_train.shape[0] / 2)

for twin_kw in ["twin_1", "twin_2"]:
    print(f"Preparing twin data directories for {raw_data_dir}: {twin_kw} ...")
    
    twin_data_dir = os.path.join(TDDPM_DIR, f"data/inf_raw_{twin_kw}")
    shutil.copytree(raw_data_dir, twin_data_dir, dirs_exist_ok=True)
    
    temp_info_dict = deepcopy(info_dict)
    temp_info_dict["name"] = f"inf_{twin_kw}"
    temp_info_dict["id"] = f"inf_{twin_kw}"
    
    if twin_kw == "twin_1":
        X_temp, y_temp = X_train[:m], y_train[:m]
        temp_info_dict["train_size"] = m
    else:
        X_temp, y_temp = X_train[m:], y_train[m:]
        temp_info_dict["train_size"] = X_train.shape[0] - m
    
    np.save(os.path.join(twin_data_dir, "X_num_train.npy"), X_temp)
    np.save(os.path.join(twin_data_dir, "y_train.npy"), y_temp)
    json.dump(temp_info_dict, open(os.path.join(twin_data_dir, "info.json"), "w"))
    
    
############ Get pre-trained generator ############

train_tabddpm(
    pipeline_config_path="./ckpt/base_config.toml",
    real_data_dir=pretrain_data_dir,
    steps=50000,
    temp_parent_dir=synthetic_sample_dir,
    device=f"cuda:{expconfig.cuda_id}",
    seed=expconfig.seed,
)


############ Fine-tune to get twin generators ############

ckpt_dir = f"./ckpt/{keyword}"

# when generating samples, one should use the SAME PRE-PROCESSING pipeline during fine-tuning
for twin_kw in ["twin_1", "twin_2"]:
    print(f"Fine-tuning pre-trained generator {keyword} for {twin_kw} ...")
    train_tabddpm(
        pipeline_config_path=os.path.join(ckpt_dir, "config.toml"),
        real_data_dir= os.path.join(TDDPM_DIR, f"data/inf_raw_{twin_kw}"),
        ckpt_path=os.path.join(ckpt_dir, "model.pt"),
        pipeline_dict_path=os.path.join(ckpt_dir, "pipeline_dict.joblib"),
        steps=1000,
        lr=3e-6,
        # lr=1e-7,
        temp_parent_dir=f"./ckpt/{keyword}_{twin_kw}",
        device=f"cuda:{expconfig.cuda_id}",
        seed=expconfig.seed,
    )


############ Configurations for Syn-Test ############

# kwargs for training CatBoost model
cb_kwargs = {
    "num_features_list": [f"num_{i}" for i in range(8)],
    "iterations": 2000,
    "loss_function": "MAE",
    "verbose": False,
}


# kwargs for generating samples using twin generators
tddpm_kwargs_twin = {
    twin_kw: {
        "pipeline_config_path": f"./ckpt/{expconfig.keyword}_{twin_kw}/config.toml",
        "ckpt_path": f"./ckpt/{expconfig.keyword}_{twin_kw}/model.pt",
        "pipeline_dict_path": os.path.join(
            expconfig.ckpt_dir, "pipeline_dict.joblib"
        ),  # same as the pre-processing pipeline during its fine-tuning
        "temp_parent_dir": (
            expconfig.cache_dir if twin_kw == "twin_1" else expconfig.cache_dir_spare
        ),
        "device": f"cuda:{expconfig.cuda_id}",
    }
    for twin_kw in ["twin_1", "twin_2"]
}


############## Syn-Test ##############

# Generate synthetic samples ONCE AND FOR ALL for twin folders

regenerate = False  # whether to regenerate synthetic samples

if regenerate:
    seed_everything(expconfig.seed)

    for twin_kw in ["twin_1", "twin_2"]:
        temp_sample_kwargs = deepcopy(tddpm_kwargs_twin[twin_kw])
        temp_sample_kwargs["temp_parent_dir"] = f"./ckpt/{expconfig.keyword}_{twin_kw}/"

        print("Generating synthetic samples for", twin_kw)

        m = 20 * expconfig.n_inf  # rho_max = 20
        generate_sample(
            num_samples=int(expconfig.D * m),
            batch_size=int(expconfig.D * m / 10),
            seed=random.randint(0, 100000),
            **temp_sample_kwargs,
        )
        print("Synthetic sample saved at", temp_sample_kwargs["temp_parent_dir"])
else:
    df_twin_1 = concat_data(f"./ckpt/{expconfig.keyword}_twin_1/")
    print("Size for twin_1:", df_twin_1.shape)
    df_twin_2 = concat_data(f"./ckpt/{expconfig.keyword}_twin_2/")
    print("Size for twin_2:", df_twin_2.shape)


# Get f* and g* for constructing test statistic

df_train = concat_data(expconfig.raw_data_dir, split="train")
df_raw_all = concat_data(expconfig.raw_data_dir, split="val")

model_full = catboost_pred_model(df_train, df_raw_all, **cb_kwargs)
model_partial = catboost_pred_model(
    df_train, df_raw_all, null_features_list=expconfig.null_features_list, **cb_kwargs
)


print("Full model:", model_full.best_score_)
print("Partial model:", model_partial.best_score_)


# Perform Syn-Test tuning with fine-tuned SYNTHETIC GENERATOR, used for getting m_hat


rho_max, num_rhos = 20, 20

result_dict = {}
for i, rho in enumerate(np.linspace(0, rho_max, num_rhos + 1)[1:]):
    m = int(rho * expconfig.n_inf)

    print("-" * 100)
    print(f"rho: {rho}, m: {m}")

    temp_dict = {
        "n_inf": expconfig.n_inf,
        "rho": rho,
        "m": m,
    }
    for twin_kw in ["twin_1", "twin_2"]:
        print("Processing", twin_kw)

        df_twin = df_twin_1 if twin_kw == "twin_1" else df_twin_2
        test_stat_list, test_stat_null_list = [], []

        for d in tqdm(range(expconfig.D)):
            temp_df = df_twin.iloc[d * m : (d + 1) * m].copy()
            test_stat = blackbox_test_stat(
                temp_df,
                model_full,
                model_partial,
                null_feature_names=expconfig.null_features_list,
                **cb_kwargs,
            )
            test_stat_list.append(test_stat)

            temp_df_null = temp_df.copy()
            temp_df_null[expconfig.null_features_list] = np.random.rand(
                temp_df_null.shape[0], len(expconfig.null_features_list)
            )
            test_stat_null = blackbox_test_stat(
                temp_df_null,
                model_full,
                model_partial,
                null_feature_names=expconfig.null_features_list,
                **cb_kwargs,
            )
            test_stat_null_list.append(test_stat_null)

        temp_dict[twin_kw] = {
            "test_stat": test_stat_list,
            "test_stat_null": test_stat_null_list,
        }

    result_dict[str(rho)] = temp_dict
    pickle.dump(result_dict, open(expconfig.result_dict_save_path, "wb"))


# Perform Syn-Test tuning with TRUE GENERATOR, used as reference

seed_everything(expconfig.seed)

rho_max, num_rhos = 20, 20
total_size = rho_max * expconfig.n_inf * expconfig.D

true_sampler = TrueSampler(sigma=expconfig.sigma, null_feature=True)
temp_df = true_sampler.sample(2 * total_size, return_df=True)
raw_df_twin_1 = temp_df.iloc[:total_size].copy()
raw_df_twin_2 = temp_df.iloc[total_size:].copy()

raw_result_dict = {}
for i, rho in enumerate(np.linspace(0, rho_max, num_rhos + 1)[1:]):
    m = int(rho * expconfig.n_inf)

    print("-" * 100)
    print(f"rho: {rho}, m: {m}")

    temp_dict = {
        "n_inf": expconfig.n_inf,
        "rho": rho,
        "m": m,
    }
    for twin_kw in ["twin_1", "twin_2"]:
        print("Processing", twin_kw)

        df_twin = raw_df_twin_1 if twin_kw == "twin_1" else raw_df_twin_2
        test_stat_list, test_stat_null_list = [], []

        for d in tqdm(range(expconfig.D)):
            temp_df = df_twin.iloc[d * m : (d + 1) * m].copy()
            test_stat = blackbox_test_stat(
                temp_df,
                model_full,
                model_partial,
                null_feature_names=expconfig.null_features_list,
                **cb_kwargs,
            )
            test_stat_list.append(test_stat)

            temp_df_null = temp_df.copy()
            temp_df_null[expconfig.null_features_list] = np.random.rand(
                temp_df_null.shape[0], len(expconfig.null_features_list)
            )
            test_stat_null = blackbox_test_stat(
                temp_df_null,
                model_full,
                model_partial,
                null_feature_names=expconfig.null_features_list,
                **cb_kwargs,
            )
            test_stat_null_list.append(test_stat_null)

        temp_dict[twin_kw] = {
            "test_stat": test_stat_list,
            "test_stat_null": test_stat_null_list,
        }

    raw_result_dict[str(rho)] = temp_dict
    pickle.dump(raw_result_dict, open(expconfig.raw_result_dict_save_path, "wb"))


############## Traditional ##############

# true test statistic and p value

df_inf = concat_data(expconfig.raw_data_dir, split="val")
raw_test_stat = blackbox_test_stat(
    df_inf,
    model_full,
    model_partial,
    null_feature_names=expconfig.null_features_list,
    **cb_kwargs,
)

print("Test statistic from raw sample:", raw_test_stat)

rv = norm()
p_value = 1 - rv.cdf(raw_test_stat)
print("Raw p-value:", p_value)
