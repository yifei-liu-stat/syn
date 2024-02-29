"""
Temporary script for running Syn-Test framework for inference.
"""


CUDA_ID = 7
temp_parent_dir = "./temp2/"  # saving cached samples
null_features_list = ["num_7"]


SEED = 2023
TDDPM_DIR = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/"
SYNINF_DIR = (
    "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference/syninf"
)


import torch

torch.cuda.set_device(CUDA_ID)


import random
import numpy as np

from lightning.pytorch import seed_everything

import time
import os
from tqdm import tqdm
import json
import pickle


import sys

sys.path.insert(0, SYNINF_DIR)

from utils_syninf import (
    generate_sample,
    concat_data,
    catboost_pred_model,
    blackbox_test_stat,
)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


sigma = 0.2
n_pretrain = 10000  # pre-training size
keyword = f"inf_{n_pretrain}"
synthetic_sample_dir = f"./ckpt/{keyword}/"
ckpt_dir = f"./ckpt/{keyword}"


n_train = 1000  # raw training size
n_val = 200  # validation size
n_test = 1000  # test or evaluation size


raw_data_dir = os.path.join(TDDPM_DIR, f"data/inf_raw")
pretrain_data_dir = os.path.join(TDDPM_DIR, f"data/inf_{n_pretrain}")


kwargs = {
    "num_features_list": [f"num_{i}" for i in range(8)],
    "iterations": 2000,
    "loss_function": "MAE",
    "verbose": False,
}


# kwargs for generating samples using twin generators
tddpm_kwargs_twin = {
    twin_kw: {
        "pipeline_config_path": f"./ckpt/{keyword}_{twin_kw}/config.toml",
        "ckpt_path": f"./ckpt/{keyword}_{twin_kw}/model.pt",
        "pipeline_dict_path": os.path.join(
            ckpt_dir, "pipeline_dict.joblib"
        ),  # same as the pre-processing pipeline during its fine-tuning
        "temp_parent_dir": temp_parent_dir,
        "device": f"cuda:{CUDA_ID}",
    }
    for twin_kw in ["twin_1", "twin_2"]
}


seed_everything(SEED)


# Get f* and g* for constructing test statistic

generate_sample(
    num_samples=10 * n_train,
    batch_size=10 * n_train,
    seed=random.randint(0, 100000),
    **tddpm_kwargs_twin["twin_1"],
)
df_train = concat_data(temp_parent_dir, split="train")

df_inf = concat_data(raw_data_dir, split="val")

model_full = catboost_pred_model(df_train, df_inf, **kwargs)
model_partial = catboost_pred_model(
    df_train, df_inf, null_features_list=null_features_list, **kwargs
)


print("Full model:", model_full.best_score_)
print("Partial model:", model_partial.best_score_)


# Setup for the MC experiments

inf_dir = "./results/inf/"
if not os.path.exists(inf_dir):
    os.makedirs(inf_dir)


rho_max = 20
num_rhos = 20
D = 1000


suffix = "_".join(null_features_list)
result_dict = {}
result_dict_save_path = os.path.join(inf_dir, f"result_dict_{suffix}.pkl")


# Run MC experiments for null distribution, type-I error rate and test statistic under learned true distribution

for i, rho in enumerate(np.linspace(0, rho_max, num_rhos + 1)[1:]):
    m = int(rho * n_val)
    print(f"rho: {rho}, m: {m}")

    # Generate samples once for all given a specific m
    start = time.time()
    with HiddenPrints():
        generate_sample(
            num_samples=D * m,
            batch_size=int(D * m / 10),
            seed=random.randint(0, 100000),
            **tddpm_kwargs_twin["twin_1"],
        )
    df_twin_1 = concat_data(temp_parent_dir, split="train")
    print(df_twin_1.shape)

    with HiddenPrints():
        generate_sample(
            num_samples=2 * D * m,
            batch_size=int(2 * D * m / 10),
            seed=random.randint(0, 100000),
            **tddpm_kwargs_twin["twin_2"],
        )
    df_twin_2 = concat_data(temp_parent_dir, split="train")
    print(df_twin_2.shape)

    print(f"Time elapsed: {time.time() - start:.2f} seconds")

    # get null distribution of the test statistic
    null_T_list = []
    for D1 in tqdm(range(D)):
        temp_df = df_twin_1.iloc[D1 * m : (D1 + 1) * m].copy()
        temp_df[null_features_list] = np.random.rand(
            temp_df.shape[0], len(null_features_list)
        )

        test_stat_null = blackbox_test_stat(
            temp_df, model_full, model_partial, **kwargs
        )
        null_T_list.append(test_stat_null)

    # estimate the type-I error rate
    # and calculate the test statistic under learned true distribution
    type1_T_list, learned_true_T_list = [], []
    for D2 in tqdm(range(D)):
        # for estimating type-I error
        temp_df = df_twin_2.iloc[D2 * m : (D2 + 1) * m].copy()
        temp_df[null_features_list] = np.random.rand(
            temp_df.shape[0], len(null_features_list)
        )
        test_stat_null = blackbox_test_stat(
            temp_df, model_full, model_partial, **kwargs
        )
        type1_T_list.append(test_stat_null)

        # the test statistic under learned true distribution
        temp_df = df_twin_2.iloc[(D + D2) * m : (D + D2 + 1) * m].copy()
        test_stat_true = blackbox_test_stat(
            temp_df, model_full, model_partial, **kwargs
        )
        learned_true_T_list.append(test_stat_true)

    # save the null test statistics, as well as the one under learned true distribution
    result_dict[str(rho)] = {
        "m": m,
        "null_dist": null_T_list,
        "type1_test_stat": type1_T_list,
        "learned_true_test_stat": learned_true_T_list,
    }

    pickle.dump(result_dict, open(result_dict_save_path, "wb"))
