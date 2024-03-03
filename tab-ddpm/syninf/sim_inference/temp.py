"""
Temporary script for running in tmux and getting the comparison results using true/learned/learned_null generators.
"""


# %%
CUDA_ID = 6
SEED = 2023
TDDPM_DIR = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/"
SYNINF_DIR = (
    "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference/syninf"
)

# %%

import torch

torch.cuda.set_device(CUDA_ID)


import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform

from lightning.pytorch import seed_everything


from copy import deepcopy
import os
import shutil
from tqdm import tqdm
import json
import pickle

import seaborn as sns
import matplotlib.pyplot as plt


from sample import TrueSampler

import sys

sys.path.insert(0, SYNINF_DIR)

from utils_syninf import (
    train_tabddpm,
    generate_sample,
    concat_data,
    catboost_pred_model,
    blackbox_test_stat,
)
from utils_num import wasserstein_2_distance
from syninf.utils.utils_viz import compare_distributions_grid, heatmap_correlation


# %%
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


# %% [markdown]
# # Get original data and pre-training data ready
# * Original data: 1000 training + 200 inference
# * Pre-training size $n_h = 2000$ + **SAME** inference set (as validation)

# %%
sigma = 0.2
n_pretrain = 10000  # pre-training size
keyword = f"inf_{n_pretrain}"
synthetic_sample_dir = f"./ckpt/{keyword}/"


n_train = 1000  # raw training size
n_val = 200  # validation size
n_test = 1000  # test or evaluation size

# %%
true_sampler = TrueSampler(sigma=sigma, null_feature=True)


# %% [markdown]
# # Syn-Test

# %%
ckpt_dir = f"./ckpt/{keyword}"


# %% [markdown]
# ## Compare distributions of the test statistic
#
# * Under size: 1000 (twin: 500 + 500) + 200
# * Of three generators: true generator, learned generator (**twin_1**) and learned null generator (**twin_1**)
# * For two candidate features: $X_7$ (significant, "num_6", on a second thought, this one is not indicative of the null distribution, so use the 8th feature only) and $X_8$ (insignificant, "num_7")

# %%
D = 500
null_features_list = ["num_7"]  # versus ["num_6"]

kwargs = {
    "num_features_list": [f"num_{i}" for i in range(8)],
    "iterations": 1000,
    "loss_function": "MAE",
    "verbose": False,
}

# kwargs for generating samples using twin_1 tddpm
tddpm_kwargs = {
    "pipeline_config_path": f"./ckpt/{keyword}_twin_1/config.toml",
    "ckpt_path": f"./ckpt/{keyword}_twin_1/model.pt",
    "pipeline_dict_path": os.path.join(
        ckpt_dir, "pipeline_dict.joblib"
    ),  # same as the pre-processing pipeline during its fine-tuning
    "temp_parent_dir": "./temp/",
    "device": f"cuda:{CUDA_ID}",
}

comparison_dir = "./results/comparison/"

# %% [markdown]
# ### True generator

# %%
np.random.seed(SEED)

test_stat_list = []
temp_save_dir = os.path.join(comparison_dir, "true")
if not os.path.exists(temp_save_dir):
    os.makedirs(temp_save_dir)

for d in tqdm(range(D)):
    with HiddenPrints():
        df_train = true_sampler.sample(n_train, return_df=True)
        df_inf = true_sampler.sample(n_val, return_df=True)

        model_full = catboost_pred_model(df_train, df_inf, **kwargs)
        model_partial = catboost_pred_model(
            df_train, df_inf, null_features_list=null_features_list, **kwargs
        )

    test_stat = blackbox_test_stat(df_inf, model_full, model_partial, **kwargs)
    test_stat_list.append(test_stat)
    pickle.dump(
        test_stat_list,
        open(
            os.path.join(temp_save_dir, f"{n_train}_{D}_{null_features_list[0]}.pkl"),
            "wb",
        ),
    )

# %% [markdown]
# ### Learned generator

# %%
seed_everything(SEED)

test_stat_list = []
temp_save_dir = os.path.join(comparison_dir, "learned")
if not os.path.exists(temp_save_dir):
    os.makedirs(temp_save_dir)


for d in tqdm(range(D)):
    with HiddenPrints():
        generate_sample(
            num_samples=n_train,
            batch_size=n_train,
            seed=random.randint(0, 100000),
            **tddpm_kwargs,
        )
        df_train = concat_data("./temp/", split="train")

        generate_sample(
            num_samples=n_val,
            batch_size=n_val,
            seed=random.randint(0, 100000),
            **tddpm_kwargs,
        )
        df_inf = concat_data("./temp/", split="train")

        model_full = catboost_pred_model(df_train, df_inf, **kwargs)
        model_partial = catboost_pred_model(
            df_train, df_inf, null_features_list=null_features_list, **kwargs
        )

    test_stat = blackbox_test_stat(df_inf, model_full, model_partial, **kwargs)
    test_stat_list.append(test_stat)
    pickle.dump(
        test_stat_list,
        open(
            os.path.join(temp_save_dir, f"{n_train}_{D}_{null_features_list[0]}.pkl"),
            "wb",
        ),
    )

# %% [markdown]
# ### Learned null generator
#
# For illustration and simplicity, we use marginal Uniform[0, 1] to replace the 8th feature.

# %%
seed_everything(SEED)

test_stat_list = []
temp_save_dir = os.path.join(comparison_dir, "learned_null")
if not os.path.exists(temp_save_dir):
    os.makedirs(temp_save_dir)


for d in tqdm(range(D)):
    with HiddenPrints():
        generate_sample(
            num_samples=n_train,
            batch_size=n_train,
            seed=random.randint(0, 100000),
            **tddpm_kwargs,
        )
        df_train = concat_data("./temp/", split="train")
        df_train[null_features_list] = np.random.rand(
            df_train.shape[0], len(null_features_list)
        )

        generate_sample(
            num_samples=n_val,
            batch_size=n_val,
            seed=random.randint(0, 100000),
            **tddpm_kwargs,
        )
        df_inf = concat_data("./temp/", split="train")
        df_inf[null_features_list] = np.random.rand(
            df_inf.shape[0], len(null_features_list)
        )

        model_full = catboost_pred_model(df_train, df_inf, **kwargs)
        model_partial = catboost_pred_model(
            df_train, df_inf, null_features_list=null_features_list, **kwargs
        )

    test_stat = blackbox_test_stat(df_inf, model_full, model_partial, **kwargs)
    test_stat_list.append(test_stat)
    pickle.dump(
        test_stat_list,
        open(
            os.path.join(temp_save_dir, f"{n_train}_{D}_{null_features_list[0]}.pkl"),
            "wb",
        ),
    )
