"""
This script generate synthetic samples for evaluation purpose
"""

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
from copy import deepcopy

import matplotlib.pyplot as plt


import sys

sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

from utils_tabddpm import (
    train_tabddpm,
    generate_sample,
)

from utils_syn import (
    concat_data,
)

######## Set up configurations ########

n = 32650

dataset_name = "adult"
dataset_dir = os.path.join(TDDPM_DIR, f"data/{dataset_name}")

num_features_list = [
    "age",
    "fnlwgt",
    "educationl-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

cat_features_list = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]

y_feature = "income"  # <= 50K or > 50K

is_y_cat = True

names_dict = {
    "num_features_list": num_features_list,
    "cat_features_list": cat_features_list,
    "y_feature": y_feature,
    "is_y_cat": is_y_cat,
}

pickle.dump(names_dict, open(os.path.join(dataset_dir, "names_dict.pkl"), "wb"))


########## Synthetic data generation ##########
gender_names_dict = deepcopy(names_dict)
gender_names_dict["cat_features_list"].remove("gender")

# raw test female dataset
female_df_twin1_test = concat_data(
    f"{TDDPM_DIR}/data/adult_female_3000_twin_1",
    split="test",
    **gender_names_dict,
)

# raw adult male dataset
temp_train = concat_data(
    f"{TDDPM_DIR}/data/adult_male",
    split="train",
    **gender_names_dict,
)
temp_val = concat_data(
    f"{TDDPM_DIR}/data/adult_male",
    split="val",
    **gender_names_dict,
)
temp_test = concat_data(
    f"{TDDPM_DIR}/data/adult_male",
    split="test",
    **gender_names_dict,
)
male_df = pd.concat([temp_train, temp_val, temp_test], axis=0)

# synethetic female sample from pre-trained model
synthetic_sample_dir = generate_sample(
    pipeline_config_path=f"{TDDPM_DIR}/exp/adult_female_pretraining_twin1/ddpm_cb_best/config.toml",
    ckpt_path=f"{TDDPM_DIR}/exp/adult_female_pretraining_twin1/ddpm_cb_best/model.pt",
    pipeline_dict_path=f"{TDDPM_DIR}/exp/adult_female_pretraining_twin1/ddpm_cb_best/pipeline_dict.joblib",
    num_samples=n,
    batch_size=n,
)
fake_female_df_pt = concat_data(synthetic_sample_dir, **gender_names_dict)
fake_female_df_pt["income"].cat.categories = ["0", "1"]


# synethetic female sample from fine-tuned model
synthetic_sample_dir = generate_sample(
    pipeline_config_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/config.toml",
    ckpt_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/model.pt",
    pipeline_dict_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/pipeline_dict.joblib",
    num_samples=n,
    batch_size=n,
)
fake_female_df = concat_data(synthetic_sample_dir, **gender_names_dict)
fake_female_df["income"].cat.categories = ["0", "1"]


# synethetic male sample from pre-trained model
synthetic_sample_dir = generate_sample(
    pipeline_config_path=f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/config.toml",
    ckpt_path=f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/model.pt",
    pipeline_dict_path=f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/pipeline_dict.joblib",
    num_samples=n,
    batch_size=n,
)
fake_male_df = concat_data(synthetic_sample_dir, **gender_names_dict)
fake_male_df["income"].cat.categories = ["0", "1"]


########## Save the dfs for later evaluation ##########

eval_df_dict = {
    "male_df": male_df,
    "female_df_twin1_test": female_df_twin1_test,
    "fake_female_df_pt": fake_female_df_pt,
    "fake_female_df": fake_female_df,
    "fake_male_df": fake_male_df,
    "gender_names_dict": gender_names_dict,
}

pickle.dump(eval_df_dict, open("temp_df_dict.pkl", "wb"))

print("Evaluation dataframes saved to temp_df_dict.pkl")
