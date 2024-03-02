"""
Tune m to control Type-I error.

Run script_prep_model.py with proper dataset name before running this script.
"""

import os
import pickle
import json

import pandas as pd

import sys
from tqdm import tqdm
import random

from utils_syninf import *


# currently only support numerical null features

dataset_name = "adult_female_3000"


# null_features_list = ["age", "capital-gain"]
# null_features_list = ["age", "educationl-num"]
null_features_list = ["age", "educationl-num", "hours-per-week"]


# # the new fine-tuning pipeline might not work for the previous california experiment
# dataset_name = "california"
# null_features_list = ["MedInc"]


# dataset paths
dataset_dir = os.path.join(TDDPM_DIR, f"data/{dataset_name}")
dataset_dir_twin_1 = os.path.join(TDDPM_DIR, f"data/{dataset_name}_twin_1")
dataset_dir_twin_2 = os.path.join(TDDPM_DIR, f"data/{dataset_name}_twin_2")

# experiment paths
exp_dir_twin_1 = os.path.join(TDDPM_DIR, f"exp/{dataset_name}_twin_1/ddpm_cb_best")
exp_dir_twin_2 = os.path.join(TDDPM_DIR, f"exp/{dataset_name}_twin_2/ddpm_cb_best")

# meta info, names, pred & null model dictionaries
data_info_dict = json.load(open(os.path.join(dataset_dir_twin_1, "info.json"), "rb"))
names_dict = pickle.load(open(os.path.join(dataset_dir, "names_dict.pkl"), "rb"))
twin_null_model_dict = load_twin_null_models(dataset_name, null_features_list)
pred_model_dict = load_pred_models(dataset_name, null_features_list, **names_dict)

# inference result directory
inf_result_dir = os.path.join(SYNINF_DIR, dataset_name, "syngen_inf_result")
if not os.path.exists(inf_result_dir):
    os.makedirs(inf_result_dir)


# # -- test the replace null features stuff --- #

# temp_df = concat_data(f"{dataset_name}/syn_pred_sample", **names_dict)
# temp_df_copy = temp_df.copy()
# temp_df_null = replace_null_features(
#     temp_df_copy, twin_null_model_dict, twin_folder="twin_1"
# )

# test_stat_null = blackbox_test_stat(
#     temp_df_null,
#     pred_model_dict["full"],
#     pred_model_dict["partial"],
#     null_feature_names=null_features_list,
#     **names_dict,
# )

# breakpoint()

# # --- end of the test --- #


df_dist_t = concat_data(exp_dir_twin_1, **names_dict)
print(df_dist_t.shape)

df_type_i = concat_data(exp_dir_twin_2, **names_dict)
print(df_type_i.shape)


# tune m to control Type-I error
SEED = 2023
np.random.seed(SEED)
random.seed(SEED)

# n = 1000
n = data_info_dict["val_size"]
# n = (
#     data_info_dict["train_size"]
#     + data_info_dict["val_size"]
#     + data_info_dict["test_size"]
# )
rho_max = 20
num_rhos = 20
D_null_T = 1000
D_type_I = 1000


suffix = "_".join(null_features_list)

result_dict = {}
result_dict_save_path = os.path.join(inf_result_dir, f"result_dict_{suffix}.pkl")
for i, rho in enumerate(np.linspace(0, rho_max, num_rhos + 1)[1:]):
    m = int(rho * n)
    print(f"rho: {rho}, m: {m}")

    # get null distribution of the test statistic
    null_T_list = []
    for D1 in tqdm(range(D_null_T)):
        # extract the df chunk for calculating test statistic under the null
        temp_df = df_dist_t[m * D1 : m * (D1 + 1)].copy()

        # replace the null feature by the prediction of the null feature using the rest of the features as predictors
        temp_df_null = replace_null_features(
            temp_df, twin_null_model_dict, twin_folder="twin_1"
        )

        # calculate test statistic
        test_stat_null = blackbox_test_stat(
            temp_df_null,
            pred_model_dict["full"],
            pred_model_dict["partial"],
            null_feature_names=null_features_list,
            **names_dict,
        )
        null_T_list.append(test_stat_null)

    # estimate the type-I error rate
    # and calculate the test statistic under learned true distribution
    type1_T_list, learned_true_T_list = [], []
    for D2 in tqdm(range(D_type_I)):
        # extract the df chunk for estimating the type-I error
        temp_df = df_type_i[m * D2 : m * (D2 + 1)].copy()

        # replace the null feature by the prediction of the null feature using the rest of the features as predictors
        temp_df_null = replace_null_features(
            temp_df, twin_null_model_dict, twin_folder="twin_2"
        )

        # calculate the test statistic under the null
        test_stat_null = blackbox_test_stat(
            temp_df_null,
            pred_model_dict["full"],
            pred_model_dict["partial"],
            null_feature_names=null_features_list,
            **names_dict,
        )
        type1_T_list.append(test_stat_null)

        # calculate the test statistic under the learned true distribution
        test_stat_true = blackbox_test_stat(
            temp_df,
            pred_model_dict["full"],
            pred_model_dict["partial"],
            null_feature_names=null_features_list,
            **names_dict,
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
