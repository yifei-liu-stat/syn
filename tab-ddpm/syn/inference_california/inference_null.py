"""
SynGen: tune m to achieve the maximum power while controlling Type-I error rate
"""

import random
import numpy as np
import pandas as pd
import pickle
import subprocess
import os
import matplotlib.pyplot as plt
from dython.nominal import associations
import seaborn as sns

from catboost import CatBoostRegressor, Pool

from tqdm import tqdm
import pickle

import statsmodels.api as sm

PASS_INFERENCE_PATH = (
    "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference"
)


NULL_DATA_PATH_DIST_T = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/exp/california_twin_1/ddpm_cb_best"

NULL_DATA_PATH_TYPE_I = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/exp/california_twin_2/ddpm_cb_best"

FAKE_DATA_PATH = (
    "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/exp/california/ddpm_cb_best"
)


num_features_list = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


cat_features_list = None

y_features_list = ["MedHouseVal"]


SEED = 2023
SEED = 1234
np.random.seed(SEED)
random.seed(SEED)


def concat_data(
    data_path,
    num_features_list=None,
    cat_features_list=None,
    y_features_list=None,
    is_y_cat=False,
):
    """
    Aggregate generated features and the response to a dataframe.
    """
    concat_list, col_names = [], []
    y_test_syn = np.load(os.path.join(data_path, "y_train.npy"))
    concat_list.append(y_test_syn[:, None])
    col_names += y_features_list
    if num_features_list is not None:
        X_num_test_syn = np.load(os.path.join(data_path, "X_num_train.npy"))
        concat_list.append(X_num_test_syn)
        col_names += num_features_list
    else:
        num_features_list = []
    if cat_features_list is not None:
        X_cat_test_syn = np.load(
            os.path.join(data_path, "X_cat_train.npy"), allow_pickle=True
        )
        concat_list.append(X_cat_test_syn)
        col_names += cat_features_list
    else:
        cat_features_list = []
    temp_df = pd.DataFrame(
        np.concatenate(concat_list, axis=1),
        columns=col_names,
    )
    cat_list = (
        cat_features_list if is_y_cat == False else cat_features_list + y_features_list
    )
    new_types = {
        col_name: "category" if col_name in cat_list else "float"
        for col_name in col_names
    }
    temp_df = temp_df.astype(new_types)
    return temp_df


def california_null_catboost(
    target_folders="/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_1",
    X_label_idx=0,
):
    """Using twin folders, train a CatBoost model on the null feature using the rest of the features as predictors."""
    if not isinstance(target_folders, list):
        target_folders = [target_folders]
    temp_dict = {
        "label_train": [],
        "predictors_train": [],
    }
    for target_folder in target_folders:
        X_num_train = np.load(os.path.join(target_folder, "X_num_train.npy"))
        X_num_val = np.load(os.path.join(target_folder, "X_num_val.npy"))
        # use the null feature as the response
        label_train = X_num_train[:, X_label_idx]
        label_val = X_num_val[:, X_label_idx]
        # use the rest of the features as predictors
        predictors_train = np.delete(X_num_train, X_label_idx, axis=1)
        predictors_val = np.delete(X_num_val, X_label_idx, axis=1)
        # append to the temp_dict
        temp_dict["label_train"].append(label_train)
        temp_dict["predictors_train"].append(predictors_train)
    # aggregate the training data (no need for validation data since they are the same based on twin folder structure)
    label_train = np.concatenate(temp_dict["label_train"], axis=0)
    predictors_train = np.concatenate(temp_dict["predictors_train"], axis=0)
    # train the model
    train_dataset = Pool(data=predictors_train, label=label_train)
    eval_dataset = Pool(data=predictors_val, label=label_val)
    model = CatBoostRegressor(iterations=2000, loss_function="RMSE")
    model.fit(train_dataset, use_best_model=True, eval_set=eval_dataset)
    return model


def replace_null_feature_by_prediction(
    temp_df,
    null_feature_name=None,
    cat_features_list=None,
    y_features_list=None,
    pred_model=None,
):
    """Replace the null feature by the prediction of the null feature using the rest of the features as predictors."""
    df_copy = temp_df.copy()
    if cat_features_list is not None:
        df_copy = pd.get_dummies(df_copy, columns=cat_features_list, drop_first=True)
    # predictors in X for predicting the null feature
    pred_df = df_copy.drop([null_feature_name, y_features_list[0]], axis=1)
    # predict the null feature as the new generated feature
    df_copy[null_feature_name] = pred_model.predict(pred_df)
    return df_copy


def ols_t_test_statistic(temp_df, null_feature_name=None):
    """
    T test statistic of the null feature based on OLS model.
    - temp_df: dataframe of all features and response (first column). Categorical features should be one-hot encoded. It can be the one directly returned from replace_null_feature_by_prediction()
    """
    df_copy = temp_df.copy()
    new_names = list(df_copy.columns)
    olm = sm.OLS(df_copy[new_names[0]], df_copy[new_names[1:]]).fit()
    return olm.tvalues[null_feature_name]


def one_split_test_stat(
    temp_df,
    null_feature_name=None,
    train_inf_ratio=0.8,
    cat_features_list=None,
    y_features_list=None,
):
    """
    One-split test statistic for feature relevance: https://arxiv.org/pdf/2103.04985.pdf
    - temp_df: dataframe with all features and response with column names corresponding to xxx_list
    - null_feature_idx: the index of the null feature in the num_features_list
    - train_inf_ratio: the ratio of training data to inference data for splitting

    Example:
    >>> temp_df = concat_data(
    >>>     FAKE_DATA_PATH, num_features_list, cat_features_list, y_features_list
    >>> )
    >>>
    >>> test_stat = one_split_test_stat(
    >>>     temp_df,
    >>>     null_feature_name="HouseAge",
    >>>     cat_features_list=cat_features_list,
    >>>     y_features_list=y_features_list,
    >>> )
    >>>
    >>> print(test_stat)
    """
    df_copy = temp_df.copy()
    # 1. split into two parts: training and inference
    train_len = int(train_inf_ratio * df_copy.shape[0])
    df_copy_train = df_copy[:train_len]
    df_copy_inf = df_copy[train_len:]
    # 2. train a partial model and get residulas
    predictors_train = df_copy_train.drop(
        [null_feature_name, y_features_list[0]], axis=1
    )
    label_train = df_copy_train[y_features_list[0]]
    train_dataset = Pool(data=predictors_train, label=label_train)
    predictors_val = df_copy_inf.drop([null_feature_name, y_features_list[0]], axis=1)
    label_val = df_copy_inf[y_features_list[0]]
    eval_dataset = Pool(data=predictors_val, label=label_val)
    partial_model = CatBoostRegressor(iterations=2000, loss_function="RMSE")
    partial_model.fit(
        train_dataset, use_best_model=True, eval_set=eval_dataset, verbose=False
    )
    partial_preds = partial_model.predict(predictors_val)
    true_labels = label_val.to_numpy()
    partial_residuals = np.abs(true_labels - partial_preds)
    # 3. train a full model and get residuals
    predictors_train = df_copy_train.drop([y_features_list[0]], axis=1)
    label_train = df_copy_train[y_features_list[0]]
    train_dataset = Pool(data=predictors_train, label=label_train)
    predictors_val = df_copy_inf.drop([y_features_list[0]], axis=1)
    label_val = df_copy_inf[y_features_list[0]]
    eval_dataset = Pool(data=predictors_val, label=label_val)
    full_model = CatBoostRegressor(iterations=2000, loss_function="RMSE")
    full_model.fit(
        train_dataset, use_best_model=True, eval_set=eval_dataset, verbose=False
    )
    full_preds = full_model.predict(predictors_val)
    true_labels = label_val.to_numpy()
    full_residuals = np.abs(true_labels - full_preds)
    # 4. calculate the test statistic
    # ## correlation: I don't think it makes sense
    # test_stat = np.corrcoef(partial_residuals, full_residuals)[0, 1]
    ## t-value based on residual difference
    ### supposely to be significantly larger if the null feature is important in terms of prediction
    residual_diff = partial_residuals - full_residuals
    test_stat = (
        np.mean(residual_diff) / np.std(residual_diff) * np.sqrt(residual_diff.shape[0])
    )
    return test_stat


# generated data for estimating null distribution of the test statistic
df_dist_t = concat_data(
    NULL_DATA_PATH_DIST_T, num_features_list, cat_features_list, y_features_list
)  # (99070000, 9)

# generated data for estimating type-I error
df_type_i = concat_data(
    NULL_DATA_PATH_TYPE_I, num_features_list, cat_features_list, y_features_list
)  # (99070000, 9)


# --- sanity check of the learned all generative model --- #

# TRUE_DATA_PATH = (
#     "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_2"
# )
# temp = np.load(os.path.join(TRUE_DATA_PATH, "X_num_train.npy"))

# true_df = pd.DataFrame(temp, columns=num_features_list)
# true_df.corr()


# df_dist_t[:10000].corr()
# df_type_i[:10000].corr()

# --- end of sanity check --- #


n = 9907
null_feature_idx = 0
null_feature_name = num_features_list[null_feature_idx]


# test run
rho_max = 10
num_rhos = 1
D_null_T = 100
D_type_I = 100
# rho_max, num_rhos = 5, 5: [0.05 0.03 0.05 0.05 0.03]

# # real run
# rho_max = 10
# num_rhos = 20
# D_null_T = 1000
# D_type_I = 1000


# model_twin_0 = california_null_catboost(
#     target_folders=[
#         "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_1",
#         "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_2",
#     ],
#     X_label_idx=null_feature_idx,
# )

model_twin_1 = california_null_catboost(
    target_folders="/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_1",
    X_label_idx=null_feature_idx,
)

model_twin_2 = california_null_catboost(
    target_folders="/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california_twin_2",
    X_label_idx=null_feature_idx,
)


result_dict = {}  # rho: {"m": ..., "null_dist": ..., "type1_test_stat": ...}

for i, rho in enumerate(np.linspace(0, rho_max, num_rhos + 1)[1:]):
    m = int(rho * n)
    print(f"{i} / {num_rhos}, rho: {rho}, m: {m}")
    # get null distribution of the test statistic
    null_T_list = []
    for D1 in tqdm(range(D_null_T)):
        # extract the df chunk for calculating test statistic under the null
        temp_df = df_dist_t[m * D1 : m * (D1 + 1)].copy()
        # replace the null feature by the prediction of the null feature using the rest of the features as predictors
        temp_df_null = replace_null_feature_by_prediction(
            temp_df,
            null_feature_name=null_feature_name,
            cat_features_list=cat_features_list,
            y_features_list=y_features_list,
            pred_model=model_twin_1,
        )
        # test_stat = ols_t_test_statistic(
        #     temp_df_null, null_feature_name
        # )  # OLS test statistic
        test_stat = one_split_test_stat(
            temp_df_null,
            null_feature_name=null_feature_name,
            train_inf_ratio=0.8,
            cat_features_list=cat_features_list,
            y_features_list=y_features_list,
        )  # one-split test statistic
        null_T_list.append(test_stat)
    # estimate the type-I error rate
    type1_T_list = []
    for D2 in tqdm(range(D_type_I)):
        # extract the df chunk for estimating the type-I error
        temp_df = df_type_i[m * D2 : m * (D2 + 1)].copy()
        # replace the null feature by the prediction of the null feature using the rest of the features as predictors
        temp_df_null = replace_null_feature_by_prediction(
            temp_df,
            null_feature_name=null_feature_name,
            cat_features_list=cat_features_list,
            y_features_list=y_features_list,
            pred_model=model_twin_2,
        )
        # test_stat = ols_t_test_statistic(
        #     temp_df_null, null_feature_name
        # )  # OLS test statistic
        test_stat = one_split_test_stat(
            temp_df_null,
            null_feature_name=null_feature_name,
            train_inf_ratio=0.8,
            cat_features_list=cat_features_list,
            y_features_list=y_features_list,
        )  # one-split test statistic
        type1_T_list.append(test_stat)
    # save the result
    result_dict[str(rho)] = {
        "m": m,
        "null_dist": null_T_list,
        "type1_test_stat": type1_T_list,
    }
    pickle.dump(
        result_dict,
        open(
            os.path.join(
                PASS_INFERENCE_PATH,
                f"syngen_inf_res/california_{null_feature_name}.pkl",
            ),
            "wb",
        ),
    )


# load the saved result
alpha = 0.05
result_dict = pickle.load(
    open(
        os.path.join(
            PASS_INFERENCE_PATH,
            f"syngen_inf_res/california_{null_feature_name}.pkl",
        ),
        "rb",
    )
)

type_I_error_list = []
for k, v_dict in result_dict.items():
    null_dist = np.array(v_dict["null_dist"])
    type1_test_stat = np.array(v_dict["type1_test_stat"])
    p_values = []
    for t in type1_test_stat:
        # p_value = 2 * min(np.mean(null_dist >= t), np.mean(null_dist <= t))
        p_value = np.mean(null_dist >= t)
        p_values.append(p_value)
    type_I_error = np.mean(np.array(p_values) <= alpha)
    type_I_error_list.append(type_I_error)

type_I_error_list = np.array(type_I_error_list)


print(type_I_error_list)

# test statistic using one split test statistic: n = 9907

# test statistic using OLS t-value: n = 826
# 1-2: [0.087 0.083 0.085 0.073 0.075 0.099 0.091 0.086 0.089 0.104 0.124 0.094 0.123 0.137 0.118 0.146 0.158 0.155 0.167 0.172]
# 0-0: [0.085 0.11  0.103 0.098 0.106 0.137 0.148 0.166 0.156 0.197 0.27  0.227 0.239 0.266 0.295 0.299 0.297 0.325 0.356 0.342]
# perm: [0.066 0.048 0.037 0.068 0.064 0.06  0.041 0.052 0.037 0.047 0.063 0.068 0.066 0.052 0.052 0.045 0.05  0.053 0.057 0.056]
