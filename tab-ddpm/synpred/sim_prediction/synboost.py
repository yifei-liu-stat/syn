import os

REPO_DIR = os.environ.get("REPO_DIR")
TDDPM_DIR = os.path.join(REPO_DIR, "tab-ddpm")

import numpy as np

from tqdm import tqdm
import pickle

import sys
sys.path.insert(0, os.path.join(TDDPM_DIR, "utils/"))

from utils_tabddpm import (
    generate_sample,
)

from utils_syn import (
    concat_data,
    catboost_pred_model,
    test_rmse,
)



n_pretrain = 5000

keyword = f"reg_{n_pretrain}"
ckpt_dir = f"./ckpt/{keyword}"
raw_data_dir = os.path.join(TDDPM_DIR, f"data/reg_raw")
pretrain_data_dir = os.path.join(TDDPM_DIR, f"data/reg_{n_pretrain}")



########## Get raw data and its splits ready ##########

num_features_list = [f"num_{i}" for i in range(7)]

train_df = concat_data(raw_data_dir, split="train")
val_df = concat_data(raw_data_dir, split="val")
test_df = concat_data(raw_data_dir, split="test")

########## Train CatBoost model on raw data ##########

raw_model = catboost_pred_model(
    train_df,
    val_df,
    num_features_list=num_features_list,
    iterations=2000,
    loss_function="RMSE",
    verbose=False,
)

test_rmse_raw = test_rmse(raw_model, test_df)
print("Regression using raw training data:")
print("Validation:", raw_model.get_best_score())
print("Test RMSE:", test_rmse_raw)



pretrain_df = concat_data(pretrain_data_dir, split="train")

pretrain_model = catboost_pred_model(
    pretrain_df,
    val_df,
    num_features_list=num_features_list,
    iterations=2000,
    loss_function="RMSE",
    verbose=False,
)

print("Regression using pre-training data:")
print("Validation:", pretrain_model.get_best_score())
print("Test RMSE:", test_rmse(pretrain_model, test_df))


########## Syn-Boost tuning ############
rho_min, rho_max, step_size = 1, 30, 1
rho_list = np.linspace(rho_min, rho_max, int((rho_max - rho_min) / step_size) + 1)

result_dict = {"rhos": rho_list, "scores": []}

for rho in tqdm(rho_list):
    m = int(len(train_df) * rho)

    temp_dir = generate_sample(
        pipeline_config_path=f"./ckpt/{keyword}_finetuned/config.toml",
        ckpt_path=f"./ckpt/{keyword}_finetuned/model.pt",
        pipeline_dict_path=f"./ckpt/{keyword}_finetuned/pipeline_dict.joblib",
        num_samples=m,
        batch_size=m,
        temp_parent_dir="./temp",
    )

    fake_train_df = concat_data(temp_dir, split="train")

    fake_train_model = catboost_pred_model(
        fake_train_df,
        val_df,
        num_features_list=num_features_list,
        iterations=2000,
        loss_function="RMSE",
        verbose=False,
    )

    score = test_rmse(fake_train_model, test_df)
    result_dict["scores"].append(score)

    pickle.dump(result_dict, open(f"./results/{keyword}_finetuned.pkl", "wb"))
    print(f"rho = {rho}, m = {m}: Test RMSE is {score}.")