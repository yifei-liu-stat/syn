"""
Divide data into three folds: 0.48 (train generative/predictive model) + 0.48 (same) + 0.04 (validation)
- (TBD) generalize this script to othe datasets: name-agonostic
- cd to under the data folder: "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/data/california"
- This split is irrelavant to the original stuff
- Will used the derived results in two copies of the data (for null generator)
- In addition, it will create names dictionary under the original data folder
"""

import numpy as np
import os
import shutil
import pickle
import json

from catboost import CatBoostRegressor, Pool, metrics, cv


# Aggregate all splits
for prefix in ["X_num", "X_cat", "y"]:
    if os.path.exists(f"{prefix}_train.npy"):
        train_data = np.load(f"{prefix}_train.npy")
        test_data = np.load(f"{prefix}_test.npy")
        val_data = np.load(f"{prefix}_val.npy")
        agg_data = np.concatenate([train_data, test_data, val_data], axis=0)
        np.save(f"{prefix}_agg.npy", agg_data)


# Create a dictionary for the twin folders
train_split = 0.48
val_split = 1 - 2 * train_split

train_len = int(len(agg_data) * train_split)
val_len = len(agg_data) - 2 * train_len


np.random.seed(2023)
temp = np.random.choice(np.arange(0, len(agg_data)), 2 * train_len, replace=False)
val_idx = np.array(list(set(np.arange(0, len(agg_data))) - set(temp)))

inf_idx_dict = {
    "train_idx_1": temp[:train_len],
    "train_idx_2": temp[train_len:],
    "val_idx": val_idx,
}


pickle.dump(inf_idx_dict, open("inf_idx_dict.pkl", "wb"))


# Create twin folders for the dataset

source_folder = "./california"

## Twin folder 1
target_folder = "../california_twin_1"
os.mkdir(target_folder)

### Copy the data meta information JSON file
info = json.load(open("info.json", "r"))
info["train_size"] = train_len
info["val_size"] = val_len
info["test_size"] = train_len

json.dump(info, open(os.path.join(target_folder, "info.json"), "w"))


### Save splits
for prefix in ["X_num", "X_cat", "y"]:
    try:
        agg_data = np.load(f"{prefix}_agg.npy")
    except:
        pass
    else:
        np.save(
            os.path.join(target_folder, f"{prefix}_train.npy"),
            agg_data[inf_idx_dict["train_idx_1"]],
        )
        np.save(
            os.path.join(target_folder, f"{prefix}_test.npy"),
            agg_data[inf_idx_dict["train_idx_2"]],
        )
        np.save(
            os.path.join(target_folder, f"{prefix}_val.npy"),
            agg_data[inf_idx_dict["val_idx"]],
        )


## Twin folder 2
target_folder = "../california_twin_2"
os.mkdir(target_folder)

### Copy the data meta information JSON file
info = json.load(open("info.json", "r"))
info["train_size"] = train_len
info["val_size"] = val_len
info["test_size"] = train_len

json.dump(info, open(os.path.join(target_folder, "info.json"), "w"))


### Save splits (switch idx_1 and idx_2)
for prefix in ["X_num", "X_cat", "y"]:
    try:
        agg_data = np.load(f"{prefix}_agg.npy")
    except:
        pass
    else:
        np.save(
            os.path.join(target_folder, f"{prefix}_train.npy"),
            agg_data[inf_idx_dict["train_idx_2"]],
        )
        np.save(
            os.path.join(target_folder, f"{prefix}_test.npy"),
            agg_data[inf_idx_dict["train_idx_1"]],
        )
        np.save(
            os.path.join(target_folder, f"{prefix}_val.npy"),
            agg_data[inf_idx_dict["val_idx"]],
        )


# Train a predictive model on twin datasets

## Twin data 1
target_folder = "../california_twin_1"

X_num_train = np.load(os.path.join(target_folder, "X_num_train.npy"))
X_num_val = np.load(os.path.join(target_folder, "X_num_val.npy"))
y_train = np.load(os.path.join(target_folder, "y_train.npy"))
y_val = np.load(os.path.join(target_folder, "y_val.npy"))


train_dataset = Pool(data=X_num_train, label=y_train)
eval_dataset = Pool(data=X_num_val, label=y_val)

model = CatBoostRegressor(iterations=2000, loss_function="RMSE")
model.fit(train_dataset, use_best_model=True, eval_set=eval_dataset)
model.save_model(os.path.join(target_folder, "california.dump"))


## Twin data 2
target_folder = "../california_twin_2"

X_num_train = np.load(os.path.join(target_folder, "X_num_train.npy"))
X_num_val = np.load(os.path.join(target_folder, "X_num_val.npy"))
y_train = np.load(os.path.join(target_folder, "y_train.npy"))
y_val = np.load(os.path.join(target_folder, "y_val.npy"))


train_dataset = Pool(data=X_num_train, label=y_train)
eval_dataset = Pool(data=X_num_val, label=y_val)

model = CatBoostRegressor(iterations=2000, loss_function="RMSE")
model.fit(train_dataset, use_best_model=True, eval_set=eval_dataset)
model.save_model(os.path.join(target_folder, "california.dump"))


# names dictionary

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

is_y_cat = False

names_dict = {
    "num_features_list": num_features_list,
    "cat_features_list": cat_features_list,
    "y_features_list": y_features_list,
    "is_y_cat": is_y_cat,
}

pickle.dump(names_dict, open("names_dict.pkl", "wb"))
