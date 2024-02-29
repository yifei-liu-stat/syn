import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lib
import os

SEED = 0
ds_name = "ad21"
data_folder = "data/ad21"
num_numerical_features = 20  # select between 1 and 20 (the first column will be regarded as the target variable, even though it is not)


df = pd.read_csv(os.path.join(data_folder, ds_name + ".csv"))
df = df.iloc[
    :, : num_numerical_features + 1
]  # select the first (num_numerical_features + 1) columns

column_names = df.columns.values.tolist()


df_train_temp, df_test = train_test_split(df, test_size=0.1, random_state=SEED)
df_train, df_val = train_test_split(df_train_temp, test_size=0.2, random_state=SEED)


for split in ["train", "val", "test"]:
    temp_df = locals()["df_" + split]

    # save the split indices
    np.save(os.path.join(data_folder, "idx_" + split + ".npy"), temp_df.index.values)

    # save the split data
    temp_y = temp_df.iloc[:, 0].to_numpy()
    np.save(os.path.join(data_folder, "y_" + split + ".npy"), temp_y)
    temp_num_X = temp_df.iloc[:, 1:].to_numpy()
    np.save(os.path.join(data_folder, "X_num_" + split + ".npy"), temp_num_X)


# save the meta data of the dataset
info_dict = {
    "name": ds_name,
    "id": ds_name + "--default",
    "task_type": "regression",
    "n_num_feautures": num_numerical_features,
    "n_cat_features": 0,
    "train_size": len(df_train),
    "val_size": len(df_val),
    "test_size": len(df_test),
    "y_name": column_names[0],
    "X_names": column_names[1 : num_numerical_features + 1],
}
lib.dump_json(info_dict, os.path.join(data_folder, "info.json"))
