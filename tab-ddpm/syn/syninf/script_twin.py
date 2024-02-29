"""
Prepare twin folders, fine tune models and generate fake data for inference
- twin folders split: train_ratio, val_ratio, test_ratio: 0.45, 0.1, 0.45 (default)
"""

import pickle
from utils_syninf import *

dataset_name = "adult_female_3000"
twin_split = (0.45, 0.1, 0.45)  # train, val, test

# for generating synthetic data for syninf
rho_max = 20
D = 1000


# for fine-tuning purpose
finetune_lr = 0.002
pipeline_config_path = f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/config.toml"
ckpt_path = f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/model.pt"
pipeline_dict_path = f"{TDDPM_DIR}/exp/adult_male/ddpm_cb_best/pipeline_dict.joblib"


dataset_dir = os.path.join(TDDPM_DIR, f"data/{dataset_name}")
names_dict = pickle.load(open(os.path.join(dataset_dir, "names_dict.pkl"), "rb"))

seed = 2023
device = "cuda:0"
twin_name_list = ["twin_1", "twin_2"]


# read all data from the dataset folder
train_df = concat_data(dataset_dir, split="train", **names_dict)
val_df = concat_data(dataset_dir, split="val", **names_dict)
test_df = concat_data(dataset_dir, split="test", **names_dict)

overall_df = pd.concat([train_df, val_df, test_df], axis=0)


# create twin data folders
reverse_flag = False
train_data_dir_dict = {}
for twin_name in twin_name_list:
    twin_data_dir = os.path.join(TDDPM_DIR, f"data/{dataset_name}_{twin_name}")
    twin_data_dir = prepare_train_data(
        temp_df=overall_df,
        train_data_dir=twin_data_dir,
        split=twin_split,
        seed=seed,
        reverse=reverse_flag,
        **names_dict,
    )
    pickle.dump(names_dict, open(os.path.join(twin_data_dir, "names_dict.pkl"), "wb"))

    reverse_flag = not reverse_flag
    train_data_dir_dict[twin_name] = twin_data_dir


# create twin generative models via fine-tuning (if ckpt_path is not None. otherwise, train from scratch)
train_asset_dir_dict = {}
for twin_name in twin_name_list:
    temp_parent_dir = os.path.join(
        TDDPM_DIR, f"exp/{dataset_name}_{twin_name}/ddpm_cb_best"
    )
    if not os.path.exists(temp_parent_dir):
        temp_parent_dir = train_tabddpm(
            pipeline_config_path=pipeline_config_path,
            real_data_dir=train_data_dir_dict[twin_name],
            ckpt_path=ckpt_path,
            pipeline_dict_path=pipeline_dict_path,
            lr=finetune_lr,
            temp_parent_dir=temp_parent_dir,
            device=device,
        )
    train_asset_dir_dict[twin_name] = temp_parent_dir


# generate fake data for inference
for twin_name in twin_name_list:
    data_info_dict = json.load(
        open(os.path.join(train_data_dir_dict[twin_name], "info.json"), "rb")
    )
    total_size = (
        data_info_dict["train_size"]
        + data_info_dict["val_size"]
        + data_info_dict["test_size"]
    )
    num_samples = int(total_size * rho_max * D)
    batch_size = int(total_size * rho_max * D / 100)

    _ = generate_sample(
        pipeline_config_path=os.path.join(
            train_asset_dir_dict[twin_name], "config.toml"
        ),
        ckpt_path=os.path.join(train_asset_dir_dict[twin_name], "model.pt"),
        pipeline_dict_path=pipeline_dict_path,
        num_samples=num_samples,
        batch_size=batch_size,
        temp_parent_dir=train_asset_dir_dict[twin_name],
        device=device,
    )
