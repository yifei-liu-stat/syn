import optuna
from optuna.samplers import TPESampler

import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json


ds_dict = {
    "insurance": "rmse",
    "abalone": "rmse",
    "gesture": "acc",
    "churn2": "acc",
    "california": "rmse",
    "house": "rmse",
    "adult": "acc",
    "fb-comments": "rmse",
}


INFERENCE_FOLDER = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference"

for DS_NAME, EVALUATION_METRIC in ds_dict.items():
    if EVALUATION_METRIC in ["r2", "f1"]:
        # default metrics
        storage_name = f"sqlite:///{INFERENCE_FOLDER}/ratio_optuna_studies/{DS_NAME}.db"
        pickle_path = f"{INFERENCE_FOLDER}/fake_to_real_ratio/ratio_list_{DS_NAME}.pkl"
    else:
        storage_name = f"sqlite:///{INFERENCE_FOLDER}/ratio_optuna_studies/{DS_NAME}_{EVALUATION_METRIC}.db"
        pickle_path = f"{INFERENCE_FOLDER}/fake_to_real_ratio/ratio_list_{DS_NAME}_{EVALUATION_METRIC}.pkl"

    fake_to_real_ratio_list = pickle.load(open(pickle_path, "rb"))
    fake_to_real_ratio_list.sort()

    best_score_list = []
    for fake_to_real_ratio in tqdm(fake_to_real_ratio_list):
        study_name = f"catboost_{DS_NAME}_fake_to_real_ratio_{fake_to_real_ratio}"
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_score_list.append(study.best_value)

    if EVALUATION_METRIC == "rmse":
        best_score_list = -np.array(best_score_list)
    else:
        # error mesuure: 1 - r2, 1 - f1, 1 - accuracy
        best_score_list = 1 - np.array(best_score_list)

    print(
        f"{DS_NAME} ({EVALUATION_METRIC}): {best_score_list[0]} (True) versus {best_score_list.min()} (SynGen) "
    )
