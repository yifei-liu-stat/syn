import optuna
from optuna.samplers import TPESampler

import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json


# DS_NAME = "insurance"       # regression
# DS_NAME = "abalone"  # regression
# DS_NAME = "gesture"  # classification
# DS_NAME = "churn2"  # classification
# DS_NAME = "california"  # regression
# DS_NAME = "house"  # regression
# DS_NAME = "adult"  # classification
# DS_NAME = "fb-comments"  # regression

INFERENCE_FOLDER = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference"
DS_NAME = "adult"
EVALUATION_METRIC = "acc"  # if sets to "rmse", it is in negative scale

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


nn_result_path = f"/home/liu00980/Documents/multimodal/tabular/tab-ddpm/tuned_models/mlp/{DS_NAME}_val_metrics.json"
metric_report = json.load(open(nn_result_path, "r"))
nn_best_score = metric_report["test"][EVALUATION_METRIC]

if EVALUATION_METRIC == "rmse":
    best_score_list = -np.array(best_score_list)
else:
    # error mesuure: 1 - r2, 1 - f1, 1 - accuracy
    best_score_list = 1 - np.array(best_score_list)
    nn_best_score = 1 - nn_best_score


plt.style.use("bmh")


fig, ax = plt.subplots(figsize=(15, 7))
# plt.figure(figsize=(15, 7))

ax.plot(
    fake_to_real_ratio_list,
    best_score_list,
    marker="s",
    mew=2,
    linestyle="-",
    markersize=8,
    linewidth=2,
)


raw_size, pretrain_size = 1, 4
syngen_size = fake_to_real_ratio_list[np.argmin(best_score_list)]

y_min, y_max = ax.get_ylim()
y_min, y_max = min(y_min, nn_best_score), max(y_max, nn_best_score)
y_min, y_max = y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)
marker_kargs = {
    "y": y_min,
    "marker": "*",
    "zorder": 10,
    "clip_on": False,
    "s": 250,
}

size_1 = ax.scatter(x=raw_size, color="grey", label="raw size", **marker_kargs)
size_2 = ax.scatter(x=pretrain_size, color="C1", label="pretrain size", **marker_kargs)
size_3 = ax.scatter(x=syngen_size, color="C0", label="Syn-Boost size", **marker_kargs)

ax.set_ylim(y_min, y_max)


base_line_value = best_score_list[0]
tuned_value = (
    min(best_score_list) if EVALUATION_METRIC != "rmse" else min(best_score_list)
)
error_1 = ax.axhline(base_line_value, linestyle="--", color="grey", label="CatBoost")
error_2 = ax.axhline(tuned_value, linestyle="--", color="C0", label="Syn-Boost")
error_nn = ax.axhline(nn_best_score, linestyle="--", color="C1", label="FNN")

ax2 = ax.twinx()
ax2.set_yticks([base_line_value, tuned_value, nn_best_score])

ax.tick_params(axis="both", which="major", labelsize=14)
ax2.set_ylim(ax.get_ylim())
ax2.tick_params(axis="both", which="major", labelsize=14)


first_legend = plt.legend(
    handles=[error_1, error_2, error_nn],
    loc="upper right",
    fontsize=14,
    ncol=3,
    bbox_to_anchor=[1, 1.123],
)

plt.gca().add_artist(first_legend)

size_title = plt.plot([], marker="", ls="")[0]
plt.legend(
    handles=[size_title, size_1, size_2, size_3],
    loc="upper center",
    bbox_to_anchor=[0.5, -0.12],
    ncol=4,
    fontsize=14,
    columnspacing=8,
    labels=["Data size:", "raw size", "pretrain size", "Syn-Boost size"],
)

if EVALUATION_METRIC == "rmse":
    y_label = "RMSE"
elif EVALUATION_METRIC == "r2":
    y_label = "1 - R2"
elif EVALUATION_METRIC == "f1":
    y_label = "1 - F1"
else:
    y_label = "misclassification error"

ax.set_xlabel("synthetic to raw ratio", fontsize=18)
ax.set_ylabel(y_label, fontsize=16)
ax.set_title(
    f"Statistical error curve: {DS_NAME.capitalize()}",
    weight="bold",
    fontsize=24,
    loc="left",
    y=1.03,
)


plt.subplots_adjust(bottom=0.2)

if EVALUATION_METRIC in ["r2", "f1"]:
    plt.savefig(f"{INFERENCE_FOLDER}/tuned_ratio_{DS_NAME}.png")
else:
    plt.savefig(f"{INFERENCE_FOLDER}/tuned_ratio_{DS_NAME}_{EVALUATION_METRIC}.png")

plt.close()


# test + train for tuning and evaluate on val data
# (insurance) rmse: 4534.32; r2: 0.8469908732477203
# (california) rmse: 0.42122391450735913; r2: 0.863920768483055
# (gesture) f1: 0.7156108402332849
