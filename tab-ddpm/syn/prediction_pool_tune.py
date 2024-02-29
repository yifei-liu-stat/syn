import numpy as np
import optuna
from catboost import CatBoostClassifier, CatBoostRegressor

import os
import sys
import zero

sys.path.insert(0, "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/scripts/")
from sample import sample

import lib
from lib import concat_features, read_pure_data, get_catboost_config, read_changed_val
from pathlib import Path
import pickle

from pprint import pprint

SEED = 0
TABDDPM_FOLDER = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm"
INFERENCE_FOLDER = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference"


# DS_NAME = "insurance"       # regression
# DS_NAME = "abalone"  # regression
# DS_NAME = "gesture"  # classification
# DS_NAME = "churn2"  # classification
# DS_NAME = "california"  # regression
# DS_NAME = "house"  # regression
DS_NAME = "adult"  # classification
# DS_NAME = "fb-comments"  # regression

# choose from "r2" (default) and "rmse" for regression, and "f1" (default) and "acc" for classification
# (cont.) if sets to "rmse", it is in negative scale
EVALUATION_METRIC = "acc"
print(f"Evaluating metric: {EVALUATION_METRIC}")

# GPU training no available due to insufficient driver version
DEVICE_ID = 6
THREAD_COUNT = 10


device = f"cuda:{DEVICE_ID}"
real_data_path = Path(f"{TABDDPM_FOLDER}/data/{DS_NAME}")
model_path = Path(f"{TABDDPM_FOLDER}/exp/{DS_NAME}/ddpm_cb_best/model.pt")
config_path = Path(f"{TABDDPM_FOLDER}/exp/{DS_NAME}/ddpm_cb_best/config.toml")
storage_name = f"sqlite:///{INFERENCE_FOLDER}/ratio_optuna_studies/{DS_NAME}.db"
if EVALUATION_METRIC in ["r2", "f1"]:
    # default metrics
    storage_name = f"sqlite:///{INFERENCE_FOLDER}/ratio_optuna_studies/{DS_NAME}.db"
    pickle_path = f"{INFERENCE_FOLDER}/fake_to_real_ratio/ratio_list_{DS_NAME}.pkl"
else:
    storage_name = f"sqlite:///{INFERENCE_FOLDER}/ratio_optuna_studies/{DS_NAME}_{EVALUATION_METRIC}.db"
    pickle_path = f"{INFERENCE_FOLDER}/fake_to_real_ratio/ratio_list_{DS_NAME}_{EVALUATION_METRIC}.pkl"


MAX_FAKE_TO_REAL_RATIO = 30
NUM_RATIOS = 30
fake_to_real_ratio_list = np.linspace(0, MAX_FAKE_TO_REAL_RATIO, NUM_RATIOS + 1)


if not os.path.exists(pickle_path):
    fake_to_real_ratio_list_raw = []
else:
    print("pickle file already exists, using ratios not in the original pickle file...")
    fake_to_real_ratio_list_raw = pickle.load(open(pickle_path, "rb"))

    diff = set(fake_to_real_ratio_list) - set(fake_to_real_ratio_list_raw)
    union = set(fake_to_real_ratio_list) | set(fake_to_real_ratio_list_raw)

    fake_to_real_ratio_list = list(diff)


# # toy arguments for testing
# T_dict = {
#     "seed": SEED,
#     "normalization": None,
#     "num_nan_policy": None,
#     "cat_nan_policy": None,
#     "cat_min_frequency": None,
#     "cat_encoding": None,
#     "y_policy": "default",
# }


# params = {}
# params["learning_rate"] = 0.01  # Default value
# params["depth"] = 5  # Default value
# params["l2_leaf_reg"] = 1.0  # Default value
# params["bagging_temperature"] = 0.5  # Default value
# params["leaf_estimation_iterations"] = 5  # Default value

# params = params | {
#     "iterations": 2000,
#     "early_stopping_rounds": 50,
#     "od_pval": 0.001,
#     "task_type": "CPU",
#     "thread_count": 4,
# }


def train_catboost_simple(
    T_dict,
    seed=0,
    params=None,
    device=None,
    fake_to_real_ratio=0,
    add_original_train=False,
):
    zero.improve_reproducibility(seed)

    info = lib.load_json(os.path.join(real_data_path, "info.json"))
    T = lib.Transformations(**T_dict)

    X = None
    print("-" * 100)
    # ??? shall we use validation set as training and test as validation?
    # the current approach uses validation as val, but the ddpm is tuned to maximize the val score originally, so natually better than tuning using test?
    # another idea: small ratio must have improvement since n_tune << n_holdout (I am using insurance dataset to check this)
    print("loading real data...")
    X_num, X_cat, y = read_pure_data(
        real_data_path, "test"
    )  # use test data for training catboost model
    X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, "val")
    # this part will not be used in the very end (since we are calling get_val_score), but it remains here for consistency and avoid some errors
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, "test")

    if add_original_train:
        # add original training data to the test set for augmented training
        X_num_train, X_cat_train, y_train = read_pure_data(real_data_path, "train")
        if X_num is not None:
            X_num = np.concatenate([X_num, X_num_train], axis=0)

        if X_cat is not None:
            X_cat = np.concatenate([X_cat, X_cat_train], axis=0)

        y = np.concatenate([y, y_train], axis=0)

    if fake_to_real_ratio > 0:
        print("fake_to_real_ratio is positive, sampling synthetic data...")

        raw_config = lib.load_config(config_path)

        if "perturb_dict" in raw_config["sample"]:
            del raw_config["sample"]["perturb_dict"]

        # raw_config["sample"].keys()  # num_samples, batch_size, seed

        raw_config["sample"]["num_samples"] = int(len(y) * fake_to_real_ratio)
        raw_config["sample"]["batch_size"] = 2**14

        result_dict = sample(
            num_samples=raw_config["sample"]["num_samples"],
            batch_size=raw_config["sample"]["batch_size"],
            disbalance=raw_config["sample"].get("disbalance", None),
            **raw_config["diffusion_params"],
            parent_dir=raw_config["parent_dir"],
            real_data_path=real_data_path,
            model_path=model_path,
            model_type=raw_config["model_type"],
            model_params=raw_config["model_params"],
            T_dict=raw_config["train"]["T"],
            num_numerical_features=raw_config["num_numerical_features"],
            device=device,
            seed=seed,
            change_val=False,
            deterministic=raw_config["sample"].get("deterministic", False),
            ddim=raw_config["sample"].get("ddim", False),
            perturb_dict=None,
            save=False,
        )
        X_num_fake, X_cat_fake, y_fake = (
            result_dict["X_num_train"],
            result_dict["X_cat_train"],
            result_dict["y_train"],
        )

        if X_num is not None:
            X_num = np.concatenate([X_num, X_num_fake], axis=0)

        if X_cat is not None:
            X_cat = np.concatenate([X_cat, X_cat_fake], axis=0)

        y = np.concatenate([y, y_fake], axis=0)

    D = lib.Dataset(
        {"train": X_num, "val": X_num_val, "test": X_num_test}
        if X_num is not None
        else None,
        {"train": X_cat, "val": X_cat_val, "test": X_cat_test}
        if X_cat is not None
        else None,
        {"train": y, "val": y_val, "test": y_test},
        {},
        lib.TaskType(info["task_type"]),
        info.get("n_classes"),
    )

    D = lib.transform_dataset(D, T, None)
    # X is a dict with keys "train", "val", "test", each of which is a DataFrame
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    if params is None:
        # if no params is specified, specify the path to save the catboost configuration file
        # even though is_cv is set to be true, it is not used in the function
        # so it will get the configuration file ending with _cv.json anyway
        catboost_config = get_catboost_config(real_data_path, is_cv=True)
    else:
        catboost_config = params

    if "cat_features" not in catboost_config:
        catboost_config["cat_features"] = list(range(D.n_num_features, D.n_features))

    for col in range(D.n_features):
        for split in X.keys():
            if col in catboost_config["cat_features"]:
                X[split][col] = X[split][col].astype(str)
            else:
                X[split][col] = X[split][col].astype(float)
    print(T_dict)
    pprint(catboost_config, width=100)
    print("-" * 100)

    if D.is_regression:
        model = CatBoostRegressor(
            **catboost_config,
            eval_metric="RMSE",
            random_seed=seed,
        )
        predict = model.predict
    else:
        model = CatBoostClassifier(
            loss_function="MultiClass" if D.is_multiclass else "Logloss",
            **catboost_config,
            eval_metric="TotalF1",
            random_seed=seed,
            class_names=[str(i) for i in range(D.n_classes)]
            if D.is_multiclass
            else ["0", "1"],
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(X["train"], D.y["train"], eval_set=(X["val"], D.y["val"]), verbose=100)
    predictions = {k: predict(v) for k, v in X.items()}

    report = {}
    # report["eval_type"] = 'real'
    report["dataset"] = real_data_path
    report["metrics"] = D.calculate_metrics(
        predictions, None if D.is_regression else "probs"
    )

    metrics_report = lib.MetricsReport(report["metrics"], D.task_type)
    metrics_report.print_metrics()

    return metrics_report


def suggest_catboost_params(trial):
    params = {}
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", 0.001, 1.0)
    params["depth"] = trial.suggest_int("depth", 3, 10)
    params["l2_leaf_reg"] = trial.suggest_uniform("l2_leaf_reg", 0.1, 10.0)
    params["bagging_temperature"] = trial.suggest_uniform(
        "bagging_temperature", 0.0, 1.0
    )
    params["leaf_estimation_iterations"] = trial.suggest_int(
        "leaf_estimation_iterations", 1, 10
    )

    # # GPU training: CUDA driver version is insufficient for CUDA runtime version
    # params = params | {
    #     "iterations": 2000,
    #     "early_stopping_rounds": 50,
    #     "od_pval": 0.001,
    #     "task_type": "GPU",
    #     "devices": str(DEVICE_ID),
    # }

    # CPU training
    params = params | {
        "iterations": 2000,
        "early_stopping_rounds": 50,
        "od_pval": 0.001,
        "task_type": "CPU",
        "thread_count": THREAD_COUNT,
    }

    return params


def objective(trial, fake_to_real_ratio=0, add_original_train=False):
    # params defines the hyparameters to be tuned for the catboost model
    params = suggest_catboost_params(trial)

    # T_dict defines the transformations to be applied to the data
    T_dict = {
        "seed": SEED,
        "normalization": None,
        "num_nan_policy": None,
        "cat_nan_policy": None,
        "cat_min_frequency": None,
        "cat_encoding": None,
        "y_policy": "default",
    }
    trial.set_user_attr("params", params)

    metrics_report = train_catboost_simple(
        T_dict=T_dict,
        seed=SEED,
        params=params,
        device=device,
        fake_to_real_ratio=fake_to_real_ratio,
        add_original_train=add_original_train,
    )
    # For binary classification problems, the metrics are: acc, (macro-)f1 (DEFAULT score), roc-auc
    # For regression problems, the metrics are: rmse, r2 (DEFAULT score)

    # score = metrics_report.get_val_score()
    score = metrics_report.get_metric("val", EVALUATION_METRIC)
    if EVALUATION_METRIC == "rmse":
        score = -score

    return score


if __name__ == "__main__":
    study_list = []
    for fake_to_real_ratio in fake_to_real_ratio_list:
        study_name = f"catboost_{DS_NAME}_fake_to_real_ratio_{fake_to_real_ratio}"

        # maximize either r2 (regression) or macro-f1 (classification)
        study_list.append(
            optuna.create_study(
                storage=storage_name,
                study_name=study_name,
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
            )
        )
        study_list[-1].optimize(
            lambda trial: objective(trial, fake_to_real_ratio=fake_to_real_ratio),
            n_trials=100,
            show_progress_bar=True,
        )

        fake_to_real_ratio_list_raw.append(fake_to_real_ratio)
        pickle.dump(fake_to_real_ratio_list_raw, open(pickle_path, "wb"))


# # a sample run
# fake_to_real_ratio = 0
# add_original_train = True

# study = optuna.create_study(
#     direction="maximize",
#     sampler=optuna.samplers.TPESampler(seed=SEED),
# )

# study.optimize(
#     lambda trial: objective(
#         trial,
#         fake_to_real_ratio=fake_to_real_ratio,
#         add_original_train=add_original_train,
#     ),
#     n_trials=100,
#     show_progress_bar=True,
# )

# pprint(study.best_trial.user_attrs["params"])
# pprint(study.best_value)


# # retrieve the best parameters and save them
# bets_params = study.best_trial.user_attrs["params"]

# # change this path
# best_params_path = f"tuned_models/catboost/{DS_NAME}_val.json"

# lib.dump_json(bets_params, best_params_path)
