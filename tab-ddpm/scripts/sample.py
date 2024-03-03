import torch
import numpy as np
import zero
import os

REPO_DIR = os.environ.get("REPO_DIR")

import sys
sys.path.insert(0, os.path.join(REPO_DIR, "tab-ddpm"))
sys.path.insert(0, os.path.join(REPO_DIR, "tab-ddpm/scripts"))
sys.path.insert(0, os.path.join(REPO_DIR, "tab-ddpm/tab_ddpm"))

from gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion

from utils_train import get_model, make_dataset
from lib import round_columns
import lib
import joblib

from collections import defaultdict


def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1] : indices[i]], axis=1)
        t = X[:, indices[i - 1] : indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


# if save=False, return data in original scale
def sample(
    parent_dir,
    real_data_path="data/higgs-small",
    batch_size=2000,
    num_samples=0,
    model_type="mlp",
    model_params=None,
    model_path=None,
    pipeline_dict_path=None,
    num_timesteps=1000,
    gaussian_loss_type="mse",
    scheduler="cosine",
    T_dict=None,
    num_numerical_features=0,
    disbalance=None,
    device=torch.device("cuda:1"),
    seed=0,
    change_val=False,
    deterministic=False,
    ddim=False,
    perturb_dict=None,
    save=True,
):
    """
    Some arguments:
    - parent_dir: the parent directory to save the generated synthetic data
    - model_path: path to the model checkpoint. Set as None if one wants to sample from a randomly initialized model (not recommended)
    - pipeline_dict_path: path to the preprocessing pipelines dictionary. Set as None if constructing pipelines from scratch.
        - "num_transform": preprocessing pipeline for numerical features. None if no numerical features.
        - "cat_transform": preprocessing pipeline for categorical features. None if no categorical features.
        - "category_sizes": a list of category sizes for categorical features
    """

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params["num_classes"],
        is_y_cond=model_params["is_y_cond"],
        change_val=change_val,
        pipeline_dict_path=pipeline_dict_path,
    )
    # features in D already transformed with T

    K = np.array(D.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    # when provided, load preprocessing pipelines: num_transform and cat_transform
    if pipeline_dict_path is not None:
        pipeline_dict = joblib.load(pipeline_dict_path)
        K = pipeline_dict["category_sizes"]
        print("Use category sizes from the loaded pipeline:", K)
    else:
        print("Use category sizes from the training set:", K)

    # num_numerical_features is the number of numerical features in the predictors
    # num_numerical_features_ is the number of numerical features in the predictors + the number of numerical features in the targets (1 or 0 in most cases)
    num_numerical_features_ = D.X_num["train"].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params["d_in"] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes("train"),
    )  # last two arguments are not used

    # the model is a MLP with time embedding (and potentially conditioning if applicable) when model_type == 'mlp'
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
    )

    diffusion.to(device)
    diffusion.eval()

    _, empirical_class_dist = torch.unique(
        torch.from_numpy(D.y["train"]), return_counts=True
    )
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    # disbalance == None so far, so directly go the final part. In other words, we are not doing any disbalance sampling, and the corresonding sample_all arguments haven't been changed
    if disbalance == "fix":
        empirical_class_dist[0], empirical_class_dist[1] = (
            empirical_class_dist[1],
            empirical_class_dist[0],
        )
        x_gen, y_gen = diffusion.sample_all(
            num_samples, batch_size, empirical_class_dist.float(), ddim=False
        )

    elif disbalance == "fill":
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        for i in range(empirical_class_dist.shape[0]):
            if i == ix_major:
                continue
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[i] = 1
            num_samples = val_major - empirical_class_dist[i].item()
            x_temp, y_temp = diffusion.sample_all(
                num_samples, batch_size, distrib.float(), ddim=False
            )
            x_gen.append(x_temp)
            y_gen.append(y_temp)

        x_gen = torch.cat(x_gen, dim=0)
        y_gen = torch.cat(y_gen, dim=0)

    else:
        x_gen, y_gen = diffusion.sample_all(
            num_samples,
            batch_size,
            empirical_class_dist.float(),
            ddim=ddim,
            deterministic=deterministic,
            perturb_dict=perturb_dict,
            dataset=D,
            parent_dir=parent_dir,
        )

    # Save the generated data

    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    # updated, so num_numerical_features is the number of numerical features in the predictors + the number of numerical features in the targets (1 or 0 in most cases)
    num_numerical_features = num_numerical_features + int(
        D.is_regression and not model_params["is_y_cond"]
    )

    X_num_ = X_gen
    # store the generated data in the result_dict and return it when save is False
    result_dict = defaultdict(lambda: None)
    # get X_cat from X_gen when categorical features are present
    if num_numerical_features < X_gen.shape[1]:
        # unnormalized generated logits (continuous)
        if save:
            np.save(
                os.path.join(parent_dir, "X_cat_unnorm"),
                X_gen[:, num_numerical_features:],
            )
        else:
            result_dict["X_cat_unnorm"] = X_gen[:, num_numerical_features:]
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict["cat_encoding"] == "one-hot":
            X_gen[:, num_numerical_features:] = to_good_ohe(
                D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:]
            )

        # convert numerical multinomial labels to real categorical labels (e.g. 0 -> "Male")
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    # get X_num from X_gen when numerical features are present
    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        if save:
            np.save(
                os.path.join(parent_dir, "X_num_unnorm"),
                X_gen[:, :num_numerical_features],
            )
        else:
            result_dict["X_num_unnorm"] = X_gen[:, :num_numerical_features]
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(
            os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True
        )
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)

        # if we are doing regression, the first column is the target (recall how we constructed the data using make_data)
        if model_params["num_classes"] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        if save:
            np.save(os.path.join(parent_dir, "X_num_train"), X_num)
        else:
            result_dict["X_num_train"] = X_num

    # actually save X_cat constructed from the unnormalized continuous logits
    if num_numerical_features < X_gen.shape[1]:
        if save:
            np.save(os.path.join(parent_dir, "X_cat_train"), X_cat)
        else:
            result_dict["X_cat_train"] = X_cat

    if save:
        np.save(os.path.join(parent_dir, "y_train"), y_gen)
    else:
        result_dict["y_train"] = y_gen

    if not save:
        return result_dict
