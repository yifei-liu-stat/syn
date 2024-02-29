"""
Calculate 2-Wasserstein distance between synthetic data and true data with multiprocessing.
"""

SEED = 2023
SYNINF_DIR = (
    "/home/liu00980/Documents/multimodal/tabular/tab-ddpm/pass-inference/syninf"
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import quantile_transform
import multiprocessing as mp

from sample import TrueSampler


import sys

sys.path.insert(0, SYNINF_DIR)

from utils_syninf import (
    generate_sample,
    concat_data,
)
from utils_num import wasserstein_2_distance


keyword_list = [
    "reg_1000",
    "reg_1000_finetuned",
    "reg_5000",
    "reg_5000_finetuned",
]


n = 50000


# Generate synthetic samples for comparison
df_processed_dict = {}
for keyword in keyword_list:
    print(f"Generating synthetic data for {keyword} ...")
    temp_dir = generate_sample(
        pipeline_config_path=f"./ckpt/{keyword}/config.toml",
        ckpt_path=f"./ckpt/{keyword}/model.pt",
        num_samples=n,
        batch_size=n,
        temp_parent_dir="./temp/",
    )

    synthetic_df = concat_data(temp_dir, split="train")
    synthetic_df_processed = pd.DataFrame(
        quantile_transform(
            synthetic_df, output_distribution="uniform", random_state=SEED
        )
    )
    df_processed_dict[keyword] = synthetic_df_processed


# Generate the test sample from the true distribution
np.random.seed(SEED)
print("Preparing true data ...")
true_sampler = TrueSampler(sigma=0.2)
X, y = true_sampler.sample(n)
true_df = pd.DataFrame(
    np.concatenate([y.reshape(-1, 1), X], axis=1),
    columns=["y"] + [f"num_{i}" for i in range(X.shape[1])],
)
true_df_processed = pd.DataFrame(
    quantile_transform(true_df, output_distribution="uniform", random_state=SEED)
)


# Experiment function to be forked
def experiment(keyword):
    print(f"Calculating 2-Wasserstein distance for {keyword} ...")

    df_processed = df_processed_dict[keyword]
    w2_distance = wasserstein_2_distance(df_processed, true_df_processed)

    print(keyword, ":", w2_distance)


if __name__ == "__main__":
    print("Begin multiprocessing")
    ctx = mp.get_context("fork")

    processes = []
    for kw in keyword_list:
        process = ctx.Process(target=experiment, args=(kw,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("Multiprocessing complete")
