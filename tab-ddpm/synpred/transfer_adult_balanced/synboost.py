"""
Syn-Boost tuning with the fine-tuned generator on Adult-Female, which was originally pre-trained on Adult-Male.
"""


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
    test_acc,
    test_scores_catboost,
)



######## Configurations ########

n = 1350
rho_max, num_rhos = 30, 30
rho_list = np.linspace(0, rho_max, num_rhos + 1)[1:]

save_path = f"synboost_transfer_adult_female.pkl"

df_dict = pickle.load(open("temp_df_dict.pkl", "rb"))
gender_names_dict = df_dict["gender_names_dict"]

# Get the test female data (twin_2 or twin_1 test) as the validation set
df_female_test = concat_data(
    f"{TDDPM_DIR}/data/adult_female_3000_twin_1",
    split="test",
    **gender_names_dict,
)



######### Generate synthetic female data (fine-tuned generator) #########
synthetic_sample_dir = generate_sample(
    pipeline_config_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/config.toml",
    ckpt_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/model.pt",
    pipeline_dict_path=f"{TDDPM_DIR}/exp/adult_female_3000_twin_1/ddpm_cb_best/pipeline_dict.joblib",
    num_samples=int(n * rho_max),
    batch_size=int(n * rho_max / 10),
    seed=2023,
)
fake_df_female_allinone = concat_data(synthetic_sample_dir, **gender_names_dict)
fake_df_female_allinone["income"].cat.categories = ["0", "1"]


fake_df_train = fake_df_female_allinone.copy()
df_test = df_female_test.copy()


########## Syn-Boost tuning ##########

result_dict = {"rhos": rho_list, "scores": []}
for rho in tqdm(rho_list):
    m = int(n * rho)

    fake_train_df = fake_df_train[:m]
    
    classes = fake_train_df[["income"]].to_numpy(dtype=int).flatten()
    npos = np.sum(classes)
    nneg = len(classes) - npos
    
    cat_model = catboost_pred_model(
        fake_train_df,
        df_test,
        **gender_names_dict,
        iterations=1000,
        verbose=False,
        scale_pos_weight=nneg/npos,
    )
    
    # overall accuracy, macro F1 score, AUROC, and AUPRC
    scores = test_scores_catboost(cat_model, df_test, **gender_names_dict)
    
    result_dict["scores"].append(scores)

    pickle.dump(result_dict, open(save_path, "wb"))
    print(f"rho: {rho} | m: {m} | Overall accuracy: {scores['accuracy']} | Macro F1: {scores['f1_macro']} | AUROC: {scores['auroc']} | AUPRC: {scores['auprc']}.")
    

print(f"Syn-Boost tuning is done. The result is saved in {save_path}.")