# TODO: aggregate results from privacy_distance.py and dataset information
import os
import json
from pprint import pprint


def retrieve_result_data_privacy(data_keywords=[], cwd=None):
    overall_dict = {}
    for data_keyword in data_keywords:
        data_folder_path = os.path.join(cwd, f"data/{data_keyword}")
        exp_folder_path = os.path.join(cwd, f"exp/{data_keyword}/ddpm_cb_best")

        # information about the dataset
        data_info_path = os.path.join(data_folder_path, "info.json")
        data_info = json.load(open(data_info_path, "r"))
        data_info["total_size"] = (
            data_info["train_size"] + data_info["val_size"] + data_info["test_size"]
        )
        for split in ["train", "val", "test"]:
            key = f"{split}_size"
            data_info[key] = [data_info[key], data_info[key] / data_info["total_size"]]

        # information about the privacy distance
        dist_dict_path = os.path.join(exp_folder_path, "distance_dict.json")
        dist_dict = json.load(open(dist_dict_path, "r"))

        for key, value in dist_dict.items():
            if key != "data_keyword":
                data_info[key] = value

        overall_dict[data_keyword] = data_info

    return overall_dict


if __name__ == "__main__":
    tabddpm_folder = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm"

    data_keywords = [
        "house",
        "adult",
        "churn2",
        "california",
        "abalone",
        "fb-comments",
        "insurance",
        "gesture",
    ]

    result = retrieve_result_data_privacy(data_keywords, cwd=tabddpm_folder)
    result_list_sorted = dict(
        sorted(
            list(result.items()),
            key=lambda x: x[1]["train_size"][0]
            + x[1]["val_size"][0]
            + x[1]["test_size"][0],
            reverse=True,
        )
    )
    pprint(result_list_sorted, sort_dicts=False)
