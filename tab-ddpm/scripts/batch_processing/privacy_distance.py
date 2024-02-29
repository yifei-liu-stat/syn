import os
import json
import lib
import subprocess

tabddpm_dir = "/home/liu00980/Documents/multimodal/tabular/tab-ddpm"


def generate_experiment_cmd(data_keyword="house", device_id=0):
    data_folder_path = os.path.join(tabddpm_dir, f"data/{data_keyword}")
    exp_folder_path = os.path.join(tabddpm_dir, f"exp/{data_keyword}/ddpm_cb_best")

    # change generation size and batch size to be the same as test size
    data_info_path = os.path.join(data_folder_path, "info.json")
    data_info = json.load(open(data_info_path, "r"))
    generation_size, batch_size = [data_info["test_size"]] * 2

    # add config-pass.toml based on config.toml
    toml_path = os.path.join(exp_folder_path, "config.toml")
    toml_config = lib.load_config(toml_path)

    toml_config["sample"]["num_samples"] = generation_size
    toml_config["sample"]["batch_size"] = batch_size
    toml_config["sample"]["perturb_dict"] = {"tau": 0.5}
    toml_config["device"] = f"cuda:{device_id}"
    toml_config_pass_path = os.path.join(
        tabddpm_dir, toml_config["parent_dir"], "config-pass.toml"
    )
    lib.dump_config(
        toml_config,
        toml_config_pass_path,
    )

    cmd = [
        "python",
        "scripts/pipeline.py",
        "--config",
        toml_config_pass_path,
    ]

    train_flag = True  # whether to train the model
    sample_flag = True  # whether to sample from the model
    for filename in os.listdir(exp_folder_path):
        if filename == "distance_dict.json":
            sample_flag = False
            cmd = ["echo", "distance_dict.json already exists, skipping sampling."]
            break

        if filename.endswith(".pt"):
            train_flag = False

    if sample_flag:
        if train_flag:
            cmd.extend(["--train", "--sample"])
        else:
            cmd.append("--sample")

    return cmd


if __name__ == "__main__":
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

    processes = [
        subprocess.Popen(
            generate_experiment_cmd(data_keyword, device_id=i), cwd=tabddpm_dir
        )
        for i, data_keyword in enumerate(data_keywords)
    ]

    for process in processes:
        process.wait()
