# must run from tab-ddpm folder (logics from the tddpm library)

import subprocess

# ds_name: [device_num, metric]
map_dict = {
    "adult": [0, "acc"],
    "insurance": [1, "rmse"],
    "abalone": [2, "rmse"],
    "california": [3, "rmse"],
    "house": [4, "rmse"],
    "gesture": [5, "acc"],
    "churn2": [6, "acc"],
    "fb-comments": [7, "rmse"],
}


processes = []

for ds_name, [device_num, metric] in map_dict.items():
    cmd = [
        "python",
        "scripts/tune_evaluation_model.py",
        ds_name,
        "mlp",
        "val",
        f"cuda:{device_num}",
        metric,
    ]
    processes.append(subprocess.Popen(cmd))

for process in processes:
    process.wait()
