import json
import os.path
from pathlib import Path

import numpy as np
import pandas as pd

output_directory = "process_results"
artifacts_root = "."
seeds = list(range(5))

algorithms = ["gd", "em_soft", "em_hard", "gd_variational"]

datasets = {
    "toy": ["pinwheel", "moons", "circles", "two_banana"],
    "image": ["mnist5", "mnist"]
}

dataset_types = datasets.keys()

keys = ["dataset", "algorithm", "seed", "time", "purity_score", "adjusted_rand_score", "normalized_mutual_info_score",
        "accuracy", "loss", "nll", "-elbo"]

processed_result = []
for dataset_type in dataset_types:
    for dataset in datasets[dataset_type]:
        print(dataset)
        for algorithm in algorithms:
            for seed in seeds:
                path_metrics = f"{artifacts_root}/artifacts-seed{seed}/{dataset}/{algorithm}/evaluate/metrics.json"
                path_logs = f"{artifacts_root}/artifacts-seed{seed}/{dataset}/{algorithm}/train/logs.txt"

                if os.path.isfile(path_metrics):
                    with open(path_metrics) as f:
                        metrics = json.load(f)

                    with open(path_logs, "r") as f:
                        lines = f.readlines()
                        time = None
                        if len(lines) > 3:
                            time = float(lines[3].split(" | ")[-1].split(": ")[-1].split()[0])
                        else:
                            print(path_logs)

                    metrics["time"] = time

                    if algorithm == "gd_variational":
                        metrics["-elbo"] = metrics["loss"]
                    else:
                        metrics["nll"] = metrics["loss"]
                        metrics["-elbo"] = None

                    metrics["dataset"] = dataset
                    metrics["algorithm"] = algorithm
                    metrics["seed"] = seed

                    d = {key: metrics[key] for key in keys}
                else:
                    print(f"{path_metrics} does not exist!")
                    d = {key: None for key in keys}
                processed_result.append(d)

processed_result = pd.DataFrame(processed_result)
processed_result = processed_result[keys]

Path(output_directory).mkdir(parents=True, exist_ok=True)
processed_result.to_csv(f"{output_directory}/result.csv")
processed_result.to_excel(f"{output_directory}/result.xlsx")

with pd.ExcelWriter(f"{output_directory}/result_sheets.xlsx") as writer:
    n_seeds = len(seeds)
    for i in range(0, len(processed_result), n_seeds):
        data = processed_result[i:i + n_seeds]
        data.style \
            .highlight_min(color='lightgreen', axis=0, subset="loss") \
            .to_excel(writer,
                      sheet_name="all",  # f'{data.iloc[0]["dataset"]}_{data.iloc[0]["algorithm"]}',
                      startrow=writer.sheets["all"].max_row if "all" in writer.sheets else 1)

processed_result = processed_result.loc[processed_result.groupby(["dataset", "algorithm"])["loss"].idxmin()]

processed_result.seed = processed_result.seed.astype(int)

processed_result['dataset'] = processed_result['dataset'].str.capitalize()
processed_result['dataset'] = processed_result['dataset'].str.replace("_", " ")
for image_dataset in datasets["image"]:
    processed_result['dataset'].replace(image_dataset.capitalize(), image_dataset.upper(), inplace=True)

processed_result['algorithm'].replace("gd", "GD", inplace=True)
processed_result['algorithm'].replace("em_soft", "SoftEM", inplace=True)
processed_result['algorithm'].replace("em_hard", "HardEM", inplace=True)
processed_result['algorithm'].replace("gd_variational", "VarGD", inplace=True)

processed_result.rename(columns={
    "dataset": "Dataset",
    "algorithm": "Algorithm",
    "seed": "Seed",
    "time": "Time",
    "purity_score": "Purity",
    "adjusted_rand_score": "ARI",
    "normalized_mutual_info_score": "NMI",
    "accuracy": "ACC",
    "loss": "Loss",
    "nll": "NLL",
    "-elbo": "-ELBO"
}, inplace=True)

processed_result_styled = processed_result.style.format(
    na_rep="NaN",
    precision=3
)
for min_col in ["Time", "Loss", "NLL"]:
    idx = np.array(processed_result.groupby(["Dataset"])[min_col].idxmin())
    processed_result_styled = processed_result_styled.apply(lambda x, idx: ['font-weight: bold'
                                                                            if x.name in idx
                                                                            else '' for i in x], idx=idx,
                                                            subset=min_col, axis=1)

for max_col in ["Purity", "ARI", "NMI", "ACC"]:
    idx = np.array(processed_result.groupby(["Dataset"])[max_col].idxmax())
    processed_result_styled = processed_result_styled.apply(lambda x, idx: ['font-weight: bold'
                                                                            if x.name in idx
                                                                            else '' for i in x], idx=idx,
                                                            subset=max_col, axis=1)

processed_result_styled.to_excel(f"{output_directory}/table.xlsx", index=False)
processed_result_styled.to_latex(f"{output_directory}/table.tex", convert_css=True, environment="table",
                                 position_float="centering")
