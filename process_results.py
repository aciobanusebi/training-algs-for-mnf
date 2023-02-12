import json
from pathlib import Path

import pandas as pd

output_directory = "process_results"
artifacts_root = "."
seeds = list(range(10))

algorithms = ["gd", "em_soft", "em_hard", "gd_variational"]

datasets = {
    "toy": ["two_banana", "smile", "moons", "circles", "pinwheel"],
    "image": ["mnist", "mnist5", "fmnist"]
}

dataset_types = datasets.keys()

keys = ["dataset", "algorithm", "seed", "time", "purity_score", "adjusted_rand_score", "normalized_mutual_info_score",
        "accuracy", "loss", "nll", "elbo"]

processed_result = []
for dataset_type in dataset_types:
    for dataset in datasets[dataset_type]:
        print(dataset)
        for algorithm in algorithms:
            for seed in seeds:
                path_metrics = f"{artifacts_root}/artifacts-seed{seed}/{dataset}/{algorithm}/evaluate/metrics.json"
                path_logs = f"{artifacts_root}/artifacts-seed{seed}/{dataset}/{algorithm}/train/logs.txt"

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
                    metrics["elbo"] = metrics["loss"]
                else:
                    metrics["nll"] = metrics["loss"]
                    metrics["elbo"] = None

                metrics["dataset"] = dataset
                metrics["algorithm"] = algorithm
                metrics["seed"] = seed

                d = {key: metrics[key] for key in keys}
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
