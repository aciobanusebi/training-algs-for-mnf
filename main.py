import os

algorithms = ["gd", "em_soft", "em_hard", "gd_variational"]

datasets = {
    "toy": ["pinwheel", "moons", "circles", "two_banana"],
    "image": ["mnist5", "mnist"]
}
maf_hidden_units = {
    "toy": [8],
    "image": [8]
}
maf_number_of_blocks = 2
encoder_hidden_units = {
    "toy": [10, 10],
    "image": [10, 10]
}
encoder_activations = {
    "toy": ["relu", "relu"],
    "image": ["relu", "relu"]
}
epochs = {
    "toy": {
        "gd": 400,
        "em_soft": 40,
        "em_hard": 40,
        "gd_variational": 400
    },
    "image": {
        "gd": 20,
        "em_soft": 5,
        "em_hard": 5,
        "gd_variational": 20
    }
}
m_step_epochs = {
    "toy": 10,
    "image": 4
}
learning_rate = {
    "toy": 0.001,
    "image": 0.0001
}
seeds = list(range(5))

dataset_types = ["toy", "image"]
for dataset_type in dataset_types:
    local_datasets = datasets[dataset_type]
    for dataset in local_datasets:
        for seed in seeds:
            for algorithm in algorithms:
                command = f"python -m tools.gd_em.train " \
                          f"--algorithm {algorithm} " \
                          f"--maf_hidden_units {' '.join(map(str, maf_hidden_units[dataset_type]))} " \
                          f"--maf_activation relu " \
                          f"--maf_number_of_blocks {maf_number_of_blocks} " \
                          f"--prior_trainable True " \
                          f"--encoder_hidden_units {' '.join(map(str, encoder_hidden_units[dataset_type]))} " \
                          f"--encoder_activations {' '.join(encoder_activations[dataset_type])} " \
                          f"--learning_rate f{learning_rate} " \
                          f"--epochs {epochs[dataset_type][algorithm]} " \
                          f"--m_step_epochs {m_step_epochs[dataset_type]} " \
                          f"--dataset_name {dataset} " \
                          f"--batch_size 512 " \
                          f"--patience 5 " \
                          f"--m_step_patience 5 " \
                          f"--validation_split 0.2 " \
                          f"--e_step_cache_directory tmp " \
                          f"--output_directory artifacts-seed{seed} " \
                          f"--dtype float32 " \
                          f"--seed {seed} " \
                          f"--suppress_warnings False " \
                          f">> main_logs_train.txt 2>&1"
                print(command + "\n")
                os.system(command)

                command = f"python -m tools.gd_em.evaluate " \
                          f"--algorithm {algorithm} " \
                          f"--output_directory artifacts-seed{seed} " \
                          f"--dataset_name {dataset} " \
                          f"--batch_size 4000 " \
                          f">> main_logs_evaluate.txt 2>&1"
                print(command + "\n")
                os.system(command)
