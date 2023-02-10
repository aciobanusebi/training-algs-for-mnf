import os

algorithms = ["gd", "em_soft", "em_hard", "gd_variational"]

datasets = {
    "toy": ["two_banana", "smile", "moons", "circles", "pinwheel"],
    "image": ["mnist", "mnist5", "fmnist", "cifar10"]
}
maf_hidden_units = {
    "toy": 10,
    "image": 1024
}
encoder_hidden_units = {
    "toy": 10,
    "image": 1024
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
seeds = list(range(10))

dataset_types = ["toy", "image"]
for dataset_type in dataset_types:
    local_datasets = datasets[dataset_type]
    for dataset in local_datasets:
        for seed in seeds:
            for algorithm in algorithms:
                command = f"python -m tools.gd_em.train " \
                          f"--algorithm {algorithm} " \
                          f"--maf_hidden_units {maf_hidden_units[dataset_type]} " \
                          f"--maf_activation tanh " \
                          f"--prior_trainable True " \
                          f"--encoder_hidden_units {encoder_hidden_units[dataset_type]} " \
                          f"--encoder_activations relu " \
                          f"--learning_rate 0.001 " \
                          f"--epochs {epochs[dataset_type][algorithm]} " \
                          f"--m_step_epochs {m_step_epochs[dataset_type]} " \
                          f"--dataset_name {dataset} " \
                          f"--batch_size 4000 " \
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
