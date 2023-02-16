import os
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mixture_of_normalizing_flows_algorithms.data.datasets import Datasets
from mixture_of_normalizing_flows_algorithms.model.creators.base_distributions.base_multivariate_normal_diag_creator import \
    BaseMultivariateNormalDiagCreator
from mixture_of_normalizing_flows_algorithms.model.creators.bijectors.bijector_masked_autoregressive_flow_creator import \
    BijectorMaskedAutoregressiveFlowCreator
from mixture_of_normalizing_flows_algorithms.model.creators.mixture_of_normalizing_flows_creator import \
    MixtureOfNormalizingFlowsCreator
from mixture_of_normalizing_flows_algorithms.model.creators.normalizing_flows.normalizing_flow_creator import \
    NormalizingFlowCreator
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_hard_em import MixtureTrainerHardEm
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_soft_em import MixtureTrainerSoftEm
from mixture_of_normalizing_flows_algorithms.trainers.variational_mixture_trainer import VariationalMixtureTrainer

tfd = tfp.distributions
tfb = tfp.bijectors

import argparse


def str2bool(v):
    # from https://stackoverflow.com/a/43357954/7947996
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=["gd", "em_soft", "em_hard", "gd_variational"])

    parser.add_argument('--maf_hidden_units', type=int, nargs="+", default=[10])
    parser.add_argument('--maf_activation', type=str, default="tanh")
    parser.add_argument('--maf_number_of_blocks', type=int, default=1)
    parser.add_argument('--prior_trainable', type=str2bool, default=True,
                        help="Should the weights of the mixture components be trainable?")
    parser.add_argument('--encoder_hidden_units', type=int, nargs="+", default=[10],
                        help="Hyperparameter for the encoder used in gd_variational: q(z|x)")
    parser.add_argument('--encoder_activations', type=str, nargs="+", default=["relu"],
                        help="Hyperparameter for the encoder used in gd_variational: q(z|x)")
    parser.add_argument('--encoder_temperature_decay_steps', type=int, default=1,
                        help="Hyperparameter for the encoder used in gd_variational: q(z|x)")
    parser.add_argument('--encoder_temperature_decay_rate', type=float, default=0.96,
                        help="Hyperparameter for the encoder used in gd_variational: q(z|x)")
    parser.add_argument('--encoder_temperature_initial_rate', type=float, default=None,
                        help="Hyperparameter for the encoder used in gd_variational: q(z|x)")

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10, help="GD epochs or EM iterations")
    parser.add_argument('--m_step_epochs', type=int, default=10, help="GD epochs for each M step of the EM algorithm")

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--patience', type=int, default=10,
                        help="Patience arg used for early stopping in the main loop.")
    parser.add_argument('--m_step_patience', type=int, default=10,
                        help="Patience arg used for early stopping in each M step of the EM algorithm.")
    parser.add_argument('--validation_split', type=float, default=None)

    parser.add_argument('--e_step_cache_directory', type=str, default=None,
                        help="Used for each E step of the EM algorithm. It caches a dataset. If it is not set, then the dataset is cached in memory.")
    parser.add_argument('--output_directory', type=str, default="artifacts")
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--seed', type=int, default=11)

    parser.add_argument('--suppress_warnings', type=str2bool, default=False)

    return parser


args = get_parser().parse_args()
print(args)

if args.suppress_warnings:
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

if args.suppress_warnings:
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(1)

from tensorflow import keras

input_data, _ = getattr(Datasets, args.dataset_name)()
input_data = np.array(input_data, dtype=args.dtype)
input_data_dimensionality = input_data.shape[1]
number_of_clusters = len(np.unique(_))

tf.random.set_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

encoder = None
if args.algorithm == "gd_variational":
    encoder = tf.keras.Sequential([
        keras.Input(shape=(input_data_dimensionality,)),
        *[keras.layers.Dense(units, activation=activation) for units, activation in
          zip(args.encoder_hidden_units, args.encoder_activations)],
        keras.layers.Dense(number_of_clusters, activation='linear')
    ])

tf.random.set_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

output_directory = f"{args.output_directory}/{args.dataset_name}/{args.algorithm}/train"
Path(output_directory).mkdir(parents=True, exist_ok=True)
with open(f"{output_directory}/logs.txt", "w") as f:
    print(args, file=f)
with open(f"{output_directory}/train_args.pickle", 'wb') as file:
    pickle.dump(args, file)

standard_scaler = StandardScaler()
if args.validation_split is not None:
    data_train, data_validation = train_test_split(input_data, test_size=args.validation_split, shuffle=True,
                                                   random_state=args.seed)
    standard_scaler.fit(data_train)
    data_train = standard_scaler.transform(data_train)
    data_validation = standard_scaler.transform(data_validation)
else:
    data_train, data_validation = input_data, None
    standard_scaler.fit(data_train)
    data_train = standard_scaler.transform(data_train)
with open(f"{output_directory}/standard_scaler.pickle", 'wb') as f:
    pickle.dump(standard_scaler, f)

mnf = MixtureOfNormalizingFlowsCreator(
    dtype=args.dtype,
    normalizing_flow_creators=[
        NormalizingFlowCreator(
            base_distribution_creator=BaseMultivariateNormalDiagCreator(
                dimensionality=input_data_dimensionality,
                dtype=args.dtype
            ),
            bijector_creator=BijectorMaskedAutoregressiveFlowCreator(
                dtype=args.dtype,
                hidden_units=args.maf_hidden_units,
                activation=args.maf_activation,
                number_of_blocks=args.maf_number_of_blocks
            )
        )
        for _ in range(number_of_clusters)],
    trainable_categorical_logits=args.prior_trainable
).create()

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

batched_dataset_train = tf.data.Dataset.from_tensor_slices(data_train). \
    batch(args.batch_size)
batched_dataset_validation = None
if data_validation is not None:
    batched_dataset_validation = tf.data.Dataset.from_tensor_slices(data_validation). \
        batch(args.batch_size)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
with open(f"{output_directory}/logs.txt", "a") as f:
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), file=f)

if args.algorithm == "gd":
    DistributionTrainer(mnf).fit_via_gd(
        batched_dataset_train=batched_dataset_train,
        optimizer=optimizer,
        epochs=args.epochs,
        batched_dataset_validation=batched_dataset_validation,
        patience=args.patience,
        output_directory=output_directory
    )
elif args.algorithm in ["em_hard", "em_soft"]:
    em_class = MixtureTrainerSoftEm
    if args.algorithm == "em_hard":
        em_class = MixtureTrainerHardEm
    em_class(mnf).fit_via_em(
        unshuffled_batched_dataset_train=batched_dataset_train,
        em_iterations=args.epochs,
        m_step_optimizer_getters=[
            lambda: tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
            for _ in range(number_of_clusters)
        ],
        m_step_epochs=args.m_step_epochs,
        unshuffled_batched_dataset_validation=batched_dataset_validation,
        m_step_patience=args.m_step_patience,
        em_patience=args.patience,
        e_step_cache_directory=args.e_step_cache_directory,
        output_directory=output_directory
    )
elif args.algorithm == "gd_variational":
    decay_steps = args.encoder_temperature_decay_steps
    decay_rate = args.encoder_temperature_decay_rate
    initial_rate = args.encoder_temperature_initial_rate
    if initial_rate is None:
        initial_rate = 1 / ((decay_rate / decay_steps) ** args.epochs)

    temperature_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)

    VariationalMixtureTrainer(mnf, encoder, temperature_scheduler).fit_via_gd(
        batched_dataset_train=batched_dataset_train,
        optimizer=optimizer,
        epochs=args.epochs,
        batched_dataset_validation=batched_dataset_validation,
        patience=args.patience,
        output_directory=output_directory
    )

# for batch in batched_dataset_train.take(1):
#     print(MixturePredictor(mnf).predict_proba(batch))
#     print(MixturePredictor(mnf).predict_proba_via_exp_log(batch))
#     print(MixturePredictor(mnf).predict(batch))
#     print(MixturePredictor(mnf).predict_log_unnormalized(batch))
