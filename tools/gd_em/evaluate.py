# the evaluation script is not necessarily scalable: for the evaluation metrics/plots we need the whole data in memory
import json
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from mixture_of_normalizing_flows_algorithms.predictors.variational_mixture_predictor import VariationalMixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer
from mixture_of_normalizing_flows_algorithms.trainers.variational_mixture_trainer import VariationalMixtureTrainer

tfkl = keras.layers

from mixture_of_normalizing_flows_algorithms.data.datasets import Datasets
from mixture_of_normalizing_flows_algorithms.evaluate.clustering_evaluator import ClusteringEvaluator
from mixture_of_normalizing_flows_algorithms.model.creators.base_distributions.base_multivariate_normal_diag_creator import \
    BaseMultivariateNormalDiagCreator
from mixture_of_normalizing_flows_algorithms.model.creators.bijectors.bijector_masked_autoregressive_flow_creator import \
    BijectorMaskedAutoregressiveFlowCreator
from mixture_of_normalizing_flows_algorithms.model.creators.mixture_of_normalizing_flows_creator import \
    MixtureOfNormalizingFlowsCreator
from mixture_of_normalizing_flows_algorithms.model.creators.normalizing_flows.normalizing_flow_creator import \
    NormalizingFlowCreator
from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor

tfd = tfp.distributions
tfb = tfp.bijectors

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, required=True, choices=["gd", "em_soft", "em_hard", "gd_variational"])
    parser.add_argument('--output_directory', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    return parser


args = get_parser().parse_args()
print(args)

output_directory = f"{args.output_directory}/{args.dataset_name}/{args.algorithm}/evaluate"
Path(output_directory).mkdir(parents=True, exist_ok=True)

checkpoint_filepath = f'{args.output_directory}/{args.dataset_name}/{args.algorithm}/train/best_model/'
checkpoint_filepath_encoder = f'{args.output_directory}/{args.dataset_name}/{args.algorithm}/train/best_encoder/'
epoch_filepath = f'{args.output_directory}/{args.dataset_name}/{args.algorithm}/train/epoch.txt'
standard_scaler_path = f'{args.output_directory}/{args.dataset_name}/{args.algorithm}/train/standard_scaler.pickle'

train_args_filepath = f'{args.output_directory}/{args.dataset_name}/{args.algorithm}/train/train_args.pickle'

with open(train_args_filepath, 'rb') as file:
    train_args = pickle.load(file)

with open(standard_scaler_path, 'rb') as f:
    standard_scaler = pickle.load(f)

input_data, output_data = getattr(Datasets, args.dataset_name)()
input_data = np.array(input_data, dtype=train_args.dtype)
input_data_dimensionality = input_data.shape[1]
number_of_clusters = len(np.unique(output_data))

input_data = standard_scaler.transform(input_data)

tf.random.set_seed(train_args.seed)
random.seed(train_args.seed)
np.random.seed(train_args.seed)

encoder = None
if args.algorithm == "gd_variational":
    encoder = tf.keras.Sequential([
        keras.Input(shape=(input_data_dimensionality,)),
        *[keras.layers.Dense(units, activation=activation) for units, activation in
          zip(train_args.encoder_hidden_units, train_args.encoder_activations)],
        keras.layers.Dense(number_of_clusters, activation='linear')
    ])

# seed is important; if not set accordingly, the results will differ from the ones obtained in the training phase
tf.random.set_seed(train_args.seed)
random.seed(train_args.seed)
np.random.seed(train_args.seed)

mnf = MixtureOfNormalizingFlowsCreator(
    dtype=train_args.dtype,
    normalizing_flow_creators=[
        NormalizingFlowCreator(
            base_distribution_creator=BaseMultivariateNormalDiagCreator(
                dimensionality=input_data_dimensionality,
                dtype=train_args.dtype
            ),
            bijector_creator=BijectorMaskedAutoregressiveFlowCreator(
                dtype=train_args.dtype,
                hidden_units=train_args.maf_hidden_units,
                activation=train_args.maf_activation,
                number_of_blocks=train_args.maf_number_of_blocks
            )
        )
        for _ in range(number_of_clusters)],
    trainable_categorical_logits=train_args.prior_trainable
).create()

inputs = tfkl.Input(shape=mnf.event_shape, dtype=mnf.dtype)
mnf.log_prob(inputs)

checkpoint = tf.train.Checkpoint(mnf)
checkpoint.read(
    checkpoint_filepath).expect_partial()  # .expect_partial is called so that an ignored exception won't appear

epoch = None
if args.algorithm == "gd_variational":
    checkpoint_encoder = tf.train.Checkpoint(encoder)
    checkpoint_encoder.read(
        checkpoint_filepath_encoder).expect_partial()  # .expect_partial is called so that an ignored exception won't appear

    with open(epoch_filepath, "r") as f:
        epoch = int(f.readline())

batched_dataset_train = tf.data.Dataset.from_tensor_slices(input_data). \
    batch(args.batch_size)

predictor_object = MixturePredictor(mnf)
temperature_scheduler = None
if args.algorithm == "gd_variational":
    decay_steps = train_args.encoder_temperature_decay_steps
    decay_rate = train_args.encoder_temperature_decay_rate
    initial_rate = train_args.encoder_temperature_initial_rate
    if initial_rate is None:
        initial_rate = 1 / ((decay_rate / decay_steps) ** train_args.epochs)

    temperature_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate)
    predictor_object = VariationalMixturePredictor(encoder, temperature_scheduler(epoch))
predicted_clustering = list(batched_dataset_train.map(predictor_object.predict).unbatch().as_numpy_iterator())
trainer_object = DistributionTrainer(mnf)
if args.algorithm == "gd_variational":
    trainer_object = VariationalMixtureTrainer(mnf, encoder, temperature_scheduler, epoch)
loss = trainer_object._compute_loss(batched_dataset_train, None)
nll = None
if args.algorithm == "gd_variational":
    nll = DistributionTrainer(mnf)._compute_loss(batched_dataset_train, None)

computed_metrics = ClusteringEvaluator.evaluate(y_true=output_data, y_pred=predicted_clustering)
computed_metrics["loss"] = loss
if args.algorithm == "gd_variational":
    computed_metrics["nll"] = nll
with open(f"{output_directory}/metrics.json", 'w') as outfile:
    json.dump(computed_metrics, outfile)

if input_data_dimensionality == 2:
    ClusteringEvaluator.plot_density_2d(input_data, mnf, log=False)
    plt.savefig(f'{output_directory}/plot_density_2d.png')
    ClusteringEvaluator.plot_decision_boundary_2d(
        lambda x: predictor_object.predict(x).numpy(),
        input_data,
        output_data
    )
    plt.savefig(f'{output_directory}/plot_decision_boundary_2d.png')
