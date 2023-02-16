import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.model.creators.base_distributions.base_multivariate_normal_diag_creator import \
    BaseMultivariateNormalDiagCreator
from mixture_of_normalizing_flows_algorithms.model.creators.bijectors.bijector_masked_autoregressive_flow_creator import \
    BijectorMaskedAutoregressiveFlowCreator
from mixture_of_normalizing_flows_algorithms.model.creators.mixture_of_normalizing_flows_creator import \
    MixtureOfNormalizingFlowsCreator
from mixture_of_normalizing_flows_algorithms.model.creators.normalizing_flows.normalizing_flow_creator import \
    NormalizingFlowCreator
from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer

tfd = tfp.distributions
tfb = tfp.bijectors

distribution = MixtureOfNormalizingFlowsCreator(
    dtype="float32",
    normalizing_flow_creators=[
        NormalizingFlowCreator(
            base_distribution_creator=BaseMultivariateNormalDiagCreator(
                dimensionality=1,
                dtype="float32",
                scale=1.0
            ),
            bijector_creator=BijectorMaskedAutoregressiveFlowCreator(
                dtype="float32",
                hidden_units=[5],
                activation=None,
                number_of_blocks=1
                # for more expressiveness, set a higher number of blocks, which will be chained via tfb.Chain
            )
        )
        for _ in range(2)],
    trainable_categorical_logits=True
).create()

DistributionTrainer(distribution).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=100)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([[1.0], [2.0], [3.0]]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([[1.0], [2.0], [3.0]]))
