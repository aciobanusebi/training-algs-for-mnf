import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer

tfd = tfp.distributions

distribution = tfd.Mixture(
    cat=tfd.Categorical(logits=tf.Variable([1.0, 1.0])),
    components=[
        tfd.Normal(loc=tf.Variable(0.0), scale=tf.Variable(1.0)),
        tfd.Normal(loc=tf.Variable(10.0), scale=tf.Variable(1.0))
    ]
)

DistributionTrainer(distribution).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=100)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([1.0, 2.0, 3.0]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([1.0, 2.0, 3.0]))
