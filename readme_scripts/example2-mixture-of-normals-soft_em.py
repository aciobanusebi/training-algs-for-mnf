import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_soft_em import MixtureTrainerSoftEm

tfd = tfp.distributions

distribution = tfd.Mixture(
    cat=tfd.Categorical(logits=tf.Variable([1.0, 1.0])),
    components=[
        tfd.Normal(loc=tf.Variable(0.0), scale=tf.Variable(1.0)),
        tfd.Normal(loc=tf.Variable(10.0), scale=tf.Variable(1.0))
    ]
)

MixtureTrainerSoftEm(distribution).fit_via_em(
    unshuffled_batched_dataset_train=tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0]).batch(3),
    em_iterations=10,
    m_step_optimizer_getters=[
        lambda: tf.keras.optimizers.Adam(learning_rate=0.1)
        for _ in range(2)
    ],
    m_step_epochs=10)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([1.0, 2.0, 3.0]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([1.0, 2.0, 3.0]))
