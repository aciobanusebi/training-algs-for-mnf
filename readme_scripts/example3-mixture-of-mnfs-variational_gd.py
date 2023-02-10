import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.variational_mixture_predictor import VariationalMixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.variational_mixture_trainer import VariationalMixtureTrainer

tfd = tfp.distributions
tfb = tfp.bijectors

distribution = tfd.Mixture(
    cat=tfd.Categorical(logits=tf.Variable([1.0, 1.0])),
    components=[
        tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0], scale_diag=[1.0]),
            bijector=tfb.MaskedAutoregressiveFlow(tfb.AutoregressiveNetwork(params=2, hidden_units=[5]))
        ),
        tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[10.0], scale_diag=[1.0]),
            bijector=tfb.MaskedAutoregressiveFlow(tfb.AutoregressiveNetwork(params=2, hidden_units=[5]))
        )
    ]
)

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(3, activation="relu"),
    tf.keras.layers.Dense(2, activation='softmax')
])

VariationalMixtureTrainer(distribution, encoder).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=100)

print(encoder.trainable_variables)
print(distribution.trainable_variables)

print("hard clustering:")
print(VariationalMixturePredictor(encoder).predict(tf.convert_to_tensor([[1.0], [2.0], [3.0]])))

print("soft clustering:")
print(VariationalMixturePredictor(encoder).predict_proba(tf.convert_to_tensor([[1.0], [2.0], [3.0]])))
