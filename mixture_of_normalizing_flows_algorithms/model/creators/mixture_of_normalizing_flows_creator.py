import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class MixtureOfNormalizingFlowsCreator:
    def __init__(self, dtype, normalizing_flow_creators, categorical_logits=None, trainable_categorical_logits=True):
        self.dtype = dtype
        self.number_of_clusters = len(normalizing_flow_creators)
        self.normalizing_flow_creators = normalizing_flow_creators

        logits_initial_value = categorical_logits
        if logits_initial_value is None:
            logits_init = tf.ones_initializer()
            logits_initial_value = logits_init(shape=(self.number_of_clusters,), dtype=self.dtype)

        if trainable_categorical_logits:
            self.logits = tf.Variable(
                initial_value=logits_initial_value,
                trainable=True
            )
        else:
            self.logits = logits_initial_value

    def create(self):
        return tfd.Mixture(
            cat=tfd.Categorical(logits=self.logits, dtype=self.dtype),
            components=[normalizing_flow_creator.create() for normalizing_flow_creator in
                        self.normalizing_flow_creators]
        )
