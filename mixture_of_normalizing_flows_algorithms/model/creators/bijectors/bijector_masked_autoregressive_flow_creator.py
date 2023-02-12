import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class BijectorMaskedAutoregressiveFlowCreator:
    def __init__(self, dtype, hidden_units, activation="tanh"):
        self.dtype = dtype
        self.hidden_units = hidden_units
        self.activation = activation

    def create(self):
        return tfb.MaskedAutoregressiveFlow(
            tfb.AutoregressiveNetwork(params=2,
                                      hidden_units=self.hidden_units,
                                      activation=self.activation,
                                      dtype=self.dtype)
        )
