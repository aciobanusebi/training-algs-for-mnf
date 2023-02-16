import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class BijectorMaskedAutoregressiveFlowCreator:
    def __init__(self, dtype, hidden_units, activation="tanh", number_of_blocks=1):
        self.dtype = dtype
        self.hidden_units = hidden_units
        self.activation = activation
        self.number_of_blocks = number_of_blocks

    def create(self):
        flows = []
        for i in range(self.number_of_blocks):
            bijector = tfb.MaskedAutoregressiveFlow(
                tfb.AutoregressiveNetwork(params=2,
                                          hidden_units=self.hidden_units,
                                          activation=self.activation,
                                          dtype=self.dtype)
            )
            flows.append(bijector)
        return tfb.Chain(list(reversed(flows)))
