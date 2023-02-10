import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class NormalizingFlowCreator:
    def __init__(self, base_distribution_creator, bijector_creator):
        self.base_distribution_creator = base_distribution_creator
        self.bijector_creator = bijector_creator

    def create(self):
        return tfd.TransformedDistribution(
            distribution=self.base_distribution_creator.create(),
            bijector=self.bijector_creator.create()
        )
