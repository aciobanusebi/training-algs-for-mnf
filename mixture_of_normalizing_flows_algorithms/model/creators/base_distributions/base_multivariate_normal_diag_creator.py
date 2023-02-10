import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseMultivariateNormalDiagCreator:
    def __init__(self,
                 dimensionality,
                 dtype,
                 mean=None,
                 min_random=0,
                 max_random=1,
                 scale=0.1):
        self.dimensionality = dimensionality
        self.dtype = dtype
        self.mean = mean
        self.base_distributions_loc_init = tf.random_uniform_initializer(min_random, max_random)
        self.base_distributions_scale_diag_init = tf.constant_initializer(scale)

    def create(self):
        loc = self.mean
        if self.mean is None:
            loc = self.base_distributions_loc_init(shape=(self.dimensionality,), dtype=self.dtype)
        return tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=self.base_distributions_scale_diag_init(shape=(self.dimensionality,), dtype=self.dtype)
        )
