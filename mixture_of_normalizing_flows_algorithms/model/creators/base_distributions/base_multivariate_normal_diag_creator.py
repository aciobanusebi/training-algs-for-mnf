import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseMultivariateNormalDiagCreator:
    def __init__(self,
                 dimensionality,
                 dtype,
                 mean=None,
                 loc_init_mean=0.0,
                 loc_init_stddev=1.0,
                 scale=1.0):
        self.dimensionality = dimensionality
        self.dtype = dtype
        self.mean = mean
        self.base_distributions_loc_init = tf.random_normal_initializer(mean=loc_init_mean, stddev=loc_init_stddev)
        self.base_distributions_scale_diag_init = tf.constant_initializer(scale)

    def create(self):
        loc = self.mean
        if self.mean is None:
            loc = self.base_distributions_loc_init(shape=(self.dimensionality,), dtype=self.dtype)
        return tfd.MultivariateNormalDiag(
            loc=loc,
            scale_diag=self.base_distributions_scale_diag_init(shape=(self.dimensionality,), dtype=self.dtype)
        )
