import tensorflow_probability as tfp

tfd = tfp.distributions

fitted_distribution = tfd.Normal.experimental_fit([1.0, 2.0, 3.0])

print(fitted_distribution.loc.numpy())
print(fitted_distribution.scale.numpy())
