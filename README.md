[//]: # (For code from files, run `npx embedme README.md`)

[//]: # (For TOC go to https://luciopaiva.com/markdown-toc/ and paste the result here.)

# Training Algorithms for Mixtures of Normalizing Flows

# Table of contents

- [Important Warning](#important-warning)
- [Install requirements](#install-requirements)
- [Training algorithms for probabilistic distributions: sample code](#training-algorithms-for-probabilistic-distributions-sample-code)
  - [Normal distribution](#normal-distribution)
    - [via tfd.Distribution.experimental_fit](#via-tfddistributionexperimental_fit)
    - [via our code: gradient descent](#via-our-code-gradient-descent)
  - [Mixture of normal distributions](#mixture-of-normal-distributions)
    - [via our code: gradient descent](#via-our-code-gradient-descent)
    - [via our code: hard EM](#via-our-code-hard-em)
    - [via our code: soft EM](#via-our-code-soft-em)
    - [via our code: variational gradient descent](#via-our-code-variational-gradient-descent)
  - [Mixture of normalizing flows (1 block)](#mixture-of-normalizing-flows-1-block)
    - [via our code: gradient descent](#via-our-code-gradient-descent)
    - [via our code: hard EM](#via-our-code-hard-em)
    - [via our code: soft EM](#via-our-code-soft-em)
    - [via our code: variational gradient descent](#via-our-code-variational-gradient-descent)
    - [via our code: gradient descent (create the `distribution` object using our own classes)](#via-our-code-gradient-descent-create-the-distribution-object-using-our-own-classes)
- [Mixture of normalizing flows: How to run the scripts](#mixture-of-normalizing-flows-how-to-run-the-scripts)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Run all the experiments](#run-all-the-experiments)
  - [Process the experiments' results](#process-the-experiments-results)

## Important Warning

For hard EM and soft EM, you MUST NOT call `.shuffle(...)` on the dataset (tf.data.Dataset) unless you
set `reshuffle_each_iteration` to False, i.e. you can only call `.shuffle(..., reshuffle_each_iteration=False)`. **This
was not forced in the code, so it is the user's reponsibility to obey this rule.**

## Install requirements

```
pip install -r requirements.txt
```

## Training algorithms for probabilistic distributions: sample code

### Normal distribution

#### via tfd.Distribution.experimental_fit

```py
# readme_scripts/example1-normal-experimental_fit.py

import tensorflow_probability as tfp

tfd = tfp.distributions

fitted_distribution = tfd.Normal.experimental_fit([1.0, 2.0, 3.0])

print(fitted_distribution.loc.numpy())
print(fitted_distribution.scale.numpy())

```

#### via our code: gradient descent

```py
# readme_scripts/example1-normal-gd.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer

tfd = tfp.distributions

distribution = tfd.Normal(loc=tf.Variable(0.0),
                          scale=tf.Variable(1.0))

DistributionTrainer(distribution).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=500)

print(distribution.trainable_variables)

```

### Mixture of normal distributions

#### via our code: gradient descent

```py
# readme_scripts/example2-mixture-of-normals-gd.py

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

```

#### via our code: hard EM

```py
# readme_scripts/example2-mixture-of-normals-hard_em.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_hard_em import MixtureTrainerHardEm

tfd = tfp.distributions

distribution = tfd.Mixture(
    cat=tfd.Categorical(logits=tf.Variable([1.0, 1.0])),
    components=[
        tfd.Normal(loc=tf.Variable(0.0), scale=tf.Variable(1.0)),
        tfd.Normal(loc=tf.Variable(10.0), scale=tf.Variable(1.0))
    ]
)

MixtureTrainerHardEm(distribution).fit_via_em(
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

```

#### via our code: soft EM

```py
# readme_scripts/example2-mixture-of-normals-soft_em.py

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

```

#### via our code: variational gradient descent

```py
# readme_scripts/example2-mixture-of-normals-variational_gd.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.variational_mixture_predictor import VariationalMixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.variational_mixture_trainer import VariationalMixtureTrainer

tfd = tfp.distributions

distribution = tfd.Mixture(
    cat=tfd.Categorical(logits=tf.Variable([1.0, 1.0])),
    components=[
        tfd.Normal(loc=tf.Variable(0.0), scale=tf.Variable(1.0)),
        tfd.Normal(loc=tf.Variable(10.0), scale=tf.Variable(1.0))
    ]
)

encoder = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(3, activation="relu"),
    tf.keras.layers.Dense(2, activation='softmax')
])

VariationalMixtureTrainer(distribution, encoder).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([1.0, 2.0, 3.0]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=100)

print(encoder.trainable_variables)
print(distribution.trainable_variables)

print("hard clustering:")
print(VariationalMixturePredictor(encoder).predict(tf.convert_to_tensor([1.0, 2.0, 3.0])))

print("soft clustering:")
print(VariationalMixturePredictor(encoder).predict_proba(tf.convert_to_tensor([1.0, 2.0, 3.0])))

```

### Mixture of normalizing flows (1 block)

The following examples use just **one block**/flow (i.e. no chained flows). For more blocks, use `tfb.Chain` or set
our `number_of_blocks` parameter in the constructor of `BijectorMaskedAutoregressiveFlowCreator` to a number greater
than 1 (see the last example).

#### via our code: gradient descent

```py
# readme_scripts/example3-mixture-of-mnfs-gd.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer

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

DistributionTrainer(distribution).fit_via_gd(
    batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    epochs=100)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([[1.0], [2.0], [3.0]]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([[1.0], [2.0], [3.0]]))

```

#### via our code: hard EM

```py
# readme_scripts/example3-mixture-of-mnfs-hard_em.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_hard_em import MixtureTrainerHardEm

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

MixtureTrainerHardEm(distribution).fit_via_em(
    unshuffled_batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
    em_iterations=10,
    m_step_optimizer_getters=[
        lambda: tf.keras.optimizers.Adam(learning_rate=0.1)
        for _ in range(2)
    ],
    m_step_epochs=10)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([[1.0], [2.0], [3.0]]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([[1.0], [2.0], [3.0]]))

```

#### via our code: soft EM

```py
# readme_scripts/example3-mixture-of-mnfs-soft_em.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer_soft_em import MixtureTrainerSoftEm

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

MixtureTrainerSoftEm(distribution).fit_via_em(
    unshuffled_batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
    em_iterations=10,
    m_step_optimizer_getters=[
        lambda: tf.keras.optimizers.Adam(learning_rate=0.1)
        for _ in range(2)
    ],
    m_step_epochs=10)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([[1.0], [2.0], [3.0]]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([[1.0], [2.0], [3.0]]))

```

#### via our code: variational gradient descent

```py
# readme_scripts/example3-mixture-of-mnfs-variational_gd.py

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

```

#### via our code: gradient descent (create the `distribution` object using our own classes)

```py
# readme_scripts/example4-mixture-of-mnfs-gd.py

import tensorflow as tf
import tensorflow_probability as tfp

from mixture_of_normalizing_flows_algorithms.model.creators.base_distributions.base_multivariate_normal_diag_creator import \
    BaseMultivariateNormalDiagCreator
from mixture_of_normalizing_flows_algorithms.model.creators.bijectors.bijector_masked_autoregressive_flow_creator import \
    BijectorMaskedAutoregressiveFlowCreator
from mixture_of_normalizing_flows_algorithms.model.creators.mixture_of_normalizing_flows_creator import \
    MixtureOfNormalizingFlowsCreator
from mixture_of_normalizing_flows_algorithms.model.creators.normalizing_flows.normalizing_flow_creator import \
    NormalizingFlowCreator
from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer

tfd = tfp.distributions
tfb = tfp.bijectors

distribution = MixtureOfNormalizingFlowsCreator(
  dtype="float32",
  normalizing_flow_creators=[
    NormalizingFlowCreator(
      base_distribution_creator=BaseMultivariateNormalDiagCreator(
        dimensionality=1,
        dtype="float32",
        scale=1.0
      ),
      bijector_creator=BijectorMaskedAutoregressiveFlowCreator(
        dtype="float32",
        hidden_units=[5],
        activation=None,
        number_of_blocks=1
        # for more expressiveness, set a higher number of blocks, which will be chained via tfb.Chain
      )
    )
    for _ in range(2)],
  trainable_categorical_logits=True
).create()

DistributionTrainer(distribution).fit_via_gd(
  batched_dataset_train=tf.data.Dataset.from_tensor_slices([[1.0], [2.0], [3.0]]).batch(3),
  optimizer=tf.optimizers.Adam(learning_rate=0.1),
  epochs=100)

print(distribution.trainable_variables)

print("hard clustering:")
print(MixturePredictor(distribution).predict([[1.0], [2.0], [3.0]]))

print("soft clustering:")
print(MixturePredictor(distribution).predict_proba([[1.0], [2.0], [3.0]]))

```

## Mixture of normalizing flows: How to run the scripts

### Training

Call with only the required arguments:
```
python -m tools.gd_em.train --algorithm gd --dataset_name two_banana
```

Call with all the arguments:

```
python -m tools.gd_em.train --algorithm em_soft --maf_hidden_units 10 4 --maf_activation tanh --maf_number_of_blocks 2 --prior_trainable True --encoder_hidden_units 10 --encoder_activations relu --encoder_temperature_decay_steps 1 --encoder_temperature_decay_rate 0.96 --encoder_temperature_initial_rate 10000 --learning_rate 0.001 --epochs 10 --m_step_epochs 10 --dataset_name two_banana --batch_size 32 --patience 10 --m_step_patience 10 --validation_split 0.2 --e_step_cache_directory tmp --output_directory artifacts --dtype float32 --seed 11 --suppress_warnings False
```

### Evaluation

```
python -m tools.gd_em.evaluate --algorithm em_hard --output_directory artifacts --dataset_name two_banana --batch_size 32
```

### Run all the experiments

```
python main.py
```

### Process the experiments' results

This creates a directory called "process_results".

```
python process_results.py
```

```
python process_results_generate_figure.py
```
