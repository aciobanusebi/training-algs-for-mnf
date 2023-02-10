import tensorflow as tf

from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer import MixtureTrainer


class MixtureTrainerHardEm(MixtureTrainer):

    def __init__(self, mixture):
        super().__init__(mixture)

    def _e_step(self, batched_dataset_inputs):
        return batched_dataset_inputs.map(lambda x: MixturePredictor(self.distribution).predict(x)) \
            .map(lambda batch_cluster_id: tf.one_hot(batch_cluster_id, self.number_of_clusters))

    def _divide_clustering(self, batched_dataset_inputs, batched_dataset_clustering_matrix, cluster_index):
        return tf.data.Dataset.zip(
            (batched_dataset_inputs,
             batched_dataset_clustering_matrix.map(lambda x: x[:, cluster_index]))). \
            map(
            lambda batch, batch_cluster_onehot: tf.boolean_mask(batch, tf.cast(batch_cluster_onehot, tf.bool))). \
            unbatch().batch(batched_dataset_inputs._batch_size.numpy())

    def _dataset_and_weights_for_m_step_per_cluster(self, batched_dataset_inputs,
                                                    batched_dataset_weights_for_a_cluster):
        return batched_dataset_weights_for_a_cluster, None
