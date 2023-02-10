from mixture_of_normalizing_flows_algorithms.predictors.mixture_predictor import MixturePredictor
from mixture_of_normalizing_flows_algorithms.trainers.mixture_trainer import MixtureTrainer


class MixtureTrainerSoftEm(MixtureTrainer):

    def __init__(self, mixture):
        super().__init__(mixture)

    def _e_step(self, batched_dataset_inputs):
        return batched_dataset_inputs.map(MixturePredictor(self.distribution).predict_proba)

    def _divide_clustering(self, batched_dataset_inputs, batched_dataset_clustering_matrix, cluster_index):
        return batched_dataset_clustering_matrix.map(lambda x: x[:, cluster_index])

    def _dataset_and_weights_for_m_step_per_cluster(self, batched_dataset_inputs,
                                                    batched_dataset_weights_for_a_cluster):
        return batched_dataset_inputs, batched_dataset_weights_for_a_cluster
