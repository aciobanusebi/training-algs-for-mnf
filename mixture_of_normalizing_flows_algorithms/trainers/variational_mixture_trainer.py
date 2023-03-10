import tensorflow as tf

from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer


class VariationalMixtureTrainer(DistributionTrainer):
    def __init__(self, mixture, encoder, temperature_scheduler, epoch=None):
        super().__init__(mixture)
        self.number_of_clusters = mixture.num_components
        self.encoder = encoder
        self.temperature_scheduler = temperature_scheduler
        self.epoch = epoch

    def _get_trainable_variables(self):
        return list(self.distribution.trainable_variables) + list(self.encoder.trainable_variables)

    def _loss(self, data, sample_weights=None):
        log_priors = [self.distribution.cat.log_prob(i) for i in range(self.number_of_clusters)]
        temperature = self.temperature_scheduler(self.epoch)
        posteriors = tf.nn.softmax(self.encoder(data) / temperature)
        log_posteriors = tf.math.log(posteriors)
        components_log_probs = [self.distribution.components[cluster_index].log_prob(data) for cluster_index in
                                range(self.number_of_clusters)]

        elbos = 0
        for cluster_index in range(self.number_of_clusters):
            elbos += posteriors[:, cluster_index] * (
                    components_log_probs[cluster_index] + log_priors[cluster_index] - log_posteriors[:,
                                                                                      cluster_index])

        if sample_weights is not None:
            elbos = sample_weights * elbos
        return -tf.reduce_mean(elbos)

    def _save_model(self, output_directory):
        super(VariationalMixtureTrainer, self)._save_model(output_directory)
        checkpoint = tf.train.Checkpoint(self.encoder)
        checkpoint.write(f'{output_directory}/best_encoder/')
        with open(f'{output_directory}/epoch.txt', "w") as f:
            f.write(f"{self.epoch}")
