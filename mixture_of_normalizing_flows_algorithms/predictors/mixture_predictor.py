import tensorflow as tf


class MixturePredictor:
    def __init__(self, mixture):
        self.mixture = mixture
        self.number_of_clusters = self.mixture.num_components

    @staticmethod
    def __row_normalization(matrix):
        return tf.linalg.normalize(
            matrix, ord=1, axis=1
        )[0]

    def predict_proba(self, data):
        # this should be efficiently implemented in order to avoid a soft clustering with all zeros
        # (due to very low log probs, e.g.
        # [-11790.2568, -11614.082] =>
        # [exp(-11790.2568)/(exp(-11790.2568) + exp(-11614.082)), exp(-11614.082)/(exp(-11790.2568) + exp(-11614.082))] =>
        # [0, 0], which is wrong; it should have been [0, 1]);
        # this problem is solved if we use log-sum-exp trick:https://en.wikipedia.org/wiki/LogSumExp#:~:text=Therefore%2C%20many%20math%20libraries%20such%20as%20IT%2B%2B%20provide%20a%20default%20routine%20of%20LSE%20and%20use%20this%20formula%20internally.
        # using tf.nn.softmax solved this issue

        # manual, non-stable implementation:
        # mixture_cat_probs = [self.mixture.cat.prob(i) for i in range(self.number_of_clusters)]
        # soft_clustering_matrix = [None for _ in range(self.number_of_clusters)]
        # for i in range(self.number_of_clusters):
        #     soft_clustering_matrix[i] = mixture_cat_probs[i] * self.mixture.components[i].prob(data)
        # soft_clustering_matrix = tf.transpose(tf.convert_to_tensor(soft_clustering_matrix, dtype=self.mixture.dtype))
        # soft_clustering_matrix = MixturePredictor.__row_normalization(soft_clustering_matrix)
        # return soft_clustering_matrix

        # stable implementation:
        soft_clustering_matrix = tf.nn.softmax(self.predict_log_unnormalized(data))
        return soft_clustering_matrix

    def predict(self, data):
        mixture_components_weighted_log_probs = self.predict_log_unnormalized(data)
        clustering = tf.math.argmax(mixture_components_weighted_log_probs, axis=1)
        return clustering

    def predict_log_unnormalized(self, data):
        mixture_cat_log_probs = [self.mixture.cat.log_prob(i) for i in range(self.number_of_clusters)]
        mixture_components_weighted_log_probs = [None for _ in range(self.number_of_clusters)]
        for i in range(self.number_of_clusters):
            mixture_components_weighted_log_probs[i] = mixture_cat_log_probs[i] + self.mixture.components[i].log_prob(
                data)
        mixture_components_weighted_log_probs = tf.transpose(
            tf.convert_to_tensor(mixture_components_weighted_log_probs, dtype=self.mixture.dtype))
        return mixture_components_weighted_log_probs
