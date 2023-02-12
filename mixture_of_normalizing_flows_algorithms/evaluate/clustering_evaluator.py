import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score


class ClusteringEvaluator:
    @staticmethod
    def __purity_score(y_true, y_pred):  # from https://stackoverflow.com/a/51672699/7947996
        # compute contingency matrix (also called confusion matrix)
        matrix = contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

    @staticmethod
    def __normalized_contingency_matrix(y_true, y_pred):
        matrix = contingency_matrix(y_true, y_pred)
        return matrix / matrix.sum(axis=1, keepdims=True)

    @staticmethod
    def __accuracy(true_row_labels, predicted_row_labels):
        # from coclust package
        def _make_cost_m(cm):
            s = np.max(cm)
            return (- cm + s)

        cm = confusion_matrix(true_row_labels, predicted_row_labels)
        indexes = linear_assignment(_make_cost_m(cm))
        total = 0
        for row, column in zip(*indexes):
            value = cm[row][column]
            total += value

        return (total * 1. / np.sum(cm))

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "purity_score": ClusteringEvaluator.__purity_score(y_true, y_pred),
            "adjusted_rand_score": adjusted_rand_score(y_true, y_pred),
            "normalized_mutual_info_score": normalized_mutual_info_score(y_true, y_pred),
            "accuracy": ClusteringEvaluator.__accuracy(y_true, y_pred),
            "contingency_matrix": contingency_matrix(y_true, y_pred).tolist(),
            "normalized_contingency_matrix": ClusteringEvaluator.__normalized_contingency_matrix(y_true,
                                                                                                 y_pred).tolist()
        }

    @staticmethod
    def plot_density_2d(data, distribution, log=False):
        # from https://tiao.io/post/building-probability-distributions-with-tensorflow-probability-bijector-api/
        u1_lim = min(data[:, 0]), max(data[:, 0])
        u2_lim = min(data[:, 1]), max(data[:, 1])

        u1 = tf.linspace(*u1_lim, 100)
        u2 = tf.linspace(*u2_lim, 100)
        u_grid = tf.stack(tf.meshgrid(u1, u2), axis=-1)

        fig, ax = plt.subplots(figsize=(6, 5))

        if log:
            ax.set_title('log density log$p_{X}(\mathbf{x})$')
            cb = ax.pcolormesh(u1, u2, distribution.log_prob(u_grid), cmap='inferno')
        else:
            ax.set_title('density $p_{X}(\mathbf{x})$')
            cb = ax.pcolormesh(u1, u2, distribution.prob(u_grid), cmap='inferno')

        fig.colorbar(cb, ax=ax)

        ax.set_xlim(u1_lim)
        ax.set_ylim(u2_lim)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')

    @staticmethod
    def plot_decision_boundary_2d(model, X, y):
        # from https://raw.githubusercontent.com/aciobanusebi/ml2/master/planar_utils_modified.py
        X = X.T
        y = y[np.newaxis, ...]
        # Set min and max values and give it some padding
        x_min, x_max = -0.25, 1.25
        y_min, y_max = -0.25, 1.25
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)
