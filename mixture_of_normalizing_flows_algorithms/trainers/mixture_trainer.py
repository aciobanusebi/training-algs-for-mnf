import os
import shutil
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mixture_of_normalizing_flows_algorithms.trainers.distribution_trainer import DistributionTrainer


class MixtureTrainer(DistributionTrainer, ABC):
    def __init__(self, mixture):
        super().__init__(mixture)
        self.number_of_clusters = mixture.num_components

    warning = "Generally, you should not call .shuffle(...) on the dataset. Still, you can call .shuffle(..., reshuffle_each_iteration=False), but cannot call .shuffle(..., reshuffle_each_iteration=True)."

    @abstractmethod
    def _e_step(self, batched_dataset_inputs):
        pass

    @abstractmethod
    def _divide_clustering(self, batched_dataset_inputs, batched_dataset_clustering_matrix, cluster_index):
        pass

    @abstractmethod
    def _dataset_and_weights_for_m_step_per_cluster(self, batched_dataset_inputs,
                                                    batched_dataset_weights_for_a_cluster):
        pass

    def _e_step_and_postprocessing(self, e_step_cache_directory, unshuffled_batched_dataset, phase):
        batched_dataset_clustering_matrix = self._e_step(unshuffled_batched_dataset)

        cache_e_step_filename = ""
        if e_step_cache_directory is not None:
            if os.path.exists(e_step_cache_directory):
                shutil.rmtree(e_step_cache_directory)
            Path(e_step_cache_directory).mkdir(parents=True, exist_ok=True)
            cache_e_step_filename = f"{e_step_cache_directory}/e_step_{phase}"
        batched_dataset_clustering_matrix = batched_dataset_clustering_matrix.cache(cache_e_step_filename)

        clustering_matrix_sum_per_cluster = 0
        clustering_matrix_count_per_cluster = 0
        for batch_clustering_matrix in batched_dataset_clustering_matrix:
            clustering_matrix_sum_per_cluster += tf.math.reduce_sum(batch_clustering_matrix, axis=0)
            clustering_matrix_count_per_cluster += batch_clustering_matrix.shape[0]

        clustering_matrix_per_cluster = [None for _ in range(self.number_of_clusters)]
        for cluster_index in range(self.number_of_clusters):
            clustering_matrix_per_cluster[cluster_index] = self._divide_clustering(unshuffled_batched_dataset,
                                                                                   batched_dataset_clustering_matrix,
                                                                                   cluster_index)

        return clustering_matrix_sum_per_cluster, clustering_matrix_count_per_cluster, clustering_matrix_per_cluster

    def fit_via_em(self, unshuffled_batched_dataset_train, em_iterations, m_step_optimizer_getters, m_step_epochs,
                   unshuffled_batched_dataset_validation=None,
                   m_step_patience=None,
                   em_patience=None,
                   e_step_cache_directory=None,
                   output_directory=None,
                   print_final_best=True):
        warnings.warn(self.warning)

        losses_train = []
        losses_validation = []
        best_loss = np.inf
        best_iteration = -1
        early_stopping = 0
        start = time.time()
        for iteration in (pbar := tqdm(range(em_iterations), desc="EM", position=0)):
            # E step:
            clustering_matrix_sum_per_cluster_train, clustering_matrix_count_per_cluster_train, clustering_matrix_per_cluster_train = self._e_step_and_postprocessing(
                e_step_cache_directory, unshuffled_batched_dataset_train, phase="train")
            clustering_matrix_per_cluster_validation = None
            if unshuffled_batched_dataset_validation is not None:
                _, _, clustering_matrix_per_cluster_validation = self._e_step_and_postprocessing(e_step_cache_directory,
                                                                                                 unshuffled_batched_dataset_validation,
                                                                                                 phase="validation")

            # M step:
            cat_probs = clustering_matrix_sum_per_cluster_train / clustering_matrix_count_per_cluster_train
            ## update cat only if there is a trainable tf.Variable:
            if self.distribution.cat.probs is not None:
                if isinstance(self.distribution.cat.probs, tf.Variable):
                    if self.distribution.cat.probs.trainable:
                        self.distribution.cat.probs.assign(cat_probs)
            else:
                cat_logits = np.log(cat_probs)
                if isinstance(self.distribution.cat.logits, tf.Variable):
                    if self.distribution.cat.logits.trainable:
                        self.distribution.cat.logits.assign(cat_logits)
            for cluster_index in range(self.number_of_clusters):
                batched_dataset_train, batched_dataset_sample_weights_train = self._dataset_and_weights_for_m_step_per_cluster(
                    unshuffled_batched_dataset_train, clustering_matrix_per_cluster_train[cluster_index])
                batched_dataset_validation, batched_dataset_sample_weights_validation = None, None
                if unshuffled_batched_dataset_validation is not None:
                    batched_dataset_validation, batched_dataset_sample_weights_validation = self._dataset_and_weights_for_m_step_per_cluster(
                        unshuffled_batched_dataset_validation, clustering_matrix_per_cluster_validation[cluster_index])

                DistributionTrainer(self.distribution.components[cluster_index]).fit_via_gd(
                    optimizer=m_step_optimizer_getters[cluster_index](),
                    batched_dataset_train=batched_dataset_train,
                    epochs=m_step_epochs,
                    batched_dataset_sample_weights_train=batched_dataset_sample_weights_train,
                    batched_dataset_validation=batched_dataset_validation,
                    batched_dataset_sample_weights_validation=batched_dataset_sample_weights_validation,
                    tqdm_kwargs={
                        "desc": f"Iteration {iteration}: M step for cluster {cluster_index}",
                        "position": 1,
                        "leave": False
                    },
                    print_final_best=False,
                    patience=m_step_patience
                )

            postfix = {}
            loss_train = self._compute_loss(unshuffled_batched_dataset_train, batched_dataset_sample_weights=None)
            self._process_loss(loss_train, losses_train, postfix, "train_loss")

            loss_to_compare = loss_train

            if unshuffled_batched_dataset_validation is not None:
                loss_validation = self._compute_loss(unshuffled_batched_dataset_validation,
                                                     batched_dataset_sample_weights=None)
                self._process_loss(loss_validation, losses_validation, postfix, "val_loss")

                loss_to_compare = loss_validation

            if best_loss > loss_to_compare:
                early_stopping = 0
                best_loss = loss_to_compare
                best_iteration = iteration

                if output_directory is not None:
                    self._save_model(output_directory)
            else:
                early_stopping += 1
            if em_patience is not None:
                if early_stopping == em_patience:
                    print(f"Early stopping at EM iteration/epoch {iteration}.")
                    break

            pbar.set_postfix(postfix)

        self._plot_and_print(output_directory, losses_train, unshuffled_batched_dataset_validation, losses_validation,
                             print_final_best, best_iteration, best_loss, start)

        if e_step_cache_directory is not None:
            if os.path.exists(e_step_cache_directory):
                shutil.rmtree(e_step_cache_directory)
