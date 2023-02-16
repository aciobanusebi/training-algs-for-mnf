import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from mixture_of_normalizing_flows_algorithms.evaluate.avg_meter import AvgMeter

tfkl = keras.layers


class DistributionTrainer:
    def __init__(self, distribution):
        self.distribution = distribution
        inputs = tfkl.Input(shape=self.distribution.event_shape,
                            dtype=self.distribution.dtype)
        self.distribution.log_prob(
            inputs)  # if not called, then the initial self.distribution.trainable_variables contains only the "cat" variables

    def _get_trainable_variables(self):
        return self.distribution.trainable_variables

    def _loss(self, data, sample_weights=None):
        log_probs = self.distribution.log_prob(data)
        if sample_weights is not None:
            log_probs = sample_weights * log_probs
        return -tf.reduce_mean(log_probs)

    def _compute_loss(self, batched_dataset_inputs, batched_dataset_sample_weights):
        loss = AvgMeter()
        batched_sample_weights_iterator = None
        if batched_dataset_sample_weights is not None:
            batched_sample_weights_iterator = iter(batched_dataset_sample_weights)
        for batch in batched_dataset_inputs:
            loss.update(self._loss(batch, next(
                batched_sample_weights_iterator) if batched_sample_weights_iterator is not None else None).numpy(),
                        n=len(batch))
        return loss.avg

    def _process_loss(self, loss, list_to_append_to, postfix, key):
        list_to_append_to.append(loss)
        postfix[key] = f"{loss}"

    @staticmethod
    @tf.function
    def _get_loss_and_grads(self, data, sample_weights=None):
        # https://ekamperi.github.io/mathematics/2020/12/26/tensorflow-trainable-probability-distributions.html
        with tf.GradientTape() as tape:
            tape.watch(self._get_trainable_variables())
            loss = self._loss(data, sample_weights)
            grads = tape.gradient(loss, self._get_trainable_variables())
        return loss, grads

    def _plot_and_print(self, output_directory, losses_train, batched_dataset_validation, losses_validation,
                        print_final_best, best_epoch, best_loss, start):
        if output_directory is not None:
            history = {
                "Train loss": losses_train
            }
            if batched_dataset_validation is not None:
                history["Validation loss"] = losses_validation
            pd.DataFrame(history).plot()
            plt.savefig(f'{output_directory}/plot.png')
            with open(f'{output_directory}/history.json', "w") as file:
                json.dump(history, file)

        if print_final_best:
            data_type = "Val" if batched_dataset_validation is not None else "Train"
            text = f'\nBest {data_type} Epoch:{best_epoch} | {data_type} Loss:{best_loss} | time: {(time.time() - start) / 60} minutes'
            print(text)
            if output_directory is not None:
                with open(f'{output_directory}/logs.txt', 'a') as f:
                    f.write(text)

    def _save_model(self, output_directory):
        checkpoint = tf.train.Checkpoint(self.distribution)
        checkpoint.write(f'{output_directory}/best_model/')

    def fit_via_gd(self, batched_dataset_train, optimizer, epochs,
                   batched_dataset_sample_weights_train=None,
                   batched_dataset_validation=None, batched_dataset_sample_weights_validation=None,
                   patience=None,
                   output_directory=None, tqdm_kwargs=None, print_final_best=True):
        losses_train = []
        losses_validation = []
        best_loss = np.inf
        best_epoch = -1
        early_stopping = 0
        start = time.time()
        for epoch in (pbar := tqdm(range(epochs), **(tqdm_kwargs if tqdm_kwargs is not None else {}))):
            self.epoch = epoch
            loss_train_per_epoch = AvgMeter()
            batched_sample_weights_train_iterator = None
            if batched_dataset_sample_weights_train is not None:
                batched_sample_weights_train_iterator = iter(batched_dataset_sample_weights_train)
            for batch_train in batched_dataset_train:
                loss_train_per_batch, grads_train_per_batch = DistributionTrainer._get_loss_and_grads(self, batch_train,
                                                                                                      next(
                                                                                                          batched_sample_weights_train_iterator) if batched_sample_weights_train_iterator is not None else None
                                                                                                      )
                optimizer.apply_gradients(zip(grads_train_per_batch, self._get_trainable_variables()))
                loss_train_per_epoch.update(loss_train_per_batch.numpy(), n=batch_train.shape[0])

            if np.isnan(loss_train_per_epoch.avg):
                raise Exception("loss is nan => exit program")

            postfix = {}
            self._process_loss(loss_train_per_epoch.avg, losses_train, postfix, "train_loss")

            loss_to_compare = loss_train_per_epoch.avg

            if batched_dataset_validation is not None:
                loss_validation = self._compute_loss(batched_dataset_validation,
                                                     batched_dataset_sample_weights_validation)
                self._process_loss(loss_validation, losses_validation, postfix, "val_loss")

                loss_to_compare = loss_validation

            if best_loss > loss_to_compare:
                early_stopping = 0
                best_loss = loss_to_compare
                best_epoch = epoch

                if output_directory is not None:
                    self._save_model(output_directory)
            else:
                early_stopping += 1
            if patience is not None:
                if early_stopping == patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

            pbar.set_postfix(postfix)

        self._plot_and_print(output_directory, losses_train, batched_dataset_validation, losses_validation,
                             print_final_best, best_epoch, best_loss, start)
