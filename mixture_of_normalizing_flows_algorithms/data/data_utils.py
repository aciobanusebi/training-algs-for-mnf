import tensorflow as tf


class DataUtils:
    @staticmethod
    def get_dummy_labels(data, dtype):
        return tf.zeros((data.shape[0], 0), dtype=dtype)
