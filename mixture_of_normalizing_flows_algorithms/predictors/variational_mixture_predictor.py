import tensorflow as tf


class VariationalMixturePredictor:
    def __init__(self, encoder):
        self.encoder = encoder

    def predict(self, data):
        return tf.math.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        return self.encoder(data)
