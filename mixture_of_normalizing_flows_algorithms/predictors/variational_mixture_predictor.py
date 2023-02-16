import tensorflow as tf


class VariationalMixturePredictor:
    def __init__(self, encoder, temperature):
        self.encoder = encoder
        self.temperature = temperature

    def predict(self, data):
        return tf.math.argmax(self.predict_proba(data), axis=1)

    def predict_proba(self, data):
        return tf.nn.softmax(self.encoder(data) / self.temperature)
