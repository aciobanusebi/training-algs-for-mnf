class Datasets:
    @staticmethod
    def news20():
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        _20news = fetch_20newsgroups(subset="all")
        data = _20news.data
        target = _20news.target

        vectorizer = TfidfVectorizer(max_features=2000)
        data = vectorizer.fit_transform(data)
        data = data.toarray().astype(np.float32)
        # data = StandardScaler().fit_transform(data).astype(np.float32)

        return data, target

    @staticmethod
    def mnist():
        import numpy as np
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))

        real_labels = y_train

        samples = (x_train.reshape((x_train.shape[0], -1)) / 255.).astype(np.float32)
        # samples = StandardScaler().fit_transform(samples).astype(np.float32)

        return samples, real_labels

    @staticmethod
    def cifar10():
        import numpy as np
        import tensorflow as tf
        mnist = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))
        y_train = y_train.squeeze()

        real_labels = y_train

        samples = (x_train.reshape((x_train.shape[0], -1)) / 255.).astype(np.float32)
        # samples = StandardScaler().fit_transform(samples).astype(np.float32)

        return samples, real_labels

    @staticmethod
    def mnist5():
        import numpy as np
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))

        indices = y_train < 5
        x_train = x_train[indices]
        y_train = y_train[indices]

        real_labels = y_train

        samples = (x_train.reshape((x_train.shape[0], -1)) / 255.).astype(np.float32)
        # samples = StandardScaler().fit_transform(samples).astype(np.float32)

        return samples, real_labels

    @staticmethod
    def fmnist():
        import numpy as np
        import tensorflow as tf
        fmnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fmnist.load_data()

        x_train = np.concatenate((x_train, x_test))
        y_train = np.concatenate((y_train, y_test))

        real_labels = y_train

        samples = (x_train.reshape((x_train.shape[0], -1)) / 255.).astype(np.float32)
        # samples = StandardScaler().fit_transform(samples).astype(np.float32)

        return samples, real_labels

    @staticmethod
    def usps(path="data/usps.h5"):
        # from https://www.kaggle.com/datasets/bistaumanga/usps-dataset
        import numpy as np
        import h5py
        with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]

        samples = np.concatenate((X_tr, X_te))
        real_labels = np.concatenate((y_tr, y_te))
        # samples = StandardScaler().fit_transform(samples).astype(np.float32)
        return samples.astype(np.float32), real_labels

    # @staticmethod
    # def __rescale_columns(data):
    #     for i in range(data.shape[1]):
    #         mmin = min(data[:, i])
    #         mmax = max(data[:, i])
    #         data[:, i] = (data[:, i] - mmin) / (mmax - mmin)
    #     return data

    @staticmethod
    def two_banana():
        # from https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/AutoregressiveNetwork?hl=zh-cn
        # Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
        # [2]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive Flow for Density Estimation. In Neural Information Processing Systems, 2017. https://arxiv.org/abs/1705.07057
        import numpy as np
        np.random.seed(1)
        n = 2000
        x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
        x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
        x2 = (x2 - min(x2)) / max(x2)
        x1 = (x1 - min(x1)) / max(x1)
        data = np.stack([x1, x2], axis=-1)
        data = np.concatenate((data, data + 2))
        # # data = Datasets.__rescale_columns(data).astype(np.float32)
        # data = StandardScaler().fit_transform(data).astype(np.float32)
        labels = np.array([0 for _ in range(len(x1))] + [1 for _ in range(len(x2))])
        np.random.seed(None)

        return data.astype(np.float32), labels

    @staticmethod
    def smile(path="data/smile.csv"):
        # from https://profs.info.uaic.ro/~pmihaela/DM/datasets%20clustering/
        import numpy as np
        import pandas
        data = pandas.read_csv(path).to_numpy().astype(np.float32)
        samples = data[:, :2]
        labels = data[:, 2]
        # # samples = Datasets.__rescale_columns(samples)
        # samples = StandardScaler().fit_transform(samples)
        return samples, labels.astype(np.uint8)

    @staticmethod
    def moons():
        import numpy as np
        from sklearn.datasets import make_moons
        np.random.seed(1)
        data = make_moons(1000, noise=0.1)
        samples = data[0].astype(np.float32)
        labels = data[1]
        # # samples = Datasets.__rescale_columns(samples)
        # samples = StandardScaler().fit_transform(samples)
        np.random.seed(None)

        return samples, labels

    @staticmethod
    def circles():
        import numpy as np
        np.random.seed(1)
        from sklearn.datasets import make_circles
        data = make_circles(1000, noise=0.01)
        samples = data[0].astype(np.float32)
        labels = data[1]
        # # samples = Datasets.__rescale_columns(samples)
        # samples = StandardScaler().fit_transform(samples)
        np.random.seed(None)

        return samples, labels

    @staticmethod
    def pinwheel():
        # from https://github.com/emtiyaz/vmp-for-svae/blob/master/data.py
        import numpy as np

        def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
            # code from Johnson et. al. (2016)
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            np.random.seed(1)

            features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))

            feats = 10 * np.einsum('ti,tij->tj', features, rotations)

            data = np.random.permutation(np.hstack([feats, labels[:, None]]))

            np.random.seed(None)

            return data[:, 0:2], data[:, 2].astype(np.int64)

        nb_spokes = 5
        samples, labels = make_pinwheel_data(0.3, 0.05, nb_spokes, 512, 0.25)

        samples = samples.astype(np.float32)

        # samples = Datasets.__rescale_columns(samples)
        # # samples = StandardScaler().fit_transform(samples)

        return samples, labels
