import numpy as np
from sklearn.model_selection import train_test_split
from cloud_predictor.data_loader import DataConverter, get_raster_vector


class _DataInterface(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def next_batch(self, num):

        idx = np.arange(0, len(self.X))
        np.random.shuffle(idx)
        idx = idx[:num]
        data_shuffle = [self.X[i] for i in idx]
        labels_shuffle = [self.y[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(labels_shuffle)


class ImageLoader(object):

    def __init__(self, x_paths, y_paths, coords):

        x = []
        y = []

        for i in range(len(x_paths)):
            x_full = DataConverter(x_paths[i])
            y_matrix = get_raster_vector(y_paths[i])

            for h in range(coords[i][0], coords[i][0]+y_matrix.shape[0]):
                for w in range(coords[i][1], coords[i][1] + y_matrix.shape[1]):
                    x.append(x_full.get_vector([h, w]))

            y_one_hot = np.asarray(list(map(lambda x: [0, 1] if x>0 else [1, 0], y_matrix.reshape(y_matrix.shape[0]*y_matrix.shape[1]))))
            y_old = y.copy()
            y = np.concatenate([y_old, y_one_hot]) if len(y) else y_one_hot

            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        self.train = _DataInterface(X_train, y_train)
        self.test = _DataInterface(X_test, y_test)