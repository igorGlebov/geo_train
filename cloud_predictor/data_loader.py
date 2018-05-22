from PIL import Image
import numpy as np


class DataConverter:
    """Docs for DataConverter"""

    def __init__(self, path):
        """Constructor for DataConverter"""

        # self.scale = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 6, 2, 2]
        # self.files = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

        # self.max_size = [4000, 1024, 25000, 5000]

        self.scale = [6, 6, 2, 2]
        self.files = ['B09', 'B10', 'B11', 'B12']
        self.matrixs = []

        print('Loading spectra')
        for i in range(len(self.files)):
            self.matrixs.append(self.jb2_to_np_matrix(path + '\\' + self.files[i] + '.jp2'))
        print('Spectra complete')

    def get_vector(self, coords):
        vec = []
        for i in range(len(self.files)):
            vec.append(self.matrixs[i][int(coords[0] / self.scale[i]), int(coords[1] / self.scale[i])])
        return vec

    def jb2_to_np_matrix(self, path):
        img = Image.open(path)
        matrix = np.asarray(img, dtype=np.float32)
        return matrix


def get_raster_vector(path):
    image = Image.open(path)
    matrix_3D = np.asarray(image, dtype=np.float32)
    matrix = matrix_3D[:,:,1]
    f = np.vectorize(lambda x: 1 if x else 0)
    matrix = f(matrix)

    return matrix
