import run
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':

    matrix = run.get_matrix(r"C:\Users\Host\Desktop\Clouds\Training\39UUB_20170624", [1683, 7038], [408, 503], r"data\model.ckpt-339", 0.5)
    f = np.vectorize(lambda x: 255 if x else 0)
    plt.imshow(f(matrix))
    print(sum(f(matrix)))
    plt.show()


