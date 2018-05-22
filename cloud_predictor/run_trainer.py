from cloud_predictor.batch_loader import ImageLoader
from cloud_predictor.FC import  FC
import numpy  as np

training_path = r"C:\Users\Host\Desktop\Clouds\Training\\"
labels_path = r"C:\Users\Host\Desktop\v2\\"

x_paths = [
    # training_path+"39UUB_20170125",
    #        training_path+"39UUB_20170425",
    #        training_path+"39UUB_20170425",
           training_path+"39UUB_20170624",
           training_path+"39UUB_20170624",
           # training_path+"39UUB_20170624",
           training_path+"39UUB_20170813",
           training_path+"39UUB_20170813",
           # training_path+"39UUB_20171027",
           # training_path+"39UUB_20171027"
           ]
y_paths = [
        # labels_path+"cl_20170125_v2.tif",
        #    labels_path+"cl_20170425_1.tif",
        #    labels_path+"cl_20170425_2.tif",
           labels_path+"cl_20170624_1.tif",
           labels_path+"cl_20170624_2.tif",
           # labels_path+"cl_20170624_3.tif",
           labels_path+"cl_20170813_1.tif ",
           labels_path+"cl_20170813_2.tif",
           # labels_path+"cl_20171027_1.tif",
           # labels_path+"cl_20171027_2.tif"
           ]
coords = [
    # (7077, 5514),
    # (7575, 1134),
    # (4517, 4065),
    (9947, 673),
    (7038, 1683),
    # (7838, 4988),
    (7135, 1709),
    (4923, 4311),
    # (7163, 2020),
    # (3307, 3774)
          ]



t = ImageLoader(x_paths, y_paths, coords)
print(np.max(t.train.X, 0))

a = FC()

a.train(t)





