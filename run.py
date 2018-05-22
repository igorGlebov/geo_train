from cloud_predictor import FC, data_loader


def get_matrix(path, coords, ksize,  model_path, threshold_const):

    loader = data_loader.DataConverter(path)
    predictor = FC.FC(path=model_path)
    threshold = lambda x: 1 if x[0][1] > threshold_const else 0
    list = []

    for h in range(coords[0], coords[0] + ksize[0]):
        temp = []
        for w in range(coords[1], coords[1] + ksize[1]):
            pr = predictor.predict(loader.get_vector([h, w]))
            temp.append(threshold(pr))
            print(pr)
        list.append(temp)

    return list


