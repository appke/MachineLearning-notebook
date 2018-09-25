import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

iris = datasets.load_iris()

x = iris.data[:, :2]
y = iris.target
step = 0.02

color_map_front = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
color_map_back = ListedColormap(['#FF9999', '#99FF99', '#9999FF'])

def knn_predict(weights):
    sample_count = 5
    clf = neighbors.KNeighborsClassifier(sample_count, weights=weights)
    clf.fit(x, y)

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    x_range, y_range = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    zinput = np.c_[x_range.ravel(), y_range.ravel()]
    z = clf.predict(zinput)
    z = z.reshape(x_range.shape)

    plt.figure()
    plt.pcolormesh(x_range, y_range, z, cmap=color_map_back)
    plt.scatter(x[:, 0], x[:, 1], c = y, cmap=color_map_front, edgecolors='k', s = 20)

    plt.title(weights)
    plt.show()

knn_predict('uniform')
knn_predict('distance')