import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets._samples_generator import make_classification
from sklearn.datasets import make_blobs

def classification():
    X, y = make_classification(n_samples=6, n_features=5, n_informative=2, n_redundant=2,
                               n_classes=2, n_clusters_per_class=2, scale=1.0, random_state=20)

    for x, y in zip(X, y):
        print(y, end=":")
        print(x)


def blobs():
    # X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
    # print(X.shape)
    # print(y.shape)
    data, label = make_blobs(n_samples=100, n_features=2, centers=5)

    # 绘制样本显示
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.show()



if __name__ == '__main__':
    blobs()