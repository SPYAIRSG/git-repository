import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets._samples_generator import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles, make_moons

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
    data, label = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=[1.0, 2.0, 3.0])

    # 绘制样本显示
    plt.scatter(data[:, 0], data[:, 1], c=label)
    plt.show()

def circle_and_moons():
    fig = plt.figure(1)

    plt.subplot(121)
    x1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.1)
    plt.title('make_circle function example')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)

    plt.subplot(122)
    x1, y1 = make_moons(n_samples=1000, noise=0.1)
    plt.title('make_moons function example')
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
    plt.show()



if __name__ == '__main__':
    circle_and_moons()