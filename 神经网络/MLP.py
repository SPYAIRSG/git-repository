import os
import struct
import numpy as np

# 把分类转化为独热码
def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# 加载数据
def load_mnist(path, kind='train', normal=False, onehot=False):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>2I', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
        if onehot:
            labels = dense_to_one_hot(labels)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">4I", imgpath.read(16))
            if normal:
                images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(-labels), 784)/255
            else:
                images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(-labels), 784)

        return images, labels

x_train, y_train = load_mnist(r'mnist', kind='train', normal=True)
print('rows: %d, columns: %d' % (x_train.shape[0], x_train.shape[1]))
print('rows: %d' % (y_train.shape[0]))

x_test, y_test = load_mnist(r'mnist', kind='t10k', normal=True)
print('rows: %d, columns: %d' % (x_test.shape[0], x_test.shape[1]))

# 初始化参数
def initialize_with_zeros(n_x, n_h, n_y, std=0.001):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x)*std
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * std
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }
    return parameters
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

# 构建神经网络
def forward(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X)+b1
    # A1 = sigmoid(Z1)
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    dict = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

    return A2, dict

# 损失函数
def loss(A2, Y, parameters):

    t = 1e-6
    logprobs = np.multiply(np.log(A2+t), Y)+np.multiply(np.log(1-A2+t), (1-Y))
    loss1 = np.sum(logprobs ,axis=0, keepdims=True)/A2.shape[0]
    return loss1*(-1)

def backward(parameters, dict, X, Y):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = dict["A1"]
    A2 = dict["A2"]
    Z1 = dict["Z1"]

    dZ2 = A2 - Y

    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2}
    return grads

# 梯度更新
def gradient(parameters, grads, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


# 训练模型
if __name__ == '__main__':
    train_images = x_train
    train_labels = y_train
    test_images = x_test
    test_labels = y_test

    count = 0
    n_x = 28 * 28
    n_h = 100
    n_y = 10

    lr = 0.01
    loss_all = []
    train_size = 60000
    parameters = initialize_with_zeros(n_x, n_h, n_y)
    for i in range(10000):

        img_train = train_images[i]
        label_train1 = train_labels[i]
        label_train = np.zeros((10, 1))

        # 动态修改学习率
        if i % 2000 == 0:
            lr = lr * 0.99
        label_train[int(train_labels[i])] = 1
        imgvector = np.expand_dims(img_train, axis=1)
        A2, dict = forward(imgvector, parameters)
        pre_labels = np.argmax(A2)

        loss1 = loss(A2, label_train, parameters)
        grads = backward(parameters, dict, imgvector, label_train)
        parameters = gradient(parameters, grads, learning_rate=lr)
        grads["dW1"] = 0
        grads["dW2"] = 0
        grads["db1"] = 0
        grads["db2"] = 0

        if i%200 == 0:
            print("迭代：{}次的损失：{:.6f}".format(i, loss1[0][0]))
            loss_all.append(loss1[0][0])

    for i in range(10000):
        img_test = test_images[i]
        vector_image = np.expand_dims(img_test, axis=1)
        label_trainx = test_labels[i]
        aa2, xxx = forward(vector_image, parameters)
        predict_value = np.argmax(aa2)
        if predict_value == int(label_trainx):
            count += 1
            print("准确率：", count/10000)















