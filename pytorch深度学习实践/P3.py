import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += np.square(y_pred-y)
        # cost = (y_pred-y)**2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)



loss = []
lr = 0.01

for epoch in range(100):
    loss_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    loss.append(loss_val)
    print('epoch: ', epoch, 'w: ', w, 'cost: ', loss_val)

epoch = np.arange(0, 100)
epoch = list(epoch)

plt.plot(epoch, loss)
plt.xlabel(epoch)
plt.ylabel(loss)
plt.show()

