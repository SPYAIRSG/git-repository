import torch

x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 13.0, 25.0]

w1 = torch.tensor([1.0])
w2 = torch.tensor([1.0])
b = torch.tensor([0.0])

w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return w1 * x * x + w2 * x + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)**2

print("predict", forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)           # forward
        l.backward()             # backward
        print("\tgrad:", x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("process:", epoch, l.item())

print("predict", forward(4).item())
print("w1 w2 b:", w1.item(), w2.item(), b.item())