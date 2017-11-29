#%%
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# a random guess for weight
w = 1.0
w2 = 0.5
b = 1.0


def forward(x):  # model forward pass
    return x * x * w2 + x * w + b


def loss(x, y):  # los function
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x, y):  # gradient computation function
    return 2 * w2 * x + w



#%%
print("predict (before training", 4, forward(4))

w_list = []
l_list = []
# training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        w_list.append(w)
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
        l_list.append(l)

    print("progress:", epoch, "w=", w, "loss=", l)


# after training
print("predict (after training)", 4, forward(4))

plt.plot(w_list, l_list)
plt.xlabel('weights')
plt.ylabel('loss')
plt.show()
