#%%

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# a random guess for weight
w = 1.0


def forward(x):  # model forward pass
    return x * w


def loss(x, y):  # los function
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


def gradient(x, y):  # gradient computation function
    return 2 * x * (x * w - y)



#%%
print("predict (before training", 4, forward(4))

# training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)

    print("progress:", epoch, "w=", w, "loss=", l)


# after training
print("predict (after training)", 4, forward(4))
