#%%

import numpy as np

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)


def nonlin(x, deriv=False):

    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
#%%


syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for iter in range(1):

    # define layers
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # resolve errors
    l2_error = y - l2
    l2_delta = l2_error * nonlin(l2, deriv=True)
    print("\n\nLayer 2 error\n", l2_error, "\n and delta \n", l2_delta)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)
    print("Layer 1 error \n", l1_error, "\n and delta \n", l1_delta)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    if(iter % 1000) == 0:
        print("Error ", np.mean(np.abs(l2_error)))
        print("\nWeights\n", syn1, "\n", syn0)
