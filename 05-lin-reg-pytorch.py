#%%

import torch
from torch.autograd import Variable

#%%

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

#%%


class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 1 in 1 out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


#%%
model = Model()
# construct a loss function
loss = torch.nn.MSELoss()
# model.parameters() contain the learnable parameters of the two nn.Linear modules
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#%%

for epoch in range(500):
    # forward pass: compute predicted y by passing x to the model
    y_pred = model(x_data)

    # compute and print loss
    l = loss(y_pred, y_data)
    # print("progress:", epoch, l.data[0])

    # zero gradients, perform backward pass and update weights
    optimizer.zero_grad()
    l.backward()
    optimizer.step()


#%%
to_predict = Variable(torch.Tensor([[4.0]]))
print("predict (after training)", 4, model.forward(to_predict).data[0][0])
