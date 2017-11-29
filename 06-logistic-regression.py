#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#%%


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

#%%


model = Model()

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


#%%
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [1.], [1.]]))

for epoch in range(1000):
    y_pred = model(x_data)

    # compute loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # pytorch learning step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#%%
# after training
test = Variable(torch.Tensor([[1.0]]))
print("predict 1 hour", 1.0, model.forward(test).data[0][0] > 0.5)

test = Variable(torch.Tensor([[7.0]]))
print("predict 7 hours", 7.0, model.forward(test).data[0][0] > 0.5)
