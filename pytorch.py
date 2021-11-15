
##### basics ######
import torch

# creating arrays
x = torch.empty(4,5)
x = torch.rand(4,5)
y = torch.ones(4,5, dtype = torch.float32)

# arithmetic
x + y 
x.add(y) #using methods
x.add_(y) # method ending with "_" is in place

# slicing
x[:,1]
x[1,1].item() # can only be used on a 1x1 tensor

# reshaping
y = x.view(1,4) # reshaping

# converting to numpy arrays
a = torch.ones(5)
b = a.numpy() # tensor to numpy array
a.add_(1) # changing a affects b
b
a = torch.from_numpy(b) # convert numpy array to tensor

# creating tensor on gpu, note only cpu tensors can be convert to numpy
if torch.cuda.is_avaialable():
    device = torch.device.("cuda")
    x = torch.ones(5, device = device) # create on gpu
    y = torch.ones(5)
    y = y.to(device) # move to gpu
    y.to("cpu") # move back to cpu

# autograd
# if z is a scalar function:
x = torch.rand(3, requires_grad=True) # gradient tracked: we want df/dx later
y = x + 2
print(y) # y now has grad_fn attribute <AddBackward>
z = sum(y**2 )
z.backward() # dz/dx is calculated
x.grad == 2*(x+2) # xgrad is dz/dx i.e. 2*(x+2)
# if z is a vector function then we need to pass a vector
z = y**2
z.backward(torch.tensor([0.1,0.1,0.1]))
x.grad

# stop tracking gradients (3 ways)
x.requires_grad_(False)
x.detach()
with torch.no_grad(): # wrap code in this

# reset gradients, this is important as otherwise looping will add to the gradient
x.grad.zero_() # do this after each training loop

##### linear regression using autograd #####

X = torch.rand(10, 2)
y = 2*X[:,0] + 3*X[:,1]
# we need to record the derivative w.r.t w:
w = torch.tensor([0,0], dtype= torch.float32, requires_grad = True)
learning_rate = 0.1
n_iters = 100

def forward(X):
    return X @ w

def loss(y, y_pred):
    return ((y-y_pred)**2).mean()

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(y, y_pred)
    l.backward() # this calculate dl/dw
    with torch.no_grad(): # this calc should not effect w.grad
        w -= learning_rate * w.grad
    w.grad.zero_() # otherwise the gradient will be added to
    if((epoch + 1) %20 == 0): # console update every 20 iterations
        print(f'w0 = {w[0]: .2f}, w1 = {w[1]: .2f}')
        print(f'epoch: {epoch + 1}, loss: {l:.2f}')
print(w)

##### linear regression using pytorch functions #####
# General pytorch process:
# 1) Design model (input_dim, output_dim, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop: forward: prediction + loss, backward: grads, update weights

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
X_np, y_np = datasets.make_regression(
    n_samples = 100, n_features = 1, noise = 20, random_state = 1)
plt.scatter(X_np, y_np)

X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape y into a column vector

# 1) model
n_samples, n_features = X.shape
input_size, output_size = n_features, 1
model = nn.Linear(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # MSE Loss fn
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # backward pass
    loss.backward()

    # weight update
    optimizer.step()
    optimizer.zero_grad()
    
# plot
predicted = model(X).detach().numpy() # stop gradient calcs
plt.plot(X_np, y_np, 'ro')
plt.plot(X_np, predicted)
plt.show()

##### logistic regression using pytorch class structure #####
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 123)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1,1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1,1)

# 1) model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, X):
        return torch.sigmoid(self.linear(X))

model = LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss() # binary cross entropy loss
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
num_epochs = 200
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    #backward pass
    loss.backward()
    
    #updates
    optimizer.step()
    optimizer.zero_grad()
    
    if((epoch +1) % 20 == 0):
        print(f'epoch: {epoch +1}, loss: {loss:.2f}')

with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred_cls = y_test_pred.round()
    test_acc = (y_test_pred_cls.eq(y_test).sum())/len(y_test)
    y_pred_cls = y_pred.round()
    train_acc = (y_pred_cls.eq(y_train).sum())/len(y_train) 
    print(f'test accuracy: {train_acc:.3f}, train accuracy = {train_acc:.3f}')


