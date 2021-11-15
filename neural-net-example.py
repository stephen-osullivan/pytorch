import torch
import torch.nn as nn
import torchvision # datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# General pytorch process:
# 1) Design model (input_dim, output_dim, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop: forward: prediction + loss, backward: grads, update weights


# hyper parameters:
input_size = 784 # images are 28x28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# download mnist data, with tensor transform applied
train_dataset = torchvision.datasets.MNIST(root='./data', train = True,
   transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False,
   transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size = batch_size, shuffle=False)

#test loader
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

#plot images from sample
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap = 'gray')
plt.show()


# 1) define model which applies the feed forward method
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) # layer 1
        self.relu = nn.ReLU() # activation 1 = ReLU
        self.l2 = nn.Linear(hidden_size, num_classes) # layer 2
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        #we don't apply softmax activation fn as CrossEntropyLoss does this
        return out
    
model = NeuralNet(input_size, hidden_size, num_classes)

# 2) loss and optimizer
criterion = nn.CrossEntropyLoss() # applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 3) training loop
n_total_steps = len(train_loader) # 6000 images = 600 steps
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size)
        # forward pass
        output = model.forward(images)
        loss = criterion(output, labels)
        
        # backward pass
        loss.backward()
        
        # update
        optimizer.step()
        optimizer.zero_grad()
        if ((i+1) % 100 == 0):        
            print(f'epoch: {epoch+1}, step: {i+1}, loss = {loss:.2f}')
        

#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.view(-1, input_size)
        outputs = model(images)
        
        _, predictions = torch.max(outputs,1) # returns val, index
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    test_acc = (n_correct/n_samples) * 100

print(f'test acc: {test_acc:.2f}')

