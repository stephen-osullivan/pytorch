# Here we defeine a custom dataset and cusomt transformer and demonstrate loading them into a dataloader

import torch, math, torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets #used to get wine data

# Datasets need at minimum an __init__, __getitem__ and __length__ method
class WineDataset(Dataset):
    def __init__(self, transform=None):
        #data loading
        wine = datasets.load_wine()
        X, y = wine.data, wine.target
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1,1)
        self.n_samples = X.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        # allows calling of dataset[i]
        sample = self.X[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        
        return sample
        
    def __len__(self):
        # allows calling len(dataset)
        return self.n_samples

#define a tensor transform, transforms must have a __call__ method.
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dataset = WineDataset(transform = ToTensor()) # length 178

# create the data loader: 4 obs in each batch, obj length = 45
dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True)

#we can use the dataloader as an iterater
num_epochs = 2
total_samples = len(dataset)
num_iterations = math.ceil(total_samples/4)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        if(i % 5 == 0):
            print(f'epoch: {epoch + 1}, batch: {i}, batchsize: {len(labels)}')
            
