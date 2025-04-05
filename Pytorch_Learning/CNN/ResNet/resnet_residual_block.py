  # @file    resnet_residual_block.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/05 14:34:46
  # @version 1.0
  # @brief ResetNet的残差块

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.01
num_epochs = 10
batch_size = 128

# Architecture
num_classes = 10


##########################
### MNIST DATASET
##########################

# Note transforms.ToTensor() scales input images
# to 0-1 range
train_dataset = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='/root/autodl-fs/ml_learning/Pytorch_Learning/datasets', 
                              train=False, 
                              transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

##########################
### MODEL
##########################



class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        #########################
        ### 1st residual block
        #########################
        # 28x28x1 => 28x28x4
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=4,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)
        self.conv_1_bn = torch.nn.BatchNorm2d(4)
                                    
        # 28x28x4 => 28x28x1
        self.conv_2 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=1,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)   
        self.conv_2_bn = torch.nn.BatchNorm2d(1)
        
        
        #########################
        ### 2nd residual block
        #########################
        # 28x28x1 => 28x28x4
        self.conv_3 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=4,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)
        self.conv_3_bn = torch.nn.BatchNorm2d(4)
                                    
        # 28x28x4 => 28x28x1
        self.conv_4 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=1,
                                      kernel_size=(3, 3),
                                      stride=(1, 1),
                                      padding=1)   
        self.conv_4_bn = torch.nn.BatchNorm2d(1)

        #########################
        ### Fully connected
        #########################        
        self.linear_1 = torch.nn.Linear(28*28*1, num_classes)

        
    def forward(self, x):
        
        #########################
        ### 1st residual block
        #########################
        shortcut = x
        
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = F.relu(out)

        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### 2nd residual block
        #########################
        
        shortcut = out
        
        out = self.conv_3(out)
        out = self.conv_3_bn(out)
        out = F.relu(out)

        out = self.conv_4(out)
        out = self.conv_4_bn(out)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### Fully connected
        #########################   
        logits = self.linear_1(out.view(-1, 28*28*1))
        probas = F.softmax(logits, dim=1)
        return logits, probas

    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)
model = model.to(device)
    
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


start_time = time.time()
for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
        
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model = model.eval() # eval mode to prevent upd. batchnorm params during inference
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

# Resenet with convolutional block for resizing
##########################
### MODEL
##########################



class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        #########################
        ### 1st residual block
        #########################
        # 28x28x1 => 14x14x4 
        self.conv_1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=4,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(4)
                                    
        # 14x14x4 => 14x14x8
        self.conv_2 = torch.nn.Conv2d(in_channels=4,
                                      out_channels=8,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(8)
        
        # 28x28x1 => 14x14x8
        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=1,
                                               out_channels=8,
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(8)
        
        #########################
        ### 2nd residual block
        #########################
        # 14x14x8 => 7x7x16 
        self.conv_3 = torch.nn.Conv2d(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_3_bn = torch.nn.BatchNorm2d(16)
                                    
        # 7x7x16 => 7x7x32
        self.conv_4 = torch.nn.Conv2d(in_channels=16,
                                      out_channels=32,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_4_bn = torch.nn.BatchNorm2d(32)
        
        # 14x14x8 => 7x7x32 
        self.conv_shortcut_2 = torch.nn.Conv2d(in_channels=8,
                                               out_channels=32,
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_2_bn = torch.nn.BatchNorm2d(32)

        #########################
        ### Fully connected
        #########################        
        self.linear_1 = torch.nn.Linear(7*7*32, num_classes)

        
    def forward(self, x):
        
        #########################
        ### 1st residual block
        #########################
        shortcut = x
        
        out = self.conv_1(x) # 28x28x1 => 14x14x4 
        out = self.conv_1_bn(out)
        out = F.relu(out)

        out = self.conv_2(out) # 14x14x4 => 714x14x8
        out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### 2nd residual block
        #########################
        
        shortcut = out
        
        out = self.conv_3(out) # 14x14x8 => 7x7x16 
        out = self.conv_3_bn(out)
        out = F.relu(out)

        out = self.conv_4(out) # 7x7x16 => 7x7x32
        out = self.conv_4_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_2(shortcut)
        shortcut = self.conv_shortcut_2_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)
        
        #########################
        ### Fully connected
        #########################   
        logits = self.linear_1(out.view(-1, 7*7*32))
        probas = F.softmax(logits, dim=1)
        return logits, probas

    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_loader), cost))

    model = model.eval() # eval mode to prevent upd. batchnorm params during inference
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))
        
print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))

## ResNet with convolutional blocks for resizing (using a helper class)

class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        
        super(ResidualBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=channels[0],
                                      out_channels=channels[1],
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=1)
        self.conv_1_bn = torch.nn.BatchNorm2d(channels[1])
                                    
        self.conv_2 = torch.nn.Conv2d(in_channels=channels[1],
                                      out_channels=channels[2],
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      padding=0)   
        self.conv_2_bn = torch.nn.BatchNorm2d(channels[2])

        self.conv_shortcut_1 = torch.nn.Conv2d(in_channels=channels[0],
                                               out_channels=channels[2],
                                               kernel_size=(1, 1),
                                               stride=(2, 2),
                                               padding=0)   
        self.conv_shortcut_1_bn = torch.nn.BatchNorm2d(channels[2])

    def forward(self, x):
        shortcut = x
        
        out = self.conv_1(x)
        out = self.conv_1_bn(out)
        out = F.relu(out)

        out = self.conv_2(out)
        out = self.conv_2_bn(out)
        
        # match up dimensions using a linear function (no relu)
        shortcut = self.conv_shortcut_1(shortcut)
        shortcut = self.conv_shortcut_1_bn(shortcut)
        
        out += shortcut
        out = F.relu(out)

        return out

##########################
### MODEL
##########################



class ConvNet(torch.nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.residual_block_1 = ResidualBlock(channels=[1, 4, 8])
        self.residual_block_2 = ResidualBlock(channels=[8, 16, 32])
    
        self.linear_1 = torch.nn.Linear(7*7*32, num_classes)

        
    def forward(self, x):

        out = self.residual_block_1.forward(x)
        out = self.residual_block_2.forward(out)
         
        logits = self.linear_1(out.view(-1, 7*7*32))
        probas = F.softmax(logits, dim=1)
        return logits, probas

    
torch.manual_seed(random_seed)
model = ConvNet(num_classes=num_classes)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):            
        features = features.to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100
    

for epoch in range(num_epochs):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        
        features = features.to(device)
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, num_epochs, batch_idx, 
                     len(train_dataset)//batch_size, cost))

    model = model.eval() # eval mode to prevent upd. batchnorm params during inference
    with torch.set_grad_enabled(False): # save memory during inference
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, num_epochs, 
              compute_accuracy(model, train_loader)))
print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))