  # @file    wgan.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/06 16:29:48
  # @version 1.0
  # @brief  使用wasserstein距离的生成对抗网络

# ​​移除判别器的Sigmoid激活​​，输出实数分数。
# ​​定义Wasserstein损失函数​​，直接优化分布间距离。
# ​​多次更新判别器并进行权重裁剪​​，确保Lipschitz约束。
# ​​标签设计​​：真实样本标签为-1，生成样本为1，适配损失计算。

import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 0
generator_learning_rate = 0.0005
discriminator_learning_rate = 0.0005
NUM_EPOCHS = 100
BATCH_SIZE = 128
LATENT_DIM = 50
IMG_SHAPE = (1, 28, 28)
IMG_SIZE = 1
for x in IMG_SHAPE:
    IMG_SIZE *= x

## WGAN-specific settings
num_iter_critic = 5 # 每训练一次生成器，更新判别器多次（如5次），确保判别器接近最优
weight_clip_value = 0.01 # 强制判别器的权重在有限范围内（如[-0.01, 0.01]），近似满足Lipschitz连续性条件



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
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

##########################
### MODEL
##########################


def wasserstein_loss(y_true, y_pred):
    """计算Wasserstein距离损失（WGAN的核心损失函数）
    
    Args:
        y_true: 真实标签张量（真实样本为-1，生成样本为1）
        y_pred: 判别器输出的未经过激活的分数（critic分数）,这里是分数不是概率
        
    Returns:
        标量损失值，表示分布间的近似Wasserstein距离
    
    数学原理：
    对于真实样本：最大化D(x)即最小化 -D(x) → y_true=-1时，损失=-D(x)
    对于生成样本：最小化D(G(z)) → y_true=1时，损失=D(G(z))
    因此总损失 = E[D(G(z))] - E[D(x)] ≈ Wasserstein距离
    """
    return torch.mean(y_true * y_pred)


class GAN(torch.nn.Module):

    def __init__(self):
        super(GAN, self).__init__()
        
        
        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.Linear(128, IMG_SIZE),
            nn.Tanh()
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(IMG_SIZE, 128),
            nn.LeakyReLU(inplace=True),
            #nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            #nn.Sigmoid() # WGAN should have linear activation
        )

            
    def generator_forward(self, z):
        img = self.generator(z)
        return img
    
    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred.view(-1)


torch.manual_seed(random_seed)

model = GAN()
model = model.to(device)

optim_gener = torch.optim.Adam(model.generator.parameters(), lr=generator_learning_rate)
optim_discr = torch.optim.Adam(model.discriminator.parameters(), lr=discriminator_learning_rate)

start_time = time.time()    

discr_costs = []
gener_costs = []
for epoch in range(NUM_EPOCHS):
    model = model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        
        
        features = (features - 0.5)*2.
        features = features.view(-1, IMG_SIZE).to(device) 
        targets = targets.to(device)

        # Regular GAN:
        # valid = torch.ones(targets.size(0)).float().to(device)
        # fake = torch.zeros(targets.size(0)).float().to(device)
        
        # WGAN:
        valid = -(torch.ones(targets.size(0)).float()).to(device)
        fake = torch.ones(targets.size(0)).float().to(device)
        

        ### FORWARD AND BACK PROP
        
        
        # --------------------------
        # Train Generator
        # --------------------------
        
        # Make new images
        z = torch.zeros((targets.size(0), LATENT_DIM)).uniform_(-1.0, 1.0).to(device)
        generated_features = model.generator_forward(z)
        
        # Loss for fooling the discriminator
        discr_pred = model.discriminator_forward(generated_features)
        
        
        # Regular GAN:
        # gener_loss = F.binary_cross_entropy_with_logits(discr_pred, valid)
        
        # WGAN:
        gener_loss = wasserstein_loss(valid, discr_pred) # 目标：最大化D(G(z))
        
        optim_gener.zero_grad()
        gener_loss.backward()
        optim_gener.step()
        
        # --------------------------
        # Train Discriminator
        # --------------------------        

        
        # WGAN: 5 loops for discriminator
        for _ in range(num_iter_critic):
        
            discr_pred_real = model.discriminator_forward(features.view(-1, IMG_SIZE))
            # Regular GAN:
            # real_loss = F.binary_cross_entropy_with_logits(discr_pred_real, valid)
            # WGAN:
            real_loss = wasserstein_loss(valid, discr_pred_real) # 目标：最大化D(x_real)

            discr_pred_fake = model.discriminator_forward(generated_features.detach())

            # Regular GAN:
            # fake_loss = F.binary_cross_entropy_with_logits(discr_pred_fake, fake)
            # WGAN:
            fake_loss = wasserstein_loss(fake, discr_pred_fake) # 目标：最小化D(G(z))

            # Regular GAN:
            discr_loss = (real_loss + fake_loss)  # 等价于 max(D(x_real) - D(G(z)))
            # WGAN:
            #discr_loss = -(real_loss - fake_loss)

            optim_discr.zero_grad()
            discr_loss.backward()
            optim_discr.step()        

            # WGAN:
            for p in model.discriminator.parameters():
                p.data.clamp_(-weight_clip_value, weight_clip_value)

        
        discr_costs.append(discr_loss.item())
        gener_costs.append(gener_loss.item())
        
        
        
        
        ### LOGGING
        if not batch_idx % 100:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Gen/Dis Loss: %.4f/%.4f' 
                   %(epoch+1, NUM_EPOCHS, batch_idx, 
                     len(train_loader), gener_loss, discr_loss))

    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    
print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

import matplotlib.pyplot as plt

ax1 = plt.subplot(1, 1, 1)
ax1.plot(range(len(gener_costs)), gener_costs, label='Generator loss')
ax1.plot(range(len(discr_costs)), discr_costs, label='Discriminator loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss')
ax1.legend()

###################
# Set scond x-axis
ax2 = ax1.twiny()
newlabel = list(range(NUM_EPOCHS+1))
iter_per_epoch = len(train_loader)
newpos = [e*iter_per_epoch for e in newlabel]

ax2.set_xticklabels(newlabel[::10])
ax2.set_xticks(newpos[::10])

ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 45))
ax2.set_xlabel('Epochs')
ax2.set_xlim(ax1.get_xlim())
###################

plt.show()

##########################
### VISUALIZATION
##########################


model.eval()
# Make new images
z = torch.zeros((5, LATENT_DIM)).uniform_(-1.0, 1.0).to(device)
generated_features = model.generator_forward(z)
imgs = generated_features.view(-1, 28, 28)

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 2.5))


for i, ax in enumerate(axes):
    axes[i].imshow(imgs[i].to(torch.device('cpu')).detach(), cmap='binary')


