  # @file    mobilenet-v2-cifar10.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 21:30:03
  # @version 1.0
  # @brief 

import torch
import torchvision 
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 或 ':16:8'

# 获取当前脚本的绝对路径（示例值）
script_path = os.path.abspath(__file__)
print("[1] 当前脚本路径:", script_path)
# 计算正确的项目根目录（Pytorch_Learning）
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
# 分解步骤：
# 1. os.path.dirname(script_path) → D:\ml_learning\Pytorch_Learning\CNN\AlexNet
# 2. 再调用一次 → D:\ml_learning\Pytorch_Learning\CNN
# 3. 再调用一次 → D:\ml_learning\Pytorch_Learning

# 构建工具目录路径
utils_dir = os.path.join(project_root, "utils")
utils_dir = os.path.normpath(utils_dir)
print("计算后的工具目录:", utils_dir)  # 输出: D:\ml_learning\Pytorch_Learning\utils

# 验证路径是否存在
if not os.path.exists(utils_dir):
    raise FileNotFoundError(f"❌ 目录不存在: {utils_dir}")
print("✅ 目录存在性验证通过")

# 添加到 sys.path
sys.path.insert(0, utils_dir)

from helper_utils import set_all_seeds, set_deterministic
from helper_evaluate import compute_confusion_matrix, compute_accuracy
from helper_train import train_classifier_simple_v2
from helper_plotting import plot_training_loss, plot_accuracy, show_examples, plot_confusion_matrix
from helper_data import get_dataloaders_cifar10, UnNormalize

RANDOM_SEED = 123
BATCH_SIZE = 128
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_all_seeds(RANDOM_SEED)
#set_deterministic()

##########################
### CIFAR-10 DATASET
##########################

### Note: Network trains about 2-3x faster if you don't
# resize (keeping the orig. 32x32 res.)
# Test acc. I got via the 32x32 was lower though; ~77%

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),
    torchvision.transforms.RandomCrop((64, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((70, 70)),        
    torchvision.transforms.CenterCrop((64, 64)),            
    torchvision.transforms.ToTensor(),                
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
    batch_size=BATCH_SIZE,
    validation_fraction=0.1,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    num_workers=2)

# Checking the dataset
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    print('Class labels of 10 examples:', labels[:10])
    break

plt.figure(figsize=(8, 8))          # 创建8x8英寸的画布
plt.axis("off")                     # 关闭坐标轴显示
plt.title("Training Images")        # 设置图像标题

# 核心可视化流程：形成图像网格
plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            images[:64],    # 取前64张图像
            padding=2,      # 图像间留2像素间隔
            normalize=True  # 将像素值归一化到[0,1]范围
        ),
        (1, 2, 0)  # 调整维度顺序：从CxHxW转换为HxWxC
    )
)

##########################
### MODEL
##########################
# torch.hub.load()：

# PyTorch提供的模型加载接口
# 可以从官方仓库或社区直接加载预定义模型
# 'pytorch/vision:v0.9.0'：

# 指定模型来源：PyTorch官方vision库
# v0.9.0表示使用该特定版本的定义
# 'mobilenet_v2'：

# 要加载的模型名称
# MobileNetV2是轻量级CNN模型，适合移动端/嵌入式设备
# pretrained=False：

# 不加载预训练权重
# 模型参数会随机初始化
# 适用于从零开始训练的场景
# 典型使用场景：

# 快速获取标准模型架构
# 避免手动实现复杂网络结构
# 方便进行迁移学习（可改为pretrained=True加载预训练权重）
# 注意事项：

# 需要联网下载模型定义（首次使用）
# 版本号影响模型具体实现细节
# 输入尺寸需符合模型要求（通常224x224 RGB）

model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2',
                       pretrained=False)

# model.classifier[-1]：

# MobileNetV2模型的分类器部分是一个序列结构
# [-1]表示取分类器部分的最后一层
# 原始MobileNetV2最后一层是为ImageNet设计的1000类分类层
# in_features=1280：

# 保持与原始模型相同的输入特征维度(1280维)
# 这是MobileNetV2最后一个卷积层输出的通道数
# 这种修改方式是PyTorch中迁移学习的常见做法，通常配合以下步骤：
# 新层的权重会随机初始化，需要重新训练
# 加载预训练模型
# 冻结前面层的参数（可选）
# 修改最后一层适应新任务
# 只训练最后一层或全部层（取决于需求）
model.classifier[-1] = torch.nn.Linear(in_features=1280, # as in original
                                       out_features=10) # number of class labels in Cifar-10)

model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

minibatch_loss_list, train_acc_list, valid_acc_list = train_classifier_simple_v2(
    model=model,
    num_epochs=NUM_EPOCHS,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    best_model_save_path='mobilenet-v2-best-1.pt',
    device=DEVICE,
    scheduler_on='valid_acc',
    logging_interval=100)


plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=NUM_EPOCHS,
                   iter_per_epoch=len(train_loader),
                   results_dir=None,
                   averaging_iterations=200)
plt.show()

plot_accuracy(train_acc_list=train_acc_list,
              valid_acc_list=valid_acc_list,
              results_dir=None)
plt.ylim([60, 100])
plt.show()

model.load_state_dict(torch.load('mobilenet-v2-best-1.pt'))
model.eval()
test_acc = compute_accuracy(model, test_loader, device=DEVICE)
print(f'Test accuracy: {test_acc:.2f}%')

model.cpu()
unnormalizer = UnNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
class_dict = {0: 'airplane',
              1: 'automobile',
              2: 'bird',
              3: 'cat',
              4: 'deer',
              5: 'dog',
              6: 'frog',
              7: 'horse',
              8: 'ship',
              9: 'truck'}

show_examples(model=model, data_loader=test_loader, unnormalizer=unnormalizer, class_dict=class_dict)

mat = compute_confusion_matrix(model=model, data_loader=test_loader, device=torch.device('cpu'))
plot_confusion_matrix(mat, class_names=class_dict.values())
plt.show()
