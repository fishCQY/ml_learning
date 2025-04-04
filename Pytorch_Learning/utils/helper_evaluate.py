  # @file    helper_evaluate.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/02 17:07:16
  # @version 1.0
  # @brief 


import torch
import torch.distributed
import torch.distributed.rpc
import torch.nn.functional as F 
import numpy as np
from itertools import product

def compute_accuracy(model, data_loader, device):
    model.eval()  # 设置模型为评估模式（关闭Dropout/BatchNorm等训练模式特有的层）
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗
        correct_pred, num_examples = 0, 0  # 初始化正确预测数和总样本数
        
        # 遍历数据加载器中的每个批次
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)  # 将输入数据移动到指定设备（如GPU）
            targets = targets.to(device)    # 将标签移动到指定设备
            logits = model(features)        # 前向传播获取模型输出
            
            # 处理分布式训练中的远程引用（RRef）
            # RRef（Remote Reference）的作用
            # ​定义：RRef（远程引用）是PyTorch分布式RPC框架中的核心概念，用于在分布式系统中跨节点透明地访问远程对象。
            # ​场景：当模型部分组件（如某层）部署在远程节点时，前向传播的输出可能被封装为RRef，而非本地张量。
            # ​操作：通过RRef可以在本地代码中透明地操作远程数据，无需显式处理网络通信。
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()  # 获取本地值返回RRef所指向的本地对象（仅在当前进程拥有该对象时有效）
            
            # 获取预测标签（最大logit值对应的索引）
            _, predicted_labels = torch.max(logits, 1)  # 按第1维度（类别维度）取最大值
            
            # 累加统计量
            num_examples += targets.size(0)  # 当前批次的样本数
            correct_pred += (predicted_labels == targets).sum()  # 统计正确预测数
        
        # 计算并返回准确率（百分比形式）
        return correct_pred.float() / num_examples * 100

def compute_epoch_loss(model, data_loader, device):
    model.eval()  # 评估模式
    curr_loss, num_examples = 0.0, 0  # 初始化总损失和样本数
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)  # 前向传播
            
            # # 处理分布式训练中的RRef
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            
            # 计算交叉熵损失（使用sum模式累加批次损失）
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)  # 累加样本数
            curr_loss += loss  # 累加总损失
        
        # 返回平均损失（总损失 / 总样本数）
        curr_loss = curr_loss / num_examples
        return curr_loss
    
def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []  # 存储所有真实标签和预测标签
    
    with torch.no_grad():
        # 遍历数据集，收集预测结果
        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            
            # 获取预测标签
            _, predicted_labels = torch.max(logits, 1)
            
            # 将结果移回CPU并转为Python列表
            all_targets.extend(targets.to('cpu'))     # 真实标签列表
            all_predictions.extend(predicted_labels.to('cpu'))  # 预测标签列表
    
    # 转换为NumPy数组以便处理
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 获取所有唯一的类别标签（合并真实和预测标签）合并真实和预测标签后去重，得到所有可能的类别（如 [0, 1, 2]）
    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    
    # 处理单一类别标签的特殊情况,处理标签只有一种的情况
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])  # 添加0类避免混淆矩阵为1x1
        else:
            class_labels = np.array([class_labels[0], 1])   # 添加1类
    
    # 生成所有可能的（真实标签，预测标签）组合
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))  # 将标签对转换为元组列表，列表中的每个元素是一个元组
    
    # 统计每个组合的出现次数
    # product(class_labels, repeat=2)：生成类别标签的笛卡尔积。
    # 若 class_labels = [0, 1]，则生成 [(0,0), (0,1), (1,0), (1,1)]。
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))  # 计算每个组合的频次，统计元组 combi 在列表 z 中出现的次数，O(n)，其中 n 是样本数（对大数据集效率极低）
    
    # 重塑为混淆矩阵格式 [n_classes, n_classes]
    # np.asarray(lst)**：将列表 lst 转换为NumPy数组。[:, None]**：
    # ​功能：增加一个维度，将形状从 (4,) 变为 (4, 1)。
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat