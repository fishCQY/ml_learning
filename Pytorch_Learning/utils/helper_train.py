import torch.distributed
from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss

import time 
import torch
import torch.nn.functional as F
print(f"PyTorch版本: {torch.__version__}")
print(f"RRef导入成功！")
print(torch.distributed.rpc)     # 检查模块是否存在

# 提供神经网络函数式接口，包含激活函数、损失函数、卷积操作等，无需实例化 nn.Module 类即可直接调用。
# ​常见用途：
# ​激活函数：F.relu, F.sigmoid, F.softmax。
# ​损失函数：F.cross_entropy, F.mse_loss。
# ​卷积操作：F.conv2d, F.max_pool2d。
#创建有序字典，保留键值对的插入顺序（Python 3.7+ 后普通 dict 也默认有序，但此模块仍用于兼容性）

from collections import OrderedDict
import json
# 处理 ​JSON 数据​（JavaScript Object Notation），用于数据序列化、配置文件读写和 API 交互。
# ​常用函数：
# json.loads()：解析 JSON 字符串 → Python 对象。
# json.dumps()：将 Python 对象 → JSON 字符串。
# json.load() / json.dump()：从文件读取/写入 JSON。
import subprocess
# 创建和管理子进程，执行外部命令或脚本，并与其输入/输出交互。
# ​常见函数：
# subprocess.run()：运行命令并等待完成。
# subprocess.Popen()：异步执行命令。
# subprocess.check_output()：捕获命令输出。
import sys
import xml.etree.ElementTree
# 解析和生成 ​XML 数据，适用于处理配置文件、标注数据（如 Pascal VOC 格式）。
# ​常用方法：
# ET.parse()：解析 XML 文件 → 元素树。
# element.find() / element.findall()：按标签名查找元素。
# ET.Element() / ET.SubElement()：创建 XML 节点。

def train_classifier_simple_v1(num_epochs, model, optimizer, device, 
                               train_loader, valid_loader=None, 
                               loss_fn=None, logging_interval=100, 
                               skip_epoch_stats=False):
    
    log_dict = {'train_loss_per_batch': [],
                'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': []}
    
    if loss_fn is None:
        loss_fn = F.cross_entropy

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # FORWARD AND BACK PROP
            logits = model(features)
            # windows 不支持rpc分布式训练
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())

            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))

        if not skip_epoch_stats:
            model.eval()

            with torch.set_grad_enabled(False):  # save memory during inference

                train_acc = compute_accuracy(model, train_loader, device)
                train_loss = compute_epoch_loss(model, train_loader, device)
                print('***Epoch: %03d/%03d | Train. Acc.: %.3f%% | Loss: %.3f' % (
                      epoch+1, num_epochs, train_acc, train_loss))
                log_dict['train_loss_per_epoch'].append(train_loss.item())
                log_dict['train_acc_per_epoch'].append(train_acc.item())

                if valid_loader is not None:
                    valid_acc = compute_accuracy(model, valid_loader, device)
                    valid_loss = compute_epoch_loss(model, valid_loader, device)
                    print('***Epoch: %03d/%03d | Valid. Acc.: %.3f%% | Loss: %.3f' % (
                          epoch+1, num_epochs, valid_acc, valid_loss))
                    log_dict['valid_loss_per_epoch'].append(valid_loss.item())
                    log_dict['valid_acc_per_epoch'].append(valid_acc.item())

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    return log_dict


def train_classifier_simple_v2(num_epochs, model, optimizer, device,
                               train_loader, valid_loader, test_loader,
                               logging_interval=50, best_model_save_path=None,
                               scheduler=None, skip_train_acc=False,
                               scheduler_on='valid_acc'):
    """增强版分类模型训练函数v2
    
    参数:
        num_epochs: 训练总轮数
        test_loader: 测试集数据加载器
        best_model_save_path: 最佳模型保存路径 (可选)
        scheduler: 学习率调度器 (可选)
        skip_train_acc: 是否跳过训练集准确率计算
        scheduler_on: 调度器触发依据 ('valid_acc'或'minibatch_loss')
    """
    
    # 初始化训练记录
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    best_valid_acc, best_epoch = -float('inf'), 0  # 最佳验证准确率跟踪

    for epoch in range(num_epochs):
        # 训练阶段 ----------------------------------------------------------
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            # 数据迁移和设备处理
            features = features.to(device)
            targets = targets.to(device)
            
            # 前向传播与损失计算
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            
            # 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失值
            minibatch_loss_list.append(loss.item())  # 改为记录数值而非张量

            # 定期日志输出
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
                      f' | Batch {batch_idx:04d}/{len(train_loader):04d}'
                      f' | Loss: {loss:.4f}')

        # 评估阶段 ----------------------------------------------------------
        model.eval()
        with torch.set_grad_enabled(False):
            # 训练集准确率计算（可跳过）
            if not skip_train_acc:
                train_acc = compute_accuracy(model, train_loader, device=device)
            else:
                train_acc = float('nan')  # 使用NaN表示跳过的值
                
            # 验证集准确率计算
            valid_acc = compute_accuracy(model, valid_loader, device=device).item()
            
            # 更新最佳模型
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)
            
            # 记录准确率
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            # 打印epoch统计信息
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc:.2f}% '
                  f'| Validation: {valid_acc:.2f}%'
                  f'| Best Validation (Ep. {best_epoch:03d}): {best_valid_acc:.2f}%')

        # 学习率调度 --------------------------------------------------------
        if scheduler is not None:
            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])  # 根据最新验证准确率调整
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])  # 根据最新损失调整
            else:
                raise ValueError('Invalid `scheduler_on` choice. Must be "valid_acc" or "minibatch_loss"')

        # 时间统计
        elapsed = (time.time() - start_time) / 60
        print(f'Time elapsed: {elapsed:.2f} min')

    # 最终测试 ------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print(f'Total Training Time: {elapsed:.2f} min')
    
    # 在测试集上评估最终模型
    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc:.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list