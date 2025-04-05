  # @file    helper_plotting.py
  # @author  cqy 3049623863@qq.com
  # @date    2025/04/04 21:31:53
  # @version 1.0
  # @brief 

# imports from installed libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch,
                       results_dir=None, averaging_iterations=100):
    """绘制训练损失曲线（包含双x轴：迭代次数和epoch数）
    
    参数:
        minibatch_loss_list: 所有mini-batch的损失值列表
        num_epochs: 总训练轮数
        iter_per_epoch: 每个epoch的迭代次数
        results_dir: 结果保存路径（None不保存）
        averaging_iterations: 滑动平均的窗口大小
    """
    plt.figure()
    # 主坐标轴（迭代次数）
    ax1 = plt.subplot(1, 1, 1)
    # 绘制原始mini-batch损失曲线
    ax1.plot(range(len(minibatch_loss_list)), 
             minibatch_loss_list, label='Minibatch Loss')

    # 跳过前1000次迭代的显示波动（如果数据量足够）
    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([0, np.max(minibatch_loss_list[1000:])*1.5])
    
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    # 添加滑动平均曲线（平滑损失曲线）
    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # 添加第二个x轴（epoch数）
    ax2 = ax1.twiny()  # 创建共享y轴的第二个坐标轴
    newlabel = list(range(num_epochs+1))  # 生成epoch标签
    newpos = [e*iter_per_epoch for e in newlabel]  # 计算epoch对应的迭代位置
    
    # 设置刻度位置和标签（每10个epoch显示一次）
    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])
    
    # 调整第二个x轴的位置和样式
    ax2.xaxis.set_ticks_position('bottom')  # 刻度线显示在下方
    ax2.xaxis.set_label_position('bottom')   # 标签显示在下方
    ax2.spines['bottom'].set_position(('outward', 45))  # 下移第二个x轴45像素
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())  # 与主坐标轴范围对齐
    ###################

    plt.tight_layout()

    # 保存矢量图到指定路径
    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
        plt.savefig(image_path)

def plot_accuracy(train_acc_list, valid_acc_list, results_dir):
    """绘制训练和验证准确率曲线
    
    参数:
        train_acc_list: 各epoch的训练准确率列表
        valid_acc_list: 各epoch的验证准确率列表  
        results_dir: 结果保存路径（None不保存）
    """
    # 转换 CUDA 张量为 NumPy（如果列表中有残留 CUDA 张量）
    train_acc_list = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_acc_list]
    valid_acc_list = [acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in valid_acc_list]
    num_epochs = len(train_acc_list)  # 根据训练准确率列表长度确定epoch数

    # 绘制两条曲线：训练集（蓝色）和验证集（橙色）
    plt.plot(np.arange(1, num_epochs+1),  # x轴：1到num_epochs
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')

    # 坐标轴标签设置
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图间距

    # 保存矢量图到PDF文件
    if results_dir is not None:
        image_path = os.path.join(results_dir, 'plot_acc_training_validation.pdf')
        plt.savefig(image_path)


def show_examples(model, data_loader, unnormalizer=None, class_dict=None):
    """可视化模型预测样本（3x5网格）
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器（获取样本）
        unnormalizer: 数据反规范化处理器（还原原始图像）
        class_dict: 类别标签字典（id到名称的映射）
    """
    # 获取第一个batch的数据
    for batch_idx, (features, targets) in enumerate(data_loader):
        with torch.no_grad():  # 禁用梯度计算
            features = features  # 保持张量格式
            targets = targets    # 真实标签
            logits = model(features)  # 前向传播
            predictions = torch.argmax(logits, dim=1)  # 取概率最大类别
        break  # 只取第一个batch

    # 创建3行5列的子图布局
    fig, axes = plt.subplots(nrows=3, ncols=5, sharex=True, sharey=True)
    
    # 反规范化处理（还原标准化后的图像）
    if unnormalizer is not None:
        for idx in range(features.shape[0]):
            features[idx] = unnormalizer(features[idx])
    
    # 调整通道顺序为NHWC（Matplotlib显示需要）
    nhwc_img = np.transpose(features, axes=(0, 2, 3, 1))
    
    # 处理单通道图像（灰度图）
    if nhwc_img.shape[-1] == 1:
        nhw_img = np.squeeze(nhwc_img.numpy(), axis=3)  # 去除通道维度
        
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhw_img[idx], cmap='binary')  # 使用二值色图
            # 设置标题（预测 vs 真实）
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False  # 隐藏坐标轴

    # 处理三通道图像（RGB）
    else:
        for idx, ax in enumerate(axes.ravel()):
            ax.imshow(nhwc_img[idx])  # 显示原始RGB图像
            # 设置标题（预测 vs 真实）
            if class_dict is not None:
                ax.title.set_text(f'P: {class_dict[predictions[idx].item()]}\nT: {class_dict[targets[idx].item()]}')
            else:
                ax.title.set_text(f'P: {predictions[idx]} | T: {targets[idx]}')
            ax.axison = False  # 隐藏坐标轴
    
    plt.tight_layout()  # 自动调整子图间距
    plt.show()          # 显示图像

def plot_confusion_matrix(conf_mat,
                          hide_spines=False,
                          hide_ticks=False,
                          figsize=None,
                          cmap=None,
                          colorbar=False,
                          show_absolute=True,
                          show_normed=False,
                          class_names=None):
    """绘制混淆矩阵的可视化图表
    
    参数：
        conf_mat: 二维数组格式的混淆矩阵
        hide_spines: 是否隐藏坐标轴脊
        hide_ticks: 是否隐藏刻度线
        figsize: 图表尺寸（宽，高）
        cmap: 颜色映射（默认使用plt.cm.Blues）
        colorbar: 是否显示颜色条
        show_absolute: 显示绝对值
        show_normed: 显示归一化值
        class_names: 类别名称列表
    
    返回：
        matplotlib的fig和ax对象
    """
    
    # 参数校验逻辑
    if not (show_absolute or show_normed):
        raise AssertionError('Both show_absolute and show_normed are False')
    if class_names is not None and len(class_names) != len(conf_mat):
        raise AssertionError('len(class_names) should be equal to number of'
                             'classes in the dataset')

    # 计算归一化混淆矩阵（按行归一化）
    total_samples = conf_mat.sum(axis=1)[:, np.newaxis]
    normed_conf_mat = conf_mat.astype('float') / total_samples

    # 初始化图表
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)
    if cmap is None:
        cmap = plt.cm.Blues  # 默认使用蓝色系颜色映射

    # 设置默认图表尺寸
    if figsize is None:
        figsize = (len(conf_mat)*1.25, len(conf_mat)*1.25)

    # 绘制混淆矩阵热力图
    if show_normed:
        matshow = ax.matshow(normed_conf_mat, cmap=cmap)
    else:
        matshow = ax.matshow(conf_mat, cmap=cmap)

    # 添加颜色条
    if colorbar:
        fig.colorbar(matshow)

    # 在单元格中添加文本标签
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            cell_text = ""
            if show_absolute:
                num = conf_mat[i, j].astype(np.int64)  # 原始计数值
                cell_text += format(num, 'd')
                if show_normed:
                    cell_text += "\n" + '('            # 添加括号格式
                    cell_text += format(normed_conf_mat[i, j], '.2f') + ')'
            else:
                cell_text += format(normed_conf_mat[i, j], '.2f')
            # 根据背景色深浅自动调整文字颜色
            ax.text(x=j,
                    y=i,
                    s=cell_text,
                    va='center',
                    ha='center',
                    color="white" if normed_conf_mat[i, j] > 0.5 else "black")
    
    # 设置类别标签（如果提供）
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)  # x轴标签旋转90度
        plt.yticks(tick_marks, class_names)

    # 控制坐标轴显示
    if hide_spines:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    if hide_ticks:
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([])

    # 添加轴标签
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    return fig, ax