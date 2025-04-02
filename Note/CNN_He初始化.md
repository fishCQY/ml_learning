CNN-He初始化

**He初始化（Kaiming初始化）** 是何恺明（Kaiting He）等人提出的针对深度神经网络中 **ReLU激活函数** 的权重初始化方法。它通过调整权重的方差，缓解了深度网络训练中梯度消失或爆炸的问题，尤其适用于使用ReLU及其变体（如Leaky ReLU）的网络。

------

### **1. 核心原理**

#### **(1) 问题背景**

- 在深度神经网络中，**权重初始化不当** 会导致前向传播的输出方差逐层累积（指数级变化），进而引发梯度不稳定（过大或过小）。
- **ReLU激活函数** 对输入分布的敏感性更高，因为其负半区输出为0，会破坏权重的对称性。

#### **(2) 数学推导**

He初始化目标是 **保持各层输出的方差一致**，推导基于以下假设：

- 权重 *W* 初始化为均值为0、方差为 *σ*2 的正态分布。

- 输入 *x* 的均值为0，方差为 
  $$
  {\sigma_x^2}
  $$
  ，且与权重独立。

- 使用ReLU激活函数，其输出的期望值为 
  $$
  0.5{\sigma_x^2}
  $$
  。

对于前向传播，输出的方差应满足：
$$
Var(y) = Var(x) \cdot \sum_{i=1}^{n} Var(W_i) \cdot E\left[(\text{ReLU}(x))^2\right]
$$
通过推导得出权重的方差应为：

*σ*2=*n*in2

其中 *n*in 是输入神经元的数量（全连接层）或输入通道数 × 卷积核面积（卷积层）。

------

### **2. 公式形式**

- **正态分布初始化**：
  $$
  W \sim N(0, \sqrt\frac{2}{n_i})
  $$
  

- **均匀分布初始化**（等效近似）：
  $$
  W \sim N(-\sqrt\frac{6}{n_i}, \sqrt\frac{6}{n_i})
  $$
  

------

### **3. 代码实现**

在PyTorch中，使用 `kaiming_normal_` 或 `kaiming_uniform_` 实现：

python

复制

```python
import torch.nn as nn

# 对卷积层权重使用He初始化（默认针对ReLU）
nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')

# 对全连接层权重使用He初始化（Leaky ReLU需指定negative_slope）
nn.init.kaiming_normal_(fc_layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
```

#### **关键参数**：

- `mode='fan_in'`：根据输入神经元数 *n*in 调整方差（推荐默认）。
- `nonlinearity`：指定激活函数类型（如 `'relu'` 或 `'leaky_relu'`）。
- `a`：当使用Leaky ReLU时，指定负半区斜率（默认0）。

------

### **4. 优势对比**

|    初始化方法     |          适用场景           |       激活函数假设       |              特点              |
| :---------------: | :-------------------------: | :----------------------: | :----------------------------: |
| **Xavier/Glorot** |    全连接层（浅层网络）     | 线性或对称激活（如Tanh） |    保持输入和输出的方差一致    |
|  **He/Kaiming**   | 卷积层/ReLU网络（深层网络） |     ReLU族非对称激活     | 针对ReLU修正方差，避免梯度消失 |

#### **实验效果**：

- 在ResNet等深层网络中，He初始化比Xavier初始化收敛更快且准确率更高。
- 使用ReLU时，He初始化的训练损失下降更稳定（见下图）。

https://miro.medium.com/v2/resize:fit:720/format:webp/1*Q4fzvjWq5k-fOgeM-CM2TQ.png

------

### **5. 使用场景**

- 网络类型：
  - 卷积神经网络（CNN）
  - 残差网络（ResNet、DenseNet）
  - 使用ReLU/Leaky ReLU的深度网络（≥10层）
- 激活函数：
  - ReLU
  - Leaky ReLU（需指定 `nonlinearity='leaky_relu'` 和参数 `a`）
  - PReLU（Parametric ReLU）

------

### **6. 实际应用技巧**

1. **模式选择（`mode`）**：
   - `fan_in`：默认，适用于大部分情况（保持前向传播稳定）。
   - `fan_out`：适用于反向传播（如转置卷积）。
2. **偏置初始化**：
   - 通常初始化为0（`nn.init.zeros_(bias)`），避免破坏初始对称性。
3. **与批归一化（BatchNorm）配合**：
   - 若网络包含批归一化层，权重初始化的影响会减弱，但He初始化仍能加速初期收敛。

------

### **总结**

He初始化通过修正权重方差，解决了ReLU激活函数在深度网络中的梯度不稳定问题，是训练现代深度模型（尤其是CNN）的最佳实践之一。其数学简洁性和实际效果使其成为深度学习框架中的标准初始化方法。