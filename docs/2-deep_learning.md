# 深度学习 {#dl}

这一章节主要介绍深度学习领域各个神经网络的原理及实现方法——参考《动手学深度学习：PyTorch版》。

> 鉴于神经网络的可解释性完全不如其他机器学习方法，故一开始对其并不感冒，但没办法，它太powerful了，得学。

参考资料：

[菜鸟教程](https://www.runoob.com/pytorch/pytorch-tutorial.html)

[PyTorch文档](https://docs.pytorch.org/docs/stable/index.html)

## 预备知识 {#dl_1}

### 数据操作 {#dl_1_1}


``` default
import torch
```

1. 形状

   - `shape`
      
      输出形状
   
   - `reshape()`
      
      更改形状，`-1`表示自适应。默认按行排列，必要时可先改变形状，后转置得到按列排列的结果
      
   - `numel()`
   
      元素数量有多少
   
2. 拼接

   - `torch.cat()`
      
      在已有维度上拼接
   
   - `torch.stack()` `torch.vstack()` `torch.stack()`
   
      在新的维度上堆叠

3. 逐元素操作

   - 传统运算符`+ - * / **`
   
   - `sum()`
   
      若为空则对所有元素求和；`dim`指定轴方向，`0,1,2...`表示维度从外到内
      

``` default
x = torch.arange(24).reshape(2,3,-1)
x

"""
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
"""

x.sum(dim=0)

"""
tensor([[12, 14, 16, 18],
        [20, 22, 24, 26],
        [28, 30, 32, 34]])
"""

x.sum(dim=1)

"""
tensor([[12, 15, 18, 21],
        [48, 51, 54, 57]])
"""

x.sum(dim=2)

"""
tensor([[ 6, 22, 38],
        [54, 70, 86]])
"""
```
   
   - 广播机制
   
      维度对齐，从尾部(最右边)开始逐维度比较，形状不足的张量在左边补1个维度
      
      维度大小为1的轴自动"复制"以匹配较大尺寸


``` default
A = torch.tensor([[1, 2, 3], 
                  [4, 5, 6]])  # (2, 3)
b = torch.tensor([10, 20, 30])  # (3,) 

result = A * b

"""
tensor([[ 10,  40,  90],
        [ 40, 100, 180]])
"""

# 广播过程：b → (1,3) → (2,3)
# 由于张量b维度为1，A的维度为2，相当于张量b先复制了一行，再与A逐元素相乘
```

4. 节省内存

   对于形如`X=X+Y`的操作，事实上赋值前的X和赋值后的X占用了两个地方的内存（即使变量名相同），建议改为`X[:]=X+Y`，这样前后X的内存地址就一致了

### 自动微分 {#dl_1_2}

深度学习框架能够自动计算导数：先将梯度附加到想要计算偏导数的变量上，然后对目标值进行反向传播`backward()`，访问得到的梯度。


``` default
x = torch.arange(4)
x.requires_grad_(True)   # 等价于x = torch.arange(4, requires_grad=True)
x.grad     # 默认值为None
y = 2 * torch.dot(x,x)
y.backward()
x.grad
x.grad.zero_()     # 变量会累积梯度，在必要时需要清空
```

对于复合函数`y=f(x), z=g(y,x)`，有时想控制y直接计算z关于x的梯度，则需要将y剥离出来。


``` default
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u
```

5. 类型转换

|类型|方法|备注|
|:---:|:---:|:---:|
|数组->张量|torch.from_numpy()|共享内存|
|数组->张量|torch.tensor()|仅复制|
|张量->数组|.numpy()|共享内存|
|张量->数组|.clone().numpy()|仅复制|
|数据框->数组|.values|内存高|
|数据框->数组|.to_numpy(copy=False)|内存低|
|数组->数据框|pd.DataFrame()|-|

> 默认张量在CPU上，若在GPU上，则先将其移到CPU上，再`.cpu().numpy()`
> 
> 存有梯度的张量不能直接转为数组，应`.detach().numpy()`或`.detach().cpu().numpy()`

### 加载数据集 {#dl_1_3}


``` default
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
```

1. 创建数据集对象

`TensorDataset(*tensor)`用于将内存中的多个张量包装为一个数据集对象。
   

``` default
x = torch.arange(12).reshape(3,4)   # 特征
y = torch.tensor([0, 1, 0])         # 标签
dataset = TensorDataset(x,y)        # 数据集对象
```

也可根据抽象类`Dataset`自定义数据集对象，切记一定要重写`__len__()`和`__getitem__()`。


``` default
# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        """
        初始化数据集，X_data 和 Y_data 是两个列表或数组
        X_data: 输入特征
        Y_data: 目标标签
        """
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        """返回数据集的大小"""
        return len(self.X_data)

    def __getitem__(self, idx):
        """返回指定索引的数据"""
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)  # 转换为 Tensor
        y = torch.tensor(self.Y_data[idx], dtype=torch.float32)
        return x, y
```
   
2. 加载数据集

`DataLoader()`用于从数据集中加载数据，并支持打乱、划分批次等操作。
   

``` default
loader = DataLoader(
    dataset,            # 数据集对象
    batch_size=128,     # 每个批次的样本数
    shuffle=True,       # 是否打乱数据顺序
    sampler=None,       # 抽样策略
    num_workers=4,      # 用于数据加载的进程数
    pin_memory=True,    # 是否使用固定内存(CUDA)
    drop_last=False     # 是否丢弃最后的不完整批次
)
```

3. 划分数据集

`random_split()`用于将一整个数据集分割为几个不重合的子集。


``` default
# generator用于精细化控制每个生成器的种子
# 也可设置全局随机数种子
# torch.manual_seed(132)

# 设置随机种子保证每次分割相同
generator = torch.Generator().manual_seed(42)  # 固定随机种子

train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15],           # 子集大小
    generator=generator          # 传递随机数生成器
)
```

## 线性神经网络 {#dl_2}

### 线性回归 {#dl_2_1}

`nn.Linear`是线性层，对输入数据进行仿射变换$y=XW^T+b$，其中$W$是权重矩阵，$b$是偏置。

对于简单的线性回归模型，故只需一层线性层即可。


``` default
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(123)   # 设置全局随机数种子

X = torch.randn(100,2)   # 标准正态抽样
beta = torch.rand(2)     # 均匀分布[0,1)抽样
intercept = torch.rand(1)
y = intercept + torch.matmul(X, beta) + torch.randn(100)*0.1
y = y.reshape(-1,1)

dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 搭建网络结构
linear_model = nn.Sequential(
    nn.Linear(2,1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.05)   # lr为学习率

for epoch in range(100):       # 迭代50次
    linear_model.train()       # 训练模式
    for batch_X, batch_y in dataloader:
        y_pred = linear_model(batch_X)     # 前向传播，计算预测值
        loss = criterion(y_pred, batch_y)  # 计算损失
        optimizer.zero_grad()  # 清零梯度
        loss.backward()        # 反向传播，计算梯度
        optimizer.step()       # 更新模型参数
    linear_model.eval()        # 评估模式
    with torch.no_grad():
        epoch_loss = criterion(linear_model(X),y)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch_{epoch + 1}: {epoch_loss.item():.6f}')

print(linear_model)                  # 查看网络结构
print(linear_model[0].weight.data)   # 查看第0层的权重
print(linear_model[0].bias.data)     # 查看第0层的偏置
```

说明：

1. `DataLoader()`将数据分批次，变为可迭代的对象，每次返回一个批次的数据。也可结合`enumerate()`适用。

2. `nn.Sequential()`**按顺序**组织多个神经网络层，相较自定义类无需定义`forward`方法，适用于简单场合的模型构建。

> 除了`nn.Sequential()`，还可以通过继承`nn.Module`来自定义模型架构，从而创建更复杂的模型

3. `nn.Linear()`是线性全连接层，用来实现仿射变换$y=XW^T+b$。

4. `nn.MSELoss()`定义了损失函数的类型，计算两个形状相同的张量之间的MSE

5. `optim.SGD()`定义了优化算法为随机梯度下降法，`linear_model.parameters()`用于传递需要优化的参数。

6. `.train()`、`.eval()`分别代表模型的训练模式和评估模式，不同模式下会影响部分层(如Dropout层)的行为，一般在训练时开始`.train()`，在计算相关指标时`.eval()`。特别的，在评估时还可配合`torch.no_grad()`来禁用梯度计算，节省内存和计算资源，从而提高计算速度。

### 二分类问题 {#dl_2_2}

对于二分类问题，输出层的维度应该为1，并且输出层的激活函数为Sigmoid函数，损失函数为`nn.BCELoss()`等适用于二分类场合的函数。

> `nn.BCELoss()`即$-y_i \log \hat y_i - (1-y_i)\log (1-\hat y_i)$


``` default
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 设置随机数种子
torch.manual_seed(123)
np.random.seed(321)

def gen_data(n_samples=2000, n_features=10, n_classes=2):

    # 生成复杂的分类型数据（有重叠）
    X, y = make_classification(
        n_samples=n_samples,    # 样本数
        n_features=n_features,  # 特征数
        n_informative=8,        # 有信息量的特征数量
        n_redundant=2,          # 冗余特征数量
        n_repeated=0,           # 重复特征数量
        n_classes=n_classes,    # 类别数
        flip_y=0.15,            # 15%的噪声
        class_sep=0.8           # 类间分离程度
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 二分类需要二维标签
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train, X_test, y_train, y_test

# 生成数据
X_train, X_test, y_train, y_test = gen_data()

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Linear(10,64),
    nn.ReLU(),
    nn.BatchNorm1d(64),  # 对该批次数据进行标准化操作，并进行缩放和平移，可在一定程度上缓解“内部协变量偏移”情况
    nn.Dropout(0.2),     # 以一定概率丢弃某些神经元，从而缓解过拟合现象
    
    nn.Linear(64,32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.2),
    
    nn.Linear(32,1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# 将模型移到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 存储训练指标
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(100):
    # 训练模式
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_x, batch_y in train_loader:
        # 移动数据到设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_x)
        
        # 计算损失
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计训练情况
        running_loss += loss.item()
        
        # 计算准确率
        predictions = (outputs > 0.5).float()
        correct_train += (predictions == batch_y).sum().item()
        total_train += batch_y.size(0)
    
    # 计算本轮训练的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_train / total_train if total_train > 0 else 0
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # 验证评估
    model.eval()
    with torch.no_grad():
        # 移动测试数据到设备
        test_x, test_y = X_test.to(device), y_test.to(device)
        # 前向传播
        outputs = model(test_x)
        # 计算损失
        val_loss = criterion(outputs, test_y).item()
        # 计算预测结果
        predictions = (outputs > 0.5).float()
        # 计算准确率
        correct_val = (predictions == test_y).sum().item()
        total_val = test_y.size(0)
        val_acc = correct_val / total_val
        # 存储验证结果
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    # 每10个epoch打印一次进度
    if (epoch + 1) % 10 == 0:
        print("="*10,
              f"Epoch_{epoch+1}",
              "="*10,
              f"\nTrain Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}\n",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

说明：

1. 在将数据存储为张量时就要统一特征和标签的数据类型为浮点数，否则后续特征为浮点数，标签为整数会报错。同时，标签的形状也应变为二维的。

2. `nn.BatchNorm1d()`对该批次数据进行标准化操作，并进行缩放和平移，可在一定程度上缓解“内部协变量偏移”情况

3. `nn.Dropout()`在训练`model.train()`时会以一定概率丢弃某些神经元，从而缓解过拟合现象，是一种正则化技术。

4. 无论如何，模型和数据都要在同一设备上。张量数据必须重新赋值`batch_x = batch_x.to(device)`，而模型则可以直接`model.to(device)`

### 多分类问题 {#dl_2_3}

对于多分类问题，输出层维度为类别数，无需添加softmax函数，因为在交叉熵损失函数`nn.CrossEntropyLoss()`中自带了softmax运算。


``` default
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 设置随机数种子
torch.manual_seed(123)
np.random.seed(321)

def gen_data(n_samples=2000, n_features=10, n_classes=5):

    # 生成复杂的分类型数据（有重叠）
    X, y = make_classification(
        n_samples=n_samples,    # 样本数
        n_features=n_features,  # 特征数
        n_informative=8,        # 有信息量的特征数量
        n_redundant=2,          # 冗余特征数量
        n_repeated=0,           # 重复特征数量
        n_classes=n_classes,    # 类别数
        flip_y=0.15,            # 15%的噪声
        class_sep=0.8           # 类间分离程度
    )
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)     # 交叉熵损失函数要求标签为整数型
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    return X_train, X_test, y_train, y_test

# 生成数据，多分类任务
X_train, X_test, y_train, y_test = gen_data()

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(
    nn.Linear(10,128),
    nn.ReLU(),
    nn.BatchNorm1d(128),  # 对该批次数据进行标准化操作，并进行缩放和平移，可在一定程度上缓解“内部协变量偏移”情况
    nn.Dropout(0.2),     # 以一定概率丢弃某些神经元，从而缓解过拟合现象
    
    nn.Linear(128,64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Dropout(0.2),

    nn.Linear(64,32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.2),
    
    nn.Linear(32,5)      # 输出维度为类别数
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 存储训练指标
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(100):
    # 训练模式
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_x, batch_y in train_loader:
        # 移动数据到设备
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_x)
        
        # 计算损失
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计训练情况
        running_loss += loss.item()
        
        # 计算准确率
        predictions = torch.argmax(outputs, dim = 1)   # logits值最大的为预测类别
        correct_train += (predictions == batch_y).sum().item()
        total_train += batch_y.size(0)
    
    # 计算本轮训练的平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_train / total_train if total_train > 0 else 0
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # 验证评估
    model.eval()
    with torch.no_grad():
        # 移动测试数据到设备
        test_x, test_y = X_test.to(device), y_test.to(device)
        # 前向传播
        outputs = model(test_x)
        # 计算损失
        val_loss = criterion(outputs, test_y).item()
        # 计算预测结果
        predictions = torch.argmax(outputs, dim = 1)
        # 计算准确率
        correct_val = (predictions == test_y).sum().item()
        total_val = test_y.size(0)
        val_acc = correct_val / total_val
        # 存储验证结果
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
    # 每10个epoch打印一次进度
    if (epoch + 1) % 10 == 0:
        print("="*10,
              f"Epoch_{epoch+1}",
              "="*10,
              f"\nTrain Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}\n",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
```

说明：

1. `nn.CrossEntropyLoss()`接收预测值logits（原值）与标签。其中logits为神经网络的原始输出，无需在输出时添加Softmax激活函数，`nn.CrossEntropyLoss()`的内部会自动进行Softmax计算，避免重复计算。同时，标签要求为整数型且维度为1，不需要独热编码。

2. 如果需要输出概率，可以在输出logits后手动计算`torch.softmax(outputs, dim=1)`。

3. 如果要输出预测类别，可以`torch.argmax(outputs, dim=1)`。







