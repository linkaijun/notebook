


# 生存分析 {#survival}

生存分析方法研究一个感兴趣的事件发生的时间。该事件可以是死亡、离婚、戒烟、设备故障等等。因此，“生存”二字不应与狭义上的“死亡”绑定，应当将“生存”视作一种“持续”，“死亡”则对应“事件的发生”。

## 基本概念 {#survival_1}

### 生存数据 {#survival_1_1}

在随访期（观察期）内，我们关心研究对象生存了多长时间，感兴趣的事件是否有发生。因此，每一个研究对象都可以由**生存时间、生存结局**这两个指标去描述。

在实践中，我们不可能有无限的时间去持续观察样本，因此会设置一段观察期。**生存时间**指的是样本自被观察起直到目标事件发生或者观察期止、中途丢失的持续时间，无论该样本是一开始就在还是中途加入。

若样本在观察期内确实发生了我们感兴趣的事件，那么我们可以记录该样本的**生存时间**及**生存结局**，不妨将其**生存结局**记为**1**(failure)。而对于那些在观察期内目标事件尚未发生或中途丢失的样本，我们同样可以记录其**生存时间**，并将其**生存结局**记为**0**(censoring)，称之为**删失**。

下面给出示例。

<div class="figure" style="text-align: center">
<img src="06-survival_analysis_files/figure-html/survival-p1-1.png" alt="生存数据示例" width="80%" />
<p class="caption">(\#fig:survival-p1)生存数据示例</p>
</div>

图中<span style='color:red'>红点</span>代表目标事件的发生，<span style='color:blue'>蓝点</span>代表样本丢失。由图可知第一个和第三个观测对象的**生存结局**是**1**，其余对象的**生存结局**是**0**。


| id | 生存时间 | 生存结局 |
|:--:|:--------:|:--------:|
| 1  |   6.0    |    1     |
| 2  |   8.0    |    0     |
| 3  |   1.5    |    1     |
| 4  |   3.0    |    0     |
| 5  |   2.5    |    0     |

### 删失 {#survival_1_2}

当我们无法准确获取研究对象自被观察起**至目标事件发生**的生存时间，便称这样的数据为**删失数据**，对应的生存时间为**不完全生存时间或截尾值**。

**删失**的原因有很多：

- 观察期结束了目标事件都还没有发生
- 观察对象失联，中途丢失
- 观察对象终止于其他事件
- ...
   
   例如我们对“肺癌”感兴趣，但患者还患有其他疾病，可能因其他疾病而死亡。
   
**删失**的类型可分为如下三种：

1. 右删失

   真正的生存时间大于或等于观测到的生存时间。
   
   例如研究对象直至观测期满都未发生目标事件，或者研究对象中途退出研究，我们不知道目标事件会在之后的什么时候发生，但至少比我们观测到的生存时间要长。
   
2. 左删失

   真正的生存时间小于或等于观测到的生存时间。
   
   例如想测试某个零件的使用寿命，但该零件内部已经有裂痕存在，因此实际观测到的使用寿命一定比新零件的使用寿命要短。
   
3. 区间删失

   只知道目标事件是在某个时间段内发生的，但不知道具体时间。
   
   例如核酸检验，第一次阴性，第二次阳性，那大概就是在这个时间段内被感染了，但不知道什么时候被感染。

### 函数 {#survival_1_3}

1. 生存函数

   记$T$为代表生存时间的随机变量，则其密度函数$f(t)$为
   
   $$
   f(t)=\lim_{\Delta t \rightarrow 0} \frac{P(t \leq T \leq t+\Delta t)}{\Delta t} (\#eq:eq1)
   $$
   
   记其累积分布函数为
   
   $$
   F(t)=P(T \leq t) (\#eq:eq2)
   $$
   
   定义**生存函数**为
   
   $$
   S(t)=P(T \gt t)=1-F(t) (\#eq:eq3)
   $$
   
   > $T \gt t$具有“生存”的意味
   
2. 危险函数

   $$
   \begin{aligned}
   h(t)&=\lim_{\Delta t \rightarrow 0}\frac{P(t \leq T \leq t+\Delta t \mid T \geq t)}{\Delta t} \\
   &= \lim_{\Delta t \rightarrow 0}\frac{F(t+\Delta t)-F(t)}{(1-F(t))\Delta t} \\
   &= \frac{f(t)}{S(t)} \\
   &= \frac{d}{dt}(-\ln S(t))
   \end{aligned} (\#eq:eq4)
   $$
   
   由定义可知，危险函数代表了即时事件发生率。
   
   故累积危险函数为
   
   $$
   H(t)=\int_0^t h(u)du=-\ln S(t)  (\#eq:eq5)
   $$

> 生存函数和危险函数、累积危险函数可以互相推导得到

----------

参考资料

1. https://zhuanlan.zhihu.com/p/497968260

## 论文复现 {#survival_2}

本节内容是对Simon等人[@survival_1]论文的复现。

观测对象的数据结构为$(y_i,x_i,\delta_i)$，分别表示生存时间、自变量向量、生存结局。其中$\delta_i$取值为1: failure time或0: censoring time。将$\delta=1$的failure time进行排序得到$t_1 \leq t_2 \leq \cdots \leq t_m$。

> 在有结点情况时才有可能取到等号

事实上，这篇论文有些地方有小错误，因此下面给出自己的推导过程。

### 推导 {#survival_2_1}

**1. 无结点情况**

$$
L(\beta)=\prod_{i=1}^m \frac{e^{x_{j(i)}^T\beta}}{\sum_{j \in R_i }e^{x_j^T\beta}}
$$

其中$R_i$表示$t_i \leq y_j$的$j$的集合，$j(i)$表示第$i$个failure time对应的观测对象$j$。

$$
\frac{2}{n}l(\beta)=\frac{2}{n}\begin{bmatrix}\sum_{i=1}^mx_{j(i)}^T\beta-\sum_{i=1}^m\log(\sum_{j \in R_i}e^{x_j^T\beta})\end{bmatrix}
$$

> 原文缺了log前面的求和号
> 
> 这里的1/n相当于是权重，2是为了消掉泰勒展开中的1/2

令$\eta=X\beta$，对对数似然函数进行二阶泰勒展开

$$
\begin{aligned}
l(\beta)&\approx l(\tilde \beta)+(\beta-\tilde \beta)^T\dot l(\tilde \beta)+(\beta-\tilde \beta)^T\ddot l(\tilde \beta)(\beta-\tilde \beta)/2 \\
&=l(\tilde \beta)+(X\beta-\tilde \eta)^Tl'(\tilde \eta)+(X\beta-\tilde \eta)^Tl''(\tilde \eta)(X\beta-\tilde \eta)/2
\end{aligned}
$$

其中

$$
\frac{\partial l(\eta)}{\partial \beta}=(X^T)_{p \times n}l'(\eta)_{n \times 1}=\dot l(\beta)_{p \times1}
$$

> $(X_{\cdot k})^Tl'(\eta)=\dot l(\beta)_k$ 

整理可得

$$
l(\beta)\approx\frac{1}{2}(z(\tilde \eta)-X\beta)^Tl''(\tilde \eta)(z(\tilde \eta)-X\beta)+C(\tilde \eta,\tilde \beta)\\
z(\tilde \eta)=\tilde \eta-l''(\tilde \eta)^{-1}l'(\tilde \eta)
$$

> 经检验，原文该表达式没错

则

$$
\frac{2}{n}l(\beta)\approx\frac{1}{n}(z(\tilde \eta)-X\beta)^Tl''(\tilde \eta)(z(\tilde \eta)-X\beta)+\frac{2}{n}C(\tilde \eta,\tilde \beta)
$$

原文指出，为了计算方便，仅取黑塞矩阵的对角线元素而无视其他元素，其余元素对最终结果的影响也较小。

故目标函数为

$$
M(\beta)=-\frac{1}{n}\sum_{i=1}^nw(\tilde \eta)_i(z(\tilde \eta)_i-x_i^T\beta)^2+\lambda(\alpha\sum_{k=1}^p|\beta_k|+\frac{1}{2}(1-\alpha)\sum_{k=1}^p\beta_k^2)
$$

其中$w(\tilde \eta)_i$是$l''(\tilde \eta)$的第$i$个对角线元素。

> 由于是最小化，对数似然函数得添负号，但原文却少了负号

下面推导$w(\eta)_i$及$z(\eta)_i$的具体表达式

$$
l(\eta)=\sum_{i=1}^m (\eta_{j(i)}-\log(\sum_{j \in R_i}e^{\eta_j}))=\sum_{i=1}^m \eta_{j(i)}-\sum_{i=1}^m \log(\sum_{j \in R_i}e^{\eta_j})
$$

对$\eta_k$求偏导

$$
l'(\eta)_k=\delta_k-\sum_{i=1}^m \frac{e^{\eta_k}\textrm{I}_{\{y_k \geq t_i\}}}{\sum_{j \in R_i}e^{\eta_j}}=\delta_k-\sum_{i \in C_k}(\frac{e^{\eta_k}}{\sum_{j \in R_i}e^{\eta_j}})
$$

由于是对$\eta_k$求偏导，因此在$\sum_{i=1}^m \eta_{j(i)}$中，若有$\eta_k$则为1，反之为0。也就是说，只要$y_k$是failure time就为1，是删失数据即为0，等价于$\delta_k$。而对于给定的$i$，$\eta_k$不一定在$\sum_{j \in R_i}e^{\eta_j}$中，因此可根据$R_i$的定义添加示性函数。综合考虑$\sum_{i=1}^m$和$\textrm{I}_{\{y_k \geq t_i\}}$即可发现，$\eta_k$仅出现在$y_k \geq t_i$的$i$的集合中，也就是$C_k$的定义。

$$
\begin{aligned}
l''(\eta)_{kk}&=-[e^{\eta_k} \cdot \sum\limits_{i \in C_k}\frac{1}{\sum_{j \in R_i}e^{\eta_j}}+e^{\eta_k}(-\sum_{i \in C_k}\frac{e^{\eta_k}}{(\sum_{j \in R_i}e^{\eta_j})^2})] \\
&=-\sum_{i \in C_k}\frac{e^{\eta_k}\sum_{j \in R_i}e^{\eta_j}-(e^{\eta_k})^2}{(\sum_{j \in R_i}e^{\eta_j})^2}
\end{aligned}
$$

当$\sum_{i=1}^m$转化为$\sum_{i \in C_k}$后，此时$i$对应的$R_i$中必定包含$\eta_k$，因此不用再加示性函数。

> 可见原文的$w(\tilde \eta)_k$缺少了负号

> $w(\tilde \eta)_k$显然小于等于0，因为$R_i$中必定包含索引$k$

$$
z(\tilde \eta)_k=\tilde \eta_k-\frac{l'(\tilde \eta)_k}{l''(\tilde \eta)_{kk}}=\tilde \eta_k-\frac{\delta_k-\sum\limits_{i \in C_k}(\frac{e^{\eta_k}}{\sum_{j \in R_i}e^{\eta_j}})}{w(\tilde \eta)_k}
$$

> 和原文相比还是符号有问题

**<span style='color:red'>事实上，这里有一个致命的错误。</span>**当$y_k \lt t_1$时，$C_k$为空集，对应的$w(\tilde \eta)_k=0$，不能取倒数！

之后，对$\beta_k$求偏导

$$
\frac{\partial M}{\partial \beta_k}=\frac{2}{n}\sum_{i=1}^nw(\tilde \eta)_ix_{ik}(z(\tilde \eta)_i-x_i^T\beta)+\lambda\alpha\cdot\textrm{sgn}(\beta_k)+\lambda(1-\alpha)\beta_k
$$

> 至此，目标函数与原文相差负号，但是$-x_{ik}$的负号与前面的负号抵消掉，所以最终是正号。但这个分子上的2不知道是忘了写了还是前面又默认乘上1/2把2消了。为了与原文一致，后面暂且忽略掉这个2。

令偏导为0，可得

$$
\frac{1}{n}\sum_{i=1}^nw_ix_{ik}(z_i-\sum_{j \neq k}x_{ij}\beta_j)-\frac{1}{n}\sum_{i=1}^nw_ix_{ik}^2\beta_k+\lambda\alpha\cdot \textrm{sgn}(\beta_k)+\lambda(1-\alpha)\beta_k=0
$$

> 行宽有限，这里简记$w(\tilde \eta)_i=w_i$、$z(\tilde \eta)_i=z_i$

> 再次强调，这里的1/n事实上就是权重，应该把$w_i/n$看出一个整体

此时可将不含$\beta_k$的第一项记作常数$C$，把后面的三项记作关于$\beta_k$的函数$f(\beta_k)$

$$
f(\beta_k)=(\lambda(1-\alpha)-\frac{1}{n}\sum_{i=1}^nw_ix_{ik}^2)\beta_k+\lambda\alpha\cdot\textrm{sgn}(\beta_k)
$$

由于$w_i \leq 0$，且$f(\beta_k)$为奇函数，则其图像大概为

<div class="figure" style="text-align: center">
<img src="06-survival_analysis_files/figure-html/survival-p2-1.png" alt="函数图" width="60%" />
<p class="caption">(\#fig:survival-p2)函数图</p>
</div>

则该问题就转化为对$C$进行分类讨论，看看$f(\beta_k)$什么时候和横轴相交，求出相交时的横坐标即可。思路已经有了，这里就不展开说了，得到结果如下所示

$$
\hat\beta_k=-\frac{\textrm{S}(\frac{1}{n}\sum_{i=1}^nw_ix_{ik}(z_i-\sum_{j \neq k}x_{ij}\beta_j),\lambda\alpha)}{-\frac{1}{n}\sum_{i=1}^nw_ix_{ik}^2+\lambda(1-\alpha)}
$$

其中$\textrm{S}(x,\lambda)=\textrm{sgn}(x)(|x|-\lambda)_+$。

> 所以原文关于$\hat\beta_k$的解是有问题的

**2. 有结点情况**

有结点情况相较无结点情况就是多了权重，其余步骤都是一样的。

$$
L(\beta)=\prod_i^m\frac{\exp{(\sum_{j \in D_i}\omega_j\eta_j})}{(\sum_{j \in R_i}\omega_je^{\eta_j})^{d_i}}
$$

其中$D_i$表示结点为$t_i$的集合，$\omega_j$表示权重，$d_i=\sum_{j \in D_i}\omega_j$。

对数似然函数为

$$
l(\beta)=\sum_i^{m}[(\sum_{j \in D_i}\omega_j\eta_j)-d_i\log(\sum_{j \in R_i}\omega_je^{\eta_j})]
$$

对$\eta_k$求一阶导及二阶导

$$
l'(\eta)_k=\delta_k\omega_k-\sum_{i \in C_k}d_i\frac{\omega_ke^{\eta_k}}{\sum_{j \in R_i}\omega_je^{\eta_j}}
$$

$$
l''(\eta)_{kk}=-\sum_{i \in C_k}d_i\frac{\omega_ke^{\eta_k}(\sum_{j \in R_i}\omega_je^{\eta_j})-(\omega_ke^{\eta_k})^2}{(\sum_{j \in R_i}\omega_je^{\eta_j})^2}
$$

> 同样和原文差了负号
> 
> 同样没有解决$w(\eta)_k$可能为0的问题。
> 
> 你可以试着将其代入到无结点的情况下，也就是把$\omega=1/n$、$d_i=1/n$带进去，就会发现无结点情况下的那个1/n就是权重，应该把那个1/n并到$l''(\tilde \eta)$中，这样无结点和有结点就一致了

则

$$
z(\tilde \eta)_k=\tilde \eta_k-\frac{\delta_k\omega_k-\sum_{i \in C_k}d_i\frac{\omega_ke^{\eta_k}}{\sum_{j \in R_i}\omega_je^{\eta_j}}}{w(\tilde \eta)_k}
$$

> z中的eta要不要带权重

$\hat\beta_k$的表达式同无结点情形。

### 自定义算法 {#survival_2_2}






