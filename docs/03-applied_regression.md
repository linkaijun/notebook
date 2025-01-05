# (PART) 模型与方法 {.unnumbered}

# 应用回归分析 {#reg}

回归模型的分类如下所示：


```{=html}
<div class="grViz html-widget html-fill-item" id="htmlwidget-d766a0a612277c55a920" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-d766a0a612277c55a920">{"x":{"diagram":"digraph {\n  graph [layout = dot, rankdir = LR]\n  node [shape = box]\n  edge [arrowsize=0.5, headport=w, arrowhead=none]\n  \n  rec1_1 [label = \"回归模型\"]\n  rec2_1 [label = \"参数回归模型\"]\n  rec2_2 [label = \"半参数回归模型\"]\n  rec2_3 [label = \"非参数回归模型\"]\n  rec3_1 [label = \"线性回归模型\"]\n  rec3_2 [label = \"非线性回归模型\"]\n  rec4_1 [label = \"一元线性回归模型\"]\n  rec4_2 [label = \"多元线性回归模型\"]\n  rec4_3 [label = \"一元回归\"]\n  rec4_4 [label = \"多元回归\"]\n  \n  rec1_1 -> {rec2_1 rec2_2 rec2_3}\n  rec2_1 -> {rec3_1 rec3_2}\n  rec3_1 -> {rec4_1 rec4_2}\n  rec3_2 -> {rec4_3 rec4_4}\n  }","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script>
```

回归模型的建模步骤如下所示：


```{=html}
<div class="grViz html-widget html-fill-item" id="htmlwidget-d82ba77cba6ee148ea65" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-d82ba77cba6ee148ea65">{"x":{"diagram":"digraph {\n  graph [layout = dot, rankdir = TB]\n  \n  node [shape = box]\n  edge [arrowsize=0.5]\n  \n  rec1_1 [label = \"实际问题\"]\n  rec1_2 [label = \"设置指标变量\"]\n  rec1_3 [label = \"收集整理数据\"]\n  rec1_4 [label = \"构造理论模型\"]\n  rec2_1 [label = \"修改\"]\n  rec2_2 [label = \"模型诊断\"]\n  rec2_3 [label = \"估计模型参数\"]\n  rec3_1 [label = \"模型运用\"]\n  rec4_1 [label = \"影响因素分析\"]\n  rec4_2 [label = \"决策预测\"]\n  rec4_3 [label = \"变量控制\"]\n  \n  rec1_1 -> rec1_2\n  rec1_2 -> rec1_3\n  rec1_3 -> rec1_4\n  rec1_4 -> rec2_3\n  rec2_3 -> rec2_2\n  rec2_2 -> rec2_1[label=\"否\"]\n  rec2_1 -> rec1_2\n  rec2_2 -> rec3_1[label=\"是\"]\n  rec3_1 -> {rec4_1 rec4_2 rec4_3}[arrowhead=none]\n  }","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script>
```

## 引言 {#reg_1}

### 变量间的相关关系 {#reg_1_1}

- 函数关系：变量间存在的确定性数量对应关系

   $$
   y=f(x_1, ..., x_p)
   $$

- 相关关系：变量间客观存在的非确定性数量对应关系

   $$
   y=f(x_1, ..., x_p, \varepsilon)
   $$

------------------------------------------------------------------------

- 相关分析

   用相关系数描述变量间相关关系的强度。以变量之间是否相 关、相关的方向和密切程度等为研究内容，不区分自变量和因变量，不 关心相关关系的表现形态。

- 回归分析

   对具有相关关系的变量，根据其相关关系的具体形态，选择 合适的回归模型描述自变量对因变量的影响方式。

> 相关分析与回归分析的联系
>
> 相关分析是回归分析的前提；回归分析是对因果关系（说是这么说，但不能忽视“伪回归”现象）的探讨

> 相关分析和回归分析的区别
>
> X与Y的地位是否平等；有无因果关系；相关分析揭示线性相关程度，回归分析给出具体的回归方程。

### 回归模型的一般形式 {#reg_1_2}

$$
y=f(x_1, ..., x_p, \varepsilon)
$$

其中y为因变量（响应变量、被解释变量），x为自变量（预测变量、解释变量），$\varepsilon$为模型误差（随机扰动项）。

$\varepsilon$包含的内容：

- 被忽略的解释变量
- 变量数值的观测误差
- 模型设定误差
- 其他随机因素的影响

## 假定 {#reg_2}

1. 零均值：$E(\varepsilon)=0$
2. 同方差：$Var(\varepsilon)=\sigma^2$
3. 无自相关：$Cov(\varepsilon_i,\varepsilon_j)=0$
4. 无内生性：$Cov(X_i,\varepsilon)=0$
5. 随机扰动项$\varepsilon$服从正态分布
6. 自变量为非随机变量，且无多重共线性
7. 模型是正确设定的，具有线性关系

其中第1、2、3条为**Gauss-Markov条件**。

1\~5条和后续的内容有关，留到后面再讲。

第6条，为什么自变量为非随机变量？我们的目标是探讨自变量与因变量之间的因果关系。在线性模型中，我们已经在因变量的生成过程中引入随机扰动项来代表随机因素的影响。如果再把生成自变量的过程也引入随机因素，则不好把握自变量与因变量之间的因果关系。在实际应用中，我们往往关注在给定自变量的时候，因变量是如何变化的。通过假设自变量为非随机变量，可以简化模型，使得数学推导和计算更加简洁。如果解释变量是随机的，那么模型的推断将变得非常复杂，因为需要考虑解释变量的概率分布，这将大大增加分析的难度。同时，若自变量为随机变量则可能会有内生性问题。

第7条，如果真实模型不具有线性关系，那么我们的模型设定就是有偏误的（做此假定只是为了让自己用得放心，谁又能知道真实模型是什么样子的呢）。

## 线性回归模型 {#reg_3}

### 一元线性回归模型 {#reg_3_1}

$$
\left\{
\begin{array}{c}
y=\beta_0+\beta_1x+\varepsilon\\
E(\varepsilon|x)=0\\
Var(\varepsilon|x)=\sigma^2   
\end{array}
\right. (\#eq:model1)
$$ 

等价于

$$
\left\{
\begin{array}{c}
E(y|x)=\beta_0+\beta_1x\\
Var(y|x)=\sigma^2   
\end{array}
\right. (\#eq:model2)
$$

> 习惯上将$E(y|x)$简记为$E(y)$，并称$E(y|x)=\beta_0+\beta_1x$为总体回归方程。

其中待估参数为$\beta_0$、$\beta_1$和$\sigma^2$。

结合假定，可知$y\sim~N(\beta_0+\beta_1x,\sigma^2)$。

***为什么考虑y的条件期望？***

- 由于$\varepsilon$的存在，我们无法直接估计出参数$\beta_0$和$\beta_1$。结合零均值的假定，我们可以对模型左右两边取期望来消掉$\varepsilon$的影响。同时需要注意的是，该期望是条件期望，我们更关注当x取固定值时y的均值。

- 如果从最优化的角度进行思考，假设对y的任意预测为$f(x)$，y的条件期望为$g(x)=E(y|x)$，则g(x)是y的最佳预测。

   $$
   \begin{align}
   E(y-f(x))^2&=E(y-g(x)+g(x)-f(x))^2\\
   &=E(y-g(x))^2+E(g(x)-f(x))^2+2E[(y-g(x))(g(x)-f(x))]\\
   &=E(y-g(x))^2+E(g(x)-f(x))^2\\
   &\geq E(y-g(x))^2 
   \end{align} (\#eq:model3)
   $$

> 拓展：分位数回归
>
> 一般的线性回归都是关注y条件均值，但有些时候我们可以对y的分位数进行回归，即分位数回归。分位数回归相较于均值回归能够获取更多的关于y的分布的信息，例如在保险行业，保险公司可以通过分位数回归来理解不同风险水平下的潜在损失。

上述都是针对总体的理论模型，而对于样本数据$(x_i,y_i)$，则有:

$$
y_i=\hat y_i + \hat \varepsilon=\hat \beta_0+ \hat \beta_1x_i+e_i (\#eq:model4)
$$

$$
\hat y_i=\hat \beta_0+ \hat \beta_1x_i (\#eq:model5)
$$

其中式\@ref(eq:model4)为样本回归模型，式\@ref(eq:model5)为样本回归方程（也称经验回归方程），$\hat y$和$e$（残差）分别是对$E(y|x)$和$\varepsilon$的估计。

> 无论总体还是样本，带随机扰动项或者残差的叫“回归模型”，不带的叫“回归方程”或“回归函数”。

### 多元线性回归模型 {#reg_3_2}

$$
\left\{
\begin{array}{c}
y=\beta_0+\beta_1x_1+...+\beta_px_p+\varepsilon\\
E(\varepsilon|x)=0\\
Var(\varepsilon|x)=\sigma^2   
\end{array}
\right. (\#eq:model6)
$$ 

等价于

$$
\left\{
\begin{array}{c}
E(y|x)=\beta_0+\beta_1x+...+\beta_px_p\\
Var(y|x)=\sigma^2   
\end{array}
\right. (\#eq:model7)
$$

其中待估参数为$\beta_0$、$\beta_1$、...、$\beta_p$和$\sigma^2$。

矩阵表达式为：

$$
\left\{
\begin{array}{ll}
Y=X\beta+\varepsilon\\
E(\varepsilon)=0\\
Var(\varepsilon)=\sigma^2I_n 
\end{array}
\right. (\#eq:model8)
$$

其中

$$
Y=
\begin{pmatrix}
y_1\\
y_2\\
\vdots\\
y_n
\end{pmatrix}
\,,
X=
\begin{pmatrix}
1 & x_{11} \cdots x_{1p}\\
1 & x_{21} \cdots x_{2p}\\
\vdots\\
1 & x_{n1} \cdots x_{np}
\end{pmatrix}
\,,
\beta=
\begin{pmatrix}
\beta_1\\
\beta_2\\
\vdots\\
\beta_n
\end{pmatrix}
\varepsilon=
\begin{pmatrix}
\varepsilon_1\\
\varepsilon_2\\
\vdots\\
\varepsilon_n 
\end{pmatrix} (\#eq:model9)
$$

此时$Y \sim N(X\beta,\sigma^2I_n)$

## 参数估计 {#reg_4}

### 最小二乘估计 {#reg_4_1}

#### 一元场合 {#reg_4_1_1}

对于离差平方和$Q(\beta_0,\beta_1)$，最小二乘法考虑寻找合适的$\hat \beta_0$与$\hat \beta_1$使得残差平方和$Q(\hat \beta_0,\hat \beta_1)$最小。

$$
\begin{align}
Q(\beta_0,\beta_1)&=\sum^n_{i=1}(y_i-\beta_0-\beta_1x_i)^2=\sum^n_{i=1}\varepsilon^2\\
\Rightarrow Q(\hat \beta_0,\hat \beta_1)&=\underset{\hat \beta_0, \hat \beta_1} {\arg\min} \sum^n_{i=1}(y_i-\hat \beta_0-\hat \beta_1x_i)^2\\
&=\underset{\hat \beta_0, \hat \beta_1} {\arg\min} \sum^n_{i=1}(y_i-\hat y_i)^2\\
&=\underset{\hat \beta_0, \hat \beta_1} {\arg\min} \sum^n_{i=1}e_i^2 
\end{align} (\#eq:model10)
$$

分别对$\beta_0$和$\beta_1$求偏导，并使其为0：

$$
\left\{
\begin{array}{ll}
\frac{\partial Q}{\partial \beta_0} \mid _{\beta_0=\hat \beta_0, \beta_1=\hat \beta_1} &= -2 \sum\limits_{i=1}^n (y_i-\hat \beta_0-\hat \beta_1x_i)=0 \\
\frac{\partial Q}{\partial \beta_1} \mid _{\beta_0=\hat \beta_0, \beta_1=\hat \beta_1} &= -2 \sum\limits_{i=1}^n (y_i-\hat \beta_0-\hat \beta_1x_i)x_i=0  
\end{array}
\right. (\#eq:model11)
$$

式\@ref(eq:model11)等价于：

$$
\left\{
\begin{array}{ll}
\sum\limits_{i=1}^n e_i=0  \\
\sum\limits_{i=1}^n x_ie_i=0 
\end{array}
\right. (\#eq:model12)
$$
   
> <span style='color: red; font-weight: bold'>这个关系式非常重要</span>

将求和式展开，并稍加整理，可得：

$$
\left\{
\begin{array}{ll}
\bar y=\hat \beta_0+\hat \beta_1 \bar x  \\
\sum\limits_{i=1}^n y_ix_i=n\bar x \hat \beta_0+\hat \beta_1\sum\limits_{i=1}^n x_i^2 
\end{array}
\right. (\#eq:model13)
$$

易得$\hat \beta_0=\bar y-\hat \beta_1 \bar x$，将其代入可得$\hat \beta_1$：

$$
\begin{align}
n\bar x \hat \beta_0+\hat \beta_1\sum_{i=1}^n x_i^2 &= \sum_{i=1}^n y_ix_i\\
n\bar x (\bar y-\hat \beta_1 \bar x) + \hat \beta_1\sum_{i=1}^n x_i^2 &= \sum_{i=1}^n y_ix_i\\
\hat \beta_1(\sum_{i=1}^n x_i^2 - n \bar x^2) &= \sum_{i=1}^n y_ix_i - n\bar x \bar y\\
\hat \beta_1 &= \frac{\sum\limits_{i=1}^n(x_i-\bar x)y_i}{\sum\limits_{i=1}^n(x_i-\bar x)^2}\\
\hat \beta_1 &= \frac{\sum\limits_{i=1}^n(x_i-\bar x)(y_i-\bar y)}{\sum\limits_{i=1}^n(x_i-\bar x)^2} 
\end{align} (\#eq:model14)
$$

> 注意有以下关系，后面还会用到
>
> $\sum\limits_{i=1}^n(x_i-\bar x)=0$
>
> $\sum\limits_{i=1}^n(x_i-\bar x)x_i=\sum\limits_{i=1}^n(x_i-\bar x)(x_i-\bar x)=\sum\limits_{i=1}^n(x_i-\bar x)^2$
>
> $\sum\limits_{i=1}^n(x_i-\bar x)(y_i-\bar y)$=$\sum\limits_{i=1}^n(x_i-\bar x)y_i$

记$L_{xx}=\sum\limits_{i=1}^n(x_i-\bar x)^2$、$L_{yy}=\sum\limits_{i=1}^n(y_i-\bar y)^2$、$L_{xy}=\sum\limits_{i=1}^n(x_i-\bar x)(y_i-\bar y)$，则最小二乘估计为：

$$
\left\{
\begin{array}{ll}
\hat \beta_0=\bar y - \hat \beta_1 \bar x \\
\hat \beta_1=\frac{L_{xy}}{L_{xx}} 
\end{array}
\right. (\#eq:model15)
$$

而对于$\sigma^2$，很自然地会用残差$e$进行估计，常用无偏估计量$\hat \sigma^2$进行估计：

$$
\hat \sigma^2=\frac{\sum\limits_{i=1}^n e_i^2}{n-2} (\#eq:model16)
$$

> 在多元回归部分会给出更为详细的证明。

#### 多元场合 {#reg_4_1_2}

对于离差平方和$Q(\beta)$有

$$
\begin{aligned}
Q(\beta) &= (Y-X \beta)'(Y-X \beta)\\
&=Y'Y- \beta'X'Y-Y'X \beta+\beta'X'X \beta\\
&=Y'Y-2\beta'X'Y+\beta'X'X \beta
\end{aligned} (\#eq:model19)
$$

> 注意$\beta'X'Y$和$Y'X \beta$都是标量。

对$\beta$求偏导可得：

$$
{\partial Q \over \partial \beta} \mid_{\beta=\hat \beta}=-2X'Y+2X'X\hat \beta=0\\
\begin{aligned}
\Rightarrow X'X\hat \beta &= X'Y\\
\hat \beta&=(X'X)^{-1}X'Y
\end{aligned} (\#eq:model20)
$$
   
> 一阶导条件等价于$X'(Y-X\hat \beta)=X'e=0$，其中$e$为残差，<span style='color: red; font-weight: bold'>牢记$X'e=0$</span>
> 
> 为了确保有解，要求自变量之间无多重共线性，即矩阵X列满秩，故矩阵X'X可逆。

故最小二乘估计为$\hat \beta=(X'X)^{-1}Y$。

对于拟合值$\hat Y$，有：

$$
\hat Y = X \hat \beta = X(X'X)^{-1}X'Y=HY (\#eq:model21)
$$

其中$H=X(X'X)^{-1}X'$为n阶**对称幂等矩阵**，即$H=H'$和$H=H^2$。$H$也称为**投影矩阵**。
   
> 实对称矩阵的特征根非0即1，故tr(H)=rank(H)=p+1

则残差向量$e$为：

$$
\begin{aligned}
e &= Y-\hat Y\\
&=Y-HY\\
&=(I-H)Y\\
&=(I-H)(X\beta+\varepsilon)\\
&=X\beta-HX\beta+(I-H)\varepsilon\\
&=(I-H)\varepsilon
\end{aligned} (\#eq:model22)
$$

之后对残差平方和$SSE=e'e=\varepsilon'(I-H)\varepsilon$取期望：

> tr(AB)=tr(BA)
>
> I和H为n阶矩阵；X为p+1阶矩阵

$$
\begin{aligned}
E(SSE)&=E(\varepsilon'(I-H)\varepsilon)\\
&=E[tr(\varepsilon'(I-H)\varepsilon)]\\
&=E[tr((I-H)\varepsilon\varepsilon')]\\
&=tr((I-H)E(\varepsilon\varepsilon'))\\
&=\sigma^2 tr(I-H)\\
&=\sigma^2 [n-tr(H)]\\
&=\sigma^2 [n-tr(X(X'X)^{-1}X')]\\
&=\sigma^2 [n-tr((X'X)^{-1}X'X)]\\
&=\sigma^2 [n-p-1]\\
\end{aligned}  (\#eq:model23)
$$

故$\sigma^2$的无偏估计为$\hat \sigma^2={SSE \over n-p-1}$

### 极大似然估计 {#reg_4_2}

#### 一元场合 {#reg_4_2_1}

在$y\sim~N(\beta_0+\beta_1x,\sigma^2)$的假定下，写出对数似然函数：

$$
\ln (L)=-{n \over 2} \ln (2\pi \sigma^2) - {1 \over 2\sigma^2} \sum_{i=1}^n [y_i-(\beta_0+\beta_1x_i)]^2 (\#eq:model17)
$$

分别对$\beta_0$、$\beta_1$、$\sigma^2$求偏导，可得对应的估计量。其中$\beta_0$、$\beta_1$与最小二乘估计的结果一致，但$\sigma^2$的估计量为$\hat \sigma^2={\sum\limits_{i=1}^n e_i^2 \over n}$，是有偏估计量。

#### 多元场合 {#reg_4_2_2}

注意到有$Y \sim N(X\beta, \sigma^2I_n)$。故对数似然函数为：

$$
\ln L=-{n \over 2}\ln (2\pi)-{n \over 2}\ln (\sigma^2)-{1 \over 2\sigma^2}(Y-X\beta)'(Y-X\beta) (\#eq:model24)
$$

要使对数似然函数取得最大值，则需最小化$(Y-X\beta)'(Y-X\beta)$，与式\@ref(eq:model19)一致，故$\hat \beta_{MLE}$结果与最小二乘估计一致。而$\sigma^2$的估计量为$\hat \sigma^2={(Y-X\beta)'(Y-X\beta) \over n}$，同一元场合。

### 矩估计 {#reg_4_3}

#### 一元场合 {#reg_4_3_1}

在前提假定中规定了$E(\varepsilon)=0$及$Cov(X_i,\varepsilon)=E(X_i\varepsilon)=0$，注意到残差$e$是对$\varepsilon$的估计，则用样本矩估计总体矩有：

$$
\left\{
\begin{array}{ll}
{1 \over n} \sum\limits_{i=1}^n (y_i-\hat \beta_0-\hat \beta_1 x_i)=0 \\
{1 \over n} \sum\limits_{i=1}^n (y_i-\hat \beta_0-\hat \beta_1 x_i)x_i=0
\end{array}
\right. (\#eq:model18)
$$

与式\@ref(eq:model8)一致，则估计结果与最小二乘估计相同。

#### 多元场合 {#reg_4_3_2}

在多元场合，注意到前提假定$E(\varepsilon)=0$和$Cov(X_i,\varepsilon)=E(X_i\varepsilon)=0$，对应的样本矩条件为：

$$
{1 \over n}X'(Y-X\hat \beta)=0\\
\Rightarrow \hat \beta =(X'X)^{-1}X'Y (\#eq:model25)
$$

可得矩估计的结果和最小二乘估计相同。

> 无论一元还是多元，最小二乘估计、极大似然估计和矩估计都用到了零均值、无内生性、无多重共线性（多元场合）的前提假定，其中极大似然估计额外运用了正态分布的假定。可以发现，估计的核心都是$X'(Y-X\hat \beta)=0$，或者说是$X'e=0$。
>
> 注意X'的第一行都是1，用来满足$E(\varepsilon)=0$的条件。其余行为不同自变量的观测值，用来满足$Cov(X_i,\varepsilon)=E(X_i\varepsilon)=0$的条件。

### 几何视角 {#reg_4_4}

<video src='./video/Projection.mp4' controls width="800px" height="600px"></video>

[B站：最小二乘与投影矩阵](https://www.bilibili.com/video/BV1eFxMeKEpM/)

## 最小二乘估计的性质 {#reg_5}

根据高斯-马尔科夫定理，在满足假定的前提下，最小二乘估计为**最优线性无偏估计(best linear unbiased estimator，BLUE)**。再探讨性质之前，请先回忆一元及多元场合的最小二乘估计值，式\@ref(eq:model15)和式\@ref(eq:model20)。

### 线性 {#reg_5_1}

- 一元场合

   式\@ref(eq:model15)给出了最小二乘估计，对其稍加整理即可发现$\beta$是$y$的线性组合。

   $$
   \begin{aligned}
   \hat{\beta_1} &= {L_{xy} \over L_{xx}} \\
   &= {\sum\limits_{i=1}^n (x_i-\bar x)(y_i - \bar y) \over L_{xx}} \\
   &= {\sum\limits_{i=1}^n (x_i-\bar x)y_i \over L_{xx}} \\
   &= \sum_{i=1}^n {(x_i-\bar x) \over L_{xx}} y_i \\
   &= \sum_{i=1}^n k_i y_i
   \end{aligned} (\#eq:model26)
   $$

   > $L_{xx}$相当于一个常数，可以放进去或提出来。

   $$
   \begin{aligned}
   \hat \beta_0 &= \bar y - \hat \beta_1 \bar x \\
   &= \sum\limits_{i=1}^n {1 \over n}y_i - \sum\limits_{i=1}^n {(x_i-\bar x)\bar x \over L_{xx}} y_i \\
   &= \sum\limits_{i=1}^n [{1 \over n} - {(x_i-\bar x)\bar x \over L_{xx}}]y_i
   \end{aligned} (\#eq:model27)
   $$
   
   对于拟合值$\hat y_i$有
   
   $$
   \hat y_i=\hat \beta_0+\hat \beta_1x_i=\sum_{j=1}^n [\frac{1}{n}+\frac{(x_i-\bar x)(x_j-\bar x)}{L_{xx}}]y_j=\sum_{j=1}^n h_{ij}y_j (\#eq:model50)
   $$
   
   > $h_{ij}=h_{ji}$
   
   对于给定的新值$x_0$，对应的预测值$\hat y_0$有
   
   $$
   \hat y_0=\hat \beta_0+\hat \beta_1x_0=\sum_{j=1}^n [\frac{1}{n}+\frac{(x_0-\bar x)(x_j-\bar x)}{L_{xx}}]y_j=\sum_{j=1}^n h_{0j}y_j (\#eq:model51)
   $$

- 多元场合

对于最小二乘估计$\hat \beta$有

   $$
   \begin{aligned}
   \hat \beta &= (X'X)^{-1}X'Y \\
   &= (X'X)^{-1}X'(X\beta+\varepsilon) \\
   &= \beta+(X'X)^{-1}X'\varepsilon
   \end{aligned} (\#eq:model28)
   $$

   注意到$\hat \beta$不仅是$y$的线性组合，还是$\varepsilon$的线性组合。

对于拟合值向量$\hat Y$，参照式\@ref(eq:model21)有$\hat Y = HY$。

对于

### 无偏性 {#reg_5_2}

- 一元场合

   $$
   \begin{aligned}
   E(\hat \beta_1)&= \sum_{i=1}^n {(x_i-\bar x) \over L_{xx}} E(y_i) \\
   &= \sum_{i=1}^n {(x_i-\bar x) \over L_{xx}}(\beta_0+\beta_1x_i) \\
   &= \sum_{i=1}^n {(x_i-\bar x)x_i \over L_{xx}}\beta_1 \\
   &= {L_{xx} \over L_{xx}}\beta_1 \\
   &= \beta_1
   \end{aligned} (\#eq:model29)
   $$

   $$
   \begin{aligned}
   E(\hat \beta_0)&=E(\bar y)-E(\hat \beta_1)\bar x \\
   &= {\sum\limits_{i=1}^n E(y_i) \over n}-\beta_1 \bar x \\
   &= {\sum\limits_{i=1}^n (\beta_0 + \beta_1 x) \over n}-\beta_1 \bar x \\
   &= \beta_0 + \beta_1 \bar x -\beta_1 \bar x \\
   &= \beta_0
   \end{aligned} (\#eq:model30)
   $$

- 多元场合

   $$
   \begin{aligned}
   E(\hat \beta)&=E((X'X)^{-1}X'Y) \\
   &= (X'X)^{-1}X'E(Y) \\
   &= (X'X)^{-1}X'X\beta \\
   &= \beta
   \end{aligned} (\#eq:model31)
   $$

### 有效性 {#reg_5_3}

- 一元场合

   不妨令$\tilde \beta_1=\sum\limits_{i=1}^n c_iy_i$也是$\beta_1$的无偏估计。

   $$
   \begin{aligned}
   E(\tilde \beta_1)&=E(\sum\limits_{i=1}^n c_iy_i) \\
   &= \sum\limits_{i=1}^n c_iE(y_i) \\
   &= \sum\limits_{i=1}^n c_i(\beta_0+\beta_1x_i) \\
   &= \beta_0 \sum\limits_{i=1}^n c_i + \beta_1 \sum\limits_{i=1}^n c_ix_i \\
   &= \beta_1
   \end{aligned} (\#eq:model32)
   $$

   根据无偏性，可知$\tilde \beta_1$满足$\sum\limits_{i=1}^n c_i=0$和$\sum\limits_{i=1}^n c_ix_i=1$。

   $$
   \begin{aligned}
   Var(\tilde \beta_1)&=\sum\limits_{i=1}^n c_i^2Var(y_i) \\
   &= \sum\limits_{i=1}^n c_i^2 \sigma^2 \\
   &= \sum\limits_{i=1}^n (c_i-k_i+k_i)^2 \sigma^2 \\
   &= \sigma^2[\sum\limits_{i=1}^n(c_i-k_i)^2+\sum\limits_{i=1}^nk_i^2+\sum\limits_{i=1}^n2(c_i-k_i)k_i] \\
   &= \sum\limits_{i=1}^n(c_i-k_i)^2 \sigma^2 + Var(\hat \beta_1) \\
   Var(\tilde \beta_1) &\geq Var(\hat \beta_1)
   \end{aligned} (\#eq:model33)
   $$

   注意到$\sum\limits_{i=1}^n c_i=0$和$\sum\limits_{i=1}^n c_ix_i=1$，其中

   $$
   \begin{aligned}
   \sum\limits_{i=1}^n2(c_i-k_i)k_i &= 2\sum\limits_{i=1}^n c_ik_i-2\sum\limits_{i=1}^n k_i^2 \\
   &=2\sum\limits_{i=1}^n c_i{(x_i-\bar x) \over \sum\limits_{i=1}^n (x_i-\bar x)^2}-2\sum\limits_{i=1}^n ({(x_i-\bar x) \over \sum\limits_{i=1}^n (x_i-\bar x)^2})^2 \\
   &= 2{1 \over \sum\limits_{i=1}^n (x_i-\bar x)^2}-2{1 \over \sum\limits_{i=1}^n (x_i-\bar x)^2} \\
   &= 0
   \end{aligned} (\#eq:model34)
   $$

   关于$\hat \beta_0$的证明也是类似的，略。

- 多元场合

   不妨令$\tilde \beta=[(X'X)^{-1}X'+A]Y$是$\beta$的无偏估计。

   $$
   \begin{aligned}
   E(\tilde \beta)&=[(X'X)^{-1}X'+A]E(Y) \\
   &= [(X'X)^{-1}X'+A]X\beta \\
   &= \beta +AX\beta \\
   &= \beta
   \end{aligned} (\#eq:model35)
   $$

   根据无偏性，可知$AX=0$。

   $$
   \begin{aligned}
   Cov(\hat \beta)&=[(X'X)^{-1}X']Cov(Y)[(X'X)^{-1}X']' \\
   &= [(X'X)^{-1}X']\sigma^2I_n[(X'X)^{-1}X']' \\
   &= \sigma^2 (X'X)^{-1} \\
   Cov(\tilde \beta)&=[(X'X)^{-1}X'+A]Cov(Y)[(X'X)^{-1}X'+A]' \\
   &= [(X'X)^{-1}X'+A]\sigma^2I_n[(X'X)^{-1}X'+A]' \\
   &= \sigma^2 [(X'X)^{-1}+AA'] 
   \end{aligned} (\#eq:model36)
   $$

   > 注意到$AX=0$

   由于$AA'$是半正定矩阵（非负定），则$Cov(\tilde \beta) \geq Cov(\hat \beta)$。

### 方差 {#reg_5_4}

- 一元场合

   - $Var(\hat{\beta_1})$
   
   $$
   \begin{aligned}
   Var(\hat \beta_1)&=Var(\sum_{i=1}^n {(x_i-\bar x) \over L_{xx}} y_i) \\
   &= \sum_{i=1}^n {(x_i-\bar x)^2 \over L_{xx}^2}\sigma^2 \\
   &= {L_{xx} \over L_{xx}^2}\sigma^2 \\
   &= {\sigma^2 \over L_{xx}}
   \end{aligned} (\#eq:model37)
   $$
   
   - $Var(\hat{\beta_0})$
   
   $$
   \begin{aligned}
   Var(\hat \beta_0)&=Var(\bar y - \hat \beta_1 \bar x) \\
   &= Var(\bar y)+\bar x^2Var(\hat \beta)-2Cov(\bar y,\hat \beta_1 \bar x) \\
   &= {\sigma^2 \over n}+\bar x^2{\sigma^2 \over L_{xx}} \\
   &= [{1 \over n}+{\bar x^2 \over L_{xx}}]\sigma^2
   \end{aligned} (\#eq:model38)
   $$

   其中
   
   $$
   \begin{aligned}
   Cov(\bar y,\hat \beta_1 \bar x)&={\bar x^2 \over n}Cov(\sum_{i=1}^n y_i,\sum_{i=1}^n k_iy_i) \\
   &= {\bar x^2 \over n} \sum_{i=1}^n k_iCov(y_i,y_i) \\
   &= {\bar x^2 \sigma^2 \over n} \sum_{i=1}^n k_i \\
   &= {\bar x^2 \sigma^2 \over n} \sum_{i=1}^n {x_i-\bar x \over L_{xx}} \\
   &= 0
   \end{aligned} (\#eq:model39)
   $$

   > 注意有$Cov(\varepsilon_i,\varepsilon_j)=0$，则$Cov(y_i,y_j)=0$
   
   - $Cov(\hat \beta_0,\hat \beta_1)$
   
   $$
   \begin{aligned}
   Cov(\hat \beta_0,\hat \beta_1) &= Cov(\bar y - \hat \beta_1 \bar x,\hat \beta_1) \\
   &= Cov(\bar y,\hat \beta_1)-\bar xCov(\hat \beta_1,\hat \beta_1) \\
   &= 0-\bar x {\sigma^2 \over L_{xx}} \\
   &= -{\bar x \over L_{xx}}\sigma^2
   \end{aligned} (\#eq:model40)
   $$

- 多元场合

   $$
   \begin{aligned}
   Cov(\hat \beta) &= E[(\hat \beta-E(\hat \beta))(\hat \beta-E(\hat \beta))'] \\
   &= E[(\hat \beta-\beta)(\hat \beta-\beta)'] \\
   &= E[(X'X)^{-1}X'\varepsilon \varepsilon ' X(X'X)^{-1}] \\
   &= (X'X)^{-1}X'E(\varepsilon \varepsilon ') X(X'X)^{-1} \\
   &= (X'X)^{-1}X'\sigma^2I_n X(X'X)^{-1} \\
   &= \sigma^2(X'X)^{-1}
   \end{aligned} (\#eq:model46)
   $$

### 正态性 {#reg_5_5}

- 一元场合 

   根据式\@ref(eq:model26)和式\@ref(eq:model27)、式\@ref(eq:model29)和式\@ref(eq:model30)、式\@ref(eq:model37)和式\@ref(eq:model38)可知，$\hat \beta_1$和$\hat \beta_0$服从正态分布。
   
   $$
   \begin{gather}
   \hat \beta_1 \sim N(\beta_1, \frac{\sigma^2}{L_{xx}}) \\
   \hat \beta_0 \sim N(\beta_0, [\frac{1}{n}+\frac{\bar x^2}{L_{xx}}]\sigma^2) 
   \end{gather} (\#eq:model48)
   $$
> $\hat \beta_1$的正态性源自y的正态性，而y的正态性又源自$\varepsilon$的正态性

- 多元场合

   根据式\@ref(eq:model28)、式\@ref(eq:model31)、式\@ref(eq:model46)可知，$\hat \beta$服从正态分布。
   
   $$
   \hat \beta \sim N(\beta, \sigma^2(X'X)^{-1}) (\#eq:model49)
   $$

### 残差 {#reg_5_6}

- 一元场合

   - 线性表示

   根据式\@ref(eq:model50)，对于残差$e_i$有
   
   $$
   e_i=y_i-\hat y_i=y_i-\sum_{j=1}^n h_{ij}y_j (\#eq:model52)
   $$

   - $E(e_i)$
   
   $$
   E(e_i)=E(y_i-\hat y_i)=(\beta_0+\beta_1 x_i)-(\beta_0+\beta_1 x_i)=0 (\#eq:model53)
   $$
   
   - $Cov(e_i,e_j)$
   
   当$i \neq j$时：

   $$
   \begin{aligned}
   Cov(e_i,e_j)&=Cov(y_i-\sum\limits_{k=1}^nh_{ik}y_k \, , \, y_j-\sum\limits_{l=1}^nh_{jl}y_l) \\
   &= -Cov(y_i \, , \, h_{ji}y_i)-Cov(y_j \, , \, h_{ij}y_j)+\sum\limits_{k=1}^n h_{ik}h_{jk}Cov(y_i \, , \, y_j) \\
   &= -h_{ji}\sigma^2-h_{ij}\sigma^2+h_{ij}\sigma^2 \\
   &= -h_{ij}\sigma^2
   \end{aligned} (\#eq:model41)
   $$

   其中

   $$
   \begin{aligned}
   \sum\limits_{k=1}^n h_{ik}h_{jk}&=\sum\limits_{k=1}^n [{1 \over n^2} + {(x_k-\bar{x})(x_j-\bar{x}+x_i-\bar{x}) \over nL_{xx}}+{(x_i-\bar{x})(x_j-\bar{x})(x_k-\bar{x})^2 \over L_{xx}^2}] \\
   &= {1 \over n} + {(x_i-\bar{x})(x_j-\bar{x}) \over L_{xx}} \\
   &= h_{ij}
   \end{aligned} (\#eq:model42)
   $$

   当$i = j$时：

   $$
   \begin{aligned}
   Cov(e_i \, , \, e_i)&=Var(e_i) \\
   &= Var(y_i-\sum\limits_{j=1}^nh_{ij}y_j) \\
   &= Var(y_i)+Var(\sum\limits_{j=1}^nh_{ij}y_j)-2Cov(y_i\, , \, \sum\limits_{j=1}^nh_{ij}y_j) \\
   &= \sigma^2 + \sigma^2 \sum\limits_{j=1}^n h_{ij}^2-2h_{ii}\sigma^2 \\
   &= \sigma^2+h_{ii}\sigma^2-2h_{ii}\sigma^2 \\
   &= (1-h_{ii})\sigma^2
   \end{aligned} (\#eq:model43)
   $$

   其中

   $$
   \begin{aligned}
   \sum\limits_{j=1}^n h_{ij}^2 &= \sum\limits_{j=1}^n [{1 \over n^2}+{(x_i-\bar{x})^2(x_j-\bar{x})^2 \over L_{xx}^2}+{2(x_i-\bar{x})(x_j-\bar{x}) \over nL_{xx}}] \\
   &= {1 \over n}+{(x_i-\bar{x})^2 \over L_{xx}} \\
&= h_{ii}
   \end{aligned} (\#eq:model44)
   $$
   

   故

   $$
   Cov(e_i\, , \, e_j)=
   \begin{cases}
   (1-h_{ii})\sigma^2, &i=j \\
   -h_{ij}\sigma^2, &i \neq j
   \end{cases} (\#eq:model45)
   $$
   
   特别的，称$h_{ii}=\frac{1}{n}+\frac{(x_i-\bar{x})^2}{L_{xx}}$为<span style='color: red; font-weight: bold'>杠杆值</span>。
   
   > 杠杆值度量了自变量空间中第$i$个观测点偏离样本中心的程度。当杠杆值越大时，对应的$Var(e_i)$越小，在几何上表现为较远的观测点会把回归线尽可能地拉到自身周边，从而降低自身的残差值。对应的观测点也称之为<span style='color: red; font-weight: bold'>高杠杆点</span>
   
   - $Cov(e_i, \hat y_j)$
   
   对任意的$i$和$j$，由式\@ref(eq:model50)和式\@ref(eq:model52)可得
   
   $$
   \begin{aligned}
   Cov(e_i, \hat y_j)&=Cov(y_i-\sum_{k=1}^n h_{ik}y_k, \sum_{k=1}^n h_{jk}y_k) \\
   &=Cov(y_i,\sum_{k=1}^n h_{jk}y_k)-Cov(\sum_{k=1}^n h_{ik}y_k,\sum_{k=1}^n h_{jk}y_k) \\
   &= h_{ji}\sigma^2-\sigma^2\sum_{k=1}^n h_{ik}h_{jk} \\
   &= h_{ji}\sigma^2-\sigma^2 h_{ij} \\
   &= 0
   \end{aligned} (\#eq:model62)
   $$
   
   - $Cov(e_i, \hat y_0)$
   
   由式\@ref(eq:model51)可得
   
   $$
   \begin{aligned}
   Cov(e_i,\hat y_0)&= Cov(y_i-\sum_{j=1}^n h_{ij}y_j, \sum_{j=1}^n h_{0j}y_j) \\
   &= Cov(y_i, \sum_{j=1}^n h_{0j}y_j)-Cov(\sum_{j=1}^n h_{ij}y_j,\sum_{j=1}^n h_{0j}y_j) \\
   &= h_{0i}\sigma^2-\sigma^2 \sum_{j=1}^n h_{ij}h_{0j} \\
   &= h_{0i}\sigma^2-\sigma^2 h_{i0} \\
   &= 0
   \end{aligned} (\#eq:model63)
   $$
   
   - $Cov(e_i, \hat \beta_0)$
   
   由式\@ref(eq:model27)可得
   
   $$
   \begin{aligned}
   Cov(e_i, \hat \beta_0)&=Cov(y_i-\sum_{j=1}^n h_{ij}y_j, \sum_{j=1}^n [{1 \over n} - {(x_j-\bar x)\bar x \over L_{xx}}]y_j) \\
   &= Cov(y_i, \sum_{j=1}^n [{1 \over n} - {(x_j-\bar x)\bar x \over L_{xx}}]y_j)-Cov(\sum_{j=1}^n h_{ij}y_j,\sum_{j=1}^n [{1 \over n} - {(x_j-\bar x)\bar x \over L_{xx}}]y_j) \\
   &= [{1 \over n} - {(x_i-\bar x)\bar x \over L_{xx}}]\sigma^2 - \sigma^2 \sum_{j=1}^n h_{ij}({1 \over n} - {(x_j-\bar x)\bar x \over L_{xx}}) \\
   &= [{1 \over n} - {(x_i-\bar x)\bar x \over L_{xx}}]\sigma^2-[{1 \over n} - {(x_i-\bar x)\bar x \over L_{xx}}]\sigma^2 \\
   &= 0
   \end{aligned}  (\#eq:model64)
   $$
   
   - $Cov(e_i, \hat \beta_1)$
   
   由式\@ref(eq:model26)可得
   
   $$
   \begin{aligned}
   Cov(e_i, \hat \beta_1) &= Cov(y_i-\sum_{j=1}^n h_{ij}y_j, \sum_{j=1}^n \frac{(x_j-\bar x)}{L_{xx}}y_j) \\
   &= Cov(y_i, \sum_{j=1}^n \frac{(x_j-\bar x)}{L_{xx}}y_j) - Cov(\sum_{j=1}^n h_{ij}y_j,\sum_{j=1}^n \frac{(x_j-\bar x)}{L_{xx}}y_j) \\
   &= \frac{(x_i-\bar x)}{L_{xx}} \sigma^2 - \frac{(x_i-\bar x)}{L_{xx}} \sigma^2 \\
   &= 0
   \end{aligned} (\#eq:model65)
   $$
   
   - $\sum_{i=1}^n e_i = \sum_{i=1}^n x_ie_i=0$
   
   参见式\@ref(eq:model12)
   

- 多元场合

   - 线性表示
   
   式\@ref(eq:model22)给出了残差的表达式为$e=Y-HY=(I-H)Y=(I-H)\varepsilon$
   
   - $E(e)$
   
   $$
   E(e)=E[(I-H)\varepsilon]=0  (\#eq:model54)
   $$
   
   - $Cov(e)$

   $$
   \begin{aligned}
   Cov(e) &= Cov((I-H)Y) \\
   &= Cov((I-H)\varepsilon) \\
   &= (I-H)E(\varepsilon\varepsilon')(I-H)' \\
   &= \sigma^2 (I-H)
   \end{aligned} (\#eq:model47)
   $$
   
   - $Cov(e,\hat Y)$
   
   $$
   \begin{aligned}
   Cov(e,\hat Y)&=Cov((I-H)Y,HY) \\
   &=(I-H)Cov(Y)H' \\
   &=(I-H)Cov(X\beta+\varepsilon)H \\
   &=(I-H)H \\
   &=H-H^2 \\
   &=0
   \end{aligned} (\#eq:model55)
   $$
   
   - $Cov(e,\hat y_0)$
   
   $$
   \begin{aligned}
   Cov(e,\hat y_0)&=Cov((I-H)Y,x_0'(X'X)^{-1}X'Y) \\
   &=(I-H)X(X'X)^{-1}x_0 \\
   &=(X(X'X)^{-1}-X(X'X)^{-1}X'X(X'X)^{-1})x_0 \\
   &=0
   \end{aligned} (\#eq:model56)
   $$
   
   - $Cov(e,\hat \beta)$
   
   $$
   \begin{aligned}
   Cov(e,\hat \beta)&=Cov((I-H)Y,(X'X)^{-1}X'Y) \\
   &= (I-H)X(X'X)^{-1} \\
   &= X(X'X)^{-1}-X(X'X)^{-1}X'X(X'X)^{-1} \\
   &= 0
   \end{aligned} (\#eq:model57)
   $$
   
   - $X'e=0$
   
      参见式\@ref(eq:model20)
      
> 在正规方程组中，我们得到的一阶导条件为$X'e=0$。从几何视角来看，残差向量$e$正交于X张成的列空间，因此凡是X列空间中的向量均与$e$不相关

## 显著性检验 {#reg_6}

由于显著性检验依赖于最小二乘估计的分布，在前述内容中已经说明最小二乘估计服从正态分布，因此该部分内容严重依赖于随机扰动项的正态性、同方差、无自相关假定。

### 区间估计 {#reg_6_1}

#### 一元场合 {#reg_6_1_1}

根据式\@ref(eq:model48)已知$\hat \beta_1$的分布，由于$\sigma^2$未知，因此采用式\@ref(eq:model16)的$\hat \sigma^2$进行替代，进而构造t统计量进行区间估计。

$$
\begin{gather}
t=\frac{\hat \beta_1-\beta_1}{\sqrt{\hat \sigma^2/L_{xx}}} \sim t(n-2) \\
P\begin{pmatrix}\begin{vmatrix}\frac{\hat \beta_1-\beta_1}{\sqrt{\hat \sigma^2/L_{xx}}}\end{vmatrix} < t_{\alpha/2}(n-2)\end{pmatrix}=1-\alpha \\
\begin{pmatrix} \hat \beta_1-t_{\alpha/2}(n-2) \sqrt{\frac{\hat \sigma^2}{L_{xx}}}, \; \hat \beta_1+t_{\alpha/2}(n-2) \sqrt{\frac{\hat \sigma^2}{L_{xx}}}\end{pmatrix}
\end{gather}  (\#eq:model58)
$$

> 我们在乎自变量是否能解释因变量的变动，因此$\hat \beta_0$的区间估计，包括下面的显著性检验都不对$\hat \beta_0$进行讨论

### t检验 {#reg_6_2}

#### 一元场合 {#reg_6_2_1}

t检验用于检验单个回归系数是否显著。

对于假设检验问题

$$
H_0:\beta_1=0 \quad vs \quad H_1:\beta_1 \neq 0
$$

在原假设下有$\hat \beta_1 \sim N(0, \sigma^2/L_{xx})$，同样用式\@ref(eq:model16)的$\hat \sigma^2$替代$\sigma^2$，进而构造t统计量进行显著性检验。

$$
t=\frac{\hat \beta_1}{\sqrt{\hat \sigma^2/L_{xx}}} (\#eq:model59)
$$

在原假设下$t \sim t(n-2)$，当$|t| \geq t_{\alpha/2}(n-2)$时拒绝原假设。

### F检验 {#reg_6_3}

#### 一元场合 {#reg_6_3_1}

F检验用于检验整个回归方程是否显著，也就是说检验因变量是否与至少一个自变量存在线性关系。特别的，一元场合只有一个自变量，因此F检验也就相当于检验$\beta_1$是否为0。

对于假设检验问题

$$
H_0:\beta_1=0 \quad vs \quad H_1:\beta_1 \neq 0
$$

构造F统计量

$$
F=\frac{SSR/1}{SSE/(n-2)} (\#eq:model60)
$$

其中$SST=\sum_{i=1}^n(y_i-\bar y)^2, \; SSR=\sum_{i=1}^n(\hat y_i -\bar y)^2, \; SSE=\sum_{i=1}^n(y_i-\hat y_i)^2$。

在原假设下$F \sim F(1,n-2)$，当$F \geq F_\alpha(1,n-2)$时，拒绝原假设。

-------
***平方和分解式***

$$
\begin{aligned}
SST&=\sum_{i=1}^n(y_i-\bar y)^2 \\
&= \sum_{i=1}^n(y_i-\hat y_i+\hat y_i-\bar y)^2 \\
&=\sum_{i=1}^n(\hat y_i - \bar y)^ + \sum_{i=1}^n (y_i-\hat y_i)^2 + 2\sum_{i=1}^n(y_i-\hat y_i)(\hat y_i - \bar y) \\
&= SSR+SSE+2\sum_{i=1}^n(y_i-\hat y_i)\hat y_i - 2\bar y \sum_{i=1}^n (y_i-\hat y_i) \\
&= SSR+SSE+2\sum_{i=1}^n e_i(\hat \beta_0+\hat \beta_1x_i)-2\bar y \sum_{i=1}^n e_i \\
&= SSR+SSE +2\hat \beta_0 \sum_{i=1}^n e_i+2\hat \beta_1 \sum_{i=1}^n e_ix_i \\
&= SSR+SSE
\end{aligned} (\#eq:model61)
$$

注意式\@ref(eq:model12)表明$\sum_{i=1}^n e_i=0, \; \sum_{i=1}^n e_ix_i=0$。

关于式\@ref(eq:model60)的原理，可参考1999年王松桂《线性统计模型：线性回归与方差分析》中的定理4.1.1，该定理包括

(a) $RSS/\sigma^2 \sim \chi^2_{n-p}$

(b) 若约束条件$A\beta=b$成立，则$(RSS_H-RSS/\sigma^2) \sim \chi^2_m$

(c) $RSS$与$RSS_H-RSS$相互独立

(d) 当约束条件$A\beta=b$成立，则

$$
F_H = \frac{(RSS_H-RSS)/m}{RSS/(n-p)} \sim F_{m,n-p}
$$

其中$RSS_H$表示受约束的最小二乘估计对应的残差平方和。

-------

注意到，在一元线性回归中，F统计量与t统计量有如下关系式

$$
t^2=\begin{pmatrix}\frac{\hat \beta_1}{\sqrt{\hat \sigma^2/L_{xx}}}\end{pmatrix}^2=\frac{\hat \beta_1^2L_{xx}}{SSE/(n-2))}=\frac{SSR}{SSE/(n-2)}=F  (\#eq:model66)
$$

其中

$$
\begin{aligned}
SSR&=\sum_{i=1}^n (\hat y_i - \bar y)^2 \\
&= \sum_{i=1}^n (\hat \beta_0 + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n (\bar y - \hat \beta_1 \bar x + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n \hat \beta_1^2(x_i-\bar x)^2 \\
&= \hat \beta_1^2 L_{xx}
\end{aligned} (\#eq:model67)
$$

### 样本决定系数 {#reg_6_4}

样本决定系数定义如下

$$
R^2=\frac{SSR}{SST} (\#eq:model68)
$$

也称拟合优度、判定系数、确定系数。

$R^2$反映了因变量的变异(SST)中可以由自变量解释(SSR)的比例.

> 关于$R^2$这里推荐阅读统计之都的文章[《为什么我不是R方的粉丝》](https://zhuanlan.zhihu.com/p/649208435)

在一元线性回归中，$R^2$与样本相关系数具有如下关系

$$
R^2=\frac{SSR}{SST}=\frac{\hat \beta_1^2 L_{xx}}{L_{yy}}=\frac{L_{xy}^2}{L_{xx}L_{yy}}=r^2 (\#eq:model69)
$$

## 预测 {#reg_7}

### 预测因变量新值的均值 {#reg_7_1}

#### 一元场合 {#reg_7_1_1}

回顾式\@ref(eq:model2)，注意我们的线性回归模型是对$E(y|x)$，即因变量的条件均值进行回归。因此，给定自变量$x_0$，对$E(y|x)$的一个自然的点估计就是

$$
\hat E(y_0|x_0)=\hat y_0=\hat \beta_0 + \hat \beta_1 x_0 (\#eq:model70)
$$

根据式\@ref(eq:model51)，可知

$$
\hat y_0 = \hat \beta_0 + \hat \beta_1x_0 \sim N(\beta_0+\beta_1 x_0, (\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})\sigma^2) (\#eq:model71)
$$

构造枢轴量

$$
\frac{\hat y_0 - E(y_0|x_0)}{\sqrt{\hat \sigma^2(\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}} \sim t(n-2)  (\#eq:model72)
$$

故$E(y_0|x_0)$的区间估计为$\hat y_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat \sigma^2(\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}$。

### 预测因变量的新值 {#reg_7_2}

#### 一元场合  {#reg_7_2_1}

因变量的新值为$y_0$，相较于因变量的均值$E(y_0|x_0)$，我们需要考虑随机扰动项的影响，即$y_0=E(y_0|x_0)+\varepsilon$。

对$y_0$的点估计依旧是$\hat y_0$。

对$y_0$的区间估计则先构造出枢轴量，有

$$
y_0 - \hat y_0 \sim N(0,(1+\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})\sigma^2) (\#eq:model73)
$$

其中方差里的“1”就是纳入了随机扰动项的影响。同样用$\hat \sigma^2$来估计$\sigma^2$，根据t分布得到区间估计为$\hat y_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat \sigma^2(1+\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}$。



## 重要的等式关系 {#reg_8}

1. 一元线性回归中F统计量与t统计量的关系

$$
t^2=\begin{pmatrix}\frac{\hat \beta_1}{\sqrt{\hat \sigma^2/L_{xx}}}\end{pmatrix}^2=\frac{\hat \beta_1^2L_{xx}}{SSE/(n-2))}=\frac{SSR}{SSE/(n-2)}=F 
$$

$$
\begin{aligned}
SSR&=\sum_{i=1}^n (\hat y_i - \bar y)^2 \\
&= \sum_{i=1}^n (\hat \beta_0 + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n (\bar y - \hat \beta_1 \bar x + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n \hat \beta_1^2(x_i-\bar x)^2 \\
&= \hat \beta_1^2 L_{xx}
\end{aligned}
$$

2. 一元线性回归中$R^2$与样本相关系数的关系

$$
R^2=\frac{SSR}{SST}=\frac{\hat \beta_1^2 L_{xx}}{L_{yy}}=\frac{L_{xy}^2}{L_{xx}L_{yy}}=r^2
$$

> 这也可以视作$\hat \beta_1$与样本相关系数的关系
