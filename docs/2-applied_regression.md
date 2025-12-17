# (PART) 模型与方法 {.unnumbered}

# 应用回归分析 {#reg}

回归模型的分类如下所示：


```{=html}
<div class="grViz html-widget html-fill-item" id="htmlwidget-47afbd597e35941a95aa" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-47afbd597e35941a95aa">{"x":{"diagram":"digraph {\n  graph [layout = dot, rankdir = LR]\n  node [shape = box]\n  edge [arrowsize=0.5, headport=w, arrowhead=none]\n  \n  rec1_1 [label = \"回归模型\"]\n  rec2_1 [label = \"参数回归模型\"]\n  rec2_2 [label = \"半参数回归模型\"]\n  rec2_3 [label = \"非参数回归模型\"]\n  rec3_1 [label = \"线性回归模型\"]\n  rec3_2 [label = \"非线性回归模型\"]\n  rec4_1 [label = \"一元线性回归模型\"]\n  rec4_2 [label = \"多元线性回归模型\"]\n  rec4_3 [label = \"一元回归\"]\n  rec4_4 [label = \"多元回归\"]\n  \n  rec1_1 -> {rec2_1 rec2_2 rec2_3}\n  rec2_1 -> {rec3_1 rec3_2}\n  rec3_1 -> {rec4_1 rec4_2}\n  rec3_2 -> {rec4_3 rec4_4}\n  }","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script>
```

回归模型的建模步骤如下所示：


```{=html}
<div class="grViz html-widget html-fill-item" id="htmlwidget-ea5907b2100d5e87d457" style="width:672px;height:480px;"></div>
<script type="application/json" data-for="htmlwidget-ea5907b2100d5e87d457">{"x":{"diagram":"digraph {\n  graph [layout = dot, rankdir = TB]\n  \n  node [shape = box]\n  edge [arrowsize=0.5]\n  \n  rec1_1 [label = \"实际问题\"]\n  rec1_2 [label = \"设置指标变量\"]\n  rec1_3 [label = \"收集整理数据\"]\n  rec1_4 [label = \"构造理论模型\"]\n  rec2_1 [label = \"修改\"]\n  rec2_2 [label = \"模型诊断\"]\n  rec2_3 [label = \"估计模型参数\"]\n  rec3_1 [label = \"模型运用\"]\n  rec4_1 [label = \"影响因素分析\"]\n  rec4_2 [label = \"决策预测\"]\n  rec4_3 [label = \"变量控制\"]\n  \n  rec1_1 -> rec1_2\n  rec1_2 -> rec1_3\n  rec1_3 -> rec1_4\n  rec1_4 -> rec2_3\n  rec2_3 -> rec2_2\n  rec2_2 -> rec2_1[label=\"否\"]\n  rec2_1 -> rec1_2\n  rec2_2 -> rec3_1[label=\"是\"]\n  rec3_1 -> {rec4_1 rec4_2 rec4_3}[arrowhead=none]\n  }","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script>
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

#### 一元场合 {#reg_5_1_1}

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

#### 多元场合 {#reg_5_1_2}

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

对于预测值$\hat y_0$，有

$$
\hat y_0 = x_0'\hat \beta=x_0'(X'X)^{-1}X'Y (\#eq:model74)
$$

对于残差向量$e$，参照式\@ref(eq:model22)，有$e=(I-H)Y=(I-H)\varepsilon$。

### 无偏性 {#reg_5_2}

#### 一元场合  {#reg_5_2_1}

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

#### 多元场合 {#reg_5_2_2}

$$
\begin{aligned}
E(\hat \beta)&=E((X'X)^{-1}X'Y) \\
&= (X'X)^{-1}X'E(Y) \\
&= (X'X)^{-1}X'X\beta \\
&= \beta
\end{aligned} (\#eq:model31)
$$

### 有效性 {#reg_5_3}

#### 一元场合 {#reg_5_3_1}

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

#### 多元场合  {#reg_5_3_2}

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

#### 一元场合  {#reg_5_4_1}

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

#### 多元场合  {#reg_5_4_2}

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

#### 一元场合  {#reg_5_5_1}

根据式\@ref(eq:model26)和式\@ref(eq:model27)、式\@ref(eq:model29)和式\@ref(eq:model30)、式\@ref(eq:model37)和式\@ref(eq:model38)可知，$\hat \beta_1$和$\hat \beta_0$服从正态分布。
   
$$
\begin{gather}
\hat \beta_1 \sim N(\beta_1, \frac{\sigma^2}{L_{xx}}) \\
\hat \beta_0 \sim N(\beta_0, [\frac{1}{n}+\frac{\bar x^2}{L_{xx}}]\sigma^2) 
\end{gather} (\#eq:model48)
$$

> $\hat \beta_1$的正态性源自y的正态性，而y的正态性又源自$\varepsilon$的正态性
> 
> 因此，凡是能被$y$线性表示的都具有正态性，详见多元场合

#### 多元场合  {#reg_5_5_2}

凡是能被$Y$线性表示的都具有正态性。
   
$$
\begin{gather}
\hat \beta \sim N(\beta, \sigma^2(X'X)^{-1}) \\
\hat Y = HY \sim N(X\beta, \sigma^2H) \\
\hat y_0 = x_0'(X'X)^{-1}X'Y \sim N(x_0'\beta, \sigma^2x_0'(X'X)^{-1}x_0) \\
e=(I-H)Y \sim N(0, \sigma^2(I-H))
\end{gather} (\#eq:model49)
$$

### 残差 {#reg_5_6}

#### 一元场合  {#reg_5_6_1}

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
   &= -Cov(y_i \, , \, h_{ji}y_i)-Cov(y_j \, , \, h_{ij}y_j)+\sum\limits_{k=1}^n h_{ik}h_{jk}Cov(y_k \, , \, y_k) \\
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
   

#### 多元场合 {#reg_5_6_2}

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
   &=\sigma^2(I-H)H \\
   &=\sigma^2(H-H^2) \\
   &=0
   \end{aligned} (\#eq:model55)
   $$
   
- $Cov(e,\hat y_0)$
   
   $$
   \begin{aligned}
   Cov(e,\hat y_0)&=Cov((I-H)Y,x_0'(X'X)^{-1}X'Y) \\
   &=\sigma^2(I-H)X(X'X)^{-1}x_0 \\
   &=\sigma^2(X(X'X)^{-1}-X(X'X)^{-1}X'X(X'X)^{-1})x_0 \\
   &=0
   \end{aligned} (\#eq:model56)
   $$
   
- $Cov(e,\hat \beta)$
   
   $$
   \begin{aligned}
   Cov(e,\hat \beta)&=Cov((I-H)Y,(X'X)^{-1}X'Y) \\
   &= \sigma^2(I-H)X(X'X)^{-1} \\
   &= \sigma^2[X(X'X)^{-1}-X(X'X)^{-1}X'X(X'X)^{-1}] \\
   &= 0
   \end{aligned} (\#eq:model57)
   $$
   
> 由于$e$和$\hat \beta$都是$Y$的线性组合，因此都服从正态分布，故协方差为0表示$e$和$\hat \beta$之间独立，同样的也有SSE或$\hat \sigma^2$与$\hat \beta$独立
   
- $X'e=0$
   
   参见式\@ref(eq:model20)
      
> 在正规方程组中，我们得到的一阶导条件为$X'e=0$。从几何视角来看，残差向量$e$正交于X张成的列空间，因此凡是X列空间中的向量均与$e$不相关

## 显著性检验 {#reg_6}

由于显著性检验依赖于最小二乘估计的分布，在前述内容中已经说明最小二乘估计服从正态分布，因此该部分内容严重依赖于随机扰动项的正态性、同方差、无自相关假定。

同时，在显著性检验中涉及t检验和F检验，这就依赖如下条件

$$
\frac{SSE}{\sigma^2} \sim \chi^2(n-p-1) (\#eq:model78) 
$$

证：

   $$
   \begin{aligned}
   \frac{SSE}{\sigma^2}&=\frac{e'e}{\sigma^2} \\
   &=\frac{\varepsilon'(I-H)\varepsilon}{\sigma^2} \\
   &=\frac{\varepsilon'}{\sigma}(I-H)\frac{\varepsilon}{\sigma} 
   \end{aligned} (\#eq:model79) 
   $$

   已知$\varepsilon \sim N(0, \sigma^2I)$，则$\frac{\varepsilon}{\sigma} \sim N(0,I)$。

   由于式\@ref(eq:model79)为二次型，且矩阵$(I-H)$为秩为$n-p-1$的对称幂等矩阵，故存在某种正交变换使得式\@ref(eq:model79)的二次型化为相互独立的变量平方和，也就是卡方分布，其中自由度就是矩阵$(I-H)$的秩。
   同时，根据式\@ref(eq:model57)可知$\hat \beta$与$SSE/\sigma^2$独立。

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

#### 多元场合 {#reg_6_1_2}

回顾式\@ref(eq:model49)，可知$\hat \beta_j \sim N(\beta_j, \sigma^2c_{jj})$，其中$c_{jj}$表示$(X'X)^{-1}$的第$j+1$个主对角线元素，故有

$$
\begin{gather}
t=\frac{\hat \beta_j-\beta_j}{\sqrt{\hat \sigma^2c_{jj}}} \sim t(n-p-1) \\
P\begin{pmatrix}\begin{vmatrix}\frac{\hat \beta_j-\beta_j}{\sqrt{\hat \sigma^2c_{jj}}}\end{vmatrix} < t_{\alpha/2}(n-p-1)\end{pmatrix}=1-\alpha \\
\begin{pmatrix} \hat \beta_j-t_{\alpha/2}(n-p-1) \sqrt{\hat \sigma^2 c_{jj}}, \; \hat \beta_j+t_{\alpha/2}(n-p-1) \sqrt{\hat \sigma^2 c_{jj}}\end{pmatrix}
\end{gather}  (\#eq:model75)
$$

> 挖坑，回归系数向量的置信域（置信椭球）

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

#### 多元场合 {#reg_6_2_2}

对于假设检验问题

$$
H_0:\beta_j=0 \quad vs \quad H_1:\beta_j \neq 0
$$

在原假设下有$\hat \beta_j \sim N(0,\sigma^2 c_{jj})$，故构造检验统计量

$$
t_j=\frac{\hat \beta_j}{\sqrt{\hat \sigma^2 c_{jj}}}  (\#eq:model76)
$$

在原假设下$t_j \sim t(n-p-1)$，当$|t_j| \geq t_{\alpha/2}(n-p-1)$时拒绝原假设。

考虑更一般的假设检验问题

$$
H_0:c'\beta=0 \quad vs \quad H_1:c'\beta \neq 0
$$
 
有$c'\hat \beta \sim N(c'\beta, \sigma^2c'(X'X)^{-1}c)$，故

$$
t=\frac{c'\hat \beta}{\sqrt{\hat \sigma^2c'(X'X)^{-1}c}} (\#eq:model77)
$$

原假设下有$t \sim t(n-p-1)$，当$|t| \geq t(n-p-1)$时拒绝原假设

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

#### 多元场合 {#reg_6_3_2}

整个回归方程的显著性检验同样采用F检验进行。

对于假设检验问题

$$
H_0: \beta_1=...=\beta_p=0 \quad vs \quad H_1: \exists \beta_i \neq 0, \; i\in \{1,...,p\}
$$

构造F统计量

$$
F=\frac{SSR/p}{SSE/(n-p-1)} (\#eq:model81)
$$

在原假设下，$F\sim F(p,n-p-1)$，当$F \geq F_\alpha(p,n-p-1)$时即可拒绝原假设。

***下面对F检验进行推广。***

考虑部分回归系数的显著性检验问题，不妨令$\beta_2$为$\beta$中假设系数为0的那部分系数，对应的自变量有$p^*$个，记为$X_2$。剩余的系数和自变量个数为$\beta_1$和$p-p^*$个，自变量记为$X_1$。

> 更一般的线性假设问题及证明可参考1999年王松桂《线性统计模型：线性回归与方差分析》中的4.1节，其中的线性假设为$A\beta=b$

对于假设检验问题

$$
H_0:\beta_2=0 \quad vs \quad H_1:\beta_2 \neq 0
$$

对于同一样本，无约束回归与有约束回归对应的$SST$都是一致的。而在约束条件$\beta_2=0$下，对应的残差平方和$SSE^*$必定大于等于无约束条件下的残差平方和$SSE$，即$SSE^* \geq SSE$。注意到有$SSR-SSR^*=SSE^*-SSE$，结合式\@ref(eq:model78)，在原假设下有

$$
F=\frac{(SSE^*-SSE)/(p-p^*)}{SSE/(n-p-1)} \sim F(p-p^*,n-p-1) (\#eq:model80)
$$

> $SSE^{\ast} / \sigma^2 \sim \chi^2(n-p^{\ast}-1)$
> 
> $SSE/\sigma^2 \sim \chi^2(n-p-1)$ 
> 
> $(SSE^\ast-SSE) / \sigma^2 \sim \chi^2(p-p^\ast)$

因此，该检验统计量通过度量$SSE^*-SSE$的差异大小来检验约束条件是否显著存在。若约束条件真的存在，则$SSE^*-SSE$之间的差异自然就小；若约束条件不存在，则$SSE^*-SSE$之间的差异自然就大。

-----

***多元场合的平方和分解式的表达***

$$
\begin{gather}
SST=\sum_{i=1}^n(y_i-\bar y)^2=\sum_{i=1}^n [(1-\frac{1}{n})y_i -\frac{1}{n}\sum_{j \neq i}y_j]^2=Y'(I-\frac{1}{n}1_n1_n')Y \\
SSE=\sum_{i=1}^n(y_i-\hat y_i)^2=Y'(I-H)Y \\
SSR=SST-SSE=Y'(H-\frac{1}{n}1_n1_n')Y
\end{gather} (\#eq:model82)
$$

其中$1_n$表示长度为n且元素均为1的向量。

-----

### 偏F检验 {#reg_6_4}

在多元场合中，根据式\@ref(eq:model80)的启示，可以假设某一自变量对应的回归系数为0，根据约束前后残差平方和的差异大小来判断该自变量的重要性，称此检验为**偏F检验**。

假设检验问题为

$H_0:\beta_j=0 \quad vs \quad H_1:\beta_j \neq 0$

则检验统计量为

$$
F_j=\frac{(SSE_{(-j)}-SSE)/1}{SSE/(n-p-1)} (\#eq:model83)
$$

> $SSE_{(-j)}=Y'(I-H_0)Y, \; H_0=X_0(X_0'X_0)^{-1}X_0'$，其中$X_0$表示剔除变量$x_j$后的设计矩阵

其中$SSE_{(-i)}$表示去掉第$i$个自变量后所拟合模型的残差平方和。在原假设下，由$F_i \sim F(1,n-p-1)$。当$F \geq F_\alpha(1,n-p-1)$时拒绝原假设。

若约束前后残差平方和变化过大，说明该自变量较为重要，此时$F_i$的值会较大，倾向于拒绝原假设。

$\beta_j$的t检验统计量与偏F检验统计量有如下关系

$$
t_j^2=F_j  (\#eq:model84)
$$

证：挖坑

### 样本决定系数 {#reg_6_5}

样本决定系数定义如下

$$
R^2=\frac{SSR}{SST}=\frac{\sum_{i=1}^n (\hat y_i-\bar y)^2}{\sum_{i=1}^n (y_i-\bar y)^2} (\#eq:model68)
$$

也称拟合优度、判定系数、确定系数。

$R^2$反映了因变量的变异(SST)中可以由自变量解释(SSR)的比例.

> 关于$R^2$这里推荐阅读统计之都的文章[《为什么我不是R方的粉丝》](https://zhuanlan.zhihu.com/p/649208435)

#### 一元场合 {#reg_6_5_1}

在一元线性回归中，$R^2$与样本相关系数具有如下关系

$$
R^2=\frac{SSR}{SST}=\frac{\hat \beta_1^2 L_{xx}}{L_{yy}}=\frac{L_{xy}^2}{L_{xx}L_{yy}}=r^2 (\#eq:model69)
$$

#### 多元场合 {#reg_6_5_2}

在多元场合中，样本决定系数$R^2$与$Cor(\hat Y, Y)$具有如下关系

$$
\begin{aligned}
Cor(\hat Y, Y)&=\frac{(\hat Y - 1_n\bar y)'(Y-1_n\bar y)}{\sqrt{SSR \times SST}} \\
&= \frac{(\hat Y - 1_n\bar y)'(\hat Y + e -1_n\bar y)}{\sqrt{SSR \times SST}} \\
&= \frac{(\hat Y - 1_n\bar y)'(\hat Y-1_n\bar y)+(\hat Y - 1_n\bar y)'e}{\sqrt{SSR \times SST}} \\
&= \frac{SSR+0}{\sqrt{SSR \times SST}} \\
&= \sqrt{\frac{SSR}{SST}} \\
&= \sqrt{R^2}
\end{aligned} (\#eq:model85)
$$

> $e$与$\hat Y$正交，且$\sum_{i=1}^n e_i=0$

定义样本复相关系数为

$$
R=\sqrt{R^2}=\sqrt{\frac{SSR}{SST}}  (\#eq:model86)
$$

> 反映了因变量与一组自变量间的相关性

定义调整的$R^2$为

$$
R_{adj}^2 = 1-\frac{SSE/(n-p-1)}{SST/(n-1)}=1-\frac{n-1}{n-p-1}(1-R^2) (\#eq:model86)
$$

普通$R^2$会随着自变量的增加而单调增加，而调整的$R^2$相较于普通$R^2$多了对自变量个数的惩罚，因此可用于不同自变量个数下不同模型之间拟合效果的比较。

## 预测 {#reg_7}

### 预测因变量新值的均值 {#reg_7_1}

#### 一元场合 {#reg_7_1_1}

回顾式\@ref(eq:model2)，注意我们的线性回归模型是对$E(y|x)$，简记为$E(y)$，即对因变量的条件均值进行回归。因此，给定自变量$x_0$，对$E(y)$的一个自然的点估计就是

$$
\hat E(y_0)=\hat y_0=\hat \beta_0 + \hat \beta_1 x_0 (\#eq:model70)
$$

根据式\@ref(eq:model51)，可知

$$
\hat y_0 = \hat \beta_0 + \hat \beta_1x_0 \sim N(\beta_0+\beta_1 x_0, (\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})\sigma^2) (\#eq:model71)
$$

构造枢轴量

$$
\frac{\hat y_0 - E(y_0)}{\sqrt{\hat \sigma^2(\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}} \sim t(n-2)  (\#eq:model72)
$$

故$E(y_0)$的区间估计为$\hat y_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat \sigma^2(\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}$。

#### 多元场合 {#reg_7_1_2}

同样，一个自然的点估计就是$\hat y_0$。

在正态假设下，有

$$
\begin{gather}
\hat y_0 = x_0'(X'X)^{-1}X'Y \sim N(x_0'\beta, \sigma^2x_0'(X'X)^{-1}x_0) \\
\hat y_0-E(y_0) \sim N(0, \sigma^2x_0'(X'X)^{-1}x_0) \\
t=\frac{\hat y_0 -E(y_0)}{\sqrt{\hat \sigma^2x_0'(X'X)^{-1}x_0}} \sim t(n-p-1) \\
\hat y_0 \pm t_{\alpha/2}(n-p-1)\sqrt{\hat \sigma^2x_0'(X'X)^{-1}x_0}
\end{gather} (\#eq:model87)
$$

### 预测因变量的新值 {#reg_7_2}

#### 一元场合  {#reg_7_2_1}

因变量的新值为$y_0$，相较于因变量的均值$E(y_0)$，我们需要考虑随机扰动项的影响，即$y_0=E(y_0)+\varepsilon$。

对$y_0$的点估计依旧是$\hat y_0$。

对$y_0$的区间估计则先构造出枢轴量，有

$$
y_0 - \hat y_0 \sim N(0,(1+\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})\sigma^2) (\#eq:model73)
$$

其中方差里的“1”就是纳入了随机扰动项的影响。同样用$\hat \sigma^2$来估计$\sigma^2$，根据t分布得到区间估计为$\hat y_0 \pm t_{\alpha/2}(n-2)\sqrt{\hat \sigma^2(1+\frac{1}{n}+\frac{(x_0-\bar x)^2}{L_{xx}})}$。

#### 多元场合  {#reg_7_2_2}

同样，点估计为$\hat y_0$。

在正态性假设下，有

$$
\begin{gather}
\hat y_0 = x_0'(X'X)^{-1}X'Y \sim N(x_0'\beta, \sigma^2x_0'(X'X)^{-1}x_0) \\
y_0-\hat y_0 \sim N(0, \sigma^2(1+x_0'(X'X)^{-1}x_0)) \\
t=\frac{y_0-\hat y_0}{\sqrt{\hat \sigma^2(1+x_0'(X'X)^{-1}x_0})} \sim t(n-p-1) \\
\hat y_0 \pm t_{\alpha/2}(n-p-1)\sqrt{\hat \sigma^2(1+x_0'(X'X)^{-1}x_0)}
\end{gather} (\#eq:model88)
$$

## 回归系数的解释 {#reg_8}

对于多元线性回归模型

$$
\left\{
\begin{array}{c}
E(y|x)=\beta_0+\beta_1x+...+\beta_px_p\\
Var(y|x)=\sigma^2   
\end{array}
\right.
$$

截距项$\beta_0$反映了当自变量均取0时因变量的期望。

而对于自变量的回归系数，**理论上来说**，<span style='color:red'>$\beta_i$表示当固定其他自变量不变时，$x_i$每增加一个单位，因变量的期望能够变化$\beta_i$个单位</span>。**实际上**，自变量之间往往具有相关性，可能无法做到固定某些自变量的值而改变其他自变量的值。也就是说，自变量之间所提供的信息是有**重叠**的部分。

考虑自变量$X_i$的影响，记其余自变量对应的设计矩阵为$X_0$，对应的帽子矩阵为$H_0=X_0(X_0'X_0)^{-1}X_0'$，则

$$
\begin{aligned}
Y&=X_i\beta_i+X_0\beta_0+\varepsilon \\
(I-H_0)Y&=(I-H_0)X_i\beta_i+(I-H_0)X_0\beta_0+(I-H_0)\varepsilon \\
e_{Y|X_0}&=e_{X_i|X_0}\beta_i+(I-H_0)\varepsilon
\end{aligned}  (\#eq:model89)
$$

> 一般的多元线性回归中$e=(I-H)Y$，即$(I-H)$的作用是为了得到残差，而$H$则决定了是对谁而言的残差。像这里的$H_0$是对$X_0$而言的，也就是经过$X_0$调整后的残差

其中$e_{a|b}$表示a对b回归得到的残差，即a中不能由b线性解释的部分，称为“经过b调整后的a”。

上式表明原始多元线性回归中的$\beta_i$与经过$X_0$调整过后的$Y$对经过$X_0$调整过后的$X_i$回归得到的回归系数是一致的。

特别的，经过$X_0$调整过后的$Y$对经过$X_0$调整过后的$X_i$回归的最小二乘估计为

$$
\hat \beta_i = (X_i'(I-H_0)X_i)^{-1}X_i'(I-H)Y  (\#eq:model90)
$$

> 注意有$(I-H_0)=(I-H_0)^2$

既然原始多元线性回归中的$\beta_i$与经过$X_0$调整过后的$Y$对经过$X_0$调整过后的$X_i$回归得到的回归系数是一致的，那么<span style='color: red'>$\beta_i$也反映了经过其余自变量线性调整后$x_i$对$y$额外的贡献，也称$\beta_i$为**偏回归系数**</span>。

于是，称$e_{Y|X_0}$与$e_{X_i|X_0}$的散点图为**偏回归图**或**附加变量图**。

> 对该图拟合最小二乘回归线，其斜率就是$\hat \beta_i$
> 
> 若附加变量图中的线性关系越强，说明新增变量$x_i$对已包含其余变量的回归方程增加的贡献就越大

## 中心化与标准化 {#reg_9}

各个变量的量纲不同，会导致原始设计矩阵的数值差异较大，基于该设计矩阵得到的最小二乘估计不具有可比性。

### 中心化 {#reg_9_1}

中心化处理，即变量减去其均值。中心化的意义能够将未知参数的个数降低1，并在一定程度上降低舍入误差。

记$X=\begin{pmatrix}1_n & \tilde X\end{pmatrix}, \; \beta = \begin{pmatrix}\beta_0 \\ \tilde \beta \end{pmatrix}, \; \gamma = \begin{pmatrix}\gamma_0 \\ \tilde \gamma \end{pmatrix}, \; \alpha = \begin{pmatrix}\alpha_0 \\ \tilde \alpha \end{pmatrix}, \;\bar X = \begin{pmatrix} \bar x_1 & \cdots & \bar x_p \end{pmatrix}'$。

> $X$的第一列均为1，$\tilde X$才是纯粹的自变量矩阵，注意区分

则

$$
\begin{gather}
\tilde X_c = \begin{pmatrix}I-\frac{1}{n}1_n1_n' \end{pmatrix}\tilde X \\
Y_c = Y-1_n\bar y
\end{gather} (\#eq:model91)
$$

> $1_n$表示长度为n且元素均为1的列向量

其中$(I-\frac{1}{n}1_n1_n')$为中心化矩阵，下标$c$表示经过中心化处理后的矩阵或向量。

对此得到如下样本回归模型及相应的最小二乘估计

- 原始模型

$$
\begin{gather}
E(Y)=1_n\beta_0+\tilde X \tilde \beta \\
\\ 
\begin{pmatrix} \hat \beta_0 \\ \hat{\tilde \beta} \end{pmatrix}=\begin{pmatrix} \bar y - \bar X'\hat{\tilde \beta} \\ (\tilde X_c'\tilde X_c)^{-1}\tilde X_c' Y \end{pmatrix}
\end{gather} (\#eq:model91)
$$

- 对X进行中心化处理

$$
\begin{gather}
E(Y)=1_n\gamma_0+\tilde X_c \tilde \gamma \\
\\ 
\begin{pmatrix} \hat \gamma_0 \\ \hat{\tilde \gamma} \end{pmatrix}=\begin{pmatrix} \bar y \\ (\tilde X_c'\tilde X_c)^{-1}\tilde X_c' Y \end{pmatrix}=\begin{pmatrix} \bar y \\ \hat{\tilde \beta} \end{pmatrix}
\end{gather} (\#eq:model92)
$$

- 对Y和X进行中心化处理

$$
\begin{gather}
E(Y_c)=1_n\alpha_0+\tilde X_c \tilde \alpha \\
\\ 
\begin{pmatrix} \hat \alpha_0 \\ \hat{\tilde \alpha} \end{pmatrix}=\begin{pmatrix} 0 \\ (\tilde X_c'\tilde X_c)^{-1}\tilde X_c' Y \end{pmatrix}=\begin{pmatrix} 0 \\ \hat{\tilde \beta} \end{pmatrix}
\end{gather} (\#eq:model93)
$$

小结：

1. 仅对X进行中心化处理，则斜率项的估计不变，截距项的估计值变为$\bar y$。

2. 对Y和X进行中心化处理，则斜率项的估计不变，截距项的估计值变为0。

3. **自变量和因变量任何形式的位移变化均不改变斜率项的估计值，继而也不改变线性回归模型的拟合优度**。

### 标准化 {#reg_9_2}

变量减去其均值并除以其标准差即为**标准化处理**。标准化处理能够消除量纲不同和数量级差异所带来的影响。

沿用[中心化](##reg_9_1)的记号，并记$D_X=diag(sd(x_1),...,sd(x_p))$。

则

$$
\begin{gather}
\tilde X^*=\tilde X_cD_X^{-1}=(I-\frac{1}{n}1_n1_n')\tilde X D_X^{-1} \\
Y^*=\frac{Y_c}{sd(y)}=\frac{1}{sd(y)}(I-\frac{1}{n}1_n1_n')Y
\end{gather} (\#eq:model94)
$$

对此得到如下样本回归模型及相应的最小二乘估计

- 仅对X进行标准化处理

$$
\begin{gather}
E(Y)=1_n\delta_0+\tilde X^*\tilde \delta \\
 \\ 
\begin{pmatrix} \hat \delta_0 \\ \hat{\tilde \delta} \end{pmatrix}= \begin{pmatrix} \bar y \\ (\tilde{X^{\ast}}'\tilde{X^\ast})^{-1}\tilde{X^{\ast}}'Y \end{pmatrix} = \begin{pmatrix} \bar y \\ D_X\hat{\tilde \beta} \end{pmatrix} 
\end{gather}  (\#eq:model95)
$$

- 对Y和X进行标准化处理

$$
\begin{gather}
E(Y^*)=1_n\eta_0+\tilde X^*\tilde \eta \\
 \\ 
\begin{pmatrix} \hat \eta_0 \\ \hat{\tilde \eta} \end{pmatrix}= \begin{pmatrix} 0 \\ (\tilde{X^{\ast}}'\tilde{X^\ast})^{-1}\tilde{X^{\ast}}'Y^* \end{pmatrix} = \begin{pmatrix} 0 \\ \frac{1}{sd(y)}D_X\hat{\tilde \beta} \end{pmatrix} 
\end{gather}  (\#eq:model96)
$$

小结：

1. 仅对X进行标准化处理，由于标准化处理中包含了中心化处理，因此截距项为$\bar y$，而斜率项则为原来的$sd(x_i)$倍。

> $\beta_i^\ast \frac{x_i}{sd(x_i)}=\frac{\beta_i^\ast}{sd(x_i)}x_i=\beta_i x_i$，故$\beta_i^*=sd(x_i)\beta_i$

2. 对Y和X进行标准化处理，则截距项变为0，斜率项为原来的$\frac{sd(x_i)}{sd(y)}$倍。

> $\frac{sd(y)}{sd(x_i)} \beta_i^\ast=\beta_i$，故$\beta_i^\ast = \frac{sd(x_i)}{sd(y)} \beta_i$

3. 标准化涉及到尺度变换和位移变化，因此既有中心化的特征（截距项），又有倍数关系（斜率项）。

## 相关系数与偏相关系数 {#reg_10}

### 样本相关系数 {#reg_10_1}

定义两个变量间的相关系数

$$
r=\frac{\sum_{i=1}^n (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_{i=1}^n (x_i-\bar x)^2\sum_{i=1}^n (y_i-\bar y)^2}}=\frac{L_{xy}}{\sqrt{L_{xx}L_{yy}}}  (\#eq:model97)
$$

样本相关系数反映了两个变量间<span style='color:red'>**线性关系**</span>的密切程度。特别的，样本相关系数为0并不意味着两个变量之间没有任何关系，只是没有线性相关关系。样本相关系数的大小与样本量有关，当样本量较小时，$|r|$容易接近1，当样本量较大时，$|r|$容易偏小。

#### 样本相关系数的显著性检验 {#reg_10_1_1}

考虑两个变量间的一元线性回归模型。回顾式\@ref(eq:model69)与式\@ref(eq:model60)，可得

$$
\begin{aligned}
F&=\frac{SSR/1}{SSE/(n-2)} \\
&= (n-2)\frac{SSR/SST}{SSE/SST} \\
&= (n-2)\frac{r^2}{1-r^2} \\
r^2&=\frac{F}{F+n-2}
\end{aligned} (\#eq:model98)
$$

故样本相关系数的显著性检验可通过一元场合的F检验进行，其中$F \sim F(1,n-2)$，原假设为$\rho=0$。

或者根据一元场合式\@ref(eq:model59)可得

$$
\begin{aligned}
t&=\frac{\hat \beta_1}{\sqrt{\hat \sigma^2/L_{xx}}} \\
&= \frac{\hat \beta_1\sqrt{L_{xx}}}{\sqrt{SSE/(n-2)}} \\
&= \frac{\sqrt{n-2}\frac{L_{xy}}{\sqrt{L_{xx}L_{yy}}}}{\sqrt{SSE/L_{yy}}} \\
&= \frac{\sqrt{(n-2)}r}{\sqrt{1-r^2}}
\end{aligned}  (\#eq:model99)
$$

> 一元场合中还有式\@ref(eq:model66)的关系:$t^2=F$

此时可根据$t \sim t(n-2)$的显著性检验，原假设为$\rho=0$。

### 样本偏相关系数 {#reg_10_2}

简单样本相关系数度量了两个变量间的相关性。但在多元相关分析中，由于受到其他变量的影响，简单样本相关系数并不能反映两个变量间纯粹的相关性，需要控制其他变量的影响，对此引入样本偏相关系数。

- 自变量间的样本偏相关系数

在样本相关阵r中，记$\Delta_{ij}$为r的第i行第i列元素的代数余子式，以$x_1$与$x_2$为例，定义样本偏相关系数为

$$
r_{12;3,...p}=\frac{-\Delta_{12}}{\sqrt{\Delta_{11}\Delta_{22}}} (\#eq:model100)
$$

- 因变量与自变量的样本偏相关系数

记除$x_1$之外的自变量为$x_{(-1)}$，$e_{x_1|x_{(-1)}}$和$e_{y|x_{(-1)}}$分别表示$x_1$和$y$对$x_{(-1)}$回归的残差，定义$y$与$x_1$的样本偏相关系数为

$$
r_{y1;2,...p}=\frac{Cov(e_{x_1|x_{(-1)}},e_{y|x_{(-1)}})}{\sqrt{Var(e_{x_1|x_{(-1)}})Var(e_{y|x_{(-1)}})}}=Cor(e_{x_1|x_{(-1)}},e_{y|x_{(-1)}}) (\#eq:model101)
$$

> 控制其他变量的影响就是考虑这些变量回归后的残差

- 因变量与自变量的样本偏决定系数

定义$y$与$x_1$的样本偏决定系数为

$$
r_{y1;2,...p}^2 = \frac{SSE_{(-1)}-SSE}{SSE_{(-1)}}  (\#eq:model102)
$$

> 若把$SSE_{(-1)}$看成$SST$，则形式同$R^2$

样本偏决定系数反映了引入该新自变量后，因变量剩余变差的相对减少了。

则$y$与$x_1$的样本偏相关系数也可为

$$
r_{y1;2,...p} = \sqrt{ \frac{SSE_{(-1)}-SSE}{SSE_{(-1)}}} (\#eq:model103)
$$

正如样本相关系数与F统计量有关系，这里的样本偏相关系数（或者说样本偏决定系数）也与偏F统计量有关系。

同式\@ref(eq:model98)，若把$SSE_{(-1)}$看成$SST$，则同理有

> $SST=\sum_{i=1}^n (y_i-\bar y)^2$又何尝不是均值模型的残差平方和呢？

$$
\begin{aligned}
F_1&=\frac{(n-p-1)r_{y1;2,...p}^2}{1-r_{y1;2,...p}^2} \\
r_{y1;2,...p}^2 &= \frac{F_1}{F_1+n-p-1}
\end{aligned} (\#eq:model104)
$$

## 重要的定义和等式 {#reg_11}

1. Gauss-Markov条件

$$
\begin{gather}
E(\varepsilon)=0 \\
Var(\varepsilon_i)=\sigma^2 \\
Cov(\varepsilon_i, \varepsilon_j)=0, \; i \neq j
\end{gather}
$$

2. 一元回归中$\hat \beta_1$与样本相关系数$r$、回归平方和$SSR$

$$
\begin{aligned}
r&=\frac{L_{xy}}{\sqrt{L_{xx}L_{yy}}} \\
&= \hat \beta_1 \sqrt{\frac{L_{xx}}{L_{yy}}} \\
SSR&=\sum_{i=1}^n (\hat y_i -\bar y)^2 \\
&=\sum_{i=1}^n (\hat \beta_0+\hat \beta_1x_i-\bar y)^2 \\
&=\sum_{i=1}^n (\hat \beta_1x_i+\bar y -\hat \beta_1 \bar x - \bar y)^2 \\
&= \hat \beta_1^2 \sum_{i=1}^n (x_i-\bar x)^2 \\
&= \hat \beta_1^2 L_{xx}
\end{aligned}
$$

3. 一元场合的线性系数$h_{ij}$

$$
\begin{gather}
h_{ij}=\frac{1}{n}+\frac{(x_i-\bar x)(x_j -\bar x)}{L_{xx}}=h_{ji}  \\
\hat y_i = \sum_{j=1}^n h_{ij}y_j \\
\hat y_0=\sum_{j=1}^n h_{0j}y_j \\
e_i = y_i - \sum_{j=1}^n h_{ij}y_j \\
\sum_{j=1}^n h_{ij}^2 = h_{ii} \\
\sum_{k=1}^n h_{ik}h_{jk} = h_{ij}
\end{gather}
$$

   特别的，称$h_{ii}$为杠杆值，度量了自变量空间中第i个数据偏离数据中心的程度。
   
   <span style='color:red'>特别的，在证明最小二乘估计的性质时基本上都要将这些估计量转化为y的线性表达，无论一元还是多元</span>。

4. 方差分析表

   列：方差来源|自由度|平方和|均方|F值|p值

5. 一元线性回归中F统计量与t统计量的关系

$$
t^2=\begin{pmatrix}\frac{\hat \beta_1}{\sqrt{\hat \sigma^2/L_{xx}}}\end{pmatrix}^2=\frac{\hat \beta_1^2L_{xx}}{SSE/(n-2))}=\frac{SSR}{SSE/(n-2)}=F 
$$

其中

$$
\begin{aligned}
SSR&=\sum_{i=1}^n (\hat y_i - \bar y)^2 \\
&= \sum_{i=1}^n (\hat \beta_0 + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n (\bar y - \hat \beta_1 \bar x + \hat \beta_1x_i-\bar y)^2 \\
&= \sum_{i=1}^n \hat \beta_1^2(x_i-\bar x)^2 \\
&= \hat \beta_1^2 L_{xx}
\end{aligned}
$$

6. 一元线性回归中$R^2$与样本相关系数$r$的关系

$$
R^2=\frac{SSR}{SST}=\frac{\hat \beta_1^2 L_{xx}}{L_{yy}}=\frac{L_{xy}^2}{L_{xx}L_{yy}}=r^2
$$

> 这也可以视作$\hat \beta_1$与样本相关系数的关系

7. 帽子矩阵或投影矩阵

$$
H=X(X'X)^{-1}X'
$$

   矩阵$H$为对称幂等矩阵，即$H'=H, \; H^2=H$。$I-H$也是对称幂等矩阵。对称幂等矩阵的**秩**和**迹**相等。
   
   帽子矩阵的元素就是前面提到的线性系数$h_{ij}$。
   
8. 中心化矩阵

$$
I-\frac{1}{n}1_n1_n'
$$

9. 多元场合的平方和分解式

$$
\begin{gather}
SST=\sum_{i=1}^n(y_i-\bar y)^2=\sum_{i=1}^n [(1-\frac{1}{n})y_i -\frac{1}{n}\sum_{j \neq i}y_j]^2=Y'(I-\frac{1}{n}1_n1_n')Y \\
SSE=\sum_{i=1}^n(y_i-\hat y_i)^2=Y'(I-H)Y \\
SSR=SST-SSE=Y'(H-\frac{1}{n}1_n1_n')Y
\end{gather}
$$

10. 偏F检验统计量

$$
F_j = \frac{(SSE_{(-j)}-SSE)/1}{SSE/(n-p-1)}
$$

11. t检验统计量与偏F统计量的关系

$$
t_j^2=F_j
$$

> 以$SSE_{(-j)}$为中介，为$t_j$检验统计量与样本偏决定系数之间建立了联系。注意$SSE$可通过标准误求得

12. 调整的$R^2$

$$
R_{adj}^2 = 1-\frac{SSE/(n-p-1)}{SST/(n-1)}=1-\frac{n-1}{n-p-1}(1-R^2)
$$

13. 样本决定系数与$Cor(\hat Y, Y)$

$$
\begin{aligned}
Cor(\hat Y, Y)&=\frac{(\hat Y - 1_n\bar y)'(Y-1_n\bar y)}{\sqrt{SSR \times SST}} \\
&= \frac{(\hat Y - 1_n\bar y)'(\hat Y + e -1_n\bar y)}{\sqrt{SSR \times SST}} \\
&= \frac{(\hat Y - 1_n\bar y)'(\hat Y-1_n\bar y)+(\hat Y - 1_n\bar y)'e}{\sqrt{SSR \times SST}} \\
&= \frac{SSR+0}{\sqrt{SSR \times SST}} \\
&= \sqrt{\frac{SSR}{SST}} \\
&= \sqrt{R^2} \\
Cor(\hat Y, Y)&=\frac{\sum_{i=1}^n(\hat y_i-\bar y)(y_i-\bar y)}{\sqrt{\sum_{i=1}^n(\hat y_i-\bar y)^2\sum_{i=1}^n(y_i-\bar y)^2}}=\frac{L_{\hat yy}}{\sqrt{L_{\hat y \hat y}L_{yy}}}
\end{aligned}
$$

14. 样本复相关系数

$$
R=\sqrt{R^2}=\sqrt{\frac{SSR}{SST}}
$$

15. 原始多元线性回归中的$\beta_i$与经过$X_0$调整过后的$Y$对经过$X_0$调整过后的$X_i$回归得到的回归系数是一致的

$$
(I-H_0)Y=(I-H_0)X_i\beta_i+(I-H_0)\varepsilon
$$

16. 相关系数的显著性检验

   考虑一元线性回归中的t检验和F检验，根据$r^2=R^2=\frac{SSR}{SST}$的关系式让t统计量和F统计量转化成对应的形式即可。
   
17. 样本偏相关系数、样本偏决定系数

   - 自变量间的样本偏相关系数
   
   $$
   r_{12;3,...p}=\frac{-\Delta_{12}}{\sqrt{\Delta_{11}\Delta_{22}}}
   $$
   
   - 因变量与自变量的样本偏相关系数
   
   $$
   r_{y1;2,...p}=\frac{Cov(e_{x_1|x_{(-1)}},e_{y|x_{(-1)}})}{\sqrt{Var(e_{x_1|x_{(-1)}})Var(e_{y|x_{(-1)}})}}=Cor(e_{x_1|x_{(-1)}},e_{y|x_{(-1)}})=\sqrt{\frac{SSE_{(-1)}-SSE}{SSE_{(-1)}}}
   $$
   
   - 因变量与自变量的样本偏决定系数
   
   $$
   r_{y1;2,...p}^2 = \frac{SSE_{(-1)}-SSE}{SSE_{(-1)}} 
   $$
   
   - 偏F统计量与样本偏决定系数
   
   $$
   \begin{aligned}
F_1&=\frac{(n-p-1)r_{y1;2,...p}^2}{1-r_{y1;2,...p}^2} \\
r_{y1;2,...p}^2 &= \frac{F_1}{F_1+n-p-1}
\end{aligned}
   $$

## 回归诊断 {#reg_12}

线性回归模型的估计、检验等操作依赖于[假定](#reg_2)。因此有必要去验证假定。

除此之外，还需对数据进行检验，看看是否存在异常点或强影响点。

### 残差分析 {#reg_12_1}

残差定义为$e_i=y_i-\hat y_i$，反映了拟合效果的好坏，是随机扰动项的“观察值”，因此可根据残差的性状来判断随机扰动项假设的合理性。

#### 不同形式的残差 {#reg_12_1_1}

1. 普通残差

   普通残差定义为$e_i=y_i-\hat y_i$，具有$E(e_i)=0, \; Var(e_i)=(1-h_{ii})\sigma^2, \; \rho(e_i,e_j)=\frac{-h_{ij}}{\sqrt{(1-h_{ii})(1-h_{jj})}}$的性质。
   
   注意$Var(e_i)$中包含着$h_{ii}$，$h_{ii}$为杠杆值，是帽子矩阵$H$的第i个对角线元素，反映了自变量空间中第i个数据偏离数据中心的程度。特别的，$h_{ii}$越大，$Var(e_i)$越小，这表明当某个数据点距离数据中心较远时，会有把拟合直线拖向自己的倾向，因而其残差也可能会较小，称这样的数据点为<span style='color:red'>**高杠杆点**</span>。
   
   已知$tr(H)=\sum_{i=1}^n h_{ii}=p+1$，一个判断高杠杆点的准则是将杠杆值超过两倍杠杆值平均值的数据点认为是高杠杆点。

2. 学生化残差

   定义**学生化残差**为
   
   $$
   r_i=\frac{e_i}{\sqrt{\widehat{Var}(e_i)}}=\frac{e_i}{\sqrt{(1-h_{ii})\hat \sigma^2}} (\#eq:model105)
   $$
   
   $r_i$的性质有$E(r_i)=0, \; Var(r_i)=1, \; \rho(r_i,r_j)=\frac{-h_{ij}}{\sqrt{(1-h_{ii})(1-h_{jj})}}$。
   
   在实际应用中可**近似**认为$r_i$相互独立且服从标准正态分布。
   
   学生化残差相较于普通残差解决了方差不等的问题，但仍会受到异常值的影响，会使$\hat \sigma^2$偏大，继而让$r_i$偏小，因此不太适合根据$|r_i|>3$的准则来判断异常值。
   
3. 删除残差

   在计算残差$e_i$时，用不包含第i组观测点的数据$Y_{(i)}$和$X_{(i)}$进行回归得到回归方程，根据该回归方程对该组观测点进行预测得到$\hat y_{(i)}$，则删除残差为$e_{(i)}=y_i-\hat y_{(i)}$。由于删除残差没有用到第i组观测点，因此能够在一定程度上减轻异常点的影响。
   
   特别的，可证
   
   $$
   e_{(i)}=\frac{e_i}{1-h_{ii}} (\#eq:model106)
   $$
   
   > 参考https://zhuanlan.zhihu.com/p/49276967
   
4. 删除学生化残差

   删除学生化残差定义为
   
   $$
   r_{(i)}=\frac{e_i}{\sqrt{1-h_{ii}}\hat \sigma_{(i)}}=r_i(\frac{n-p-2}{n-p-1-r_i^2})^{\frac{1}{2}} (\#eq:model107)
   $$
   
   一般根据$|r_{(i)}|>3$来判断异常值点。
   
#### 残差图 {#reg_12_1_2}

可令残差（可取学生化残差）为纵坐标，以任何其他有关量为横坐标绘制散点图，常见的横坐标有因变量的拟合值、自变量、时间等。

根据残差图能够看出残差的分布形态，进而可以粗略地判断其是否满足相关性质。

#### 正态性检验 {#reg_12_1_3}

对残差（可取学生化残差）进行正态性检验。

1. 图方法

   - QQ图
   
      绘制(理论分位数, 实际分位数)的散点，看散点是否处于45°线上。
      
   - PP图
   
      绘制(理论累积概率, 实际累积概率)的散点，看散点是否处于45°线上。
      
2. 假设检验

   - Kolmogorov-Smirnov检验
   
   - Shapiro-Wiks检验
   
注意，无论随机扰动项是否服从正态分布，最小二乘估计都是具有BLUE性质。但若不服从正态分布，后续的t检验、F检验和预测等都不能进行了（毕竟都是基于正态分布展开讨论的）。

### 异常点和强影响点 {#reg_12_2}

#### 异常点  {#reg_12_2_1}

异常点是从因变量的维度讨论的异常数据。

1. 基于数据删除模型的异常点检验

   数据删除模型如下所示
   
   $$
   \left\{
   \begin{array}{c}
   Y_{(i)}=X_{(i)}\beta_{(i)}+\varepsilon_{(i)} \\
   E(\varepsilon_{(i)})=0\\
   Var(\varepsilon_{(i)})=\sigma^2 I_{n-1}  
   \end{array}
   \right. (\#eq:model108)
   $$ 
   
   该方法即根据删除残差和删除学生化残差来判断是否为异常点。一般根据$|r_{(i)}|>3$所对应的数据点判定为异常点。
   
   > 数据删除模型又是新的模型形式呀

2. 基于均值漂移模型的异常点检验

   均值漂移模型如下所示
   
   $$
   \left\{
   \begin{array}{c}
   Y=X\beta+\gamma d_i+\varepsilon \\
   E(\varepsilon)=0\\
   Var(\varepsilon)=\sigma^2 I_{n}  
   \end{array}
   \right. (\#eq:model109)
   $$
   
   其中$d_i$表示第i个分量为1而其他分量均为0的n维列向量。该模型表示，如果第i个观测点明显偏高或者偏低，那么$d_i$的系数$\gamma$应该是显著异于0的，而$\gamma d_i$会影响到第i个观测点的截距项，因此称“均值漂移模型”。

在识别异常点的过程中，注意有**掩盖效应**和**淹没效应**。

掩盖效应：假定的异常点个数小于实际个数，有可能一个都找不到。

淹没效应：假定的异常点个数大于实际个数，有可能将正常点误判为异常点。

#### 强影响点  {#reg_12_2_2}

异常点是从因变量的维度讨论的异常数据，高杠杆点是从自变量的角度讨论的异常数据。而综合二者后，称能够对统计推断造成较大影响的点为**强影响点**。

> 杠杆值$h_{ii}$大于两倍杠杆值均值$2\frac{p+1}{n}$即可视为高杠杆点

识别方法：

1. Cook距离

   定义Cook距离
   
   $$
   D_i=\frac{(\hat \beta -\hat \beta_{(i)})'X'X(\hat \beta -\hat \beta_{(i)})}{(p+1)\hat \sigma^2} = \frac{1}{p+1}(\frac{h_{ii}}{1-h_{ii}})r_i^2 (\#eq:model110)
   $$
   
   Cook距离度量了删除第i个数据点前后对回归系数估计值的变化情况。
   
   一个粗略的判断准则为：当$D_i < 0.5$，则认为不是强影响点，当$D_i>1$，则认为是强影响点。
   
2. Welsch-Kuh统计量(DFFITS准则)

   Welsch-Kuh统计量定义为
   
   $$
   WK_i=\frac{\hat y_i-\hat y_{(i)}}{\sqrt{\hat \sigma^2_{(i)}h_{ii}}}=\sqrt{(\frac{h_{ii}}{1-h_{ii}})r_{(i)}^2} (\#eq:model111)
   $$
   
   DFFITS准则度量了删除第i个数据点前后该点处拟合值的变化情况。
   
   判断准则为若$|WK_i|>2\sqrt{\frac{p+1}{n-p-1}}$则视为强影响点。
   
3. Hadi统计量

   Hadi统计量定义为
   
   $$
   H_i=\frac{h_{ii}}{1-h_{ii}}+\frac{p+1}{1-h_{ii}}\cdot \frac{d_i^2}{1-d_i^2} (\#eq:model112)
   $$
   
   其中$d_i=\frac{e_i}{SSE}$称为正规化残差。
   
   称以$\frac{p+1}{1-h_{ii}}\cdot \frac{d_i^2}{1-d_i^2}$为横坐标，以$\frac{h_{ii}}{1-h_{ii}}$为纵坐标的散点图为“位势-残差图”。
   
### 异方差 {#reg_12_3}

若$Var(\varepsilon_i)=\sigma^2_i$，即不同扰动项有不同的方差，则称之为“异方差”问题。

#### 原因 {#reg_12_3_1}

1. 遗漏重要变量

   重要变量对因变量的影响被归结到随机扰动项中，而这些影响具有差异性，从而导致异方差。

2. 模型设定误差

   包括模型形式和变量选择，例如本应包含自变量的二次项但未包含，也会导致异方差问题。

3. 数据的测量误差

4. 在截面数据中个体间的差异较大

5. 存在异常点

#### 后果 {#reg_12_3_2}

1. 最小二乘估计仍是无偏的，但不是最小方差线性无偏估计

   无偏性没用到随机扰动项的同方差假定，因此仍具有无偏性。但求估计量的方差时需要用到同方差假定，因而不具有有效性。
   
2. 最小二乘估计的方差估计量是有偏的

   既然方差估计量是有偏的，那么凡是用到$\hat \sigma^2$的地方（显著性检验、预测）都会失效。
   
   例如负的偏差会低估参数估计量的真实方差，这会导致对应的t统计量偏大，从而错误地拒绝了原假设。正的偏差会高估参数估计量的真实方差，会产生相反的结果。
   
#### 识别 {#reg_12_3_3}

1. 残差图

   根据残差图观察残差的分布形态。
   
2. Spearman等级相关系数法

   求得普通最小二乘下的残差，根据$x_i$与$|e_i|$的等级（秩）差来构造等级相关系数，对等级相关系数进行显著性检验，若拒绝原假设则说明自变量和$|e_i|$之间存在系统关系，也就说明存在异方差。
   
3. Goldfeld-Quandt检验

   检验是否存在递增或递减的异方差情形。

4. Breusch-Pagan检验

   $e_i^2$对所有自变量进行回归，看看残差平方是否和某个自变量有关系。
   
5. White检验

    $e_i^2$对所有自变量、自变量平方及变量间的交互项进行回归，看看残差平方是否和某一项有关系。

#### 补救 {#reg_12_3_4}

1. 加权最小二乘法

   加权最小二乘法通过为数据加权，来消除异方差性。对方差较大的观测赋予较小的权重，以牺牲大方差项的拟合效果为代价，改善小方差项的拟合效果。这个方法关键是要确定合适的权重，实际中可尝试采用残差平方的倒数最为权重。
   
2. 采用异方差稳健标准误

   既然异方差问题会影响$\hat \sigma^2$的估计，那么就直接采用更为稳健的标准误替代$\hat \sigma^2$。
   
3. Box-Cox变换

   对因变量采取如下变换（因变量为正）
   
   $$
   y^{(\lambda)}=\begin{cases} \frac{y^\lambda-1}{\lambda}, &\lambda \neq 0 \\ \ln y , &\lambda=0 \end{cases} (\#eq:model113)
   $$
   
   可根据极大似然估计法确定$\lambda$。Box-Cox变换能够在一定程度上改善数据的非正态性、异方差性、自相关性。但除了对数变换（表示百分比变动）外其余变换都缺乏解释性。

### 自相关 {#reg_12_4}

若$Cov(\varepsilon_i, \varepsilon_j) \neq 0$，则称之为“自相关”问题。

#### 原因 {#reg_12_4_1}

1. 遗漏重要变量

   重要变量对因变量的影响被归结到随机扰动项中，而这些影响是前后相关联的，从而导致自相关。

2. 模型设定误差

3. 经济变量的滞后性会给序列带来自相关性

4. 随机误差项本身的自相关

   如地震不仅影响当期，其造成的影响还会持续一段时间。

5. 因对数据加工整理而导致扰动项之间产生自相关性

   如把月度数据合并为季度数据、对缺失值进行插值。

#### 后果 {#reg_12_4_2}

1. 最小二乘估计仍是无偏的，但不是最小方差线性无偏估计

   无偏性没用到随机扰动项的无自相关假定，因此仍具有无偏性。但求估计量的方差时需要用到无自相关假定，因而不具有有效性。
   
2. 最小二乘估计的方差被低估

   同样会造成显著性检验、预测等操作失效。
   
#### 识别 {#reg_12_4_3}

1. 图示法

   绘制$e_t$与$e_{t-1}$、$t$的散点图，看看前后是否有某种特定的趋势。
   
2. 游程检验

   对残差的符号变化情况进行游程检验，看看是否是随机变化的。
   
3. Durbin-Watson检验

   DW检验只能检验一阶自相关情形，且有一些前提条件（如模型中必须包含截距项、解释变量中不包含Y的滞后项等）。

4. Breusch-Godfrey检验

   让残差对所有自变量及残差的滞后项跑个回归，看看残差滞后项的回归系数是否显著。
   
5. 纯随机性检验

   利用时间序列中的Q统计量进行纯随机性检验。

#### 补救 {#reg_12_4_4}

1. 广义差分法
   
2. 采用异方差自相关一致标准误(HAC)

   直接用HAC替换$\hat \sigma^2$。
   
3. Box-Cox变换

### 多重共线性 {#reg_12_5}

若设计矩阵$X$的各个列向量之间是线性相关的，则称之为“完全多重共线性”。若是近似线性相关的，则称之为“多重共线性”。注意，多重共线性是一个程度轻重的问题。

#### 原因 {#reg_12_5_1}

1. 经济变量之间的内在联系是产生多重共线性的根本原因

2. 经济变量之间存在共同的变化趋势

3. 模型中存在滞后项


#### 后果 {#reg_12_5_2}

1. 若是完全多重共线性，则$X'X$不可逆，无法得到最小二乘估计
   
2. 最小二乘估计仍是线性无偏的，但多重共线性会导致各估计量的方差较大

   同样会造成显著性检验、预测等操作失效。
   
3. 回归系数的估计量的符号跟实际不符，估计量的含义变得不明确
   
#### 识别 {#reg_12_5_3}

1. 经验判断 

   - $R^2$很高，F统计量值很大，但各个回归系数显著的较少
   
   - 回归系数的符号与预期相反
   
   - 解释变量之间两两高度相关
   
   - 若模型中增加或减少一个自变量，回归系数的估计值产生较大的变化

2. 条件数
   
   考虑标准化后的矩阵$X'X$，设其特征根为$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p \geq0$，则定义条件数为
   
   $$
   CI_i=\sqrt{\frac{\lambda_1}{\lambda_i}} (\#eq:model114)
   $$
   
   当条件数大于等于10时就有相对较为严重的多重共线性。
   
   > 也可以不取根号，相对应的阈值也要改变

3. 方差膨胀因子

   定义方差膨胀因子为
   
   $$
   VIF_i = \frac{1}{1-R_i^2} (\#eq:model115)
   $$
   
   其中$R_i^2$为自变量$x_i$对其余自变量回归得到的样本决定系数。若$VIF_i \geq 10$，则认为$x_i$与其他自变量之间存在较强的多重共线性问题。

#### 补救 {#reg_12_5_4}

1. 变量筛选与处理

   可以剔除一些不重要的变量，或者对变量进行变形，如化总体指标为人均指标。
   
2. 采取适当的参数估计方法

   如岭回归、主成分回归、偏最小二乘回归。

## 变量选择与正则化  {#reg_13}

### 冗余与遗漏 {#reg_13_1}

对于某一实际问题涉及到的因变量Y，我们搜寻了m个可能与其相关的自变量。称包含这m个自变量的回归模型为**全模型**。若只从这m个自变量中选取p个自变量进行回归，则称对应的模型为**选模型**。

若真实模型是选模型，而用了全模型进行回归，则在模型中引入了不必要的变量，即**变量冗余**。

若真实模型是全模型，而用了选模型进行回归，则在模型中遗漏了关键变量，即**变量遗漏**。

### 变量选择的传统方法  {#reg_13_2}

#### 自变量选择准则 {#reg_13_2_1}

可根据如下准则进行模型选择。

1. 调整的$R^2$

2. $C_p$准则

   $$
   C_p=\frac{1}{n}(SSE_p+2p\frac{SSE_m}{n-m-1}) (\#eq:model116)
   $$
   
   其中$SSE_m$表示全模型的残差平方和，$SSE_p$表示选模型的残差平方和。
   
   > 这只是$C_p$准则的一种定义，还有另一种定义

3. AIC

   $$
   AIC=-2\ln L(\hat \theta;X)+2p  (\#eq:model117)
   $$
   
   其中$L(\cdot)$表示模型的似然函数，$\hat \theta$表示参数$\theta$的极大似然估计（在多元线性回归模型中就是$\hat \beta$和$\hat \sigma^2$），$X$表示样本。
   
4. BIC

   $$
   BIC=-2\ln L(\hat \theta;X)+p \ln n  (\#eq:model118)
   $$
   
   BIC准则相较于AIC准则增强了对变量个数的惩罚，并新增了对样本数的惩罚。
   
#### 变量选择方法  {#reg_13_2_2}

1. 最优子集法

   对自变量的所有组合（共$2^m-1$种组合）分别拟合回归方程，根据自变量选择准则从中挑选最优模型。
   
   > 费时

2. 向前回归法

   考虑偏F统计量，模型的变量从少到多，每次将偏F统计量最大的且显著的那个变量纳入到模型中，直到没有可引入的变量为止。
   
   > 当然也可选择其他自变量选择准则
   
3. 向后回归法

   考虑偏F统计量，模型的变量从多到少，每次将偏F统计量最小的且不显著的那个变量从模型中剔除，直到没有剔除的变量为止。
   
4. 向前向后法

   向前法或者向后法都是“只进不出”或者“只出不进”，没有考虑变量间的**联合效应**。而向前向后法综合了这两种方法，每引入一个自变量时对所有已纳入到模型中的自变量进行逐个检验，考察是否要剔除变量，直至既无显著的自变量引入模型，也无不显著的自变量从回归模型中剔除为止。
   
### 变量选择的正则化方法  {#reg_13_3}

在模型估计时纳入正则项（惩罚项），不同的惩罚项有不同功能与作用。

详见[变量选择与惩罚函数](#penalty)。



