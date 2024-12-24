# 应用多元统计 {#ms}

## 主成分分析 {#ms_1}

**定义：**对原始变量进行线性变换构造出互不相关的主成分，主成分涵盖了原始变量的绝大部分信息。

其中**线性变换**意味着是对原始变量的线性组合；**互不相关**意味着主成分间的协方差为零；**绝大部分信息**意味着主成分的方差占比大。

令$X=(X_1,X_2,\dots,X_p)'$为p维随机向量，均值向量为$E(X)=\mu$，协方差阵为$Cov(X)=\Sigma$，主成分为$Z=(Z_1,Z_2,\dots,Z_p)'$，则主成分表示为

$$
\begin{cases}
Z_1=a_1'X=a_{11}X_1+\dots+a_{p1}X_p \\
\qquad \qquad \qquad \quad \vdots \\
Z_p=a_p'X=a_{1p}X_1+\dots+a_{pp}X_p
\end{cases} (\#eq:ms-eq1)
$$

为了限制量纲差异对方差的影响，限制$||a||=1$，则主成分的性质有

$$
\begin{aligned}
&(1) \; a_i'a_i=1 \\
&(2) \; a_i'\Sigma a_j=0, \; j=1,2,\dots,i-1 \\
&(3) \; Var(Z_i)= \max_{a_i'a_i=1, \,a_i'\Sigma a_j=0 \\ j=1,2,\dots,i-1} Var(a_i'X)
\end{aligned} (\#eq:ms-eq2)
$$

### 总体主成分 {#ms_1_1}

#### 基于协差阵的总体主成分 {#ms_1_1_1}

总体主成分的导出如下所示。不妨先求解第一主成分。

$$
\begin{array}{c}
\max \limits_{a_1} a_1'\Sigma a_1 \\
s.t \; a_1'a_1=1 
\end{array} (\#eq:ms-eq3)
$$

将其转化为拉格朗日函数：

$$
L=a_1'\Sigma a_1 + \lambda(1-a_1'a_1) (\#eq:ms-eq4)
$$

对$a_1$求偏导得

$$
\frac{\partial L}{\partial a_1} = 2\Sigma a_1 - 2\lambda_1 a_1 = 0  (\#eq:ms-eq5)
$$

可知所求$\lambda_1$及$a_1$即为$\Sigma$的最大特征值及对应的特征向量。

进一步地，对于第1个主成分之后的所有主成分而言，都要满足与前面所有主成分均不相关的条件，于是有

$$
\begin{array}{c}
\max \limits_{a_i} a_i'\Sigma a_i \\
s.t \; a_i'a_i=1, \; a_i'a_k=0, k=1,2,...,i-1 
\end{array} (\#eq:ms-eq6)
$$

转化为拉格朗日函数：

$$
L=a_i' \Sigma a_i + \lambda_i(1-a_i'a_i) + \sum_{k=1}^{i-1}\gamma_k a_i'a_k (\#eq:ms-eq7)
$$

对$a_i$求偏导得

$$
\frac{\partial L}{\partial a_i} = 2\Sigma a_i -2\lambda_i a_i + \sum_{k=1}^{i-1}\gamma_k a_k = 0 (\#eq:ms-eq8)
$$

对上式左右两边同时乘以$a_1'$，得

$$
\begin{aligned}
2a_1'\Sigma a_i -2\lambda_i a_1'a_i + \sum_{k=1}^{i-1}\gamma_k a_1'a_k = 0 \\
\gamma_1 a_1'a_1 = 0 \\
\gamma_1=0
\end{aligned} (\#eq:ms-eq9)
$$

同理，对式\@ref(eq:ms-eq8)左右同乘$a_k', k=1,2,...,i-1$，可得$\gamma_k=0$，于是式\@ref(eq:ms-eq8)转化为

$$
2\Sigma a_i -2\lambda_i a_i=0 (\#eq:ms-eq10)
$$

可知$\lambda_i$和$a_i$分别是$\Sigma$第$i$大的特征值及对应的特征向量。

**因此，只需求得$X$的协差阵的特征向量即可得到主成分。**

> 从几何上来看，$Z_i=a_i'X$是$X$在$a_i$方向上的投影，对应的$\lambda_i$就是该方向上投影点的方差。

故令$\Sigma = P\Lambda P'$，则$Z=P'X$，其中$P=(a_1|a_2|...|a_p)$，其**性质**如下所示

1. $Cov(Z)=Cov(P'X)=P'Cov(X)P=P'\Sigma P=\Lambda$

2. $\sum \lambda_i=tr(\Lambda)=tr(P'\Sigma P)=tr(\Sigma PP')=tr(\Sigma)=\sum\sigma_{ii}$

3. $\rho(Z_k, X_i)=\frac{Cov(Z_k, X_i)}{\sqrt{\lambda_k\sigma_{ii}}}=\frac{Cov(a_k'X, e_i'X)}{\sqrt{\lambda_k\sigma_{ii}}}=\frac{a_k'\Sigma e_i}{\sqrt{\lambda_k\sigma_{ii}}}=\frac{\lambda_ka_k'e_i}{\sqrt{\lambda_k\sigma_{ii}}}=\frac{\sqrt{\lambda_k}a_{ik}}{\sqrt{\sigma_{ii}}}$

4. $\sum_{k=1}^p\rho^2(Z_k,X_i)=\sum_{k=1}^p \frac{\lambda_k a_{ik}^2}{\sigma_{ii}}=\frac{1}{\sigma_{ii}}\sum_{k=1}^p \lambda_k a_{ik}^2=1$

5. $\sum_{i=1}^p \sigma_{ii}\rho^2(Z_k,X_i)=\sum_{i=1}^p \sigma_{ii}\frac{\lambda_ka_{ik}^2}{\sigma_{ii}}=\lambda_k\sum_{i=1}^pa_{ik}^2=\lambda_k$

> 3.$e_i$为第$i$个位置上为1，其余位置为0的列向量
> 
> 3.$a_k'\Sigma = (\Sigma a_k)' = (\lambda_k a_k)'$
> 
> 4.$\sigma_{ii}=(a_{i1}, a_{i2}, ..., a_{ip})\Lambda (a_{i1}, a_{i2}, ..., a_{ip})'=\sum_{k=1}^p \lambda_k a_{ik}^2$
> 
> 5.$\sum_{i=1}^pa_{ik}^2=1$

关于主成分的指标如下所示：

1. 贡献率(第i个主成分)：$\omega_i=\frac{\lambda_i}{\sum_{k=1}^p \lambda_k}$

2. 累积贡献率(前m个主成分)：$f_m=\frac{\sum_{k=1}^m \lambda_k}{\sum_{i=1}^p \lambda_i}$

3. 方差贡献率(前m个主成分对$X_i$的贡献率)：$v_i^{(m)}=\frac{\sum_{k=1}^m \lambda_k a_{ik}^2}{\sigma_{ii}}$

#### 基于相关阵的总体主成分 {#ms_1_1_2}

考虑变量量纲不同的影响，对数据进行标准化处理，得到$X_i^*=\frac{X_i-\mu_i}{\sqrt{\sigma_{ii}}}$，此时$X^*$的协差阵即为$X$的相关阵$R$。

同理可得，$Z_k^*=a_k^{*'}X^*$，其中$a_k^*=(a_{1k}^*,...,a_{pk}^*)'$是$R$对应于特征值$\lambda_k^*$的单位正交特征向量，相关**性质**如下所示：

1. $Cov(Z^*)=Cov(P^{*'}X^*)=P^{*'}Cov(X^{*})P^{*}=P^{*'}RP^{*}=\Lambda^*$

2. $\sum \lambda_i^*=tr(\Lambda^*)=tr(P^{*'}RP^{*'})=tr(RP^*P^{*'})=tr(R)=p$

3. $\rho(Z_k^*, X_i^*)=\frac{Cov(Z_k^*, X_i^*)}{\sqrt{\lambda_k^*}}=\frac{Cov(a_k^{*'}X^*, e_i'X^*)}{\sqrt{\lambda_k^*}}=\frac{a_k^{*'}R e_i}{\sqrt{\lambda_k^*}}=\frac{\lambda_k^*a_k^{*'}e_i}{\sqrt{\lambda_k^*}}=\sqrt{\lambda_k^*}a_{ik}^{*}$

4. $\sum_{k=1}^p\rho^2(Z_k^*,X_i^*)=\sum_{k=1}^p \lambda_k^* a_{ik}^{*2}=1$

5. $\sum_{i=1}^p \rho^2(Z_k^*,X_i^*)=\sum_{i=1}^p \lambda_k^*a_{ik}^{*2}=\lambda_k^*\sum_{i=1}^pa_{ik}^{*2}=\lambda_k^*$

> 简单记，就把$\sigma_{ii}=1$代入基于协差阵的结果，对应符号添加*号即可

**什么时候需要标准化？**

1. 对变量进行标准化可以提高方差较小变量对主成分的贡献

2. 当变量量纲差异较大时需要进行标准化处理

### 样本主成分 {#ms_1_2}

定义观测矩阵为

$$
X=
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}=
\begin{pmatrix}
X'_{(1)} \\
X'_{(2)} \\
\vdots \\
X'_{(n)} \\
\end{pmatrix} (\#eq:ms-eq11)
$$

$X_{(i)}$为第i个样本，则样本均值、协方差阵、相关阵分别为

$$
\bar X = \frac{1}{n}\sum_{i=1}^nX_{(i)}=(\bar x_1, ..., \bar x_p)' (\#eq:ms-eq12)
$$

$$
S=\frac{1}{n-1}\sum_{i=1}^n (X_{(i)}-\bar X)(X_{(i)}-\bar X)'=(s_{ij})_{p\times p} (\#eq:ms-eq13)
$$

$$
\tilde R=(r_{ij})_{p \times p}, \quad r_{ij}=\frac{s_{ij}}{\sqrt{s_{ii}s_{jj}}} (\#eq:ms-eq14)
$$

#### 基于协差阵的样本主成分 {#ms_1_2_1}

记样本协差阵S的特征根和单位正交特征向量为$(\hat \lambda_i, \hat a_i)$，则第i个样本主成分为$\hat Z_i=\hat a_i'X$。

故样本$X_{(k)}$在第i个主成分上的值为$\hat Z_{ik}=\hat a_i'X_{(k)}$，称其为$X_{(k)}$在第i个主成分上的得分。

- 第i个样本主成分的样本均值：$\bar Z_i = \frac{1}{n}\sum_{k=1}^n \hat Z_{ik}=\hat a_i'\bar X$

- 第i个样本主成分的样本方差：$Cov(\hat Z_i)=\frac{1}{n-1}\sum_{k=1}^n (Z_{ik}-\bar Z_i)^2=\frac{1}{n-1}\sum_{k=1}^n (\hat a_i'X_{(k)}-\hat a_i' \bar X_{(k)})^2=\hat a_i' S \hat a_i = \hat \lambda_i$

当$i\neq j$时，各主成分之间的样本协方差为：

$$
\begin{aligned}
Cov(\hat Z_i, \hat Z_j)&=\frac{1}{n-1}\sum_{k=1}^n(Z_{ik}-\bar Z_i)(Z_{jk}-\bar Z_j) \\
&=\frac{1}{n-1}\sum_{k=1}^n (\hat a_i'X_{(k)}-\hat a_i' \bar X)(\hat a_j'X_{(k)}-\hat a_j' \bar X)' \\
&= \hat a_i'S\hat a_j \\
&= \hat \lambda_j \hat a_i \hat a_j \\
&=0
\end{aligned} (\#eq:ms-eq15)
$$

类似的，总的样本方差为$\sum_{i=1}^p s_{ii}=\sum_{i=1}^p \hat \lambda_i$，样本相关系数为$r(\hat Z_k, X_i)=\frac{\sqrt{\hat \lambda_k}\hat a_{ik}}{\sqrt{s_{ii}}}$。

#### 基于相关阵的样本主成分 {#ms_1_2_2}

同样先对数据做标准化处理，标准化后的数据的协差阵即为原始数据的相关阵，其余操作同基于相关阵的总体主成分，只需要注意使用关于样本的符号即可。

### 关于主成分 {#ms_1_3}

#### 主成分的局限性 {#ms_1_3_1}

1. 仅考虑了原始变量的正交/线性变换。

2. PCA仅依赖于样本数据的均值和协方差矩阵，有些分布无法进行刻画。

3. 当原始变量是相关的时候，使用PCA可以降低维数，若原始变量不相关，则无法有效降维。

4. PCA容易受到异常点的影响。

#### 如何选取主成分个数 {#ms_1_3_2}

1. 前m个主成分的累积贡献率达到某个阈值，如80%或85%以上。

2. 无论是从协差阵还是相关阵出发，一个经验规则是保留特征值大于其平均值（或1）的主成分

3. 绘制碎石图看拐点。

#### R语言实现 {#ms_1_3_3}

PCA相对较为简单，可以自定义函数，如下所示


``` r
my_pca <- function(sigma){
  n = dim(sigma)[1]
  var_name = paste0('var_', 1:n)
  component_name = paste0('comp_', 1:n)
  lambda_name = paste0('lambda_', 1:n)
  
  eigen = eigen(sigma)
  eigen_value = eigen$values
  names(eigen_value) = lambda_name
  eigen_vec = eigen$vectors   #特征向量矩阵&主成分矩阵
  colnames(eigen_vec) = component_name
  
  # 贡献率
  contribution_rate = eigen_value/sum(eigen_value)
  names(contribution_rate) = component_name
  # 累积贡献率
  cum_contribution_rate = cumsum(eigen_value)/sum(eigen_value)
  names(cum_contribution_rate) = component_name
  # 各个主成分对变量的贡献率 lambda*a^2/sigma
  contribution_to_var = diag(diag(sigma)^(-1)) %*% eigen_vec^2 %*% diag(eigen_value)
  dimnames(contribution_to_var) = list(var_name, component_name)
  
  # 碎石图
  library(ggplot2)
  scree_plot = ggplot()+
    geom_line(aes(x=component_name, y=eigen_value, group=1), linetype='dashed')+
    geom_point(aes(x=component_name, y=eigen_value), size=2.5)+
    theme_bw()+
    labs(x='', y='variance')+
    theme(axis.text.x = element_text(size = rel(1.5)),
          axis.title.y = element_text(size = rel(1.3)))
  
  result=list(
    lambda = eigen_value,
    vector = eigen_vec,
    contribution_rate = contribution_rate,
    cum_contribution_rate = cum_contribution_rate,
    contribution_to_var = contribution_to_var,
    scree_plot = scree_plot
  )
  result
}
```

或者调用R中的函数`princomp()`、`psych::principal()`。
