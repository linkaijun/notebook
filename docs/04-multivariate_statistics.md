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
&(3) \; Var(Z_i)= \max_{a'a=1, \,a_i'\Sigma a_j=0 \\ j=1,2,\dots,i-1} Var(a'X)
\end{aligned} (\#eq:ms-eq1)
$$

### 总体主成分 {#ms_1_1}





