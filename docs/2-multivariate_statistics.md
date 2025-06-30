# 应用多元统计 {#ms}

## 矩阵运算 {#ms_1}

### Kronecker积 {#ms_1_1}

设$A=(a_{ij})_{m \times n}, \, B=(b_{ij})_{p \times q}$，则称矩阵$C=(a_{ij}B)_{mp \times nq}$为矩阵A和矩阵B的Kronecker积，记为$C=A\otimes B$。

性质：

1. $(A_1 \otimes B_1)(A_2 \otimes B_2)=(A_1A_2)\otimes(B_1B_2)$

证：

不妨记$A_1=(a_{ij}^{(1)})=\begin{pmatrix} a_{1 \cdot }^{(1)} \\ a_{2 \cdot}^{(1)} \\ \vdots \\ a_{m \cdot}^{(1)} \end{pmatrix},\, A_2=(a_{ij}^{(2)})=\begin{pmatrix} a_{\cdot 1}^{(2)} & a_{\cdot 2}^{(2)} & \cdots & a_{\cdot n}^{(2)} \end{pmatrix}$，则

$$
\begin{aligned}
(A_1 \otimes B_1)(A_2 \otimes B_2)&=(a_{ij}^{(1)}B_1)(a_{ij}^{(2)}B_2) \\
&=(a_{i \cdot}^{(1)}a_{\cdot j}^{(2)}B_1B_2) \\
&= (a_{i \cdot}^{(1)}a_{\cdot j}^{(2)})\otimes (B_1B_2) \\
&=(A_1A_2)\otimes (B_1B_2)
\end{aligned} (\#eq:ms-eq66)
$$

> 这里进行了简写，注意只要矩阵运算中出现小写字母，代表该矩阵的元素

2. $(A\otimes B)'=A' \otimes B'$

证：

$$
\begin{aligned}
(A\otimes B)'&=(a_{ij}B)' \\
&= (a_{ji}B') \\
&= A' \otimes B'
\end{aligned} (\#eq:ms-eq67)
$$

3. $(A\otimes B)^{-1}=A^{-1} \otimes B^{-1}$

证：

$$
\begin{aligned}
(A \otimes B) \cdot (A^{-1} \otimes B^{-1})&=(AA^{-1}) \otimes (BB^{-1})\\
&=I \otimes I\\
&=I\\
\Rightarrow  (A \otimes B)^{-1} &=A^{-1} \otimes B^{-1}
\end{aligned} (\#eq:ms-eq68)
$$

4. 若A，B均为方阵，则$tr(A\otimes B)=tr(A) \cdot tr(B)$

证：

$$
\begin{aligned}
tr(A\otimes B)&=tr(a_{ij}B) \\
&= \sum_{i}\sum_{j}a_{ii}b_{jj} \\
&= \sum_{i}a_{ii}\sum_{j}b_{jj} \\
&= tr(A) \cdot tr(B)
\end{aligned} (\#eq:ms-eq69)
$$

5. $||A \otimes B||=||A|| \cdot ||B||$

证：

$$
\begin{aligned}
||A \otimes B||=...
\end{aligned}  (\#eq:ms-eq69)
$$

6. $rank(A \otimes B)=rank(A) \cdot rank(B)$

证：

不妨记矩阵$A_{m\times n}$前p行为极大线性无关组，矩阵$B_{h\times k}$前q行为极大线性无关组，对$A\otimes B$进行行化简，如下所示：

$$
\begin{aligned}
A \otimes B =
&\begin{pmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
\vdots  & \vdots  & \vdots & \vdots  \\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{pmatrix}\\
\overset {a} \Rightarrow 
&\begin{pmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
\vdots  & \vdots  & \vdots & \vdots  \\
a_{p1}B & a_{p2}B & \cdots & a_{pn}B \\
0       & 0       & \cdots & 0       \\
\vdots  & \vdots  & \vdots & \vdots  \\
0       & 0       & \cdots & 0       \\
\end{pmatrix}\\
\overset {b} \Rightarrow 
&\begin{pmatrix}
a_{11}b_{11} & \cdots & a_{11}b_{1k} &  \cdots & a_{1n}b_{11} & \cdots & a_{1n}b_{1k}\\
\vdots  & \vdots  & \vdots & \vdots & \vdots  & \vdots  & \vdots \\
a_{11}b_{q1} & \cdots & a_{11}b_{qk} & \cdots &  a_{1n}b_{q1} & \cdots & a_{1n}b_{qk} \\
\vdots  & \vdots  & \vdots & \vdots & \vdots  & \vdots  & \vdots \\
a_{p1}b_{11} & \cdots & a_{p1}b_{1k} &  \cdots & a_{pn}b_{11} & \cdots & a_{pn}b_{1k}\\
\vdots  & \vdots  & \vdots & \vdots & \vdots  & \vdots  & \vdots \\
a_{p1}b_{q1} & \cdots & a_{p1}b_{qk} & \cdots &  a_{pn}b_{q1} & \cdots & a_{pn}b_{qk} \\
0 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\
\vdots  & \vdots  & \vdots & \vdots & \vdots  & \vdots  & \vdots \\
0 & \cdots & 0 & \cdots & 0 & \cdots & 0 
\end{pmatrix}
\end{aligned} (\#eq:ms-eq70)
$$

### 拉直 {#ms_1_2}

设$A=(a_1,...,a_n)$是一个$m \times n$矩阵，其中$a_i=(a_{1i},...,a_{mi})'$。将矩阵A按列向量$a_1,...,a_n$依次排成一个$mn \times 1$的向量，即$vec(A)=\begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}$，称$vec(A)$为矩阵A的按列拉直运算，同理，记$rvec(A)$为矩阵A的按行拉直运算。显然，有$rvec(A)=(vec(A'))'$。

1. $tr(AB)=(vec(A'))'vec(B)$

证：

$$
\begin{aligned}
tr(AB)&= \sum_i a_{i \cdot}b_{\cdot i} \\
&= (a_{i \cdot})(b_{\cdot i}) \\
&= rvec(A)vec(B) \\
&= (vec(A'))'vec(B)
\end{aligned} (\#eq:ms-eq71)
$$

> 第二行表示元素为A的行向量的行向量与元素为B的列向量的列向量的内积

2. $vec(ABC)=(C' \otimes A)vec(B)$

证：

令$C_{m \times n}=(c_{ij})=(c_1,...,c_n), \, B=(b_1,...,b_m)$，则

$$
\begin{aligned}
(C' \otimes A)vec(B)&=
\begin{pmatrix} c_{11}A & c_{21}A & \cdots & c_{m1}A \\ 
c_{12}A & c_{22}A & \cdots & c_{m2}A \\
\vdots & \vdots & \ddots & \vdots \\
c_{1n}A & c_{2n}A & \cdots & c_{mn}A
\end{pmatrix}
\begin{pmatrix}
b_1 \\
b_2 \\
\vdots \\
b_m
\end{pmatrix} \\
&= \begin{pmatrix} A\sum (c_{i1} b_i) \\
A\sum (c_{i2} b_i) \\
\vdots \\ 
A\sum (c_{in} b_i)
\end{pmatrix} \\
&= \begin{pmatrix} ABc_1 \\
ABc_2 \\
\vdots \\ 
ABc_n
\end{pmatrix} \\
&= vec(ABC)
\end{aligned} (\#eq:ms-eq71)
$$

3. $tr(ABC)=(vec(A'))'(I \otimes B)vec(C)$

证：

令$A=\begin{pmatrix}a_{1 \cdot} \\ a_{2 \cdot} \\ \vdots \\ a_{m \cdot}\end{pmatrix}$

$$
\begin{aligned}
tr(ABC)&=tr(AB \cdot C) \\
&= [vec((AB)')]'vec(C) \\
&= rvec(AB)vec(C) \\
&= \begin{pmatrix}a_{1 \cdot}B & a_{2 \cdot}B & \cdots & a_{m \cdot}B \end{pmatrix} vec(C) \\
&= \begin{pmatrix}a_{1 \cdot} & a_{2 \cdot} & \cdots & a_{m \cdot} \end{pmatrix} diag\{B, B , \cdots ,B\}vec(C) \\
&=rvec(A)(I \otimes B)vec(C) \\
&= (vec(A'))'(I \otimes B)vec(C)
\end{aligned} (\#eq:ms-eq72)
$$

### 减号逆与加号逆 {#ms_1_3}

#### 减号逆 {#ms_1_3_1}

对于一个$m \times n$的矩阵A，一切满足方程组$AXA=A$的矩阵X称为矩阵A的广义逆，记为$A^{-}$，也称减号逆。

1. 减号逆不唯一

证：

令A是一个$m \times n$矩阵，rank(A)=r, 若$A=P \begin{pmatrix} I_r & 0 \\ 0 & 0  \end{pmatrix} Q$，P和Q分别为m阶、n阶矩阵，则

$$
A^{-} = Q^{-1} \begin{pmatrix} I_r & B \\ C & D  \end{pmatrix} P^{-1}  (\#eq:ms-eq73)
$$

其中B，C，D为适当阶数的任意矩阵。

2. 对任意矩阵A，有$A'A(A'A)^{-}A'=A',\;A(A'A)^{-}A'A=A$

证：

令$A'A(A'A)^{-}A'=X$，$A(A'A)^{-}A'A=Y$，则可得：
   
$$
\begin{aligned}
A'A(A'A)^{-}A'&= X \\
A'A(A'A)^{-}A'A &= XA \\
A'A &= XA \\
(A'-X)A &= 0 \\
A'(A-X') &= 0 \\
\end{aligned} (\#eq:ms-eq74)
$$
故$X=A'$必为该方程的解，即$A'A(A'A)^{-}A'=A'$。
   
$$
\begin{aligned}
A(A'A)^{-}A'A &= Y \\
A'A(A'A)^{-}A'A &= A'Y \\
A'A &= A'Y \\
A'(A-Y) &= 0
\end{aligned} (\#eq:ms-eq75)
$$

故$Y=A$必为方程的解，即$A(A'A)^{-}A'A=Y$。

3. 设相容线性方程组$Ax=b$，则

   - 对任一广义逆$A^{-}, \, x=A^{-}b$必为解
   
   证：
   
   $$
   \begin{aligned}
   Ax&=b \\
   AA^{-}Ax&=AA^{-}b \\
   Ax&=AA^{-}b \\
   A(x-A^{-}b)&=0 \\
   x&=A^{-}b
   \end{aligned} (\#eq:ms-eq76)
   $$
   
   - 齐次方程组$Ax=0$的通解为$x=(I-A^-A)z$，z为任意向量，$A^-$为任一固定的广义逆
   
   证：
   
   $$
   \begin{aligned}
   A &= AA^{-}A \\
   Az &= AA^{-}Az \\
   A(Iz-A^{-}Az) &= 0 \\
   A(I-A^{-}A)z &= 0
   \end{aligned} (\#eq:ms-eq77)
   $$
   
   - $Ax=b$的通解为$x=A^-b+(I-A^-A)z$，z为任意向量，$A^-$为任一固定的广义逆
   
   证：
   
   由上述可得，$Ax=b$的通解为$Ax=b$的特解加上$Ax=0$的通解，即$x=A^{-}b+(I-A^{-}A)z$

#### 加号逆 {#ms_1_3_2}

设A为任一矩阵，若矩阵X满足$(1)AXA=A;\;(2)XAX=X; \; (3)(AX)'=AX; \; (4)(XA)'=XA$，则称X为A的Moore-Penrose广义逆，记为$A^+$，也称加号逆或伪逆。

1. 每个矩阵均存在加号逆且唯一

存在性，证：

设A是一个$m \times n$矩阵，$rank(A)=r$，若A的奇异值分解$A=U\begin{pmatrix} \Lambda_r & 0 \\0 & 0 \end{pmatrix}V'=UDV'$，U和V分别为m阶、n阶正交矩阵。令$X=V\begin{pmatrix} \Lambda_r^{-1} & 0 \\0 & 0 \end{pmatrix}U'=V\tilde DU'$，则

$$
\begin{aligned}
&AXA=(UDV')(V\tilde D U')(UDV')=UDV'=A \\
&XAX= (V\tilde D U')(UDV')(V\tilde D U')=V\tilde D U'=X \\
&(AX)'=(UDV'V\tilde D U')'=(UU')'=UU'=UDV'V\tilde D U'=AX\\
&(XA)'=(V\tilde D U'UDV')'=(VV')'= VV'=VDU'U\tilde D V'=XA
\end{aligned} (\#eq:ms-eq78)
$$

即$X=A^+$。

唯一性，证：

令X，Y均满足伪逆的四个条件，即X，Y均为A的加号逆

$$
\begin{aligned}
X&=XAX\\
&=X(AX)' \\
&=XX'A' \\
&=XX'(AYA)' \\
&=X(AX)'(AY)' \\
&=XAX(AY) \\
&=XAY \\
&=(XA)'Y \\
&=A'X'Y \\
&=A'X'(YAY) \\
&=A'X'(YA)'Y \\
&=A'X'A'Y'Y \\
&=(AXA)'Y'Y \\
&=A'Y'Y \\
&=(YA)'Y \\
&=YAY \\
&=Y
\end{aligned} (\#eq:ms-eq79)
$$

2. $A^+=A'(AA')^+=(A'A)^+A'$

证：

令矩阵A的秩为r，则其奇异值分解为$A=U\begin{pmatrix} \Lambda_r & 0 \\ 0 & 0 \end{pmatrix}V'$，对应的有$AA'=U\begin{pmatrix} \Lambda_r^2 & 0 \\ 0 & 0 \end{pmatrix}U', \; A'A=V\begin{pmatrix} \Lambda_r^2 & 0 \\ 0 & 0 \end{pmatrix}V'$。

$$
\begin{aligned}
A'(AA')^+ &= A'U\begin{pmatrix} \Lambda_r^{-2} & 0 \\ 0 & 0 \end{pmatrix}U' \\
&=V\begin{pmatrix} \Lambda_r & 0 \\ 0 & 0 \end{pmatrix}U'U\begin{pmatrix} \Lambda_r^{-2} & 0 \\ 0 & 0 \end{pmatrix}U' \\
&=V\begin{pmatrix} \Lambda_r^{-1} & 0 \\ 0 & 0 \end{pmatrix}U' \\
&=A^+
\end{aligned}  (\#eq:ms-eq80)
$$

同理，$A^+=(A'A)^+A'$。

### 分块矩阵 {#ms_1_4}

将矩阵$A_{n \times p}$分成四块：$A=\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$，即为分块矩阵。

1. 若A为方阵，$A_{11}$也为方阵，则

   - 当$|A_{11}| \neq 0$时，$|A|=|A_{11}||A_{22 \cdot 1}|$，其中$A_{22 \cdot 1}=A_{22}-A_{21}A_{11}^{-1}A_{12}$
   
   证：

   令分块矩阵左乘$\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I\end{pmatrix}$，右乘$\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I\end{pmatrix}$，可得

   $$
   \begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I\end{pmatrix}\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I\end{pmatrix}=\begin{pmatrix} A_{11} & 0 \\ 0 & A_{22 \cdot 1} \end{pmatrix} (\#eq:ms-eq81)
   $$
   
   其中$A_{22 \cdot 1}=A_{22}-A_{21}A_{11}^{-1}A_{12}$。等式左右两边取行列式即可。
   
   - 当$|A_{22}| \neq 0$时，$|A|=|A_{11 \cdot 2}||A_{22}|$，其中$A_{11 \cdot 2}=A_{11}-A_{12}A_{22}^{-1}A_{21}$。
   
   证：
   
   同理。

2. 若A为可逆方阵，$A_{11}$和$A_{22}$均为方阵，则

   - 当$|A_{11}|\neq 0$时，$A^{-1}=\begin{pmatrix} A_{11}^{-1} + A_{11}^{-1}A_{12}A_{22 \cdot 1}^{-1}A_{21}A_{11}^{-1} & -A_{11}^{-1}A_{12}A_{22 \cdot 1}^{-1} \\ -A_{22 \cdot 1}^{-1}A_{21}A_{11}^{-1} & A_{22\cdot 1}^{-1} \end{pmatrix}$。
   
   证：
   
   由式\@ref(eq:ms-eq81)可知，
   
   $$
   \begin{aligned}
   \begin{pmatrix} A_{11} & 0 \\ 0 & A_{22 \cdot 1} \end{pmatrix}^{-1}&=\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I\end{pmatrix}^{-1}\begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}^{-1}\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I\end{pmatrix}^{-1} \\
   \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}^{-1}&=\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I\end{pmatrix}\begin{pmatrix} A_{11} & 0 \\ 0 & A_{22 \cdot 1} \end{pmatrix}^{-1}\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I\end{pmatrix} \\
   &=\begin{pmatrix} I & -A_{11}^{-1}A_{12} \\ 0 & I\end{pmatrix}\begin{pmatrix} A_{11}^{-1} & 0 \\ 0 & A_{22 \cdot 1}^{-1} \end{pmatrix}\begin{pmatrix} I & 0 \\ -A_{21}A_{11}^{-1} & I\end{pmatrix} \\
   &= \begin{pmatrix} A_{11}^{-1} + A_{11}^{-1}A_{12}A_{22 \cdot 1}^{-1}A_{21}A_{11}^{-1} & -A_{11}^{-1}A_{12}A_{22 \cdot 1}^{-1} \\ -A_{22 \cdot 1}^{-1}A_{21}A_{11}^{-1} & A_{22\cdot 1}^{-1} \end{pmatrix}
   \end{aligned} (\#eq:ms-eq82)
   $$
   
   - 当$|A_{22}|\neq 0$时，$A^{-1}=\begin{pmatrix} A_{11 \cdot 2}^{-1}  & -A_{11 \cdot 2}^{-1}A_{12}A_{22}^{-1} \\ -A_{22}^{-1}A_{21}A_{11\cdot 2}^{-1} & A_{22\cdot 1}^{-1}+ A_{22}^{-1}A_{21}A_{11 \cdot 2}^{-1}A_{12}A_{22}^{-1} \end{pmatrix}$。
   
   证：
   
   同理。
   
   - 当$|A_{11}|\neq 0, \;|A_{22}|\neq 0$时，$A^{-1}=\begin{pmatrix} A_{11 \cdot 2}^{-1}  & -A_{11}^{-1}A_{12}A_{22 \cdot 1}^{-1} \\ -A_{22}^{-1}A_{21}A_{11\cdot 2}^{-1} & A_{22\cdot 1}^{-1} \end{pmatrix}$。
   
   证：
   
   同理。

## 多元正态分布 {#ms_2}

### 多元分布的基本运算性质 {#ms_2_1}

1. $E(tr(AX))=tr(E(AX))=tr(AE(X))$

2. $Cov(AX,BY)=ACov(X,Y)B', \, Cov(AX)=ACov(X)A'$

3. $E(X'AX)=tr(A\Sigma)+\mu'A\mu$

证：

$$
\begin{aligned}
E(X'AX)&=E[tr(X'AX)] \\
&=E[tr(AXX')] \\
&=tr(AE(XX')) \\
&=tr(A(\Sigma+\mu\mu')) \\
&=tr(A\Sigma)+tr(A\mu\mu') \\
&=tr(A\Sigma)+tr(\mu'A\mu) \\
&=tr(A\Sigma)+\mu'A\mu
\end{aligned} (\#eq:ms-eq51)
$$

其中$Cov(X)=\Sigma=E(XX')-\mu\mu'$。

4. 若$X \sim \varphi_X(t), \, Y=AX+A$，则$\varphi_Y(t)=\exp(it'a)\varphi_X(A't)$

证：

$$
\begin{aligned}
\varphi_Y(t)&=E(e^{it'Y}) \\
&= E(e^{it'(AX+a)}) \\
&= e^{it'a}E(e^{it'AX}) \\
&= e^{it'a}E(e^{i(A't)'X}) \\
&= e^{it'a}\varphi_X(A't)
\end{aligned} (\#eq:ms-eq52)
$$

5. 若X，Y相互独立且维数相同，则$\varphi_{X+Y}(t)=\varphi_X(t)\varphi_Y(t)$。

证：

$$
\begin{aligned}
\varphi_{X+Y}(t)&=E(e^{it'(X+Y)}) \\
&=E(e^{it'X}e^{it'Y}) \\
&= E(e^{it'X})E(e^{it'Y}) \\
&= \varphi_X(t)\varphi_Y(t)
\end{aligned} (\#eq:ms-eq53)
$$

-----
一元正态分布$N(\mu,\sigma^2)$的特征函数

$$
\begin{aligned}
E(e^{itX})&=\int_{-\infty}^{\infty}\exp(itx)\frac{1}{\sqrt{2\pi}\sigma}\exp[-\frac{(x-\mu)^2}{2\sigma^2}]dx \\
&= \frac{1}{\sqrt{2\pi}\sigma} \int_{-\infty}^{\infty}\exp[-\frac{(x-\mu)^2-2\sigma^2itx}{2\sigma^2}]dx \\
&=\frac{1}{\sqrt{2\pi}\sigma} \int_{-\infty}^{\infty}\exp[-\frac{(x-\sigma^2it-\mu)^2+\sigma^4t^2-2\sigma^2itu}{2\sigma^2}]dx \\
&= \frac{1}{\sqrt{2\pi}\sigma} \exp(\frac{-\sigma^2t^2}{2}+itu) \int_{-\infty}^{\infty}\exp[-\frac{(x-\sigma^2it-\mu)^2}{2\sigma^2}]dx \\
&\stackrel{y=(\frac{x-\sigma^2it-\mu}{\sigma})}\Longrightarrow \frac{1}{\sqrt{2\pi}} \exp(\frac{-\sigma^2t^2}{2}+itu)\int_{-\infty}^{\infty} \exp[-\frac{y^2}{2}]dy \\
&=\exp(\frac{-\sigma^2t^2}{2}+itu)
\end{aligned} (\#eq:ms-eq54)
$$

-----

### 多元正态分布的定义 {#ms_2_2}

1. 概率密度函数

$$
f(x)=\frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-\mu)'\Sigma^{-1}(x-\mu))  (\#eq:ms-eq55)
$$

   该定义要求$\Sigma > 0$，记$X\sim N_p(\mu, \Sigma)$。

2. 特征函数

$$
\varphi_X(t)=\exp(it'\mu-\frac{1}{2}t'\Sigma t) (\#eq:ms-eq56)
$$

   该定义要求$\Sigma \geq 0$，记$X\sim N_p(\mu, \Sigma)$。

3. 线性组合1

   设$Y_1,...Y_q \stackrel{iid}\sim N(0,1)$，A时$p \times q$常数矩阵，$\mu$为$p \times 1$常数向量，称q维随机向量$Y=(Y_1,...,Y_p)'$的线性组合$X=AY+\mu$的分布为p维正态分布，记记$X\sim N_p(\mu, \Sigma)$，其中$\Sigma=AA'$。
   
4. 线性组合2

   若p维随机向量$X=(X_1,...,X_p)'$的任意线性组合均服从一元正态分布，则称X为p维正态分布，记为$X\sim N_p(\mu, \Sigma)$。

-----
二元正态分布$N(\mu_1,\mu_2,\sigma_1^2,\sigma_2^2,\rho)$的概率密度函数

$$
f(x,y)=\frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}\exp\{-\frac{[\frac{(x-\mu_1)^2}{\sigma^2}-2\rho\frac{(x-\mu_1)(y-\mu_2)}{\sigma_1\sigma_2}+\frac{(y-\mu_2)^2}{\sigma^2}]}{2(1-\rho^2)}\} (\#eq:ms-eq57)
$$

-----

### 正态分布的条件分布和独立性  {#ms_2_3}

#### 条件分布 {#ms_2_3_1}

设$X \sim N_p(\mu, \Sigma), p \geq 2$，将$X,\,\mu, \, \Sigma$进行相同的分块，即$X=\begin{pmatrix} X^{(1)} \\ X^{(2)} \end{pmatrix}, \mu=\begin{pmatrix} \mu^{(1)} \\ \mu^{(2)} \end{pmatrix}, \Sigma=\begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}>0$，其中$X^{(1)}$为q维向量，$X^{(2)}$为p-q维向量。

1. 给定$X^{(2)}=x^{(2)}$时，$(X^{(1)}|X^{(2)}=x^{(2)}) \sim N_q(\mu_{1 \cdot 2},\Sigma_{11 \cdot 2})$，其中$\mu_{1 \cdot 2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}), \, \Sigma_{11 \cdot 2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$。

证1：

由分块矩阵逆的性质可知

$$
\Sigma^{-1}=\begin{pmatrix} \Sigma_{11 \cdot 2}^{-1} & -\Sigma_{11 \cdot 2}^{-1}\Sigma_{12}\Sigma_{22}^{-1} \\ -\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11 \cdot 2}^{-1} & \Sigma_{22}^{-1}+\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11 \cdot 2}^{-1}\Sigma_{12}\Sigma_{22}^{-1} \end{pmatrix} (\#eq:ms-eq58)
$$

由条件密度函数定义可得

$$
\begin{aligned}
f(x^{(1)}|x^{(2)}) &= \frac{f(x^{(1)},x^{(2)})}{f_{X^{(2)}}(x^{(2)})} \\
&=\frac{(2\pi)^{-\frac{p}{2}}|\Sigma|^{-\frac{1}{2}} \exp(-\frac{1}{2}(x-\mu)'\Sigma^{-1}(x-\mu))}{(2\pi)^{-\frac{p-q}{2}}|\Sigma_{22}|^{-\frac{1}{2}} \exp(-\frac{1}{2}(x^{(2)}-\mu^{(2)})'\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}))}
\end{aligned} (\#eq:ms-eq58)
$$

根据$(x-\mu)'=\begin{pmatrix} x^{(1)}-\mu^{(1)} \\ x^{(2)}-\mu^{(2)} \end{pmatrix}'$，及式\@ref(eq:ms-eq58)，可得

$$
\begin{aligned}
&\quad (x-\mu)'\Sigma^{-1}(x-\mu)-(x^{(2)}-\mu^{(2)})'\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}) \\
&=(x^{(1)}-\mu^{(1)})'\Sigma_{11 \cdot 2}^{-1}(x^{(1)}-\mu^{(1)})-(x^{(2)}-\mu^{(2)})'(\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11 \cdot 2}^{-1})(x^{(1)}-\mu^{(1)}) \\
&-(x^{(1)}-\mu^{(1)})'\Sigma_{11 \cdot 2}^{-1}\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}) \\
&+ (x^{(2)}-\mu^{(2)})'(\Sigma_{22}^{-1}+\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11 \cdot 2}^{-1}\Sigma_{12}\Sigma_{22}^{-1})(x^{(2)}-\mu^{(2)}) \\
&- (x^{(2)}-\mu^{(2)})'\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}) \\
&= (x^{(1)}-\mu^{(1)})'\Sigma_{11 \cdot 2}^{-1}(x^{(1)}-\mu^{(1)}-\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})) \\
&-(x^{(2)}-\mu^{(2)})'\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11 \cdot 2}^{-1}(x^{(1)}-\mu^{(1)}-\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})) \\
&= ((x^{(1)}-\mu^{(1)})'-(x^{(2)}-\mu^{(2)})'\Sigma_{22}^{-1}\Sigma_{21})\Sigma_{11\cdot 2}^{-1}(x^{(1)}-\mu^{(1)}-\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})) \\
&\stackrel{\mu_{1 \cdot 2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})}\Rightarrow (x^{(1)}-\mu_{1\cdot 2})'\Sigma_{11 \cdot 2}^{-1}(x^{(1)}-\mu_{1\cdot 2})
\end{aligned}  (\#eq:ms-eq59)
$$

则

$$
\begin{aligned}
f(x^{(1)}|x^{(2)})&=(2\pi)^{-\frac{q}{2}}|\Sigma_{11\cdot 2}|^{-\frac{1}{2}} \exp(-\frac{1}{2}(x^{(1)}-\mu_{1\cdot 2})'\Sigma_{11 \cdot 2}^{-1}(x^{(1)}-\mu_{1\cdot 2}))
\end{aligned} (\#eq:ms-eq60)
$$

故$(X^{(1)}|X^{(2)}=x^{(2)}) \sim N_q(\mu_{1 \cdot 2},\Sigma_{11 \cdot 2})$。

证2：

令$Y^{(1)}=X^{(1)}-\Sigma_{12}\Sigma_{22}^{-1}X_2^{(2)}$，$Y^{(2)}=X^{(2)}$，则有

$$
Y=
\begin{pmatrix}
Y^{(1)} \\
Y^{(2)}
\end{pmatrix}=
\begin{pmatrix}
I & -\Sigma_{12}\Sigma_{22}^{-1} \\
0 & I
\end{pmatrix}
\begin{pmatrix}
X^{(1)} \\
X^{(2)}
\end{pmatrix}=
AX (\#eq:ms-eq61)
$$

因为$X \sim N_p(\mu, \Sigma)$，所以$Y \sim N_p(A\mu, A\Sigma A')$。

其中

$$
A\mu = 
\begin{pmatrix}
\mu_1-\Sigma_{12}\Sigma_{22}^{-1}\mu_2 \\
\mu_2
\end{pmatrix} (\#eq:ms-eq62)
$$

$$
\begin{aligned}
A\Sigma A' &= 
\begin{pmatrix}
I & -\Sigma_{12}\Sigma_{22}^{-1} \\
0 & I
\end{pmatrix}
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}
\begin{pmatrix}
I & 0 \\
-\Sigma_{22}^{-1}\Sigma_{12} & I
\end{pmatrix} \\
&= 
\begin{pmatrix}
\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} & 0 \\
0 & \Sigma_{22}
\end{pmatrix} \\
&= \begin{pmatrix}
\Sigma_{11 \cdot 2}& 0 \\
0 & \Sigma_{22}
\end{pmatrix}
\end{aligned}  (\#eq:ms-eq63)
$$

故$Y^{(1)}$与$Y^{(2)}$独立。

已知$Y^{(1)} \sim N_q(\mu_1-\Sigma_{12}\Sigma_{22}^{-1}\mu_2,\Sigma_{11 \cdot 2})$，当给定$X^{(2)}=x^{(2)}$，即$Y^{(2)}=y^{(2)}$的条件下，$X^{(1)}=Y^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}x^{(2)} \sim N_q(\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu_2),\Sigma_{11 \cdot 2})$。

> 同理，给定$X^{(2)}=x^{(2)}$时，$(X^{(1)}|X^{(2)}=x^{(2)}) \sim N_q(\mu_{1 \cdot 2},\Sigma_{11 \cdot 2})$，其中$\mu_{1 \cdot 2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)}), \, \Sigma_{11 \cdot 2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$

#### 独立性 {#ms_2_3_2}

设$X \sim N_p(\mu,\Sigma),\, Y=AX+a, \, Z= BX+b$，则Y和Z独立当且仅当$A\Sigma B'=0$。

证：

$$
W=\begin{pmatrix} Y \\ Z \end{pmatrix}=\begin{pmatrix} A \\ B \end{pmatrix}X+\begin{pmatrix} a \\ b \end{pmatrix} \\
W\sim N\begin{pmatrix} \begin{pmatrix} A\mu+a \\ B\mu +b\end{pmatrix},\begin{pmatrix} A\Sigma A' & A\Sigma B' \\ B\Sigma A' & B\Sigma B'\end{pmatrix} \end{pmatrix} (\#eq:ms-eq64)
$$

显然当$A\Sigma B'=0$时，非主对角线元素为0，此时Y与Z独立。

### 偏相关系数与全相关系数  {#ms_2_4}

#### 偏相关系数  {#ms_2_4_1}

对于$(X^{(1)}|X^{(2)}=x^{(2)})\sim N_q(\mu_{1\cdot 2},\Sigma_{11 \cdot 2})$，有$E(X^{(1)}|X^{(2)})=\mu_{1 \cdot 2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(X^{(2)}-\mu^{(2)})$，该形式与回归分析中的条件期望回归类似，因此可将$\Sigma_{12}\Sigma_{22}^{-1}$视为$X^{(1)}$对$X^{(2)}$的回归系数。记$\Sigma_{11 \cdot 2}=(\sigma_{ij \cdot q+1,...,p})_{q \times q}$，则称$r_{ij \cdot q+1,...,p}=\frac{\sigma_{ij \cdot q+1,...,p}}{(\sigma_{ii \cdot q+1,...,p}\sigma_{jj \cdot q+1,...,p})^{\frac{1}{2}}}$为在给定$X^{(2)}$条件下$X_i$和$X_j$的偏相关系数。

> 就是在给定条件下对条件协差阵求相关系数

#### 全相关系数  {#ms_2_4_2}

对于随机向量X和随机变量y，设
$$
Z=\begin{pmatrix} X \\ y \end{pmatrix} \sim N\begin{pmatrix} \begin{pmatrix} \mu_X \\ \mu_y \end{pmatrix} , \begin{pmatrix} \Sigma_{XX} & \Sigma_{Xy} \\ \Sigma_{yX} & \sigma_{yy} \end{pmatrix} \end{pmatrix}
$$

称

$$R=\begin{pmatrix} \frac{\Sigma_{yX}\Sigma_{XX}^{-1}\Sigma_{Xy}}{\sigma_{yy}} \end{pmatrix}^{\frac{1}{2}}
$$
为y与X的全相关系数。

> 随机向量拼上随机变量，其中$\Sigma_{yX}$为$1 \times p$维向量，$\Sigma_{Xy}$为$p \times 1$维向量
> 
> 可以简记为大协差阵拆出两个标量构造分式，其中分母是$\sigma_{yy}$，整个分式取根号就是全相关系数

特别的，$R=\max\limits_{Cov(a'X)=1} corr(y,a'X)$。

证：

$$
\begin{aligned}
(corr(y,a'X))^2&= \frac{Cov^2(y,a'X)}{Cov(y)Cov(a'X)} \\
&= \frac{(\Sigma_{yX}a)^2}{\sigma_{yy}a'\Sigma_{XX}a} \\
&\leq \frac{(\Sigma_{Xy}'\Sigma_{XX}^{-1}\Sigma_{Xy})(a'\Sigma_{XX}a)}{\sigma_{yy}a'\Sigma_{XX}a} \\
&= \frac{(\Sigma_{Xy}'\Sigma_{XX}^{-1}\Sigma_{Xy})}{\sigma_{yy}} \\
&= \frac{\Sigma_{yX}\Sigma_{XX}^{-1}\Sigma_{Xy}}{\sigma_{yy}} \\
&= R^2
\end{aligned} (\#eq:ms-eq65)
$$

> Cauchy-Schwarz不等式：设$B>0$，则$(x'y)^2 \leq (x'Bx)(y'B^{-1}y)$。这里选取$B=\Sigma_{XX}$是为了把分母的$a'\Sigma_{XX}a$消掉

## 主成分分析 {#ms_3}

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

### 总体主成分 {#ms_3_1}

#### 基于协差阵的总体主成分 {#ms_3_1_1}

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

> 横向的，表示所有主成分对某个原始变量的方差解释百分比为100%

5. $\sum_{i=1}^p \sigma_{ii}\rho^2(Z_k,X_i)=\sum_{i=1}^p \sigma_{ii}\frac{\lambda_ka_{ik}^2}{\sigma_{ii}}=\lambda_k\sum_{i=1}^pa_{ik}^2=\lambda_k$

> 纵向的，表示单个主成分对所有原始变量的方差贡献，即为特征根

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

#### 基于相关阵的总体主成分 {#ms_3_1_2}

考虑变量量纲不同的影响，对数据进行标准化处理，得到$X_i^*=\frac{X_i-\mu_i}{\sqrt{\sigma_{ii}}}$，此时$X^*$的协差阵即为$X$的相关阵$R$。

同理可得，$Z_k^*=a_k^{*'}X^*=a_k^{*'}D^{-\frac{1}{2}}(X-\mu)$，其中$a_k^*=(a_{1k}^*,...,a_{pk}^*)'$是$R$对应于特征值$\lambda_k^*$的单位正交特征向量，$D^{\frac{1}{2}}=diag(\sqrt{\sigma_{11}},...,\sqrt{\sigma_{pp}})$，相关**性质**如下所示：

1. $Cov(Z^*)=Cov(P^{*'}X^*)=P^{*'}Cov(X^{*})P^{*}=P^{*'}RP^{*}=\Lambda^*$

2. $\sum \lambda_i^*=tr(\Lambda^*)=tr(P^{*'}RP^{*'})=tr(RP^*P^{*'})=tr(R)=p$

3. $\rho(Z_k^*, X_i^*)=\frac{Cov(Z_k^*, X_i^*)}{\sqrt{\lambda_k^*}}=\frac{Cov(a_k^{*'}X^*, e_i'X^*)}{\sqrt{\lambda_k^*}}=\frac{a_k^{*'}R e_i}{\sqrt{\lambda_k^*}}=\frac{\lambda_k^*a_k^{*'}e_i}{\sqrt{\lambda_k^*}}=\sqrt{\lambda_k^*}a_{ik}^{*}$

4. $\sum_{k=1}^p\rho^2(Z_k^*,X_i^*)=\sum_{k=1}^p \lambda_k^* a_{ik}^{*2}=1$

5. $\sum_{i=1}^p \rho^2(Z_k^*,X_i^*)=\sum_{i=1}^p \lambda_k^*a_{ik}^{*2}=\lambda_k^*\sum_{i=1}^pa_{ik}^{*2}=\lambda_k^*$

> 简单记，就把$\sigma_{ii}=1$代入基于协差阵的结果，对应符号添加*号即可

**什么时候需要标准化？**

1. 对变量进行标准化可以提高方差较小变量对主成分的贡献

2. 当变量量纲差异较大时需要进行标准化处理

3. 一般来说，从X的协方差矩阵导出的主成分和从其相关系数矩阵导出的主成分不相同，两个主成分没有显式数量关系，因此标准化后果是显著差异的

4. 实际问题处理中，考虑是否需要避免出现方差最大主成分和原始变量呈线性比例关系的分析结果

### 样本主成分 {#ms_3_2}

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

#### 基于协差阵的样本主成分 {#ms_3_2_1}

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

#### 基于相关阵的样本主成分 {#ms_3_2_2}

同样先对数据做标准化处理，标准化后的数据的协差阵即为原始数据的相关阵，其余操作同基于相关阵的总体主成分，只需要注意使用关于样本的符号即可。

### 关于主成分 {#ms_3_3}

#### 主成分的局限性 {#ms_3_3_1}

1. 仅考虑了原始变量的正交/线性变换。

2. PCA仅依赖于样本数据的均值和协方差矩阵，有些分布无法进行刻画。

3. 当原始变量是相关的时候，使用PCA可以降低维数，若原始变量不相关，则无法有效降维。

4. PCA容易受到异常点的影响。

#### 如何选取主成分个数 {#ms_3_3_2}

1. 前m个主成分的累积贡献率达到某个阈值，如80%或85%以上。

2. 无论是从协差阵还是相关阵出发，一个经验规则是保留特征值大于其平均值（或1）的主成分

3. 绘制碎石图看拐点。

#### R语言实现 {#ms_3_3_3}

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

## 因子分析 {#ms_4}

因子分析是一种降维的方法，通过探寻变量的潜在结构，将其归纳为较少的因子，从而实现降维的目的。

### 正交因子模型 {#ms_4_1}

设X为可观测的p维随机向量，其均值为$\mu$，协方差矩阵为$\Sigma$，则正交因子模型可被表示为

$$
\begin{array}{c}
X=AF+\mu+\varepsilon \\
E(F)=0, \; Cov(F)=I_m \\
E(\varepsilon)=0, \; Cov(\varepsilon)=\Psi=diag(\psi_1,...,\psi_2) \\
Cov(F,\varepsilon)=0
\end{array} (\#eq:ms-eq16)
$$

> 可通过中心化处理而消去$\mu$

其中$A$为$p \times m$的因子载荷矩阵，$F=(F_1, ...,F_m)'$为**公共因子**向量，$\varepsilon=(\varepsilon_1,...,\varepsilon_p)'$为**特殊因子**向量。

则X的协差阵可表示为

$$
\Sigma = Cov(X)=Cov(AF+\mu+\varepsilon)=ACov(F)A'+Cov(\varepsilon)=AA'+\Psi (\#eq:ms-eq17)
$$

若X已进行标准化处理，则

$$
R=AA'+\Psi (\#eq:ms-eq18)
$$

> 由于$\Psi$是对角阵，则$\Sigma$和$R$的非对角线元素全部由因子载荷矩阵A决定

1. 因子载荷矩阵的统计意义

$$
\begin{aligned}
Cov(X,F)&=E[X-E(X)][F-E(F)]' \\
&=E[(X-\mu)F'] \\
&= E[(AF+\varepsilon)F'] \\
&= AE(FF')+E(\varepsilon F') \\
&= A
\end{aligned} (\#eq:ms-eq19)
$$

由此可知，对于原始数据而言，因子载荷矩阵A记录了X与F之间的协方差，即$a_{ij}=Cov(X_i,F_j)$。

若对原始数据X进行标准化处理得到$X^*$，则

$$
\rho_{ij}=\frac{Cov(X_i,F_j)}{\sqrt{Var(X_i)}\sqrt{Var(F_j)}}=Cov(X_i, F_j)=a_{ij} (\#eq:ms-eq20)
$$

2. 变量共同度的统计意义

由式\@ref(eq:ms-eq16)可得

$$
Var(X_i)=\sigma_{ii}=\sum_{j=1}^m a_{ij}^2+\psi_i:=h_i+\psi_i (\#eq:ms-eq21)
$$

其中$h_i$表现为全部公共因子对$X_i$的变异程度的解释，反映了$X_i$对公共因子的依赖程度，称为**变量共同度**，即因子载荷矩阵A的**行元素平方和**。

3. 公共因子方差贡献的统计意义

$$
\sum_{i=1}^p Var(X_i) = \sum_{i=1}^p a_{i1}^2Var(F_1)+...+\sum_{i=1}^pa_{im}^2Var(F_m)+\sum_{i=1}^pVar(\varepsilon):=\sum_{j=1}^m q_j+\sum_{i=1}^p \psi_i (\#eq:ms-eq22)
$$

其中$q_j=\sum_{i=1}^p a_{ij}^2$，是因子载荷矩阵的列元素之和，反映了第j个公共因子对变量X的贡献及其相对重要性。进一步地，称$\frac{q_j}{\sum_{i=1}^pVar(X_i)}$为公共因子$F_j$所解释的总方差的比例。

性质：

1. 因子分析具有尺度不变性

   即对随机向量X的各个分量进行尺度放缩$X^*=CX, \; C=diag(c_1,...,c_p)$后的随机向量$X^*$仍满足正交因子模型的条件，且公共因子$F$不变。

> $CX=C\mu+CAF+C\varepsilon$，可将$CA$视作$A^*$因此公共因子$F$不变，变的是载荷矩阵

2. 因子载荷矩阵不唯一

   对于$AF$，可以有任一正交矩阵$\Gamma$，使得$AF=(A\Gamma)(\Gamma'F)$，其中可将$A\Gamma$视作$A^*$，将$\Gamma'F$视作$F^*$，进行因子旋转后仍保持正交因子模型的条件。
   
> 正交矩阵的作用相当于进行旋转

### 参数估计 {#ms_4_3}

在参数估计前，得先判断变量之间是否存在足够强的相关关系来进行因子分析。

1. KMO检验

   KMO检验用于检查变量间的相关性和偏相关性。KMO统计量的取值越接近1，表明变量间的相关性越强，偏相关性越弱，因子分析的效果越好。KMO统计量的值一般在0.7以上比较适合做因子分析。
   
> R语言：`KMO()`

2. Bartlett球形检验

   Bartlett球形检验的原假设是变量的相关阵是单位阵，当拒绝原假设时，则说明变量间具有相关性，因子分析有效。
   
> R语言：`psych::cortest.bartlett()`

#### 主成分法 {#ms_4_3_1}

令$\Sigma$的特征值为$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_p \geq 0$，对$\Sigma$进行谱分解

$$
\begin{aligned}
\Sigma &= P\Lambda P' \\
&= \sum_{i=1}^p \lambda_ip_ip_i' \\
&= \sum_{i=1}^m \lambda_ip_ip_i'+\sum_{i=m+1}^p \lambda_ip_ip_i' \\
&\approx 
\begin{pmatrix} \sqrt{\lambda_1}p_1 & ... & \sqrt{\lambda_m}p_m \end{pmatrix} 
\begin{pmatrix} \sqrt{\lambda_1}p_1' \\ \vdots \\ \sqrt{\lambda_m}p_m' \end{pmatrix} +
\begin{pmatrix} \psi_1 & \cdots & 0 \\
\vdots & \ddots & \vdots \\
0 & \cdots & \psi_p \end{pmatrix} \\
&= \hat A \hat A'+\hat \Psi
\end{aligned} (\#eq:ms-eq23)
$$

对$\Sigma$进行谱分解，取特征值较大的前m个特征值及对应的特征向量构成因子载荷矩阵A，忽视掉其余较小的特征值及对应的特征向量。其中的$\psi_i$则通过$\hat \psi_i = \sigma_{ii}-\sum_{j=1}^m a_{ij}^2$计算得到。

> 此时因子载荷矩阵中的每一列都是$\sqrt{\lambda_i}p_i$。特别的，当$\lambda_{m+1}=...=\lambda_p=0$时，此时有$\Sigma=\hat A \hat A'$，表明$\hat A=P\Lambda^{\frac{1}{2}}$就是真实的载荷矩阵
> 
> 当然也可以对变量进行标准化后利用主成分法进行求解。此时公共因子$F_j$的贡献为$q_j=\sum_{i=1}^p \hat a_{ij}^2=||\sqrt{\lambda_j}p_j||^2 = \lambda_j$
>
> m的选择可参考主成分分析

#### 主因子法 {#ms_4_3_2}

由于因子具有尺度不变性，对X进行标准化后得到

$$
R=AA'+\Psi (\#eq:ms-eq24)
$$

若$R$和$\Psi$都已知，则称$R^*=R-\Psi=AA'$为“约相关矩阵”。

其中$R$可表示为

$$
\begin{cases}
\rho_{ij}=a_{i1}a_{j1}+a_{i2}a_{j2}+...+a_{im}a_{jm} \quad &1 \leq i \neq j \leq q \\
\rho_{ii} = a_{i1}^2 + a_{i2}^2 + ...+a_{im}^2+\psi_i = h_i+\psi_i=1 \quad &i=1,2,...,p
\end{cases} (\#eq:ms-eq25)
$$

$R^*$可表示为

$$
\begin{cases}
\rho_{ij}=a_{i1}a_{j1}+a_{i2}a_{j2}+...+a_{im}a_{jm} \quad &1 \leq i \neq j \leq q \\
\rho_{ii} = a_{i1}^2 + a_{i2}^2 + ...+a_{im}^2= h_i \quad &i=1,2,...,p
\end{cases} (\#eq:ms-eq26)
$$

> 关键点为$h_i=1-\psi_i$

设$R^*$的特征值为$\lambda_1^* \geq \lambda_2^* \geq ... \geq \lambda_p^* \geq 0$，对应的特征向量为$p_i^*$，此时可对$R^*$进行谱分解，得到

$$
R^*=R-\Psi=\hat A \hat A' \approx \begin{pmatrix} \sqrt{\lambda_1^*}p_1^* & ... & \sqrt{\lambda_m^*}p_m^* \end{pmatrix} 
\begin{pmatrix} \sqrt{\lambda_1^*}p_1^{*'} \\ \vdots \\ \sqrt{\lambda_m^*}p_m^{*'} \end{pmatrix} (\#eq:ms-eq27)
$$

此时$\hat A = \begin{pmatrix} \sqrt{\lambda_1^*}p_1^* & ... & \sqrt{\lambda_m^*}p_m^* \end{pmatrix}$，特殊因子方差的估计为$\hat \psi_i = 1-\sum_{j=1}^m \hat a_{ij}^2$

> 主因子法的前提是得知道$\Psi$，但在实际中$\Psi$一般未知，因此可根据样本进行估计。也可根据迭代算法，根据初始值$\Psi^{(0)}$开始迭代，得到$\hat A^{(1)}$，再根据$\hat A^{(1)}$得到$\Psi^{(1)}$，不断重复，直至收敛

#### 极大似然法 {#ms_4_3_3}

假设公共因子$F \sim N_m(0,I)$，特殊因子$\varepsilon\sim N_p(0,\Psi)$，且相互独立，则$X \sim N_p(\mu, \Sigma)$。构造似然函数如下所示

$$
L(\mu, \Sigma)=(2\pi)^{-\frac{np}{2}}|\Sigma|^{-\frac{n}{2}}\exp[-\frac{1}{2}\sum_{i=1}^n(X_{(i)}-\mu)'\Sigma^{-1}(X_{(i)}-\mu)] (\#eq:ms-eq28)
$$

其中$\mu$的极大似然估计为$\hat \mu = \bar X$，回代可得

$$
\begin{aligned}
L(\Sigma)&=(2\pi)^{-\frac{np}{2}}|\Sigma|^{-\frac{n}{2}}\exp[-\frac{1}{2}\sum_{i=1}^n(X_{(i)}-\bar X)'\Sigma^{-1}(X_{(i)}-\bar X)] \\
&= (2\pi)^{-\frac{np}{2}}|\Sigma|^{-\frac{n}{2}}\exp[-\frac{1}{2}tr(\sum_{i=1}^n(X_{(i)}-\bar X)'\Sigma^{-1}(X_{(i)}-\bar X))] \\
&= (2\pi)^{-\frac{np}{2}}|\Sigma|^{-\frac{n}{2}}\exp[-\frac{1}{2}tr(\sum_{i=1}^n(X_{(i)}-\bar X)(X_{(i)}-\bar X)'\Sigma^{-1})] \\
&= (2\pi)^{-\frac{np}{2}}|\Sigma|^{-\frac{n}{2}}\exp[-\frac{1}{2}tr(D\Sigma^{-1})]
\end{aligned} (\#eq:ms-eq29)
$$

其中$D=\sum_{i=1}^n(X_{(i)}-\bar X)(X_{(i)}-\bar X)'$为样本离差阵。

根据$\Sigma = AA'+\Psi$，则

$$
L(A,\Psi)=(2\pi)^{-\frac{np}{2}}|AA'+\Psi|^{-\frac{n}{2}}\exp[-\frac{1}{2}tr(D(AA'+\Psi)^{-1})] (\#eq:ms-eq30)
$$

求解$\hat A$和$\hat \Psi$使得对数似然函数$\ln L(A,\Psi)$达到最大即可。

#### R语言实现 {#ms_4_3_4}

R语言`factanal()`、`psych::fa()`

### 因子旋转 {#ms_4_4}

前面已经介绍了正交因子模型的因子载荷矩阵不唯一。为了因子载荷矩阵的可解释性，可对因子载荷矩阵进行旋转，常用的方法有**最大方差旋转法**，该方法使得各列载荷向量的方差尽可能地大，从而在一列载荷向量上同时出现较大的载荷与较小的载荷，则该因子主要表现为较大的载荷。

### 因子得分 {#ms_4_5}

Bartlett因子得分和Thomson因子得分。

### 因子分析和主成分分析的区别与联系 {#ms_4_6}

区别：

1. 主成分分析中主成分是变量的线性组合；而因子分析中自变量是因子的线性组合

2. 主成分分析中的主成分不可旋转且唯一，在解释性上稍差；因子分析中的因子不唯一，可以通过因子旋转来获取较好的解释性

3. 主成分分析只能通过对协方差阵进行特征分解来获取主成分；因子分析能够通过主成分法、主因子法、极大似然法等多种方法进行估计

联系：

1. 二者都是以降维为目的，希望通过少量彼此不相关的主成分或因子来反映原始变量的绝大部分信息
   
2. 二者分析的变量间均需要具有一定的相关性

3. 二者都可运用原始变量的协差阵或相关阵进行求解

## 判别分析 {#ms_5}

### 距离判别 {#ms_5_1}

根据样品距离各个总体的远近来判断该样品属于哪个总体。记$d(X,G_i)$表示样品$X$到总体$G_i$的距离，则判别法则可表示为

$$
\delta(X) = \mathop{\arg\max}\limits_{i} \; d(X,G_i) (\#eq:ms-eq31)
$$

距离取马氏距离$d(X,G_i)=(X-\mu)'\Sigma_i^{-1}(X-\mu)$，其中$\Sigma_i$表示总体$G_i$的协方差矩阵。

#### 两总体且具有相同协差阵 {#ms_5_1_1}

当两总体$\Sigma_1=\Sigma_2=\Sigma$时，有

$$
\begin{aligned}
&d(X,G_1)-d(X,G_2) \\
=& (X-\mu_1)'\Sigma^{-1}(X-\mu_1)-(X-\mu_2)'\Sigma^{-1}(X-\mu_2) \\
&= X'\Sigma^{-1}X-2X'\Sigma^{-1}\mu_1+\mu_1'\Sigma^{-1}\mu_1-(X'\Sigma^{-1}X-2X'\Sigma^{-1}\mu_2+\mu_2'\Sigma^{-1}\mu_2) \\
&= 2X'\Sigma^{-1}(\mu_2-\mu_1)+\mu_1'\Sigma^{-1}\mu_1-\mu_2'\Sigma^{-1}\mu_2 \\
&= 2X'\Sigma^{-1}(\mu_2-\mu_1)+(\mu_1+\mu_2)'\Sigma^{-1}(\mu_1-\mu_2) \\
&= -2(X-\frac{\mu_1+\mu_2}{2})'\Sigma^{-1}(\mu_1-\mu_2) \\
&= -2(X-\bar \mu)'a \\
&= -2a'(X-\bar \mu)
\end{aligned} (\#eq:ms-eq32)
$$

其中$\bar \mu=\frac{\mu_1+\mu_2}{2}$，$a=\Sigma^{-1}(\mu_1-\mu_2)$，称a为**判别系数向量**。

称$W(X)=a'(X-\bar \mu)$为判别函数，则

$$
\begin{cases}
X\in G_1, &\textrm{if } W(X) \geq 0 \\
X\in G_2, &\textrm{if } W(X) \lt 0
\end{cases} (\#eq:ms-eq33)
$$

进一步地，当已知总体分布时，不妨令$G_1 \sim N_p(\mu_1, \Sigma)$，则当$X \in G_1$时，$W(X)=a'(X-\bar \mu) \sim N(a'(\mu_1-\mu_2)/2,a'\Sigma a)$。

根据$a=\Sigma^{-1}(\mu_1-\mu_2)$，又有$(\mu_1-\mu_2)=\Sigma\Sigma^{-1}(\mu_1-\mu_2)=\Sigma a$，故令$\Delta^2=(\mu_1-\mu_2)'\Sigma^{-1} (\mu_1-\mu_2) = a'(\mu_1-\mu_2)=a'\Sigma a$。则$W(X) \sim N(\frac{1}{2}\Delta^2,\Delta^2)$，故误判概率为

$$
P(2|1) = P(W(X)<0|X\in G_1)=\Phi(-\frac{\Delta}{2}) (\#eq:ms-eq34)
$$

同理，对于$G_2 \sim N_p(\mu_2, \Sigma)$，当$X \in G_2$时，有$W(X)\sim N(-\frac{1}{2}\Delta^2,\Delta^2)$，误判概率为

$$
P(1|2)=P(W(X) \geq 0|X \in G_2)=1-\Phi(\frac{\Delta}{2})=\Phi(-\frac{\Delta}{2}) (\#eq:ms-eq35)
$$

> 在实践中，常用样本均值替代总体均值，样本方差替代总体方差，若假设等方差，则使用联合估计

若不能假定总体为正态分布时，可用样本的误判比例替代误判概率。

1. 回代法

   用所有样本构造判别函数，再用该判别函数去判断样品所属总体，计算误判比例。
   
2. 划分样本

   将所有样本划分为训练集和验证集，训练集用于构造判别函数，验证集用于计算误判比例。
   
3. 交叉验证法

   分别从$G_1$、$G_2$中各取出1个观测值，再用剩余观测值来构造判别函数，根据判别函数来对这两个观测值进行判断，记录判断结果。不断重复这个过程，最后计算误判比例。

#### 其他情形 {#ms_5_1_2}

这里的其他情形是指多总体的或协差阵不相同的情况，核心思想还是找到最小的$d(X,G_i)$。

例如多总体且协差阵相同时，有

$$
d(X,G_i)=X'\Sigma^{-1}X-2\mu_i' \Sigma^{-1} X+\mu_i'\Sigma^{-1} \mu_i=X'\Sigma^{-1}X-2(I_i'X+c_i) (\#eq:ms-eq35)
$$

其中$I_i=\Sigma^{-1}\mu_i, \; c_i=-\frac{1}{2}\mu_i'\Sigma^{-1}\mu_i$。可以看到，$d(X,G_i)$中的$X'\Sigma^{-1}X$是固定的，因此判别规则即为$X\in G_i, \textrm{if }I_i'X+c_i=\max\limits_{j}(I'_jX+c_j)$。

### 贝叶斯判别 {#ms_5_2}

贝叶斯判别相较于距离判别法多考虑了先验信息，并根据样本信息得到后验概率分布，通过后验概率分布来进行统计推断。

假设有k个总体G，各个总体的概率密度函数为$f_i(x)$，先验概率为$q_i$，则根据样本得到的后验概率为

$$
P(G_i|x)=\frac{q_if_i(x)}{\sum_{j=1}^kq_jf_j(x)} (\#eq:ms-eq36)
$$

> 先验概率可根据历史资料或经验得到，或者用训练样本中各类所占比例作为先验概率，或者直接令各类先验概率均相等

#### 最大后验概率法 {#ms_5_2_1}

根据样品计算得到的后验概率进行判别，取最大后验概率对应的总体作为样品的判别结果，即

$$
X \in G_i, \quad \textrm{if } P(G_i|x)=\max\limits_{j} P(G_j|x) (\#eq:ms-eq37)
$$

当总体为正态分布时，对应的密度函数为

$$
f_i(x)=(2\pi)^{-\frac{p}{2}}|\Sigma_i|^{-\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu_i)'\Sigma_i^{-1}(x-\mu_i))=(2\pi)^{-\frac{p}{2}}|\Sigma_i|^{-\frac{1}{2}}\exp(-\frac{1}{2}d(x,G_i)) (\#eq:ms-eq38)
$$

则后验概率为

$$
\begin{aligned}
P(G_i|x)&=\frac{q_i(2\pi)^{-\frac{p}{2}}|\Sigma_i|^{-\frac{1}{2}}\exp(-\frac{1}{2}d(x,G_i))}{\sum_{j=1}^k q_j(2\pi)^{-\frac{p}{2}}|\Sigma_j|^{-\frac{1}{2}}\exp(-\frac{1}{2}d(x,G_j))} \\
&= \frac{q_i|\Sigma_i|^{-\frac{1}{2}}\exp(-\frac{1}{2}d(x,G_i))}{\sum_{j=1}^k q_j |\Sigma_j|^{-\frac{1}{2}}\exp(-\frac{1}{2}d(x,G_j))} \\
&= \frac{\exp (-\frac{1}{2}d(x,G_i)+\ln q_i-\frac{1}{2}\ln |\Sigma_i|)}{\sum_{j=1}^k \exp (-\frac{1}{2}d(x,G_j)+\ln q_j-\frac{1}{2}\ln |\Sigma_j|)} \\
&= \frac{\exp [-\frac{1}{2}(d(x,G_i)-2\ln q_i+\ln |\Sigma_i|)]}{\sum_{j=1}^k \exp [-\frac{1}{2}(d(x,G_j)-2\ln q_j+\ln |\Sigma_j|)]}
\end{aligned} (\#eq:ms-eq39)
$$

当假定先验概率均相等或者协差阵均相等时，对应的$q$和$|\Sigma|$均可消掉。不妨令$D(x,G_i)=d(x,G_i)+\ln|\Sigma_i|-2\ln q_i$，称其为广义平方距离，则正态假设下的最大后验概率法等价于

$$
X \in G_i, \quad \textrm{if } D(x,G_i)=\min\limits_{j} D(x,G_j) (\#eq:ms-eq40)
$$

> 注意$\exp$里面有负号，所以最大化就变成最小化了

#### 最小期望误判代价法 {#ms_5_2_2}

以两总体的情形为例，沿用最大后验概率法中的记号，并引入误判损失，记为$C(2|1),\,C(1|2)$，分别表示来自总体$G_1$却被误判为$G_2$的损失和来自总体$G_2$却被误判为$G_1$的损失。在判别规则下，$R^p$空间被划分为两个空间$R_1$和$R_2$，且$R_1 \cup R_2=\Omega, \,R_1 \cap R_2= \phi $。

因此，若样品来自$G_1$，则正确判别的概率为

$$
P(1|1)=P(x\in R_1 | x \in G_1)=\int_{R_1}f_1(x)dx (\#eq:ms-eq41)
$$

误判的概率为

$$
P(2|1)=P(x\in R_2 | x \in G_1)=\int_{R_2}f_1(x)dx (\#eq:ms-eq42)
$$

若样品来自$G_2$同理。

定义期望误判损失(Expected Cost of Misclassification)

$$
\begin{aligned}
ECM&=E(C(l|i)) \\
&= C(2|1)P(x\in G_1, x\in R_2)+C(1|2)P(x\in G_2, x\in R_1) \\
&= C(2|1)P(x \in R_2 | x \in G_1)P(x \in G_1)+C(1|2)P(x \in R_1 | x \in G_2)P(x \in G_2) \\
&= C(2|1)P(2|1)q_1+C(1|2)P(1|2)q_2 \\
&= C(2|1)q_1\int_{R_2}f_1(x)dx+C(1|2)q_2\int_{R_1}f_2(x)dx \\
&= C(2|1)q_1\int_{R_2}f_1(x)dx+C(1|2)q_2(1-\int_{R_2}f_2(x)dx) \\
&= \int_{R_2} [C(2|1)q_1f_1(x)-C(1|2)q_2f_2(x)]dx + C(1|2)q_2 \\
&\propto \int_{R_2} [C(2|1)q_1f_1(x)-C(1|2)q_2f_2(x)]dx \\
&=1- \int_{R_1} [C(2|1)q_1f_1(x)-C(1|2)q_2f_2(x)]dx
\end{aligned} (\#eq:ms-eq43)
$$

积分是曲线下的有向面积，当$C(2|1)q_1f_1(x)-C(1|2)q_2f_2(x)\geq 0$时，对其积分会变大，为了最小化ECM，应归为$R_1$，同理，当$C(2|1)q_1f_1(x)-C(1|2)q_2f_2(x)\lt 0$时，应归为$R_2$，即

$$
R_1=\{x:C(2|1)q_1f_1(x)\geq C(1|2)q_2f_2(x)\} \\
R_2=\{x:C(2|1)q_1f_1(x) \lt C(1|2)q_2f_2(x)\} (\#eq:ms-eq44)
$$

当总体为正态分布，且协差阵相等时，则

$$
\begin{aligned}
C(2|1)q_1f_1(x) &\geq C(1|2)q_2f_2(x) \\
\frac{f_1(x)}{f_2(x)} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
\frac{\exp\{-\frac{1}{2}(x-\mu_1)'\Sigma^{-1}(x-\mu_1)\}}{\exp\{-\frac{1}{2}(x-\mu_2)'\Sigma^{-1}(x-\mu_2)\}} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
\exp\{-\frac{1}{2}[(x-\mu_1)'\Sigma^{-1}(x-\mu_1)-(x-\mu_2)'\Sigma^{-1}(x-\mu_2)]\} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
\exp\{-\frac{1}{2}[-2x'\Sigma^{-1}\mu_1+\mu_1'\Sigma^{-1}\mu_1+2x'\Sigma^{-1}\mu_2-\mu_2'\Sigma^{-1}\mu_2]\} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
\exp\{-\frac{1}{2}[-2x'\Sigma^{-1}(\mu_1-\mu_2)+(\mu_1+\mu_2)'\Sigma^{-1}(\mu_1-\mu_2)]\} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
\exp\{(x-\frac{\mu_1+\mu_2}{2})'\Sigma^{-1}(\mu_1-\mu_2) \} &\geq \frac{C(1|2)q_2}{C(2|1)q_1} \\
a'(x-\bar \mu) &\geq \ln{[\frac{C(1|2)q_2}{C(2|1)q_1}]}
\end{aligned} (\#eq:ms-eq45)
$$

其中$a=\Sigma^{-1}(\mu_1-\mu_2), \, \bar \mu =\frac{\mu_1+\mu_2}{2}$。这等价于

$$
R_1=\{x:a'(x-\bar \mu) \geq \ln{[\frac{C(1|2)q_2}{C(2|1)q_1}]}\} \\
R_2=\{x:a'(x-\bar \mu) \lt \ln{[\frac{C(1|2)q_2}{C(2|1)q_1}]}\} (\#eq:ms-eq46)
$$

> 这里出现了距离判别中的$a'(x-\bar \mu)$，当不考虑先验概率和误判代价时，距离判别和贝叶斯判别等价

上述假定了两总体是正态的，且协差阵相等的情形，倘若协差阵不相等时，同第[4.3.1.2节](@#ms_5_1_2)一样，直接把正态密度函数的指数部分看成马氏距离$d(x,G_i)$，并且保留$|\Sigma_i|$，再进行讨论。

更一般的情形则是

$$
\begin{aligned}
ECM&=E[C(l|i)] \\
&=\sum_{i=1}^k\sum_{l=1}^k C(l|i)P(x\in G_i,x \in R_l) \\
&= \sum_{i=1}^k\sum_{l=1}^k C(l|i)P(x \in R_l|x\in G_i)P(x\in G_i) \\
&= \sum_{i=1}^k\sum_{l=1}^k C(l|i)P(l|i)q_i \\
&= \sum_{l=1}^k\sum_{i=1}^k C(l|i)P(l|i)q_i
\end{aligned} (\#eq:ms-eq47)
$$

最后两行虽然都是对所有情形的遍历，但内外求和顺序的不同代表着其思想也是不同。倒数第二行的外层是对$i$求和，内层是对$l$求和，其含义是先确定样品所属的总体，再考虑把该样品判给其他总体所带来的损失和。而最后一行的外层是对$l$求和，内层是对$i$求和，其含义是先确定把样品判别给某一总体，再考虑不同总体下的样品判别给该总体的损失和。显然，后者更符合我们判别的诉求，也就是把样品归类，并计算这样做的平均代价。因此让ECM最小的判别规则为

$$
R_t=\{x:\sum_{i\neq t}^k C(t|i)P(t|i)q_i = \min_{l} \sum_{i\neq l}^k C(l|i)P(l|i)q_i\} (\#eq:ms-eq48)
$$

> 考虑$ECM=10=2+4+1+3$，这意味着现在有四个总体，把这个样品判别给这四个总体会分别带来2、4、1、3的平均损失，我们要最小化ECM，因此会选择平均损失为1的那个总体作为该样品的判别结果

## 聚类分析 {#ms_6}

### 距离与相似性的度量 {#ms_6_1}

#### 距离的度量 {#ms_6_1_1}

距离的定义应当具有如下性质

1. 非负性：$d_{ij} \geq 0$

2. 对称性：$d_{ij}=d_{ji}$

3. 三角不等式： $d_{ij} \leq d_{ik} + d_{kj}$

常用的距离有闵可夫斯基距离、欧氏距离、马氏距离、曼哈顿距离。

#### 相似性的度量 {#ms_6_1_2}

有时可能会对指标之间进行聚类，指标间的距离常用相似系数进行度量。相似系数具有如下性质

1. $c_{ij}= \pm 1 \Leftrightarrow X_i=aX_j, \, a\neq 0$

2. $|c_{ij}| \leq 1$，若$|c_{ij}|$越接近1则变量间关系越密切，越接近0则关系越疏远

3. $c_{ij}=c_{ji}$

常用的相似系数有相关系数、夹角余弦。

> 距离与相似系数之间可以相互转化

### 聚类效果的评价指标 {#ms_6_2}

#### 内部评价指标 {#ms_6_2_1}

1. 轮廓系数

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i), b(i)\}}
$$

$a(i)$表示样本i到同簇其他点的平均距离，度量簇内紧密度；$b(i)$表示样本i到最近邻簇所有点的平均距离，度量簇间分离度。

平均轮廓系数越大，表明聚类效果越好。

2. DBI

$$
DBI = \frac{1}{k}\sum_{i=1}^k \max_{j \neq i }(\frac{\sigma_i + \sigma_j}{d(c_i, c_j)})
$$

其中$\sigma_i$表示簇i内样本到质心的平均距离，$d(c_i, d_j)$表示簇i与j质心间距离。

DBI越小越好。

3. CH指数

$$
CH = \frac{SSB/(k-1)}{SSW/(n-k)}
$$

其中$SSB$表示簇间平方和，$SSW$表示簇内平方和。

CH值越大越好。

#### 外部评价指标 {#ms_6_2_2}

1. 调整兰德指数

调整兰德指数是在兰德指数的基础上考虑了随机聚类的期望值，兰德指数定义为

$RI=\frac{a+d}{a+b+c+d}$

其中a表示同属于一个簇且同属于一个真实类的样本对数，b表示同属于一个簇但不属于一个真实类的样本对数，c表示不属于一个簇但同属于一个真实类的样本对数，d表示不属于一个簇且不属于一个真实类的样本对数。

而调整兰德指数定义为

$$
ARI = \frac{RI-E(RI)}{\max(RI)-E(RI)}
$$

2. 混淆矩阵

### 系统聚类 {#ms_6_3}

#### 类与类之间的距离 {#ms_6_3_1}

1. 最短距离法：$D_{KL} = \min\limits_{i\in G_K, j \in G_L} \{d_{ij}\}$

> 适用于长条形的或不规则现状的类，对球形类的效果不是很好

2. 最长距离法：$D_{KL} = \max\limits_{i\in G_K, j \in G_L} \{d_{ij}\}$

> 倾向于产生直径相等的类，易受异常值的影响

3. 类平均法：$D_{KL}=\frac{\sum_{i\in G_K, j \in G_L}d_{ij}}{n_Kn_L}$

> 两类中所有距离的平均数，倾向于先合并方差小的类，而且偏向于产生方差相同的类

4. 中间距离法：$D_{MJ}^2=\frac{1}{2}D_{KJ}^2+\frac{1}{2}D_{LJ}^2-\frac{1}{4}D_{KL}^2$

> 平行四边形的对角线的中线距离，

5. 重心法：$D_{KL}^2 = ||\bar X_K - \bar X_L||^2 = (\bar X_K-\bar X_L)'(\bar X_K-\bar X_L)$

> 注意这里是平方距离，是重心差的内积。和其他系统聚类方法相比在处理异常值方面更稳健

6. Ward法：$D_{KL}^2 = ||\bar X_K - \bar X_L||^2/(\frac{1}{n_K}+\frac{1}{n_L})$

> 在重心法的基础上多除了各自数量的倒数和。该法倾向于先合并样品少的类，且对异常值非常敏感

下面是不同形状数据及各种距离的系统聚类方法效果。

1. 能完全分开的球状数据

   各种距离的系统聚类方法均适用。
   
2. 不能完全分开的球状数据

   - Ward法、类平均法、重心法的聚类形状差不多，Ward法比较适合样本大小相等的球状数据的聚类。
   
   - 最短距离法最差。
   
3. 样本大小不等的球状数据

   重心法的聚类效果最好，类平均法偏向产生方差相等的类。
   
4. 并排拉长的数据

   - 最短距离法效果最好。
   
   - Ward法、类平均法、重心法都不行。数据进行预处理后可以得到较好的聚类结果。

5. 非球状数据

   - 最短距离法聚类效果最好。
   
   - Ward法、类平均法、重心法都不行，即使对数据进行预处理后仍不起作用。
   
系统距离的特点：

1. 无需事先指定类的数目。

2. 需要确定相异度（距离/相似性系数）和联接度量准则。

3. 运算量较大，适用于小规模数据。

4. 一旦完成合并或分裂，则无法撤销或修正。

### Kmeans {#ms_6_4}

算法步骤：

1. 输入观测样本数据和聚类数目K。

2. 随机将所有观测样本分配到K个类中，作为样本初始类。

3. 分别计算K个类的中心$\bar x^{(i)}, \, i=1,...,K$。

4. 计算每个观测样本到其所属类的中心的平方距离$SSE=\sum_{i=1}^K\sum_{j\in G_i} ||x_j-\bar x^{(i)}||^2$。

5. 重新将每个观测样本分配到距离其最近的类中心所在的类中，使得SSE减少。

6. 重复第3-5步，直至SSE不在减少，得到最终的聚类结果。

kmeans聚类的特点：

1. 核心思想是使同类内所有样本点的总体差异性尽可能小。

2. 适用于中大规模样本数据。

3. 需要事先指定聚类的数目K。

4. 不同的初始类可能会导致不同结果。

5. 适用于发现球状类。

6. 可能会陷入局部解。

7. 对异常值非常敏感。

### 其他聚类方法 {#ms_6_5}

#### DBSCAN {#ms_6_5_1}

DBSCAN(Density-Based Spatial Clustering of Applications with Noise)基于数据点在空间分布上的密度来发现任意形状的簇。簇被定义为密度相连的点的最大集合。密度低于某个阈值的区域被视为噪声。

> 并非所有观测点都属于簇，允许噪声的存在

DBSCAN需要定义两个参数：半径`eps`与区域内最少点数`minPts`。对于任意一个点，在其半径内若存在多于区域内最少点数个点，那么就将该店标记为**核心点**，找到所有的核心点，并将所有密度可达的核心点归于一个簇中。非核心点若与核心点相邻则也属于该簇，反之则为离群点。

在R中，可用`dbscan`包的`dbscan()`实现，并通过`kNNdistplot()`来辅助判断半径大小。

> 当给定`minPts`时（一般minPts>=维度数+1），`kNNdistplot()`的横轴表示按到k近邻的距离从小到大排序的点，纵轴表示距离，找到曲线的拐点对应的距离就是较为合适的半径

#### GMM {#ms_6_5_2}

假设整个数据集是K个多元高斯分布的混合体。每个点属于某个簇的概率由该高斯分布生成的概率决定。通常使用EM算法迭代优化。

在R中，可用`mclust`包的`Mclust()`实现。

## 典型相关分析 {#ms_7}

设X为随机向量，Y为随机变量，且满足$Cov\begin{pmatrix} X \\ Y\end{pmatrix}=\begin{pmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \sigma_{Y} \end{pmatrix}, \, E\begin{pmatrix} X \\ Y\end{pmatrix}=0$，则**复相关系数**定义如下

$$
\max_{a \in R^p} \rho(Y,a'X)= \max_{a \in R^p} \frac{a' \Sigma_{XY}}{\sigma_Y\sqrt{a'\Sigma_{XX}a}}=\frac{1}{\sigma_Y}\sqrt{\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}} (\#eq:ms-eq49)
$$

设X为p维随机向量，Y为q维随机向量，且满足$Cov\begin{pmatrix} X \\ Y\end{pmatrix}=\Sigma=\begin{pmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{YX} & \Sigma_{YY} \end{pmatrix} > 0$，设$a_1,\,b_1$分别为p维和q维任意非零的常数向量，使得

$$
\max_{Var(a_1'X)=1,Var(b_1'Y)=1} \rho(a_1'X,b_1'Y)=\max_{Var(a_1'X)=1,Var(b_1'Y)=1}a_1'\Sigma_{XY}b_1 (\#eq:ms-eq50)
$$

则称$(a_1'X,b_1Y)$为第1对**典型相关变量**，他们之间的相关系数为第1**典型相关系数**。

接下来同主成分分析一样，要求第k典型相关系数，要求第k对典型变量与前k-1对典型变量均不相关，并且要求第k典型相关系数在$Var(a_k'X)=Var(b_k'Y)=1$的约束下是最大的。

