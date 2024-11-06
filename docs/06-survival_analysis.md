


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

## 算法复现 {#survival_2}

本节内容是对Simon等人[@survival_1]论文的复现。

### 符号与目标 {#survival_2_1}

#### 符号 {#survival_2_1_1}

1. $(y_i,x_i,\delta_i)$

   $y_i$：生存时间
   
   $x_i$：解释变量的向量，$x_i=(x_{i1},\cdots,x_{ip})$
   
   $\delta_i$：若为0，则为右删失数据；若为1，则为failure
   
2. $t_1<t_2<\cdots<t_m$ & $j(i)$

   $t_1<t_2<\cdots<t_m$：在$y$中选取$\delta=1$的数据，进行升序排列
   
   > 注意是从1到m
   
   $j(i)$：$t_i$对应的观测对象$j$
   
3. $R_i$

   满足$t_i \leq y_j$的$j$的集合
   
4. $\hat \eta=X\tilde \beta$

5. $z(\tilde \eta)=\tilde \eta-\mathcal{l}''(\tilde \eta)^{-1}l'(\tilde \eta)$

6. $C(\tilde \eta, \tilde \beta)$

   与$\beta$无关的项
   
7. $w(\tilde \eta)_i$

   $l''(\tilde \eta)$的第$i$个对角线元素
   
8. $C_k$

   满足$t_i \lt y_k$的$i$的集合
   
9. $D_i$

   对于有“结”的情况，failure time为$t_i$的集合

10. $\omega_i$
   
   权重
   
11. $d_i=\sum_{j \in D_i}\omega_j$

   failure time为$t_i$的观测对象的权重和
   

``` r
cox_cd <- function(y, X, lambda, alpha, beta_0=NULL, max.iter=100){
  n = dim(X)[1]
  status = y[,'status']
  y = y[,'time']
  
  failure_t = y[status==1] %>% sort()
  R = map(failure_t, ~which(y>=.)) #R中每个元素对应原文的R_i
  C = map(y, ~which(failure_t<.))
  
  # 根据是否有ties运行不同代码
  if(length(y)==length(unique(y))){
    # 无ties
    weight = 1/n #原文无ties情况的1/n就是有ties情况下权重为1/n的情形
    j_i = as.numeric(y %in% failure_t)   #即原文的j(i)
    # log_likelihood_beta用于精度判断
    log_likelihood_beta <- function(beta){
      term_1 = as.numeric(j_i %*% X %*% beta)
      # 2.4节有一个递推式，待议
      term_2 = map_vec(R, function(R_i){
        map_vec(R_i, ~exp(X[.,] %*% beta)) %>% sum() %>% log()
      }) %>% sum()
      result = term_1 - term_2
      result
    }
    
    # 初始化beta
    if(is.null(beta_0)){
      beta = rep(0,dim(X)[2])
    }else{
      beta = beta_0
    }
    
    for (i in 1:max.iter) {
      print(paste0("第", i, "次迭代"))
      
      eta = X %*% beta
      w = map2_vec(C, c(1:length(C)), function(.x, .y){
        # 计算w_k
        C_k = .x
        k = .y   # .y提供位置索引
        eta_k = as.numeric(eta[k,])
        exp_eta_k = exp(eta_k)
        exp_eta_k_2 = exp_eta_k^2
        w_k = map_vec(C_k, function(i){
          sum_exp_eta_Ri = map_vec(R[[i]], ~exp(eta[.,])) %>% sum()
          sum_exp_eta_Ri_2 = sum_exp_eta_Ri^2
          value = (exp_eta_k * sum_exp_eta_Ri - exp_eta_k_2) / sum_exp_eta_Ri_2
          value
        }) %>% sum()
        w_k = -weight * w_k   #该代码的w始终和coxgrad的grad差负号，数值上略微差异
        w_k
      })
      w_sub = map2_vec(C, c(1:length(C)), function(.x, .y){
        # 计算w_k
        C_k = .x
        k = .y   # .y提供位置索引
        eta_k = as.numeric(eta[k,])
        exp_eta_k = exp(eta_k)
        w_sub_k = map_vec(C_k, function(i){
          sum_exp_eta_Ri = map_vec(R[[i]], ~exp(eta[.,])) %>% sum()
          value = exp_eta_k / sum_exp_eta_Ri
          value
        }) %>% sum()
        w_sub_k = weight * w_sub_k
        w_sub_k
      })
      if(any(w==0)) stop('w中有零')
      z = eta + (status - w_sub) / w
      
      for (k in 1:length(beta)) {
        denominator = as.numeric(w %*% X[,k]^2 + lambda * (1-alpha))
        numerator = as.numeric(diag(w) %*% X[,k] %*% (z - X[,-k] %*% beta[-k]))
        numerator = sign(numerator) * max(abs(numerator), lambda * alpha)
        beta[k] = numerator/denominator
      }
      # 精度判断
      # 无ties情况下，l_saturated = 0
      D_null = -2 * log_likelihood_beta(rep(0,length(beta)))
      D_current = -2 * log_likelihood_beta(beta)
      if(D_current - D_null >= 0.99 * D_null){
        print('满足精度要求')
        paste0('D_current:', D_current)
        break
      }
    }
    return(beta)
  }else{
    # 有ties
    message('有ties')
  }
}
```


``` r
library(glmnet)

data(CoxExample)
X <- CoxExample[[1]][1:50,1:5]
y <- CoxExample[[2]][1:50,]

n = dim(X)[1]
status = y[,'status']
y = y[,'time']
  
failure_t = y[status==1] %>% sort()
R = map(failure_t, ~which(y>=.)) #R中每个元素对应原文的R_i
C = map(y, ~which(failure_t<.))

weight = 1/n #原文无ties情况的1/n就是有ties情况下权重为1/n的情形

# 初始化beta
beta = rep(0,dim(X)[2])

# 第一次迭代
eta = X %*% beta
w_hessian = map2_vec(C, c(1:length(C)), function(.x, .y){
# 计算w_k
C_k = .x
k = .y   # .y提供位置索引
eta_k = as.numeric(eta[k,])
exp_eta_k = exp(eta_k)
exp_eta_k_2 = exp_eta_k^2
w_k = map_vec(C_k, function(i){
  sum_exp_eta_Ri = map_vec(R[[i]], ~exp(eta[.,])) %>% sum()
  sum_exp_eta_Ri_2 = sum_exp_eta_Ri^2
  value = (exp_eta_k * sum_exp_eta_Ri - exp_eta_k_2) / sum_exp_eta_Ri_2
      value
  }) %>% sum()
w_k = -weight * w_k   #该代码的w始终和coxgrad的grad差负号，数值上略微差异
w_k
})
```


``` r
fid <- function(x,index) {
    idup=duplicated(x)
    if(!any(idup)) list(index_first=index,index_ties=NULL)
    else {
        ndup=!idup
        xu=x[ndup]# first death times
        index_first=index[ndup]
        ities=match(x,xu)
        index_ties=split(index,ities)
        nties=sapply(index_ties,length)
        list(index_first=index_first,index_ties=index_ties[nties>1])
    }
}

X <- CoxExample[[1]][1:50,1:5]
y <- CoxExample[[2]][1:50,]

w=rep(1,length(eta))
w=w/sum(w)
nobs <- nrow(y)
time <- y[, "time"]
d    <- y[, "status"]
eta <- scale(eta, TRUE, FALSE)
o <- order(time, d, decreasing = c(FALSE, TRUE))
exp_eta <- exp(eta)[o]
time <- time[o]
d <- d[o]
w <- w[o]
rskden <- rev(cumsum(rev(exp_eta*w)))
dups <- fid(time[d == 1],seq(length(d))[d == 1])
dd <- d
ww <- w
rskcount=cumsum(dd)
rskdeninv=cumsum((ww/rskden)[dd==1])
rskdeninv=c(0,rskdeninv)
grad <- w * (d - exp_eta * rskdeninv[rskcount+1])
grad[o] <- grad
rskdeninv2 <- cumsum((ww/(rskden^2))[dd==1])
rskdeninv2 <- c(0, rskdeninv2)
w_exp_eta <- w * exp_eta
diag_hessian <- w_exp_eta^2 * rskdeninv2[rskcount+1] - w_exp_eta * rskdeninv[rskcount+1]
diag_hessian[o] <- diag_hessian
```

``` r
w_hessian
```

```
##  [1] -0.0183293825 -0.0035368399 -0.0008510459 -0.0090531224 -0.0049389213
##  [6] -0.0028709660  0.0000000000 -0.0320793825 -0.0064465663 -0.0201293825
## [11] -0.0124020632 -0.0028709660 -0.0018033986 -0.0008510459 -0.0201293825
## [16] -0.0111576188 -0.0064465663 -0.0151487122 -0.0056785663 -0.0028709660
## [21]  0.0000000000 -0.0100503523 -0.0008510459 -0.0028709660 -0.0042256154
## [26] -0.0018033986 -0.0201293825  0.0000000000 -0.0201293825 -0.0072783243
## [31] -0.0013158986 -0.0023158639 -0.0081460929  0.0000000000 -0.0111576188
## [36] -0.0004164780 -0.0201293825 -0.0028709660 -0.0004164780 -0.0270793825
## [41] -0.0023158639 -0.0137285938 -0.0233293825 -0.0166764899 -0.0270793825
## [46] -0.0090531224 -0.0023158639 -0.0013158986 -0.0100503523 -0.0028709660
```

``` r
diag_hessian
```

```
##  [1] -0.0201293825 -0.0042256154 -0.0008510459 -0.0090531224 -0.0056785663
##  [6] -0.0028709660  0.0000000000 -0.0320793825 -0.0072783243 -0.0201293825
## [11] -0.0137285938 -0.0035368399 -0.0023158639 -0.0008510459 -0.0201293825
## [16] -0.0111576188 -0.0064465663 -0.0166764899 -0.0064465663 -0.0028709660
## [21]  0.0000000000 -0.0100503523 -0.0013158986 -0.0028709660 -0.0049389213
## [26] -0.0018033986 -0.0201293825 -0.0004164780 -0.0233293825 -0.0081460929
## [31] -0.0013158986 -0.0023158639 -0.0090531224  0.0000000000 -0.0124020632
## [36] -0.0008510459 -0.0201293825 -0.0028709660 -0.0004164780 -0.0320793825
## [41] -0.0028709660 -0.0151487122 -0.0270793825 -0.0183293825 -0.0270793825
## [46] -0.0100503523 -0.0023158639 -0.0018033986 -0.0111576188 -0.0028709660
```

   





