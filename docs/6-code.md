# (PART) 算法复现 {.unnumbered}

# 算法复现 {#code}

该部分内容记录本人对一些论文算法的复现结果。

## 惩罚Cox比例风险模型 {#code_1}

该节是对*Regularization paths for Cox's proportional hazards model via coordinate descent*[@survival_1]论文的复现。

该论文在Cox比例风险模型的基础上添加了弹性网惩罚函数，并利用坐标下降法进行求解，可通过`glmnet`包实现。

算法复现详见第[此处](#survival_5_x)，**注意复现结果与官方函数略有出入**。

## 异质截距项的线性模型 {#code_2}

该节是对*A Concave Pairwise Fusion Approach to Subgroup Analysis*[@code_2]论文的复现。

$$
y_i = \mu_i+\mathbf{x}_i^T\beta+\varepsilon_i (\#eq:code-1)
$$

该论文在普通线性模型的基础上，考虑了截距项的异质性，即样本内部可能存在亚组，同一亚组共享相同的截距项，不同的亚组之间的截距项不同。该论文利用增广拉格朗日法构造目标函数（引入融合惩罚项$\mu_i-\mu_j=\eta_{ij}$），并通过ADMM算法进行求解。

### 自定义算法 {#code_2_1}

算法逻辑：

1. 传入参数

   - y：向量，响应变量
   - x：矩阵，预测变量
   - scale：是否对x进行标准化，默认为`TRUE`
   - penalty：设置惩罚函数类型(L1、MCP、SCAD)
   - $\theta$：ADMM算法的惩罚系数
   - $\lambda$：惩罚函数中的惩罚系数
   - $\gamma$：惩罚函数中的正则化因子
   - tol：收敛精度，默认为0.00001
   - max_iter：最大迭代次数，默认为1000

2. 其余符号说明

   除了传入参数外，在运算过程中还有其它符号，其含义如下所示。
   
   - $\upsilon$：拉格朗日乘子
   - $Q$：$X(X^TX)^{-1}X^T$
   - $Delta$：$\{(e_i-e_j), i<j\}^T$，其中$e_i$表示第$i$个分量为1，其余分量为0的n维向量
   - $\delta\_\mu$：$\mu_i-\mu_j$
   - $\delta$：$\mu_i-\mu_j+\theta^{-1}\upsilon_{ij}$
   - $\alpha$：亚组的截距项

3. 初始值

   根据ols估计获取$\beta$、$\mu$、$\eta$、$\upsilon$的初始值。

4. 迭代

   每次循环按照$\mu, \beta, \delta, \eta, \upsilon$的顺序进行迭代。
   
   $$
   \begin{aligned}
   \mu^{(m+1)}&=(\theta \Delta^T\Delta+I_n-Q)^{-1}((I_n-Q)y+\theta \Delta^T(\eta^{(m)}-\theta^{-1}\upsilon^{(m)})) \\
   \beta^{(m+1)}&=(X^TX)^{-1}X^T(y-\mu^{(m+1)}) \\
   \delta_{ij}^{(m+1)}&=\mu_i^{(m+1)}-\mu_j^{(m+1)}+\theta^{-1}\upsilon_{ij}^{(m)} \\
   \eta_{ij}^{(m+1)}&=\textrm{Penalty} \\
   \upsilon_{ij}^{(m+1)} &= \upsilon_{ij}^{(m)}+\theta (\mu_i^{(m+1)}-\mu_j^{(m+1)}-\eta_{ij}^{(m+1)})
   \end{aligned}
   $$
   
   注意$\delta_{ij}$与$\eta_{ij}$是针对每一个分量而言的，得将每一个分量代入惩罚函数才能得到对应的结果。
   
5. 停止

   当迭代次数达到最大迭代次数或残差$r^{(m+1)}=\Delta\mu^{(m+1)}-\eta^{(m+1)}$的模长足够小时，则停止迭代。
   
6. 输出

   输出每个观测对象的截距项$\hat \mu_i$，回归系数$\hat \beta$，截距项对应的亚组标签，每个亚组的截距项均值$\hat \alpha_k$。

**注意，论文涉及到$i<j$的排序，自定义算法采取先控制i再迭代j的步骤进行排序，如(1,2)、(1,3)、(1,4)...该顺序影响了后面矩阵的排列方式与`lower.tri`**

***由于设备限制，为节省运行时间，并未实现原文中的“$\lambda$路径图”及“对$\lambda$调参”的功能***

-----------

**自定义算法如下所示**


``` r
library(tidyverse)
library(igraph)
library(R6)

SubgroupIntercept <- R6Class(
  classname <- 'SubgroupIntercept',
  
  public <- list(
    # 传入参数
    y = NULL,         # 响应变量
    x = NULL,         # 预测变量
    n = NULL,         # 样本容量
    scale = NULL,     # 是否标准化
    penalty = NULL,   # 惩罚函数类型：L1、MCP、SCAD
    theta = NULL,     # 式(4)二次项的惩罚系数
    lambda = NULL,    # 惩罚函数中的惩罚系数
    gamma = NULL,     # 惩罚函数中的正则化因子
    tol = NULL,       # 收敛精度
    max_iter = NULL,  # 最大迭代次数
    
    # 初始化
    initialize = function(y, x, scale = T, penalty, theta, lambda, gamma, tol = 0.00001, max_iter = 1000){
      self$y <- y
      self$x <- x
      self$n <- length(y)
      self$scale <- scale
      self$penalty <- penalty
      self$theta <- theta
      self$lambda <- lambda
      self$gamma <- gamma
      self$tol <- tol
      self$max_iter <- max_iter
    },
    
    # 主函数——运行
    run = function(){
      start_time <- Sys.time()
      
      # 检验输入是否合理
      private$validate()
      
      # 标准化协变量
      if(self$scale == T) self$x <- apply(self$x, 2, scale)
      
      # 获取初始值
      initial_value <- private$initial_value()
      private$beta <- initial_value[[1]]
      private$mu <- initial_value[[2]]
      private$eta <- initial_value[[3]]
      private$upsilon <- initial_value[[4]]
      
      # 计算Q矩阵
      Q <- private$gen_Q()
      
      # 计算Delta
      Delta <- private$gen_Delta()
      
      # 开始迭代
      for (i in 1:self$max_iter) {
        print(paste0('正在进行第[', i, ']次迭代'))
        private$mu <- private$iter_mu(Q, Delta, private$eta, private$upsilon)
        private$beta <- private$iter_beta(private$mu)
        private$delta_mu <- private$iter_delta_mu(private$mu)
        private$delta <- private$iter_delta(private$delta_mu, private$upsilon)
        private$eta <- private$iter_eta(private$delta, self$lambda)
        private$upsilon <- private$iter_upsilon(private$upsilon, private$delta_mu, private$eta)
        
        # 终止条件
        r <- Delta[[1]] %*% private$mu - private$eta
        if(sqrt(sum(r^2)) <= self$tol){
          print('达到精度要求')
          break
        }
      }
      
      # 获取亚组分类结果
      label <- private$cluster_eta(private$eta)
      
      # 根据亚组计算对应的alpha
      subgroup_alpha <- tibble(mu = private$mu, label = label)
      subgroup_alpha <- subgroup_alpha %>% 
        group_by(label) %>% 
        summarise(alpha = mean(mu), size = n())
      
      result <- list(mu = private$mu, beta = private$beta, label = label, alpha = subgroup_alpha)
      
      end_time <- Sys.time()
      cost_time <- end_time - start_time
      print(paste('花费时间：', round(cost_time, 3), "秒"))
      
      return(result)
    }
  ),
  
  private <- list(
    # 迭代参数
    mu = NULL,        # 截距项
    beta = NULL,      # 回归系数
    delta_mu = NULL,  # mu_i-mu_j
    delta = NULL,     # delta_ij=mu_i-mu_j+1/theta*upsilon
    eta = NULL,       # 由惩罚函数计算
    upsilon =NULL,    # 拉格朗日乘子
    
    # 验证输入是否正确
    validate = function(){
      if(!is.vector(self$y)) stop('Y要求为向量')
      if(!is.matrix(self$x)) stop('X要求为矩阵')
      if(!is.logical(self$scale)) stop('scale要求为布尔值')
      if(!self$penalty %in% c('L1', 'MCP','SCAD')) stop('请选择合适的惩罚函数')
      if(self$theta <= 0) stop('请选择合适的theta值')
      if(self$lambda <= 0) stop('请选择合适的lambda值')
      if(self$gamma <= 0) stop('请选择合适的gamma值')
      if(self$tol <= 0) stop('请选择合适的精度要求')
      if(self$max_iter <= 0) stop('请选择合适的最大迭代次数')
    },
    
    # 获取初始值
    initial_value = function(){
      df <- cbind(self$y, self$x) %>% as.data.frame()
      colnames(df) <- c('y', paste0('x_', 1:dim(x)[2]))
      ols <- lm(y~., data = df)
      beta_0 <- coef(ols)[-1]
      mu_0 <- (self$y - self$x %*% beta_0) %>% as.vector()
      eta_0 <- -outer(mu_0, mu_0, '-')   # lower.tri按列提取下三角，故添加负号
      eta_0 <- eta_0[lower.tri(eta_0)]
      upsilon_0 <- rep(0, times = self$n * (self$n - 1)/2)
      result <- list(beta_0 = beta_0, mu_0 = mu_0, eta_0 = eta_0, upsilon_0 = upsilon_0)
      return(result)
    },
    
    # 计算Q矩阵
    gen_Q = function(){
      Q <- self$x %*% solve(t(self$x) %*% self$x) %*% t(self$x)
      return(Q)
    },
    
    # 计算Delta矩阵
    gen_Delta = function(){
      gen_mat <- function(n){
        mat_1 <- if(n == self$n-1){
          NULL
        }else{
          matrix(0, nrow = self$n-1-n, ncol = n)
        }
        mat_2 <- matrix(1, nrow = 1, ncol = n, byrow = T)
        mat_3 <- diag(-1, nrow=n)
        mat <- rbind(mat_1, mat_2, mat_3)
        return(mat)
      }
      Delta <- as.list(c((self$n-1):1)) %>% map(~gen_mat(.))
      Delta <- do.call(cbind, Delta) %>% t()
      Delta_Delta <- diag(self$n, nrow = self$n) - matrix(1, nrow = self$n, ncol = self$n)
      result <- list(Delta = Delta, Delta_Delta = Delta_Delta)
      return(result)
    },
    
    # mu迭代式
    iter_mu = function(Q, Delta, eta_current, upsilon_current){
      term_1 <- solve(self$theta * Delta[[2]] + diag(1, nrow = self$n) - Q)
      term_2 <- (diag(1, nrow = self$n) - Q) %*% self$y + self$theta * t(Delta[[1]]) %*% (eta_current - 1/self$theta * upsilon_current)
      mu_next <- (term_1 %*% term_2) %>% as.vector()
      return(mu_next)
    },
    
    # beta迭代式
    iter_beta = function(mu_next){
      beta_next <- solve(t(self$x) %*% self$x) %*% t(self$x) %*% (self$y - mu_next)
      return(beta_next)
    },
    
    # eta迭代式
    iter_eta = function(delta_current, lambda){
      ST <- function(t, lambda) sign(t) * max(abs(t)-lambda,0)
      delta_list <- as.list(delta_current)
      delta_to_eta <- function(delta_ij){
        switch(self$penalty,
               'L1' = {
                 eta_ij <- ST(delta_ij, self$lambda/self$theta)
               },
               'MCP' = {
                 if(abs(delta_ij) <= (self$gamma*self$lambda)){
                   eta_ij <- ST(delta_ij, self$lambda/self$theta)/(1-1/(self$gamma*self$theta))
                 }else{
                   eta_ij <- delta_ij
                 }
               },
               'SCAD' = {
                 if(abs(delta_ij) <= (self$lambda+self$lambda/self$theta)){
                   eta_ij <- ST(delta_ij, self$lambda/self$theta)
                 }else if(abs(delta_ij) > (self$lambda+self$lambda/self$theta) & abs(delta_ij) <= (self$gamma*self$lambda)){
                   eta_ij <- ST(delta_ij, self$gamma*self$lambda/((self$gamma-1)*self$theta))/(1-1/((self$gamma-1)*self$theta))
                 }else{
                   eta_ij <- delta_ij
                 }
               }
        )
        return(eta_ij)
      }
      eta_next <- delta_list %>% map_vec(~delta_to_eta(.))
      return(eta_next)
    },
    
    # mu_i-mu_j迭代式
    iter_delta_mu = function(mu_next){
      delta_mu <- -outer(mu_next, mu_next, '-')
      delta_mu <- delta_mu[lower.tri(delta_mu)]
      return(delta_mu)
    },
    
    # delta迭代式
    iter_delta = function(delta_mu, upsilon_current){
      delta_next <- delta_mu + 1/self$theta * upsilon_current
      return(delta_next)
    },
    
    # upsilon迭代式
    iter_upsilon = function(upsilon_current, delta_mu, eta_next){
      upsilon_next <- upsilon_current + self$theta * (delta_mu - eta_next)
      return(upsilon_next)
    },
    
    # eta归类
    # 注意到根据eta为0来分组刚好符合图论中“连通分量”的概念
    cluster_eta = function(eta){
      mat <- matrix(NA, nrow = self$n, ncol = self$n)
      mat[lower.tri(mat)] <- eta             # 转化为矩阵方便提取索引
      link <- which(mat == 0, arr.ind = T)
      node <- data.frame(name = 1:self$n)
      graph <- graph_from_data_frame(link, directed = F, vertices = node)
      label <- components(graph)$membership
      return(label)
    }
  )
)
```

### 数据模拟 {#code_2_2}

下面随机生成100条观测，设置3个预测变量，均服从标准正态分布，且两两之间的相关系数为0.3，然后随机生成$\beta$值与$\varepsilon$，设置截距项以相等的概率随机分为1或-1，根据上述变量生成响应变量。


``` r
library(MASS)
set.seed(123)
sigma <- matrix(0.3, ncol = 3, nrow = 3) + diag(0.7, nrow = 3)
x <- mvrnorm(n=100, mu=rep(0,3), Sigma = sigma)
epsilon <- rnorm(100)
mu <- sample(c(1,-1), 100, replace = T, prob = rep(0.5, 2))
beta <- runif(3)
y <- (mu + x %*% beta + epsilon) %>% as.vector()
```

构建模型，并运行。


``` r
model <- SubgroupIntercept$new(y, x, penalty = 'MCP', theta = 1, lambda = 0.5, gamma = 3, max_iter = 10000)
result <- model$run()
```

```
## [1] "正在进行第[1]次迭代"
## [1] "正在进行第[2]次迭代"
## [1] "正在进行第[3]次迭代"
## [1] "正在进行第[4]次迭代"
## [1] "正在进行第[5]次迭代"
## [1] "正在进行第[6]次迭代"
## [1] "正在进行第[7]次迭代"
## [1] "正在进行第[8]次迭代"
## [1] "正在进行第[9]次迭代"
## [1] "正在进行第[10]次迭代"
## [1] "正在进行第[11]次迭代"
## [1] "正在进行第[12]次迭代"
## [1] "正在进行第[13]次迭代"
## [1] "正在进行第[14]次迭代"
## [1] "正在进行第[15]次迭代"
## [1] "正在进行第[16]次迭代"
## [1] "正在进行第[17]次迭代"
## [1] "正在进行第[18]次迭代"
## [1] "正在进行第[19]次迭代"
## [1] "正在进行第[20]次迭代"
## [1] "正在进行第[21]次迭代"
## [1] "正在进行第[22]次迭代"
## [1] "正在进行第[23]次迭代"
## [1] "正在进行第[24]次迭代"
## [1] "正在进行第[25]次迭代"
## [1] "正在进行第[26]次迭代"
## [1] "正在进行第[27]次迭代"
## [1] "正在进行第[28]次迭代"
## [1] "正在进行第[29]次迭代"
## [1] "正在进行第[30]次迭代"
## [1] "正在进行第[31]次迭代"
## [1] "正在进行第[32]次迭代"
## [1] "正在进行第[33]次迭代"
## [1] "正在进行第[34]次迭代"
## [1] "正在进行第[35]次迭代"
## [1] "正在进行第[36]次迭代"
## [1] "正在进行第[37]次迭代"
## [1] "正在进行第[38]次迭代"
## [1] "正在进行第[39]次迭代"
## [1] "正在进行第[40]次迭代"
## [1] "正在进行第[41]次迭代"
## [1] "正在进行第[42]次迭代"
## [1] "正在进行第[43]次迭代"
## [1] "正在进行第[44]次迭代"
## [1] "正在进行第[45]次迭代"
## [1] "正在进行第[46]次迭代"
## [1] "正在进行第[47]次迭代"
## [1] "正在进行第[48]次迭代"
## [1] "正在进行第[49]次迭代"
## [1] "正在进行第[50]次迭代"
## [1] "正在进行第[51]次迭代"
## [1] "正在进行第[52]次迭代"
## [1] "正在进行第[53]次迭代"
## [1] "正在进行第[54]次迭代"
## [1] "正在进行第[55]次迭代"
## [1] "正在进行第[56]次迭代"
## [1] "正在进行第[57]次迭代"
## [1] "正在进行第[58]次迭代"
## [1] "正在进行第[59]次迭代"
## [1] "正在进行第[60]次迭代"
## [1] "正在进行第[61]次迭代"
## [1] "正在进行第[62]次迭代"
## [1] "正在进行第[63]次迭代"
## [1] "正在进行第[64]次迭代"
## [1] "正在进行第[65]次迭代"
## [1] "正在进行第[66]次迭代"
## [1] "正在进行第[67]次迭代"
## [1] "正在进行第[68]次迭代"
## [1] "正在进行第[69]次迭代"
## [1] "正在进行第[70]次迭代"
## [1] "正在进行第[71]次迭代"
## [1] "正在进行第[72]次迭代"
## [1] "正在进行第[73]次迭代"
## [1] "正在进行第[74]次迭代"
## [1] "正在进行第[75]次迭代"
## [1] "正在进行第[76]次迭代"
## [1] "正在进行第[77]次迭代"
## [1] "正在进行第[78]次迭代"
## [1] "正在进行第[79]次迭代"
## [1] "正在进行第[80]次迭代"
## [1] "正在进行第[81]次迭代"
## [1] "正在进行第[82]次迭代"
## [1] "正在进行第[83]次迭代"
## [1] "正在进行第[84]次迭代"
## [1] "正在进行第[85]次迭代"
## [1] "正在进行第[86]次迭代"
## [1] "正在进行第[87]次迭代"
## [1] "正在进行第[88]次迭代"
## [1] "正在进行第[89]次迭代"
## [1] "正在进行第[90]次迭代"
## [1] "正在进行第[91]次迭代"
## [1] "正在进行第[92]次迭代"
## [1] "正在进行第[93]次迭代"
## [1] "正在进行第[94]次迭代"
## [1] "正在进行第[95]次迭代"
## [1] "正在进行第[96]次迭代"
## [1] "正在进行第[97]次迭代"
## [1] "正在进行第[98]次迭代"
## [1] "正在进行第[99]次迭代"
## [1] "正在进行第[100]次迭代"
## [1] "正在进行第[101]次迭代"
## [1] "正在进行第[102]次迭代"
## [1] "正在进行第[103]次迭代"
## [1] "正在进行第[104]次迭代"
## [1] "正在进行第[105]次迭代"
## [1] "正在进行第[106]次迭代"
## [1] "正在进行第[107]次迭代"
## [1] "正在进行第[108]次迭代"
## [1] "正在进行第[109]次迭代"
## [1] "正在进行第[110]次迭代"
## [1] "正在进行第[111]次迭代"
## [1] "正在进行第[112]次迭代"
## [1] "正在进行第[113]次迭代"
## [1] "正在进行第[114]次迭代"
## [1] "正在进行第[115]次迭代"
## [1] "正在进行第[116]次迭代"
## [1] "正在进行第[117]次迭代"
## [1] "正在进行第[118]次迭代"
## [1] "正在进行第[119]次迭代"
## [1] "正在进行第[120]次迭代"
## [1] "正在进行第[121]次迭代"
## [1] "正在进行第[122]次迭代"
## [1] "正在进行第[123]次迭代"
## [1] "正在进行第[124]次迭代"
## [1] "正在进行第[125]次迭代"
## [1] "正在进行第[126]次迭代"
## [1] "正在进行第[127]次迭代"
## [1] "正在进行第[128]次迭代"
## [1] "正在进行第[129]次迭代"
## [1] "正在进行第[130]次迭代"
## [1] "正在进行第[131]次迭代"
## [1] "正在进行第[132]次迭代"
## [1] "正在进行第[133]次迭代"
## [1] "正在进行第[134]次迭代"
## [1] "正在进行第[135]次迭代"
## [1] "正在进行第[136]次迭代"
## [1] "正在进行第[137]次迭代"
## [1] "正在进行第[138]次迭代"
## [1] "正在进行第[139]次迭代"
## [1] "正在进行第[140]次迭代"
## [1] "正在进行第[141]次迭代"
## [1] "正在进行第[142]次迭代"
## [1] "正在进行第[143]次迭代"
## [1] "正在进行第[144]次迭代"
## [1] "正在进行第[145]次迭代"
## [1] "正在进行第[146]次迭代"
## [1] "正在进行第[147]次迭代"
## [1] "正在进行第[148]次迭代"
## [1] "正在进行第[149]次迭代"
## [1] "正在进行第[150]次迭代"
## [1] "正在进行第[151]次迭代"
## [1] "正在进行第[152]次迭代"
## [1] "正在进行第[153]次迭代"
## [1] "正在进行第[154]次迭代"
## [1] "正在进行第[155]次迭代"
## [1] "正在进行第[156]次迭代"
## [1] "正在进行第[157]次迭代"
## [1] "正在进行第[158]次迭代"
## [1] "正在进行第[159]次迭代"
## [1] "正在进行第[160]次迭代"
## [1] "正在进行第[161]次迭代"
## [1] "正在进行第[162]次迭代"
## [1] "正在进行第[163]次迭代"
## [1] "正在进行第[164]次迭代"
## [1] "正在进行第[165]次迭代"
## [1] "正在进行第[166]次迭代"
## [1] "正在进行第[167]次迭代"
## [1] "正在进行第[168]次迭代"
## [1] "正在进行第[169]次迭代"
## [1] "正在进行第[170]次迭代"
## [1] "正在进行第[171]次迭代"
## [1] "正在进行第[172]次迭代"
## [1] "正在进行第[173]次迭代"
## [1] "正在进行第[174]次迭代"
## [1] "正在进行第[175]次迭代"
## [1] "正在进行第[176]次迭代"
## [1] "正在进行第[177]次迭代"
## [1] "正在进行第[178]次迭代"
## [1] "正在进行第[179]次迭代"
## [1] "正在进行第[180]次迭代"
## [1] "正在进行第[181]次迭代"
## [1] "正在进行第[182]次迭代"
## [1] "正在进行第[183]次迭代"
## [1] "正在进行第[184]次迭代"
## [1] "正在进行第[185]次迭代"
## [1] "正在进行第[186]次迭代"
## [1] "正在进行第[187]次迭代"
## [1] "正在进行第[188]次迭代"
## [1] "正在进行第[189]次迭代"
## [1] "正在进行第[190]次迭代"
## [1] "正在进行第[191]次迭代"
## [1] "正在进行第[192]次迭代"
## [1] "正在进行第[193]次迭代"
## [1] "正在进行第[194]次迭代"
## [1] "正在进行第[195]次迭代"
## [1] "正在进行第[196]次迭代"
## [1] "正在进行第[197]次迭代"
## [1] "正在进行第[198]次迭代"
## [1] "正在进行第[199]次迭代"
## [1] "正在进行第[200]次迭代"
## [1] "正在进行第[201]次迭代"
## [1] "正在进行第[202]次迭代"
## [1] "正在进行第[203]次迭代"
## [1] "正在进行第[204]次迭代"
## [1] "正在进行第[205]次迭代"
## [1] "正在进行第[206]次迭代"
## [1] "正在进行第[207]次迭代"
## [1] "正在进行第[208]次迭代"
## [1] "正在进行第[209]次迭代"
## [1] "正在进行第[210]次迭代"
## [1] "正在进行第[211]次迭代"
## [1] "正在进行第[212]次迭代"
## [1] "正在进行第[213]次迭代"
## [1] "正在进行第[214]次迭代"
## [1] "正在进行第[215]次迭代"
## [1] "正在进行第[216]次迭代"
## [1] "正在进行第[217]次迭代"
## [1] "正在进行第[218]次迭代"
## [1] "正在进行第[219]次迭代"
## [1] "正在进行第[220]次迭代"
## [1] "正在进行第[221]次迭代"
## [1] "正在进行第[222]次迭代"
## [1] "正在进行第[223]次迭代"
## [1] "正在进行第[224]次迭代"
## [1] "正在进行第[225]次迭代"
## [1] "正在进行第[226]次迭代"
## [1] "正在进行第[227]次迭代"
## [1] "正在进行第[228]次迭代"
## [1] "正在进行第[229]次迭代"
## [1] "正在进行第[230]次迭代"
## [1] "正在进行第[231]次迭代"
## [1] "正在进行第[232]次迭代"
## [1] "正在进行第[233]次迭代"
## [1] "正在进行第[234]次迭代"
## [1] "正在进行第[235]次迭代"
## [1] "正在进行第[236]次迭代"
## [1] "正在进行第[237]次迭代"
## [1] "正在进行第[238]次迭代"
## [1] "正在进行第[239]次迭代"
## [1] "正在进行第[240]次迭代"
## [1] "正在进行第[241]次迭代"
## [1] "正在进行第[242]次迭代"
## [1] "正在进行第[243]次迭代"
## [1] "正在进行第[244]次迭代"
## [1] "正在进行第[245]次迭代"
## [1] "正在进行第[246]次迭代"
## [1] "正在进行第[247]次迭代"
## [1] "正在进行第[248]次迭代"
## [1] "正在进行第[249]次迭代"
## [1] "正在进行第[250]次迭代"
## [1] "正在进行第[251]次迭代"
## [1] "正在进行第[252]次迭代"
## [1] "正在进行第[253]次迭代"
## [1] "正在进行第[254]次迭代"
## [1] "正在进行第[255]次迭代"
## [1] "正在进行第[256]次迭代"
## [1] "正在进行第[257]次迭代"
## [1] "正在进行第[258]次迭代"
## [1] "正在进行第[259]次迭代"
## [1] "正在进行第[260]次迭代"
## [1] "正在进行第[261]次迭代"
## [1] "正在进行第[262]次迭代"
## [1] "正在进行第[263]次迭代"
## [1] "正在进行第[264]次迭代"
## [1] "正在进行第[265]次迭代"
## [1] "正在进行第[266]次迭代"
## [1] "正在进行第[267]次迭代"
## [1] "正在进行第[268]次迭代"
## [1] "正在进行第[269]次迭代"
## [1] "正在进行第[270]次迭代"
## [1] "正在进行第[271]次迭代"
## [1] "正在进行第[272]次迭代"
## [1] "正在进行第[273]次迭代"
## [1] "正在进行第[274]次迭代"
## [1] "正在进行第[275]次迭代"
## [1] "正在进行第[276]次迭代"
## [1] "正在进行第[277]次迭代"
## [1] "正在进行第[278]次迭代"
## [1] "正在进行第[279]次迭代"
## [1] "正在进行第[280]次迭代"
## [1] "正在进行第[281]次迭代"
## [1] "正在进行第[282]次迭代"
## [1] "正在进行第[283]次迭代"
## [1] "正在进行第[284]次迭代"
## [1] "正在进行第[285]次迭代"
## [1] "正在进行第[286]次迭代"
## [1] "正在进行第[287]次迭代"
## [1] "正在进行第[288]次迭代"
## [1] "正在进行第[289]次迭代"
## [1] "正在进行第[290]次迭代"
## [1] "正在进行第[291]次迭代"
## [1] "正在进行第[292]次迭代"
## [1] "正在进行第[293]次迭代"
## [1] "正在进行第[294]次迭代"
## [1] "正在进行第[295]次迭代"
## [1] "正在进行第[296]次迭代"
## [1] "正在进行第[297]次迭代"
## [1] "正在进行第[298]次迭代"
## [1] "正在进行第[299]次迭代"
## [1] "正在进行第[300]次迭代"
## [1] "正在进行第[301]次迭代"
## [1] "正在进行第[302]次迭代"
## [1] "正在进行第[303]次迭代"
## [1] "正在进行第[304]次迭代"
## [1] "正在进行第[305]次迭代"
## [1] "正在进行第[306]次迭代"
## [1] "正在进行第[307]次迭代"
## [1] "正在进行第[308]次迭代"
## [1] "正在进行第[309]次迭代"
## [1] "正在进行第[310]次迭代"
## [1] "正在进行第[311]次迭代"
## [1] "正在进行第[312]次迭代"
## [1] "正在进行第[313]次迭代"
## [1] "正在进行第[314]次迭代"
## [1] "正在进行第[315]次迭代"
## [1] "正在进行第[316]次迭代"
## [1] "正在进行第[317]次迭代"
## [1] "正在进行第[318]次迭代"
## [1] "正在进行第[319]次迭代"
## [1] "正在进行第[320]次迭代"
## [1] "正在进行第[321]次迭代"
## [1] "正在进行第[322]次迭代"
## [1] "正在进行第[323]次迭代"
## [1] "正在进行第[324]次迭代"
## [1] "正在进行第[325]次迭代"
## [1] "正在进行第[326]次迭代"
## [1] "正在进行第[327]次迭代"
## [1] "正在进行第[328]次迭代"
## [1] "正在进行第[329]次迭代"
## [1] "正在进行第[330]次迭代"
## [1] "正在进行第[331]次迭代"
## [1] "正在进行第[332]次迭代"
## [1] "正在进行第[333]次迭代"
## [1] "正在进行第[334]次迭代"
## [1] "正在进行第[335]次迭代"
## [1] "正在进行第[336]次迭代"
## [1] "正在进行第[337]次迭代"
## [1] "正在进行第[338]次迭代"
## [1] "正在进行第[339]次迭代"
## [1] "正在进行第[340]次迭代"
## [1] "正在进行第[341]次迭代"
## [1] "正在进行第[342]次迭代"
## [1] "正在进行第[343]次迭代"
## [1] "正在进行第[344]次迭代"
## [1] "正在进行第[345]次迭代"
## [1] "正在进行第[346]次迭代"
## [1] "正在进行第[347]次迭代"
## [1] "正在进行第[348]次迭代"
## [1] "正在进行第[349]次迭代"
## [1] "正在进行第[350]次迭代"
## [1] "正在进行第[351]次迭代"
## [1] "正在进行第[352]次迭代"
## [1] "正在进行第[353]次迭代"
## [1] "正在进行第[354]次迭代"
## [1] "正在进行第[355]次迭代"
## [1] "正在进行第[356]次迭代"
## [1] "正在进行第[357]次迭代"
## [1] "正在进行第[358]次迭代"
## [1] "正在进行第[359]次迭代"
## [1] "正在进行第[360]次迭代"
## [1] "正在进行第[361]次迭代"
## [1] "正在进行第[362]次迭代"
## [1] "正在进行第[363]次迭代"
## [1] "正在进行第[364]次迭代"
## [1] "正在进行第[365]次迭代"
## [1] "正在进行第[366]次迭代"
## [1] "正在进行第[367]次迭代"
## [1] "正在进行第[368]次迭代"
## [1] "正在进行第[369]次迭代"
## [1] "正在进行第[370]次迭代"
## [1] "正在进行第[371]次迭代"
## [1] "正在进行第[372]次迭代"
## [1] "正在进行第[373]次迭代"
## [1] "正在进行第[374]次迭代"
## [1] "正在进行第[375]次迭代"
## [1] "正在进行第[376]次迭代"
## [1] "正在进行第[377]次迭代"
## [1] "正在进行第[378]次迭代"
## [1] "正在进行第[379]次迭代"
## [1] "正在进行第[380]次迭代"
## [1] "正在进行第[381]次迭代"
## [1] "正在进行第[382]次迭代"
## [1] "正在进行第[383]次迭代"
## [1] "正在进行第[384]次迭代"
## [1] "正在进行第[385]次迭代"
## [1] "正在进行第[386]次迭代"
## [1] "正在进行第[387]次迭代"
## [1] "正在进行第[388]次迭代"
## [1] "正在进行第[389]次迭代"
## [1] "正在进行第[390]次迭代"
## [1] "正在进行第[391]次迭代"
## [1] "正在进行第[392]次迭代"
## [1] "正在进行第[393]次迭代"
## [1] "正在进行第[394]次迭代"
## [1] "正在进行第[395]次迭代"
## [1] "正在进行第[396]次迭代"
## [1] "正在进行第[397]次迭代"
## [1] "正在进行第[398]次迭代"
## [1] "正在进行第[399]次迭代"
## [1] "正在进行第[400]次迭代"
## [1] "正在进行第[401]次迭代"
## [1] "正在进行第[402]次迭代"
## [1] "正在进行第[403]次迭代"
## [1] "正在进行第[404]次迭代"
## [1] "正在进行第[405]次迭代"
## [1] "正在进行第[406]次迭代"
## [1] "正在进行第[407]次迭代"
## [1] "正在进行第[408]次迭代"
## [1] "正在进行第[409]次迭代"
## [1] "正在进行第[410]次迭代"
## [1] "正在进行第[411]次迭代"
## [1] "正在进行第[412]次迭代"
## [1] "正在进行第[413]次迭代"
## [1] "正在进行第[414]次迭代"
## [1] "正在进行第[415]次迭代"
## [1] "正在进行第[416]次迭代"
## [1] "正在进行第[417]次迭代"
## [1] "正在进行第[418]次迭代"
## [1] "正在进行第[419]次迭代"
## [1] "正在进行第[420]次迭代"
## [1] "正在进行第[421]次迭代"
## [1] "正在进行第[422]次迭代"
## [1] "正在进行第[423]次迭代"
## [1] "正在进行第[424]次迭代"
## [1] "正在进行第[425]次迭代"
## [1] "正在进行第[426]次迭代"
## [1] "正在进行第[427]次迭代"
## [1] "正在进行第[428]次迭代"
## [1] "正在进行第[429]次迭代"
## [1] "正在进行第[430]次迭代"
## [1] "正在进行第[431]次迭代"
## [1] "正在进行第[432]次迭代"
## [1] "正在进行第[433]次迭代"
## [1] "正在进行第[434]次迭代"
## [1] "正在进行第[435]次迭代"
## [1] "正在进行第[436]次迭代"
## [1] "正在进行第[437]次迭代"
## [1] "正在进行第[438]次迭代"
## [1] "正在进行第[439]次迭代"
## [1] "正在进行第[440]次迭代"
## [1] "正在进行第[441]次迭代"
## [1] "正在进行第[442]次迭代"
## [1] "正在进行第[443]次迭代"
## [1] "正在进行第[444]次迭代"
## [1] "正在进行第[445]次迭代"
## [1] "正在进行第[446]次迭代"
## [1] "正在进行第[447]次迭代"
## [1] "正在进行第[448]次迭代"
## [1] "正在进行第[449]次迭代"
## [1] "正在进行第[450]次迭代"
## [1] "正在进行第[451]次迭代"
## [1] "正在进行第[452]次迭代"
## [1] "正在进行第[453]次迭代"
## [1] "正在进行第[454]次迭代"
## [1] "正在进行第[455]次迭代"
## [1] "正在进行第[456]次迭代"
## [1] "正在进行第[457]次迭代"
## [1] "正在进行第[458]次迭代"
## [1] "正在进行第[459]次迭代"
## [1] "正在进行第[460]次迭代"
## [1] "正在进行第[461]次迭代"
## [1] "正在进行第[462]次迭代"
## [1] "正在进行第[463]次迭代"
## [1] "正在进行第[464]次迭代"
## [1] "正在进行第[465]次迭代"
## [1] "正在进行第[466]次迭代"
## [1] "正在进行第[467]次迭代"
## [1] "正在进行第[468]次迭代"
## [1] "正在进行第[469]次迭代"
## [1] "达到精度要求"
## [1] "花费时间： 12.076 秒"
```

截距项的分组结果如下所示


``` r
result$alpha
```

```
## # A tibble: 4 × 3
##   label  alpha  size
##   <dbl>  <dbl> <int>
## 1     1 -1.39     44
## 2     2  0.995    51
## 3     3 -3.33      3
## 4     4  2.96      2
```

## （异质）线性+非线性的Cox比例风险模型 {#code_3}

该节是对*Subgroup detection in the heterogeneous partially linear additive Cox model*[@code_3]论文的复现。

$$
\lambda(t|X_i,Z_i)=\lambda_0(t)\exp\{X_i^T\beta_i+\sum_{j=1}^q f_j(Z_{ij})\} (\#eq:code-2)
$$

该论文在Cox比例风险模型的基础上，不单单引入回归系数的异质性，还引入了$f(\cdot)$来捕捉非线性效应。其中$f(\cdot)$利用B-spline去近似，同时也引入融合惩罚项$u_{ik}=\beta_i-\beta_k$与$Y'$，通过majorized ADMM算法进行求解。

### 自定义算法 {#code_3_1}

算法逻辑：

1. 传入参数

   - T：矩阵，必须包含列名“time”和“status”，分别表示观测时间和最终状态
   - X：具有线性效应、异质性的协变量矩阵
   - Z：具有非线性效应的协变量矩阵
   - penalty：惩罚函数类型，SCAD或MCP
   - K：K-means的聚类个数
   - $\lambda$：惩罚函数中的惩罚系数
   - a: 融合惩罚中的正则化因子，默认SCAD是3.7，MCP是2.5
   - $\theta$：majorized ADMM算法的惩罚系数
   - df：`splines::bs()`的参数，控制基函数个数，默认为6，详见`splines::bs()`，下同
   - degree：`splines::bs()`的参数，设置基函数的次数，默认为3
   - tol：收敛精度，默认为0.001
   - max_iter：最大迭代次数，默认为10000
   
2. 其余符号说明

   除了传入参数外，在运算过程中还有其它符号，其含义如下所示。
   
   - $\beta$：$(\beta_1^T,\cdots,\beta_n^T)^T$，长度为$np$的向量
   
   - $\gamma$：$(\gamma_1^T,\cdots,\gamma_q^T)^T$，长度为$dq$的向量，其中$\gamma_j=(\gamma_{j1},\cdots,\gamma_{jd})^T$
   
   - $u$：$(u_{ik}^T,i<k)^T$，长度为$\frac{n(n-1)}{2}p$的向量，其中$u_{ik}=\beta_i-\beta_k$
   
   - $Y$：$(Y_1,\cdots,Y_n)^T$，长度为$n$的向量，$Y_i=X_i^T\beta_i+B_i(Z_i)^T\gamma$
   
   > $Y'$的含义类似，只是与$Y$的更新规则不同
   
   - $w$：$(w_1,\cdots,w_n)^T$，长度为$n$的拉格朗日乘子向量
   
   - $\nu$：$(\nu_{ik}^T, i<k)^T$，长度为$\frac{n(n-1)}{2}p$的拉格朗日乘子向量
   
   - $B$：$(B_1(Z_1),\cdots,B_n(Z_n))^T$，$n\times (dq)$维矩阵，其中每个$B_i(Z_i)$有$q$个分量，每个分量又能由$d$个基函数表示
   
   > 就是按列拼接而成的基函数矩阵，并且经过列方向上的中心化处理
   
   - $X$：$\textrm{diag}\{X_1^T,\cdots,X_n^T\}$，应该是$n \times (pn)$维矩阵
   
   >这里的$\textrm{diag}$是针对$X_i^T$而言的，若把$X_i^T$展开就不是真正意义上的对角阵，而是类似阶梯状的矩阵
   
   - $D$：$\{(e_i-e_j), i<j\}^T$，$\frac{n(n-1)}{2} \times n$矩阵，$e_i$是第$i$个分量为1，其余分量为0并且长度为$n$的向量
   
   - $A$：$D \otimes I_p$，$\frac{n(n-1)}{2}p \times np$维矩阵
   
   - $Q$：$I_n-B(B^TB)^{-1}B^T$
   
   - $\tilde g_j$：$\sum_{i=1}^n \delta_iI_{j \in R_i}$，即第$j$个对象出现在所有风险集中的次数
   
   - $\nabla_ig(Y'^{(m+1)})$：$-\delta_i+\sum_{k=1}^n \delta_k[\exp (Y_i'^{(m+1)})\cdot I_{i \in R_k}]/[\sum_{l \in R_k} \exp (Y_l'^{(m+1)})]$
   
   - $c_{ik}$：$\beta_i-\beta_k+\nu_{ik}/\theta$

3. 初始值

   对矩阵$X$进行K-means聚类，一般聚成2-5类即可。对每一类分别拟合基础的Cox比例风险模型，将回归系数作为对应观测的初始值。其余参数的初始值分别设置为$u^{(0)}=A\beta^{(0)}, w^{(0)}=0,\nu^{(0)}=0,Y^{(0)}=Y'^{(0)}=X\beta^{(0)}+B\gamma^{(0)}$。
   
> 原文仅提到根据$X$先进行聚类，再拟合Cox模型，但由于假设$Z$是同质的，如果也先聚类再拟合，得到的各组回归系数并不相同甚至维度也不满足$\gamma$的长度，因此将协变量$Z$转化成$B$后单独拟合Cox模型，将该次回归系数作为$\gamma$的初始值

4. 迭代

   迭代顺序为$\gamma,\beta;Y',Y,u;w,\nu$
   
   $$
   \begin{aligned}
   \gamma^{(m+1)} &= (B^TB)^{-1}B^T(Y^{(m)}-X\beta^{(m)}+\frac{w^{(m)}}{\theta}) \\
   \beta^{(m+1)} &= (X^TQX+A^TA)^{-1}[X^TQ(\frac{w^{(m)}}{\theta}+Y^{(m)})+A^T(u^{(m)}-\frac{\nu ^{(m)}}{\theta})] \\
   Y_i'^{(m+1)} &= X_i^T\beta_i^{(m+1)}+B_i(Z_i)^T\gamma^{(m+1)} \\
   Y_i^{(m+1)} &= (\tilde g_i + \theta)^{-1} [-\nabla_ig(Y'^{(m+1)})+\tilde g_i Y_i'^{(m+1)} - w_i^{(m)}+\theta (X_i^T\beta_i^{(m+1)}+B_i(Z_i)^T\gamma^{(m+1)})] \\
   u^{(m+1)} &= \textrm{Penalty} \\
   w_i^{(m+1)} &= w_i^{(m)}+\theta (Y_i^{(m+1)}-X_i^T\beta^{(m+1)}-B_i(Z_i)^T\gamma^{(m+1)}) \\
   \nu_{ik}^{(m+1)} &= \nu_{ik}^{(m)}+\theta (\beta_i^{(m+1)}-\beta_k^{(m+1)}-u_{ik}^{(m+1)})
   \end{aligned}
   $$
   
5. 停止

   设置停止条件为达到最大迭代次数或者残差$r$满足一定的精度要求即可。
   
   $$
   r^{(m+1)} = ||A\beta^{(m+1)}-u^{(m+1)}||+||Y^{(m+1)}-X\beta^{(m+1)}-B\gamma^{(m+1)}||
   $$
   
6. 输出

   - beta：列表，X的异质性回归系数
   - gamma：向量，基函数的回归系数
   - label：向量，样本亚组标签
   - alpha：数据框，beta亚组


``` r
library(tidyverse)
library(survival)
library(Matrix)
library(splines)
library(R6)

SubgroupBeta <- R6Class(
  classname <-  'SubgroupBeta',
  
  public <-  list(
    # 传入参数
    T = NULL,           # 观测时间与最终状态
    X = NULL,           # 具有线性效应、异质性的协变量矩阵
    Z = NULL,           # 具有非线性效应的协变量矩阵
    n = NULL,           # 样本容量
    p = NULL,           # X维度p
    q = NULL,           # Z维度q
    penalty = NULL,     # 惩罚函数，SCAD或MCP，默认为MCP
    K = NULL,           # K-means聚类的个数，默认为2
    lambda = NULL,      # 惩罚函数的惩罚系数
    a = NULL,           # 融合惩罚中的正则化因子，默认SCAD是3.7，MCP是2.5
    theta = NULL,       # majorized ADMM算法的惩罚系数
    df = NULL,          # 基函数个数
    degree = NULL,      # 基函数的次数
    tol = NULL,         # 收敛精度，默认为0.001
    max_iter = NULL,    # 最大迭代次数，默认为10000
    
    # 初始化
    initialize = function(T, X, Z, penalty = 'MCP', K = 2, lambda, a = NULL, theta, df = 6, degree = 3, tol = 0.001, max_iter = 10000){
      self$T <- T
      self$X <- X
      self$Z <- Z
      self$n <- dim(T)[1]   # 观测数
      self$p <- dim(X)[2]   # 维度p
      self$q <- dim(Z)[2]   # 维度q
      self$penalty <- penalty
      self$K <- K
      self$lambda <- lambda
      self$a <- a
      self$theta <- theta
      self$df <- df
      self$degree <- degree
      self$tol <- tol
      self$max_iter <- max_iter
    },
    
    # 主函数——运行
    run = function(trace = TRUE){
      start_time <- proc.time()
      
      # 检验输入是否合理
      private$validate()
      
      # 获取初始值
      initial_value <- private$initial_value()
      private$beta <- initial_value[[1]]
      B <- initial_value[[2]]
      private$gamma <- initial_value[[3]]
      X_ls <- initial_value[[4]]
      private$Y <- initial_value[[5]]
      private$Y2 <- initial_value[[5]]
      A <- private$gen_A()
      private$u <- A %*% unlist(private$beta)
      private$w <- rep(0, self$n)
      private$nu <- rep(0, self$n * (self$n-1) * self$p / 2)
      
      # 计算矩阵Q
      Q <- diag(1, ncol = self$n, nrow = self$n)- B %*% solve(t(B) %*% B) %*% t(B)
      
      # 计算向量g
      g <- private$gen_g()
      
      # 计算(B'B)^{-1}B'，用于gamma更新
      B_ols <- solve(t(B) %*% B) %*% t(B)
      
      # 计算(X'QX+A'A)^{-1}与X'Q，用于beta更新
      X_diag <- map(X_ls, ~matrix(., nrow = 1, byrow = T))
      X_diag <- do.call(bdiag, X_diag) %>% as.matrix()         # 转化为对角形式的X
      XQX_AA <- solve(t(X_diag) %*% Q %*% X_diag + t(A) %*% A)
      XQ <- t(X_diag) %*% Q
      
      # 参数迭代
      for (i in 1:self$max_iter) {
        if(trace == TRUE) print(paste0('正在进行第[', i, ']次迭代'))
        private$gamma <- private$iter_gamma(X_ls, B_ols, private$Y, private$beta, private$w)
        private$beta <- private$iter_beta(XQX_AA, XQ, A, private$w, private$Y, private$u, private$nu)
        private$Y2 <- private$iter_Y2(X_ls, B, g, private$beta, private$gamma)
        private$Y <- private$iter_Y(g, private$Y2, private$w)
        private$u <- private$iter_u(self$penalty, private$beta, private$nu, A)
        private$w <- private$iter_w(private$w, private$Y, private$Y2)
        private$nu <- private$iter_nu(private$nu, private$beta, private$u, A)
        
        # 终止条件
        term_1 <- A %*% unlist(private$beta) - private$u
        term_2 <- private$Y - X_diag %*% unlist(private$beta) - B %*% private$gamma
        if((norm(term_1, type = '2') + norm(term_2, type = '2')) <= self$tol){
          if(trace == TRUE) print('达到精度要求')
          break
        }
      }
      
      # beta亚组
      private$beta <- lapply(private$beta, round, digits = 2)
      str_beta <- sapply(private$beta, paste, collapse = ',')
      label <- as.numeric(factor(str_beta, levels = unique(str_beta)))
      subgroup_beta <- tibble(label = label, beta = private$beta) %>% 
        group_by(label) %>% 
        summarise(size = n(), beta = unique(beta)) %>% 
        arrange(-size)
      K_hat <- unique(label) %>% length()
      result <- list(beta = private$beta, gamma = private$gamma, label = label, K_hat = K_hat, alpha = subgroup_beta)
      
      end_time <- proc.time()
      cost_time <- end_time - start_time
      print(cost_time)
      return(result)
    },
    
    # 主函数——调优
    tune_lambda = function(seq_lambda, trace = TRUE){
      start_time <- proc.time()
      
      # 检验输入是否合理
      private$validate()
      
      # 获取初始值
      initial_value <- private$initial_value()
      private$beta <- initial_value[[1]]
      B <- initial_value[[2]]
      private$gamma <- initial_value[[3]]
      X_ls <- initial_value[[4]]
      private$Y <- initial_value[[5]]
      private$Y2 <- initial_value[[5]]
      A <- private$gen_A()
      private$u <- A %*% unlist(private$beta)
      private$w <- rep(0, self$n)
      private$nu <- rep(0, self$n * (self$n-1) * self$p / 2)
      
      # 计算矩阵Q
      Q <- diag(1, ncol = self$n, nrow = self$n)- B %*% solve(t(B) %*% B) %*% t(B)
      
      # 计算向量g
      g <- private$gen_g()
      
      # 计算(B'B)^{-1}B'，用于gamma更新
      B_ols <- solve(t(B) %*% B) %*% t(B)
      
      # 计算(X'QX+A'A)^{-1}与X'Q，用于beta更新
      X_diag <- map(X_ls, ~matrix(., nrow = 1, byrow = T))
      X_diag <- do.call(bdiag, X_diag) %>% as.matrix()         # 转化为对角形式的X
      XQX_AA <- solve(t(X_diag) %*% Q %*% X_diag + t(A) %*% A)
      XQ <- t(X_diag) %*% Q
      
      bic_vec <- rep(0, length(seq_lambda))
      result_ls <- vector('list', length = length(seq_lambda))
      for (i in 1:length(seq_lambda)) {
        self$lambda <- seq_lambda[i]
        
        # Warm Start，仅保留上轮的beta和gamma，其余恢复为初始值设定
        private$Y <- private$Y2    # Y2 = X_beta + B_gamma，因此保留，故Y的初始值与Y2相同
        private$u <- A %*% unlist(private$beta)
        private$w <- rep(0, self$n)
        private$nu <- rep(0, self$n * (self$n-1) * self$p / 2)
        
        for (j in 1:self$max_iter) {
          if(trace == TRUE) print(paste0('lambda = ', self$lambda, '; 第[', j, ']次迭代'))
          private$gamma <- private$iter_gamma(X_ls, B_ols, private$Y, private$beta, private$w)
          private$beta <- private$iter_beta(XQX_AA, XQ, A, private$w, private$Y, private$u, private$nu)
          private$Y2 <- private$iter_Y2(X_ls, B, g, private$beta, private$gamma)
          private$Y <- private$iter_Y(g, private$Y2, private$w)
          private$u <- private$iter_u(self$penalty, private$beta, private$nu, A)
          private$w <- private$iter_w(private$w, private$Y, private$Y2)
          private$nu <- private$iter_nu(private$nu, private$beta, private$u, A)
          
          # 终止条件
          term_1 <- A %*% unlist(private$beta) - private$u
          term_2 <- private$Y - X_diag %*% unlist(private$beta) - B %*% private$gamma
          if((norm(term_1, type = '2') + norm(term_2, type = '2')) <= self$tol){
            if(trace == TRUE) cat('==========\n')
            if(trace == TRUE) print(paste0('lambda = ', self$lambda, ' 达到精度要求'))
            break
          }
        }
        
        # beta亚组
        private$beta <- lapply(private$beta, round, digits = 2)
        str_beta <- sapply(private$beta, paste, collapse = ',')
        label <- as.numeric(factor(str_beta, levels = unique(str_beta)))
        subgroup_beta <- tibble(label = label, beta = private$beta) %>% 
          group_by(label) %>% 
          summarise(size = n(), beta = unique(beta)) %>% 
          arrange(-size)
        K_hat <- unique(label) %>% length()
        
        bic_vec[i] <- private$Bic(private$Y2, K_hat)
        if(trace == TRUE) print(paste0('BIC: ', round(bic_vec[i],3)))
        if(trace == TRUE) cat('==========\n')
        result <- list(beta = private$beta, gamma = private$gamma, label = label, K_hat = K_hat, alpha = subgroup_beta, 
                       best_lambda = self$lambda, BIC = bic_vec[i])
        result_ls[[i]] <- result
      }
      
      best_index <- which.min(bic_vec)
      best_result <- result_ls[[best_index]]
      
      end_time <- proc.time()
      print(end_time - start_time)
      
      return(list(best_result = best_result, BIC = bic_vec))
    }
  ),
  
  private <- list(
    # 迭代参数
    gamma = NULL,        # B样条的回归系数
    beta = NULL,         # X的回归系数
    Y2 = NULL,           # Y'
    Y = NULL,            # Y
    u = NULL,            # 融合惩罚项
    w = NULL,            # 拉格朗日乘子
    nu = NULL,           # 拉格朗日乘子
    c = NULL,            # beta_i-beta_k+nu_ik/theta
    bic = NULL,          # BIC
    
    # 验证输入是否正确
    validate = function(){
      if(!(dim(self$T)[2] == 2 & all(colnames(self$T) %in% c('time', 'status')))) stop('T要求为包含time和status的两列矩阵')
      if(!is.matrix(self$X)) stop('X要求为矩阵')
      if(!is.matrix(self$Z)) stop('Z要求为矩阵')
      if(!self$penalty %in% c('MCP','SCAD')) stop('请选择合适的惩罚函数，SCAD或MCP')
      if(!(self$K > 0 & self$K == as.integer(self$K))) stop('确保K是正整数')
      if(self$lambda <= 0) stop('请选择合适的lambda值')
      if(self$a <= 0) stop('请选择合适的theta值')
      if(self$theta <= 0) stop('请选择合适的theta值')
      if(!(self$df > 0 & self$df == as.integer(self$df))) stop('请选择合适的df值')
      if(!(self$degree > 0 & self$degree == as.integer(self$degree))) stop('请选择合适的degree值')
      if(self$tol <= 0) stop('请选择合适的精度要求')
      if(self$max_iter <= 0) stop('请选择合适的最大迭代次数')
    },
    
    # 获取初始值
    initial_value = function(){
      # beta初始值
      result <- kmeans(self$X, centers = self$K)
      df <- cbind(self$T, self$X) %>% as.data.frame()
      df$label <- result$cluster        # 添加类别标签
      df <- df %>% 
        group_nest(label) %>%           # 分组回归，批量建模
        mutate(model = map(data, ~coxph(Surv(time, status)~., data=.x)),
               coef = map(model, ~.x[['coefficients']]))
      coef <- df$coef %>% as.list()     # 提取每个类别的回归系数向量
      beta_0 <- coef[result$cluster]    # 为了后续处理方便，beta暂时以列表形式存储
      
      # gamma初始值与B
      B <- asplit(self$Z, 2)   # 按列分割Z矩阵，分别由bs拟合
      B <- B %>% map(~bs(., df = self$df, degree = self$degree))
      B <- do.call(cbind, B) %>% scale(center = TRUE, scale = FALSE)   # 列方向的中心化处理
      colnames(B) <- paste0('b_', 1:ncol(B))
      df <- cbind(self$T, B) %>% as.data.frame()
      model <- coxph(Surv(time, status)~., data = df)
      gamma_0 <- model[['coefficients']]
      
      # Y与Y'的初始值
      X_ls <- asplit(self$X, 1)
      Y_0 <- map2_vec(X_ls, beta_0, ~.x %*% .y) + B %*% gamma_0
      
      return(list(beta_0 = beta_0, B = B, gamma_0 = gamma_0, X_ls = X_ls, Y_0 = Y_0))
    },
    
    # 计算矩阵A
    gen_A = function(){
      gen_mat <- function(n){
        mat_1 <- if(n == self$n-1){
          NULL
        }else{
          matrix(0, nrow = self$n-1-n, ncol = n)
        }
        mat_2 <- matrix(1, nrow = 1, ncol = n, byrow = T)
        mat_3 <- diag(-1, nrow=n, ncol = n)
        mat <- rbind(mat_1, mat_2, mat_3)
        return(mat)
      }
      D <- as.list(c((self$n-1):1)) %>% map(~gen_mat(.))
      D <- do.call(cbind, D) %>% t()
      
      A <- D %x% diag(1, ncol = self$p, nrow = self$p)
      
      return(A)
    },
    
    # 计算g
    gen_g = function(){
      status <- self$T[,'status'] %>% as.vector()
      time <- self$T[,'time'] %>% as.vector()
      g <- as.list(time) %>% 
        map(~ifelse(. >= time, 1, 0)) %>% 
        map_vec(~status %*% .)
      
      return(g)
    },
    
    # 计算nabla_g
    gen_nabla_g = function(Y2){
      status <- self$T[,'status'] %>% as.vector()
      time <- self$T[,'time'] %>% as.vector()
      
      exp_Y2 <- as.list(exp(Y2))
      
      # 向量sum_{l \in R_k} exp(Y2)
      sum_l_Rk <- as.list(time) %>% 
        map(~which(. <= time)) %>% 
        map_vec(~sum(exp(Y2[.])))
      
      # 计算nabla_g
      nabla_g <- as.list(time) %>% 
        map(~ifelse(. >= time, 1, 0)) %>%    # 计算I_{i \in R_k}
        map(~. %*% diag(status) %*% sum_l_Rk^(-1)) %>%     # 计算delta与I与{l \in R_k}的复合项
        map2_vec(exp_Y2, ~.x * .y) - status
      
      return(nabla_g)
    },
    
    # 计算c，输出列表
    gen_c = function(beta, nu, A){
      delta_beta <- A %*% unlist(beta)
      c <- delta_beta + nu / self$theta
      c <- matrix(c, ncol = self$p, byrow = T) %>% asplit(1)
      
      return(c)
    },
    
    # gamma迭代式
    iter_gamma = function(X_ls, B_ols, Y_current, beta_current, w_current){
      # 这里的beta是列表形式
      X_beta <- map2_vec(X_ls, beta_current, ~.x %*% .y)
      gamma_next <- B_ols %*% (Y_current - X_beta + w_current / self$theta)
      
      return(gamma_next)
    },
    
    # beta迭代式，输出列表
    iter_beta = function(XQX_AA, XQ, A, w_current, Y_current, u_current, nu_current){
      beta_next <- XQX_AA %*% (XQ %*% (w_current / self$theta + Y_current) + t(A) %*% (u_current - nu_current / self$theta))
      beta_next <- matrix(beta_next, ncol = self$p, byrow = T) %>% asplit(1)   # 输出列表形式的beta
      
      return(beta_next)
    },
    
    # Y2迭代式
    iter_Y2 = function(X_ls, B, g, beta_next, gamma_next){
      term_1 <- map2_vec(X_ls, beta_next, ~.x %*% .y)   # X与beta
      term_2 <- asplit(B,1) %>% map_vec(~. %*% gamma_next)  # B与gamma
      Y2_next <- term_1 + term_2
      
      return(Y2_next)
    },
    
    # Y迭代式
    iter_Y = function(g, Y2_next, w_current){
      nabla_g_next <- private$gen_nabla_g(Y2_next)
      Y_next <- (g + self$theta)^(-1) * (-nabla_g_next+ g * Y2_next - w_current + self$theta * Y2_next)
      
      return(Y_next)
    },
    
    # u迭代式
    iter_u = function(penalty, beta_next, nu_current, A){
      S <- function(c, lambda){
       result <- max((1 - lambda / norm(c, type = '2')), 0) * c
       
        return(result)
      }
      
      c <- private$gen_c(beta_next, nu_current, A)
      
      switch(penalty,
             'SCAD' = {
               if(is.null(self$a)) self$a <- 3.7 
               u_next <- map(c, function(c_ik){
                 norm_c_ik <- norm(c_ik, type = '2')
                 if(norm_c_ik <= self$lambda + self$lambda / self$theta){
                   u_ik <- S(c_ik, self$lambda / self$theta)
                 }else if(self$lambda + self$lambda / self$theta < norm_c_ik & norm_c_ik <= self$a * self$lambda){
                   u_ik <- S(c_ik, self$a * self$lambda / ((self$a - 1) * self$theta)) / (1-1/((self$a - 1) * self$theta))
                 }else {
                   u_ik <- c_ik
                 }
                 
                 return(u_ik)
               }) %>% unlist()
             },
             'MCP' = {
               if(is.null(self$a)) self$a <- 2.5
               u_next <- map(c, function(c_ik){
                 norm_c_ik <- norm(c_ik, type = '2')
                 if(norm_c_ik <= self$a * self$lambda){
                   u_ik <- S(c_ik, self$lambda / self$theta) / (1-1/(self$a * self$theta))
                 }else {
                   u_ik <- c_ik
                 }
                 
                 return(u_ik)
               }) %>% unlist()
             }
        
      )
      
      return(u_next)
    },
    
    # w迭代式
    iter_w = function(w_current, Y_next, Y2_next){
      w_next <- w_current + self$theta * (Y_next - Y2_next)
      
      return(w_next)
    },
    
    # nu迭代式
    iter_nu = function(nu_current, beta_next, u_next, A){
      delta_beta <- A %*% unlist(beta_next)
      nu_next <- nu_current + self$theta * (delta_beta - u_next)
      
      return(nu_next)
    },
    
    # BIC准则
    Bic = function(Y2, K){
      # X_beta+B_gamma就是Y2
      status <- self$T[,'status'] %>% as.vector()
      time <- self$T[,'time'] %>% as.vector()
      
      log_sum_l_Ri <- as.list(time) %>% 
        map(~which(. <= time)) %>% 
        map_vec(~log(sum(exp(Y2[.]))))
      term_1 <- -sum(Y2[status] - log_sum_l_Ri[status])
      term_2 <- log(self$n * K + self$q) * log(self$n) * (K * self$p + self$q) / self$n
      bic <- term_1 + term_2
      
      return(bic)
    }
  )
)
```

----------------

下面给出python版本的代码。


``` default
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from scipy import sparse
from scipy.linalg import block_diag
from lifelines import CoxPHFitter
from patsy import bs


class SubgroupBeta:
    
    def __init__(self, time_status, X, Z,  lambda_value, a = None, penalty = 'MCP', K = 2, theta = 1, df = 6, degree = 3, tol = 1e-4, max_iter = 10000):
        self.time_status = time_status
        self.X = X
        self.Z = Z
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.q = Z.shape[1]
        self.lambda_value = lambda_value
        self.a = a
        self.penalty = penalty
        self.K = K
        self.theta = theta
        self.df = df
        self.degree = degree
        self.tol = tol
        self.max_iter = max_iter
        
        self.gamma = None
        self.beta = None
        self.Y2 = None
        self.Y = None
        self.u = None
        self.w = None
        self.nu = None
        self.c = None
        self.bic = None
    
    # 拟合cox模型
    def fit_cox_model(self, data):
        cph = CoxPHFitter()
        cph.fit(data, duration_col = 'time', event_col = 'status')
        coef = np.array(cph.params_).reshape(-1,1)
        return coef
    
    # 生成B样条的基函数矩阵
    def gen_B(self, z, df = 6, degree = 3, intercept = False):
        B = bs(z, df = df, degree = degree, include_intercept = intercept)
        
        return B
    
    def initial_value(self):
        # kmeans
        kmeans = KMeans(self.K, random_state = 1)
        kmeans.fit(self.X)
        labels = kmeans.labels_
        
        # beta初始值
        col_names = ['time', 'status'] + [f'X_{i+1}' for i in range(self.p)]
        df = pd.DataFrame(np.hstack((self.time_status, self.X)), columns = col_names)
        df['label'] = labels
        beta_cox = df.groupby('label', group_keys=False).apply(self.fit_cox_model, include_groups=False)
        beta_0 = np.hstack(beta_cox.tolist())
        # 每列都是beta的系数
        beta_0 = beta_0[:, labels]
        
        # gamma初始值
        B = []
        for col_Z in range(self.q):
            z = self.Z[:, col_Z]
            B_col_Z = self.gen_B(z)
            B.append(B_col_Z)
        B = np.hstack(B)
        B = scale(B, axis = 0, with_mean = True, with_std = False)
        col_names = ['time', 'status'] + [f'b_{i+1}' for i in range(B.shape[1])]
        df = pd.DataFrame(np.hstack((self.time_status, B)), columns = col_names)
        gamma_0 = self.fit_cox_model(df)
        
        # Y与Y2的初始值
        Y_0 = np.einsum('ij,ji->i', self.X, beta_0).reshape(-1,1) + B @ gamma_0

        return beta_0, B, gamma_0, Y_0
        
    # 生成矩阵A
    def gen_A(self):
        # 生成稀疏Delta矩阵
        rows = []
        for i in range(self.n-1):
            row = sparse.lil_matrix((self.n-1-i, self.n))
            for j in range(i+1, self.n):
                row[j-i-1, i] = 1
                row[j-i-1, j] = -1
            rows.append(row)
        Delta = sparse.vstack(rows)
        A = sparse.kron(Delta, np.eye(self.p))
        return A
    
    # 计算g
    def gen_g(self):
        time_obs = self.time_status[:, 0]
        # R是风险集，每行都是第i个元素的风险集
        R = (time_obs[:, np.newaxis] >= time_obs).astype(int)
        g = R @ self.time_status[:, 1].reshape(-1,1)
        return g
    
    # 计算nabla_g
    def gen_nabla_g(self, Y2):
        time_obs = self.time_status[:, 0].flatten()
        status = self.time_status[:, 1].reshape(-1,1)
        
        
        exp_Y2 = np.exp(Y2)
        R = (time_obs[:, np.newaxis] <= time_obs).astype(int)
        sum_l_Rk = 1 / (R @ exp_Y2)
        R_rev = (time_obs[:, np.newaxis] >= time_obs).astype(int)
        nabla_g = -status + exp_Y2 * (R_rev @ (status * sum_l_Rk))
        return nabla_g
    
    # 计算c
    def gen_c(self, beta, nu, A):
        delta_beta = A @ beta.flatten(order = 'F').reshape(-1,1)
        c = delta_beta + nu / self.theta
        # 每行都是beta_i - beta_k
        c = c.reshape(-1, self.p)
        return c, delta_beta
    
    # gamma迭代式
    def iter_gamma(self, B_ols, Y_current, beta_current, w_current):
        X_beta = np.einsum('ij,ji->i', self.X, beta_current).reshape(-1, 1)
        gamma_next = B_ols @ (Y_current - X_beta + w_current / self.theta)
        return gamma_next
    
    # beta迭代式
    def iter_beta(self, XQX_AA, XQ, A, w_current, Y_current, u_current, nu_current):
        beta_next = XQX_AA @ (XQ @ (w_current / self.theta + Y_current) + A.T @ (u_current - nu_current / self.theta))
        beta_next = beta_next.reshape(self.p, -1, order = 'F')
        return beta_next
    
    # Y2迭代式
    def iter_Y2(self, B, g, beta_next, gamma_next):
        term_1 = np.einsum('ij,ji->i', self.X, beta_next).reshape(-1, 1)
        term_2 = B @ gamma_next
        Y2_next = term_1 + term_2
        return Y2_next
    
    # Y迭代式
    def iter_Y(self, g, Y2_next, w_current):
        nabla_g_next = self.gen_nabla_g(Y2_next)
        Y_next = 1/(g + self.theta) * (-nabla_g_next + g * Y2_next - w_current + self.theta * Y2_next)
        return Y_next
    
    # u迭代式
    def penalty_fun(self):
        def S(c_ik, lambda_val):
            result = np.max([(1 - lambda_val / np.linalg.norm(c_ik, ord = 2)), 0]) * c_ik
            return result
        
        def SCAD(c):
            if self.a is None:
                self.a = 3.7
            lamba_lambda_theta = self.lambda_value + self.lambda_value / self.theta
            a_lambda = self.a * self.lambda_value
            
            norm_c_vec = np.linalg.norm(c, ord = 2, axis = 1)
            
            cond_1 = norm_c_vec <= lamba_lambda_theta
            cond_2 = (lamba_lambda_theta < norm_c_vec) & (norm_c_vec <= a_lambda)
            cond_3 = norm_c_vec > a_lambda
            
            u_next = np.copy(c)
            if np.any(cond_1):
                u_next[cond_1, :] = np.apply_along_axis(S, axis = 1, arr = c[cond_1, :], lambda_val = self.lambda_value/self.theta)
            if np.any(cond_2):
                u_next[cond_2, :] = np.apply_along_axis(S, axis = 1, arr = c[cond_2, :], lambda_val = self.a * self.lambda_value/((self.a - 1) * self.theta)) / (1 - 1 / ((self.a-1) * self.theta))
            if np.any(cond_3):
                u_next[cond_3, :] = c[cond_3, :]
            return u_next
        
        def MCP(c):
            if self.a is None:
                self.a = 2.5
            a_lambda = self.a * self.lambda_value
            
            norm_c_vec = np.linalg.norm(c, ord = 2, axis = 1)
            
            cond_1 = norm_c_vec <= a_lambda
            cond_2 = norm_c_vec > a_lambda
            
            u_next = np.copy(c)
            if np.any(cond_1):
                u_next[cond_1, :] = np.apply_along_axis(S, axis = 1, arr = c[cond_1, :], lambda_val = self.lambda_value / self.theta) / (1 - 1 / (self.a * self.theta))
            if np.any(cond_2):
                u_next[cond_2, :] = c[cond_2, :]
            return u_next
        
        if self.penalty == 'SCAD':
            return SCAD
        elif self.penalty == 'MCP':
            return MCP
        else:
            raise ValueError("Invalid penalty function type")
    
    def iter_u(self, beta_next, nu_current, A, penalty_fun):
        c, delta_beta = self.gen_c(beta_next, nu_current, A)
        u_next = penalty_fun(c).flatten().reshape(-1,1)
        return u_next, delta_beta
    
    # w迭代式
    def iter_w(self, w_current, Y_next, Y2_next):
        w_next = w_current + self.theta * (Y_next - Y2_next)
        return w_next
    
    # nu迭代式
    def iter_nu(self, nu_current, delta_beta, u_next):
        u_next = u_next.flatten().reshape(-1,1)
        nu_next = nu_current + self.theta * (delta_beta - u_next)
        return nu_next
        
    # 主函数——运行
    def run(self, trace = True):
        start_time = time.time()
        
        # 获取初始值
        self.beta, B, self.gamma, self.Y = self.initial_value()
        self.Y2 = self.Y
        A = self.gen_A()
        self.u = A @ self.beta.flatten(order = 'F').reshape(-1,1)
        self.w = np.zeros((self.n, 1))
        self.nu = np.zeros((int(self.n * (self.n - 1) * self.p / 2), 1))
        
        # 计算Q矩阵
        Q = np.eye(self.n) - B @ np.linalg.inv(B.T @ B) @ B.T
        
        # 计算g
        g = self.gen_g()
        
        # 计算(B'B)^(-1)B'
        B_ols = np.linalg.inv(B.T @ B) @ B.T
        
        # 计算(X'QX+A'A)^{-1}与X'Q
        X_diag = np.split(self.X, self.n, axis = 0)
        X_diag = block_diag(*X_diag)
        XQX_AA = np.linalg.inv(X_diag.T @ Q @ X_diag + A.T @ A)
        XQ = X_diag.T @ Q
        
        # 生成惩罚函数
        penalty_fun = self.penalty_fun()
        
        for i in range(self.max_iter):
            if trace:
                print(f'第[{i+1}]次迭代')
            self.gamma = self.iter_gamma(B_ols, self.Y, self.beta, self.w)
            self.beta = self.iter_beta(XQX_AA, XQ, A, self.w, self.Y, self.u, self.nu)
            self.Y2 = self.iter_Y2(B, g, self.beta, self.gamma)
            self.Y = self.iter_Y(g, self.Y2, self.w)
            self.u, delta_beta = self.iter_u(self.beta, self.nu, A, penalty_fun)
            self.w = self.iter_w(self.w, self.Y, self.Y2)
            self.nu = self.iter_nu(self.nu, delta_beta, self.u)
            
            # 终止条件
            term_1 = delta_beta - self.u
            term_2 = self.Y - self.Y2
            norm_r = np.linalg.norm(term_1, ord=2) + np.linalg.norm(term_2, ord = 2)
            if norm_r <= self.tol:
                if trace == True:
                    print('达到精度要求')
                break
        
        # beta亚组
        self.beta = np.round(self.beta, 3)
        subgroup_beta = np.unique(self.beta, axis = 1, return_counts = True)
        size = subgroup_beta[1]
        subgroup_beta = subgroup_beta[0].T
        self.K = subgroup_beta.shape[0]
        df = pd.DataFrame([{'beta': tuple(row.tolist())} for row in subgroup_beta])
        df['size'] = size
        df = df.sort_values(by = 'size', ascending = False).reset_index(drop = True)
        df['label'] = range(self.K)
        
        df_label = pd.DataFrame([{'beta': tuple(row.tolist())} for row in self.beta.T])
        df_label = df_label.merge(df, on = 'beta', how = 'left')
        label = df_label['label'].tolist()
        
        result = {
            'beta' : self.beta.T,
            'gamma' : self.gamma,
            'alpha' : df,
            'K' : self.K,
            'label' : label
        }
        
        print(f'耗时：{time.time() - start_time:.2f}s')
        return result
    
    # 主函数——调优
    def tune_lambda(self, seq_lambda, seed = None, trace = True):
        start_time = time.time()
        
        # 获取初始值
        beta_0, B, gamma_0, Y_0 = self.initial_value()
        A = self.gen_A()
        u_0 = A @ beta_0.flatten(order = 'F').reshape(-1,1)
        w_0 = np.zeros((self.n, 1))
        nu_0 = np.zeros((int(self.n * (self.n - 1) * self.p / 2), 1))
        
        # 计算Q矩阵
        Q = np.eye(self.n) - B @ np.linalg.inv(B.T @ B) @ B.T
        
        # 计算g
        g = self.gen_g()
        
        # 计算(B'B)^(-1)B'
        B_ols = np.linalg.inv(B.T @ B) @ B.T
        
        # 计算(X'QX+A'A)^{-1}与X'Q
        X_diag = np.split(self.X, self.n, axis = 0)
        X_diag = block_diag(*X_diag)
        XQX_AA = np.linalg.inv(X_diag.T @ Q @ X_diag + A.T @ A)
        XQ = X_diag.T @ Q
        
        # 生成惩罚函数
        penalty_fun = self.penalty_fun()
        
        # 存储结果
        bic_ls = []
        bic_log_ls = []
        bic_c_ls = []
        bic_logk_ls = []
        result_ls = []
        K_ls = []
        
        for i in range(len(seq_lambda)):
            self.lambda_value = seq_lambda[i]
            
            # 初始化
            self.beta = beta_0
            self.gamma = gamma_0
            self.Y = Y_0
            self.Y2 = Y_0
            self.u = u_0
            self.w = w_0
            self.nu = nu_0
            
            for j in range(self.max_iter):
                if trace:
                    print(f'seed_[{seed+1}]: lambda={seq_lambda[i]}--[{j+1}]')
                self.gamma = self.iter_gamma(B_ols, self.Y, self.beta, self.w)
                self.beta = self.iter_beta(XQX_AA, XQ, A, self.w, self.Y, self.u, self.nu)
                self.Y2 = self.iter_Y2(B, g, self.beta, self.gamma)
                self.Y = self.iter_Y(g, self.Y2, self.w)
                self.u, delta_beta = self.iter_u(self.beta, self.nu, A, penalty_fun)
                self.w = self.iter_w(self.w, self.Y, self.Y2)
                self.nu = self.iter_nu(self.nu, delta_beta, self.u)
                
                # 终止条件
                term_1 = delta_beta - self.u
                term_2 = self.Y - self.Y2
                norm_r = np.linalg.norm(term_1, ord=2) + np.linalg.norm(term_2, ord = 2)
                if norm_r <= self.tol:
                    if trace == True:
                        print('达到精度要求')
                    break
            
            # beta亚组
            self.beta = np.round(self.beta, 3)
            subgroup_beta = np.unique(self.beta, axis = 1, return_counts = True)
            size = subgroup_beta[1]
            subgroup_beta = subgroup_beta[0].T
            self.K = subgroup_beta.shape[0]
            df = pd.DataFrame([{'beta': tuple(row.tolist())} for row in subgroup_beta])
            df['size'] = size
            df = df.sort_values(by = 'size', ascending = False).reset_index(drop = True)
            df['label'] = range(self.K)
            
            df_label = pd.DataFrame([{'beta': tuple(row.tolist())} for row in self.beta.T])
            df_label = df_label.merge(df, on = 'beta', how = 'left')
            label = df_label['label'].tolist()
            
            # 计算bic
            time_obs = self.time_status[:, 0]
            status = self.time_status[:, 1].reshape(1,-1)
            R = (time_obs[:, np.newaxis] <= time_obs).astype(int)
            term_1 = - status @ (self.Y2 - np.log(R @ np.exp(self.Y2)))
            term_2 = np.log(self.n * self.K + self.q) * np.log(self.n) * (self.K * self.p + self.q) / self.n
            bic_value = term_1 + term_2
            bic_value = round(bic_value.item(), 3)
            bic_ls.append(bic_value)
            
            
            term_2 = np.log(np.log(self.n * self.K + self.q)) * np.log(self.n) * (self.K * self.p + self.q) / self.n
            bic_log_value = term_1 + term_2
            bic_log_value = round(bic_log_value.item(), 3)
            bic_log_ls.append(bic_log_value)
            
            
            term_2 = 0.5 * np.log(self.n * self.K + self.q) * np.log(self.n) * (self.K * self.p + self.q) / self.n
            bic_c_value = term_1 + term_2
            bic_c_value = round(bic_c_value.item(), 3)
            bic_c_ls.append(bic_c_value)
            
            term_2 = np.log(self.K) * np.log(self.n * self.K + self.q) * np.log(self.n) * (self.K * self.p + self.q) / self.n
            bic_logk_value = term_1 + term_2
            bic_logk_value = round(bic_logk_value.item(), 3)
            bic_logk_ls.append(bic_logk_value)
            
            K_ls.append(self.K)
            
            result = {
                'beta' : self.beta.T,
                'gamma' : self.gamma,
                'alpha' : df,
                'K' : self.K,
                'label' : label,
                'bic' : bic_value
            }
            result_ls.append(result)
        
        best_index = bic_ls.index(min(bic_ls))
        best_result = result_ls[best_index]
        print(f'总耗时{time.time() - start_time:.2f}s')
        tune_result = {'bic' : bic_ls, 'result' : result_ls, 'best_result' : best_result, 'bic_log' : bic_log_ls, 'bic_c' : bic_c_ls, 'K_ls' : K_ls, 'bic_logk' : bic_logk_ls}
        
        return tune_result
        
if __name__ == "__main__":
    
    total_start_time = time.time()
    
    def gen_data(seed, n=100):
        np.random.seed(seed)
        X = np.random.normal(loc = 0, scale = 1, size = (n, 2))
        Z_1 = np.random.uniform(low = 0, high = 1, size = (n, 1))
        f_Z1 = np.sin(np.pi * (Z_1-0.5))
        Z_2 = np.random.uniform(low = 0, high = 1, size = (n, 1))
        f_Z2 = np.cos(np.pi * (Z_2 - 0.5)) - 2/np.pi
        Z = np.hstack((Z_1, Z_2))
        
        kmeans = KMeans(2, random_state = 1)
        kmeans.fit(X)
        labels = kmeans.labels_
        X_1 = X[labels == 0, :]
        X_2 = X[labels == 1, :]
        X = np.vstack((X_1, X_2))
        
        beta = 3 * np.ones((int(n/2),2))
        beta = np.vstack((-beta, beta))
        
        log_U = np.log(np.random.uniform(0, 1, size = (n,1)))
        X_beta = np.einsum('ij,ji->i', X, beta.T).reshape(-1,1)
        time_obs = -np.exp(-X_beta - f_Z1 - f_Z2) * log_U
        status = np.random.choice([0,1], size = (n,1), p = [0.2, 0.8])
        time_status = np.hstack((time_obs, status))
       
        return time_status, X, Z

    np.random.seed(564)
    seed_ls = np.random.randint(low = 1, high = 1000, size= 10).tolist()
    seq_lambda = np.arange(0.04, 0.075, 0.005)
    bic_mat = np.zeros((len(seed_ls), len(seq_lambda)))
    K_mat = np.zeros((len(seed_ls), len(seq_lambda)))
    result_ls = []
    
    for i in range(len(seed_ls)):
        seed = seed_ls[i]
        time_status, X, Z = gen_data(n=100,seed = seed)
        model = SubgroupBeta(time_status, X, Z, lambda_value = 0.06, tol = 1e-3)
        result_tune = model.tune_lambda(seq_lambda, seed = i)
        
        bic_mat[i] = result_tune['bic']
        K_mat[i] = result_tune['K_ls']
        result_ls.append(result_tune['result'])
    
    seed_ls = list(map(str, seed_ls))
    seq_lambda = list(map(str, seq_lambda.tolist()))
    bic_mat = pd.DataFrame(bic_mat, index = seed_ls, columns = seq_lambda)
    K_mat = pd.DataFrame(K_mat, index = seed_ls, columns = seq_lambda)
    
    total_end_time = time.time()
    
    print(f'所有种子总计耗时：{total_end_time - total_start_time}')
```


### 数据模拟 {#code_3_2}


``` r
# case_3
# 有顺序调整
set.seed(123)
x_1 <- rnorm(100)
x_2 <- rnorm(100)
X <- cbind(x_1, x_2)
X_cluster <- kmeans(X, centers = 2)
X_1 <- X[which(X_cluster$cluster == 1),]
X_2 <- X[which(X_cluster$cluster == 2),]
X <- rbind(X_1,X_2)
X_diag <- map(asplit(X,1), ~matrix(., nrow = 1, byrow = T))
X_diag <- do.call(bdiag, X_diag) %>% as.matrix()
z_1 <- runif(100)
z_2 <- runif(100)
f_z1 <- sin((z_1 - 0.5) * pi)
f_z2 <- cos((z_2 - 0.5) * pi) - 2 / pi
beta <- c(rep(c(3,3), times = 50), rep(c(-3,-3), times = 50))
time <- as.vector(-exp(-X_diag %*% beta - f_z1 - f_z2)) * log(runif(100))
status <- sample(c(1,0), size = 100, replace = TRUE, prob = c(0.8, 0.2))

T <- cbind(time, status)
Z <- cbind(z_1, z_2)

case3 <- SubgroupBeta$new(T = T, X = X, Z = Z, penalty = 'MCP', K = 2, lambda = 0.1, a = 2.5, theta = 1, df = 6, degree = 3)
result3 <- case3$run(trace = FALSE)
```

```
##   用户   系统   流逝 
## 119.50   5.81 127.66
```


``` r
result3$alpha
```

```
##   label size         beta
## 1     1   56   2.26, 2.19
## 2     2   44 -2.56, -2.54
```

``` r
result3$alpha$beta
```

```
## [[1]]
## [1] 2.26 2.19
## 
## [[2]]
## [1] -2.56 -2.54
```



