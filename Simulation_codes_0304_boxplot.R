setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source("LSLVLASSO.R")
library(MASS)

# Use the SIMU function to simulate the data for different number of observations
# the function returns 2 error statistics
SIMU <- function(N){
  # 1.1 Generate X matrix
  nobs <- N
  
  loading1 <- c(0,rep(sqrt(.6),10),rep(0,9))
  loading2 <- c(sqrt(.6),rep(0,10),rep(sqrt(.6),9))
  L <- cbind(loading1,loading2)
  
  PSY <- diag(1-rowSums(L^2)) 
  SIGMA <- L%*%t(L)+ PSY
  
  X <- mvrnorm(n = nobs, mu = rep(0,20), Sigma = SIGMA, empirical = TRUE)
  
  # 1.2 Generate Y
  
  # define regression coefficients
  beta0 <- 0
  beta1 <- 1
  beta2 <- -1
  
  Score <- X %*% L %*% solve(t(L) %*% L) # obtain factor scores of X
  Y <- rep(beta0,nobs) + beta1 * Score[,1] + beta2 * Score[,2] + rnorm(n=nobs,mean=0,sd=0.01) # generate Y by the factor scores
  
  # 1.3 Split the data
  random_order <- sample(1:nobs)
  split_point <- round(nobs/2)
  X_train <- X[random_order[1:split_point], ]
  X_test <- X[random_order[(split_point + 1):nobs], ]
  Y_train <- Y[random_order[1:split_point]]
  Y_test <- Y[random_order[(split_point + 1):nobs]]
  
  # 1.4 Run the function
  res_method <- LSLVLASSO(X_train,2,2,1e3,1e-4) # before running this, run LSLVLASSO.R first
  train_score <- res_method$scores # the factor score in training set
  res_loading <- res_method$loadings
  
  # how the 20 items load on each factors, obtain best permutation based on difference in loading for L and result
  R=2 #nr of factors
  perm <- gtools::permutations(R, R)
  absdiff <- c()
  for (p in 1:nrow(perm)) {
    corsign <- sign(diag(cor(L, res_loading[, perm[p,]])))
    L_res <- (res_loading[, perm[p,]]) %*% diag(corsign)
    absdiff[p] <- sum(rowSums(abs(L - L_res)))
  }
  bestperm <- which.min(absdiff)
  loadings <- res_loading[, perm[bestperm,]]
  corsign <- sign(diag(cor(L, loadings)))
  res_loading <- loadings %*% diag(corsign)
  
  # do a parallel transformation in train_score
  scores <- train_score[, perm[bestperm,]]
  train_score <- scores %*% diag(corsign)
  
  # 1.5 Run the regression
  res_reg <- lm(Y_train ~ train_score[,1] + train_score[,2]) # do a regression and get coefficients
  reg_coef <- unname(res_reg$coefficients)
  
  # 1.6 Test
  test_score <- X_test %*% res_loading %*% solve(t(res_loading) %*% res_loading) # get the factor scores in test set
  yhat <- rep(reg_coef[1],length(Y_test)) + test_score[,1]*reg_coef[2] + test_score[,2] * reg_coef[3] # get the prediction results
  
  cbind(Y_test, yhat) # check the results
  
  # Statistics
  sta <- c( sum(Y_test-yhat)^2 / nobs, sum(Y_test-yhat)^2 / sum(Y^2) )
  return(sta)
}

n_simulations <- 100
x_values <- seq(20, 200, by = 10)

a_matrix <- matrix(0, nrow = n_simulations, ncol = length(x_values))
b_matrix <- matrix(0, nrow = n_simulations, ncol = length(x_values))

for (i in 1:n_simulations) {
  for (j in 1:length(x_values)){
    simu_res <- SIMU(x_values[j])
    a_matrix[i, j] <- simu_res[1] 
    b_matrix[i, j] <- simu_res[2] 
  }
}

par(mfrow=c(1,2))
boxplot(a_matrix, names=seq(20, 200, by = 10))
boxplot(b_matrix, names=seq(20, 200, by = 10))


par(mfrow=c(1,2))
boxplot(a_matrix, names=seq(20, 200, by = 10), outline=FALSE)
boxplot(b_matrix, names=seq(20, 200, by = 10), outline=FALSE)



