rm(list=ls())
misclass <- function(Y, P) { mean((P >= 0) == Y)}
Rcpp::sourceCpp('functon_.cpp')
source('function_1.R'); library(pROC) ; library(Matrix)
# AUC based RRR
n = 200  # samples
l = 8   # response
p = 12   # predictors
r = 5    # true rank
set.seed(1)
C = matrix(rnorm(p*r),nr=p)%*%t(matrix(rnorm(l*r),nr=l)) 
ntrain = floor(0.8 * n)

 Flip_size = 0.2  # percentage of labels to flip

rho <- 0.5   # feature correlation
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j)) ; LL = chol(Sigma) 

auc_out = est_out = accuracy_out = list()

for (ss in 1:50) {
  set.seed(ss)
  X = matrix(rnorm(n*p), nc = p )%*% LL 
  MU0 = X%*%C    ; 
  prob = pnorm(MU0) #  1/(1+ exp(-MU0) )  #  pnorm(MU0) # 
  Y =  apply(prob, 2, function(a) rbinom(n = n, size = 1, a))  #1* (MU0 + rnorm(n*l) >= 0) 
  # apply(prob, 2, function(a) rbinom(n = n, size = 1, a))  #
  Y_noisy = Y      
  
  # Flip % of the labels randomly to simulate recording errors
  
   flip_idx = sample(1:length(Y), size = Flip_size * length(Y))
   Y_noisy[ flip_idx ] = 1 - Y_noisy[flip_idx]
  
  # train / test split
  idx = sample(n)
  train = idx[1:ntrain]
  test  = idx[(ntrain+1):n]
  
  xtrain = X[train, ]
  
  # contamiated X
  xtrain[1:20, ] <- xtrain[1:20, ] * 30  # increase SNR
  
  xtest = X[test, ]
  Ytrain = Y_noisy[train, ]
  ytest = Y_noisy[test, ]
  
  fit_log <- logistic_rrr_cpp(Ytrain, xtrain, rank = r)
  C_log <- fit_log$coef[-1, ] 
  fit_auc <- logistic_rrr_auc_fast_cpp(  Ytrain, xtrain,  rank = r, lambda = 0)
  C_auc <- fit_auc$coef[-1, ] 
  
  auc_log <- numeric(l)  ;  auc_auc <- numeric(l) ;  auc_auc2 <- numeric(l) 
  for (j in 1:l) {  auc_log[j] <- auc(ytest[, j], xtest %*% C_log[, j]);
  auc_auc[j] <- auc(ytest[, j], xtest %*% C_auc[, j]) }
  
  auc_out[[ss]] = c(mean(auc_log) , mean(auc_auc) )
  est_out[[ss]] = c( mean((C_log - C)^2 ) , mean((C_auc - C)^2 ) )
  accuracy_out[[ss]] = c( misclass(ytest, xtest %*% C_log), misclass(ytest, xtest %*% C_auc)  ) 
  cat(ss)
}
source('out_results.R')


