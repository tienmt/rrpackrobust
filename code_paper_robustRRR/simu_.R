source('functions.R') ; library(rrpack)
n_test = 1000
n <- 200 
p <- 12 
q <- 7
r_true = 2
U <- matrix(rnorm(p * r_true), p, r_true); V <- matrix(rnorm(q * r_true), q, r_true)
Btrue <- U %*% t(V) 
rho <- 0.5             # feature correlation
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j)) ; LL = chol(Sigma) 


out_rrr = out_r4 = out_scad = out_mcp = out_nucl = list()

for (ss in 1:100) {
  xx <- matrix(rnorm((n+n_test) * p), n+n_test, p)%*% LL
  X <- xx[1:n, ]
  xtest <- xx[ -(1:n), ]
  yy <- xx %*% Btrue + rnorm( (n+n_test)* q) # rcauchy( (n+n_test)* q) #  
  #  1.5*rt( (n+n_test)* q, df = 3) rnorm( (n+n_test)*q, sd = 3)   rnorm( (n+n_test)* q) # heavy-tailed errors
  Y <- yy[1:n, ]
  ytest <- yy[ -(1:n), ]

  fit_rrr <- cv.rrr(X = X, Y = Y, nfold = 5)
  out_rrr[[ss]] = c(sum( (Btrue -fit_rrr$coef )^2),mean( (ytest - xtest %*% fit_rrr$coef )^2), rankMatrix(fit_rrr$coef,tol = 1e-2)[1])
  
  fit_r4 <- r4(Y, X, maxrank = min(n, p, q), ic.type= "PIC", method = "rowl0")#  c("rowl0", "rowl1", "entrywise") 
  out_r4[[ss]] = c(sum( (Btrue - coef(fit_r4) )^2),mean( (ytest - xtest %*%coef(fit_r4) )^2) ,rankMatrix(coef(fit_r4),tol = 1e-2)[1] )
  Yna <- Y #; Yna[sample(length(Y), 0. * length(Y))] <- NA
  
  # grid of lambda
  cvres_scad <- cv_proximal_cpp(X = X, Yna, penalty = "SCAD",  verbose = FALSE)
  out_scad[[ss]] = c(sum( (Btrue - cvres_scad$B_best )^2), mean( (ytest - xtest %*%cvres_scad$B_best )^2),rankMatrix(cvres_scad$B_best,tol = 1e-2)[1] )
  
  cvres_nucl <- cv_proximal_cpp(X = X, Yna, penalty = "nuclear", verbose = FALSE)
  out_nucl[[ss]] = c(sum( (Btrue - cvres_nucl$B_best )^2),  mean( (ytest - xtest %*% cvres_nucl$B_best )^2),rankMatrix(cvres_nucl$B_best,tol = 1e-2)[1]  )
  
  cvres_mcp <- cv_proximal_cpp(X = X, Yna, penalty = "MCP", verbose = FALSE)
  out_mcp[[ss]] = c(sum( (Btrue -cvres_mcp$B_best )^2),mean( (ytest - xtest %*%cvres_mcp$B_best )^2),rankMatrix(cvres_mcp$B_best,tol = 1e-2)[1] )
 cat(ss, ',')
}  
rm(xx,X,xtest,yy,Y,ytest,fit_rrr,fit_r4)
save.image('sim_rX.rda')
