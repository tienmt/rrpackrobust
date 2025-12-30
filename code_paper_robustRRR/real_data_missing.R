source('functions.R') ; library(rrpack); library(robustHD)

data("nci60")
# define response variable
Y <- protein[,1:20] ; Y = scale(Y)
X <- gene  ;X <- scale(X)
n <- 50 ; p <- ncol(X); q <- ncol(Y)
# cross-covariance matrix (p x q)
C <- crossprod(X, Y) / n   
score <- sqrt(rowSums(C^2))                # aggregate correlation score for each predictor
top_idx <- order(score, decreasing = TRUE)[1:100] # select top 100 predictors
X_sel <- X[, top_idx]

missing = 0.1

simdata <- rrr.sim3(n = n, p = p, q.mix = c(q, 0, 0), intercept = rep(0,q), nrank = 3, mis.prop = 0)
family <- simdata$family 
control = list(epsilon = 1e-4, sv.tol = 1e-2, maxit = 1000, trace =FALSE,gammaC0 = 1.1, plot.cv = F,conv.obj=TRUE)

predt = rankest = list()

for (ss in 1:100) {
  test <- sample(1:nrow(X),size = 9,replace = FALSE)
  xtrain <- X_sel[-test,]
  ytrain <- Y[-test,]
  xtest <- X_sel[test,]
  ytest <- Y[test,]
  
  Yna <- ytrain
  Yna[sample(1:(n*q), missing *(n*q) )] <- NA
  
  fit.cv.mrrr <- cv.mrrr(Y = Yna, xtrain, family = family,control = control, penstr = list(penaltySVD = "rankPen"))
  cvres_scad <- cv_proximal_cpp(X = xtrain, Yna, penalty = "SCAD",  verbose = FALSE)
  cvres_nucl <- cv_proximal_cpp(X = xtrain, Yna, penalty = "nuclear", verbose = FALSE)
  cvres_mcp <- cv_proximal_cpp(X = xtrain, Yna, penalty = "MCP", verbose = FALSE)
  hatc = coef(fit.cv.mrrr$fit)[-1,]
  predt[[ss]] = c(mean( (ytest - xtest %*% hatc )^2) , 'mcp'= mean( (ytest - xtest %*%cvres_mcp$B_best )^2) , 
                  mean( (ytest - xtest %*%cvres_scad$B_best )^2), mean( (ytest - xtest %*% cvres_nucl$B_best )^2)) 
  rankest[[ss]] = c(rankMatrix(hatc ,tol = 1e-2)[1],  rankMatrix(cvres_mcp$B_best,tol = 1e-2)[1], 
                    rankMatrix(cvres_scad$B_best,tol = 1e-2)[1] , rankMatrix(cvres_nucl$B_best,tol = 1e-2)[1] ) 
  
  cat(ss, ',')
}  
save.image('out_real_mis_0000_.rda')
