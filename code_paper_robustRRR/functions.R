#############################################################################
# Proximal-gradient for matrix Huber + spectral SCAD/MCP/nuclear penalty (R)
# - Huber loss (threshold tau)
# - Spectral penalty R(B) = sum_j r( sigma_j(B) ), where r = SCAD or MCP
# - Scalar prox solved by a cheap 1D minimization for robustness
# - K-fold cross-validation over lambda
#
# Author: The Tien Mai
#############################################################################
Rcpp::sourceCpp("rrr_huber_ncv_fast.cpp")
library(Matrix)
########################################
# K-fold cross validation for lambda (and optional penalty-params)
# We keep scad_a / mcp_gamma fixed; grid search over lambda only.
########################################
cv_proximal_cpp <- function(
    X, Y,
    penalty = c("SCAD", "MCP", "nuclear"),
    lambda_grid = NULL,
    nfolds = 5,
    seed = 1,
    tau_grid = NULL,
    scad_a = 3.7,
    mcp_gamma = 3,
    alpha = NULL,
    max_iter = 2000,
    tol = 1e-5,
    svd_rank = NULL,
    verbose = TRUE
){
  penalty <- match.arg(penalty)
  set.seed(seed)
  n <- nrow(X)
  # ------------------------------------------------------------
  # Tau grid based on IQR(Y)
  # ------------------------------------------------------------
  if( is.null(tau_grid)){
    #IQRy <- IQR(as.numeric(Y))
    tau_grid <- c(0.1, 1 , 10) # c(IQRy/10, IQRy, IQRy*10)
    #if (IQRy/10 >1){ tau_grid = tau_grid/10}
  }

  # ------------------------------------------------------------
  # Lambda grid if needed
  if (is.null(lambda_grid)) {
    p <- ncol(X)
    q <- ncol(Y)
    C <- matrix(0, p, q)
    for (k in seq_len(q)) {
      id_x <- !is.na(Y[, k])
      nk  <- sum(id_x)
      if (nk > 0) {
        C[, k] <- crossprod(X[id_x, , drop = FALSE], Y[id_x, k, drop = FALSE]) / nk
      }   }
    sXY <- svd(C, nu = 0, nv = 0)$d
    
    lambda_max <- max(sXY)
    lambda_min <- 0.05
    lambda_grid <- exp(seq(log(lambda_max),  log(lambda_min),  length.out = 10))
  }
  
  # folds
  folds <- sample(rep(1:nfolds, length.out = n))
  # ------------------------------------------------------------
  # Storage: rows = tau_index, cols = lambda_index
  # ------------------------------------------------------------
  cv_loss <- matrix(0, nrow = length(tau_grid), ncol = length(lambda_grid))
  # ------------------------------------------------------------
  # Outer loop: over tau grid
  # ------------------------------------------------------------
  for (ti in seq_along(tau_grid)) {
    tau_val <- tau_grid[ti]
    if (verbose) {
      cat("=====================================================\n")
      cat(sprintf("Evaluating tau (%d/%d): %.4e\n", ti, length(tau_grid), tau_val))
      cat("=====================================================\n")
    }
    # ------------------------------------------------------------
    # Inner loop: over lambda grid
    # ------------------------------------------------------------
    for (li in seq_along(lambda_grid)) {
      
      lam <- lambda_grid[li]
      if (verbose) cat(sprintf(" CV tau %d, lambda %d/%d: %.4e\n",
                               ti, li, length(lambda_grid), lam))
      fold_losses <- numeric(nfolds)
      # CV folds
      for (f in 1:nfolds) {
        test_idx  <- which(folds == f)
        train_idx <- setdiff(seq_len(n), test_idx)
        
        Xtr <- X[train_idx, , drop = FALSE]
        Ytr <- Y[train_idx, , drop = FALSE]
        Xte <- X[test_idx, , drop = FALSE]
        Yte <- Y[test_idx, , drop = FALSE]
        
        # Fit proximal solver
        fit <- proximal_gradient_solver_cpp(  Xtr, Ytr,    
          tau = tau_val,
          lambda = lam,
          penalty = penalty, scad_a = scad_a, mcp_gamma = mcp_gamma,
          alpha = alpha, max_iter = max_iter, tol = tol, verbose = FALSE, svd_rank = svd_rank  )
        
        Bhat <- fit$B
        res  <- Yte - Xte %*% Bhat
        # squared loss on test fold (or huber_loss_cpp if you want)
        fold_losses[f] <- mean(res^2, na.rm = TRUE)
      }
      
      cv_loss[ti, li] <- mean(fold_losses, na.rm = TRUE)
      
      if (verbose)
        cat(sprintf("  tau=%.4e lambda=%.4e -> CV loss=%.6e\n",
                    tau_val, lam, cv_loss[ti, li]))
    }
  }
  
  idx <- which(cv_loss == min(cv_loss), arr.ind = TRUE)
  best_tau_idx    <- idx[1, 1]
  best_lambda_idx <- idx[1, 2]
  
  best_tau    <- tau_grid[best_tau_idx]
  best_lambda <- lambda_grid[best_lambda_idx]
  
  if (verbose) {
    cat("=====================================================\n")
    cat("Best tuning parameters found:\n")
    cat(sprintf("  tau    = %.6f\n", best_tau))
    cat(sprintf("  lambda = %.6e\n", best_lambda))
    cat(sprintf("  CV loss= %.6f\n", cv_loss[best_tau_idx, best_lambda_idx]))
    cat("=====================================================\n")
  }
  final_fit = proximal_gradient_solver_cpp(X, Y, tau = best_tau, 
                                           lambda = best_lambda,
                                           penalty = penalty, verbose = verbose)
  return(list(
    best_fit      = final_fit,
    B_best        = final_fit$B,
    tau_grid      = tau_grid,
    lambda_grid   = lambda_grid,
    cv_loss       = cv_loss,
    best_tau      = best_tau,
    best_lambda   = best_lambda,
    best_tau_idx  = best_tau_idx,
    best_lambda_idx = best_lambda_idx
  ))
}
