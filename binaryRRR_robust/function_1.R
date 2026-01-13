logistic_rrr <- function(Y,                # n x q binary response matrix
                        X,                # n x p predictor matrix
                        rank,             # target rank
                        lambda = 0,       # optional nuclear norm penalty
                        maxit = 1000,
                        tol = 1e-10,
                        step = 1) {
  # Dimensions  
  n <- nrow(Y)
  q <- ncol(Y)
  p <- ncol(X)
  
  # Design matrix with intercept
  X0 <- cbind(1, X)              # n x (p+1)
  p0 <- p + 1
  
  # Initialize coefficients
  B <- matrix(0, p0, q)
  
  loglik <- function(Y, eta) {
    mu <- 1 / (1 + exp(-eta))
    sum(Y * log(mu + 1e-8) + (1 - Y) * log(1 - mu + 1e-8))
  }
  
  for (iter in 1:maxit) {
    
    # Linear predictor and mean
    eta <- X0 %*% B
    mu  <- 1 / (1 + exp(-eta))
    
    # Gradient of log-likelihood
    G <- t(X0) %*% (Y - mu)
    
    # Gradient step
    B_new <- B + step * G / n
    
    # Reduced-rank projection (excluding intercept)
    sv <- svd(B_new[-1, , drop = FALSE])
    d  <- sv$d
    
    if (lambda > 0) { d <- pmax(d - lambda, 0) }
    
    r <- min(rank, sum(d > 0))
    
    B_new[-1, ] <-
      sv$u[, 1:r, drop = FALSE] %*%
      diag(d[1:r], r, r) %*%
      t(sv$v[, 1:r, drop = FALSE])
    
    # Check convergence
    diff <- sum((B_new - B)^2) / (sum(B^2) + 1e-8)
    B <- B_new
    
    if (diff < tol) break
  }
  
  eta <- X0 %*% B
  mu  <- 1 / (1 + exp(-eta))
  
  list(coef = B,
      fitted = mu,
      rank = r,
      iter = iter,
      converged = (iter < maxit),
      logLik = loglik(Y, eta)
  )
}



sigmoid <- function(z) {
  z <- pmax(pmin(z, 35), -35)
  1 / (1 + exp(-z))
}





logistic_rrr_auc_fast <- function(Y, X, rank, 
                                  lambda = 0.01, maxit = 1000, 
                                  tol = 1e-10, step = 0.1) {
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X)
  X0 <- cbind(1, X)
  
  # 1. Warm start with standard Logistic RRR is highly recommended for AUC
  B <- logistic_rrr(Y, X, rank = rank,maxit = 100)$coef
  # B <- matrix(rnorm((p + 1) * q, 0, 0.01), p + 1, q)
  
  for (iter in 1:maxit) {
    B_old <- B
    eta <- X0 %*% B
    G <- matrix(0, p + 1, q)
    
    for (j in 1:q) {
      y <- Y[, j]
      pos <- which(y == 1); neg <- which(y == 0)
      nP <- length(pos); nN <- length(neg)
      
      if (nP == 0 || nN == 0) next
      
      # Pairwise difference: eta_pos - eta_neg
      # D[i, k] = eta_pos[i] - eta_neg[k]
      D <- outer(eta[pos, j], eta[neg, j], "-")
      
      # We want to minimize Loss = sum(sigmoid(-D))
      # Derivative of sigmoid(-D) w.r.t D is -sigmoid(-D) * (1 - sigmoid(-D))
      W <- sigmoid(-D) * (1 - sigmoid(-D))
      
      # Gradient w.r.t eta:
      # For pos: sum over neg pairs. For neg: sum over pos pairs.
      grad_eta_pos <- rowSums(-W)
      grad_eta_neg <- colSums(W)
      
      # Map back to X0. Note: Intercept gradient for AUC is effectively 0.
      # We scale by (nP * nN) so the step size is invariant to class imbalance.
      G[, j] <- (crossprod(X0[pos, , drop=FALSE], grad_eta_pos) + 
                   crossprod(X0[neg, , drop=FALSE], grad_eta_neg)) / (nP * nN)
    }
    
    # 2. Gradient Descent Step
    B_new <- B - step * G
    
    # 3. Reduced-Rank Proximal Step (Apply only to predictors, not intercept)
    intercept <- B_new[1, , drop = FALSE]
    predictors <- B_new[-1, , drop = FALSE]
    
   # sv <- svd(predictors)
    sv <- irlba::irlba(predictors, nv = rank, nu = rank)
    
    d <- sv$d
    if (lambda > 0) d <- pmax(d - lambda * step, 0)
    
    r <- min(rank, sum(d > 1e-8))
    if (r > 0) {
      predictors <- sv$u[, 1:r, drop=FALSE] %*% diag(d[1:r], r, r) %*% t(sv$v[, 1:r, drop=FALSE])
    } else {
      predictors[] <- 0
    }
    
    B <- rbind(intercept, predictors)
    
    # Convergence check
    if (sum((B - B_old)^2) / (sum(B_old^2) + 1e-8) < tol) break
  }
  
  return(list(coef = B, iter = iter, converged = (iter < maxit)))
}


logistic_rrr_auc_margin  <- function(
    Y, X, rank,
    lambda = 0.01,
    maxit = 500,
    tol = 1e-8,
    step = 0.1
) {
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X)
  X0 <- cbind(1, X)
  
  # Warm start
  B <- logistic_rrr(Y, X, rank = rank, maxit = 100)$coef
  
  for (iter in 1:maxit) {
    B_old <- B
    eta <- X0 %*% B
    G <- matrix(0, p + 1, q)
    
    for (j in 1:q) {
      y <- Y[, j]
      pos <- which(y == 1)
      neg <- which(y == 0)
      nP <- length(pos); nN <- length(neg)
      if (nP == 0 || nN == 0) next
      
      # Pairwise score differences
      D <- outer(eta[pos, j], eta[neg, j], "-")
      
      # Indicator of misordered pairs
      I <- (D < 0)
      
      # Gradient w.r.t. linear predictor
      grad_eta_pos <- -rowSums(I)
      grad_eta_neg <-  colSums(I)
      
      # Map to coefficient gradient
      G[, j] <-
        (crossprod(X0[pos, , drop = FALSE], grad_eta_pos) +
           crossprod(X0[neg, , drop = FALSE], grad_eta_neg)) /
        (nP * nN)
    }
    
    # Gradient descent step
    B_new <- B - step * G
    
    # Reduced-rank proximal step (exclude intercept)
    intercept <- B_new[1, , drop = FALSE]
    predictors <- B_new[-1, , drop = FALSE]
    
    sv <- svd(predictors)
    d <- sv$d
    if (lambda > 0) d <- pmax(d - lambda * step, 0)
    
    r <- min(rank, sum(d > 1e-8))
    if (r > 0) {
      predictors <- sv$u[, 1:r, drop = FALSE] %*%
        diag(d[1:r], r, r) %*%
        t(sv$v[, 1:r, drop = FALSE])
    } else {
      predictors[] <- 0
    }
    
    B <- rbind(intercept, predictors)
    
    # Convergence check
    if (sum((B - B_old)^2) / (sum(B_old^2) + 1e-8) < tol)
      break
  }
  
  list(coef = B, iter = iter, converged = (iter < maxit))
}


logistic_rrr_auc_fast2 <- function(Y, X, rank,
                                   lambda = 0.01,
                                   maxit = 1000,
                                   tol = 1e-10,
                                   step = 0.1) {
  
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X)
  X0 <- cbind(1, X)
  
  B <- logistic_rrr(Y, X, rank = rank, maxit = 100)$coef
  G <- matrix(0, p + 1, q)
  
  for (iter in 1:maxit) {
    B_old <- B
    eta <- X0 %*% B
    G[,] <- 0
    
    for (j in 1:q) {
      y <- Y[, j]
      pos <- which(y == 1); neg <- which(y == 0)
      nP <- length(pos); nN <- length(neg)
      if (nP == 0 || nN == 0) next
      
      a <- eta[pos, j]
      b <- eta[neg, j]
      
      Ea <- exp(a)
      Eb <- exp(b)
      den <- outer(Ea, Eb, "+")^2
      W <- outer(Ea, Eb, "*") / den
      
      grad_eta_pos <- -rowSums(W)
      grad_eta_neg <-  colSums(W)
      
      G[-1, j] <-
        (crossprod(X[pos, , drop=FALSE], grad_eta_pos) +
           crossprod(X[neg, , drop=FALSE], grad_eta_neg)) /
        (nP * nN)
    }
    
    B_new <- B - step * G
    
    intercept <- B_new[1, , drop=FALSE]
    predictors <- B_new[-1, , drop=FALSE]
    
    sv <- irlba::irlba(predictors, nv = rank, nu = rank)
    d <- pmax(sv$d - lambda * step, 0)
    
    predictors <- sv$u %*% diag(d, length(d)) %*% t(sv$v)
    B <- rbind(intercept, predictors)
    
    if (sum((B - B_old)^2) / (sum(B_old^2) + 1e-8) < tol)
      break
  }
  
  list(coef = B, iter = iter, converged = (iter < maxit))
}

