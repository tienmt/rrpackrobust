#' Cross-Validated Robust Reduced Rank Regression
#'
#' Performs K-fold cross-validation for robust reduced rank regression
#' using a Huber loss and a spectral penalty on the coefficient matrix.
#' The tuning parameters are selected by minimizing the cross-validated
#' prediction error.
#'
#' The underlying estimator solves a multivariate linear regression
#' problem with coefficient matrix \eqn{B} via a proximal-gradient
#' algorithm, where robustness is induced through the Huber loss and
#' low-rank structure is encouraged via a spectral penalty applied to
#' the singular values of \eqn{B}.
#'
#' @param X Numeric design matrix of dimension \eqn{n \times p}, where
#'   rows correspond to observations and columns to predictors.
#' @param Y Numeric response matrix of dimension \eqn{n \times q}.
#'   Missing values (\code{NA}) are allowed and are handled internally.
#' @param penalty Character string specifying the spectral penalty to use.
#'   Must be one of \code{"SCAD"}, \code{"MCP"}, or \code{"nuclear"}.
#' @param lambda_grid Optional numeric vector of candidate regularization
#'   parameters \eqn{\lambda}. If \code{NULL}, a logarithmically spaced
#'   grid is constructed automatically based on the singular values of
#'   \eqn{X^\top Y}.
#' @param nfolds Integer giving the number of folds for cross-validation.
#' @param seed Integer seed used to generate the cross-validation folds.
#' @param tau_grid Optional numeric vector of Huber loss threshold
#'   parameters. If \code{NULL}, a default grid is used.
#' @param scad_a Shape parameter for the SCAD penalty. Ignored unless
#'   \code{penalty = "SCAD"}.
#' @param mcp_gamma Shape parameter for the MCP penalty. Ignored unless
#'   \code{penalty = "MCP"}.
#' @param alpha_in Optional step-size parameter for the proximal-gradient
#'   algorithm. If \code{NULL}, the step size is selected internally by
#'   the solver.
#' @param max_iter Maximum number of proximal-gradient iterations used
#'   for each model fit.
#' @param tol Convergence tolerance for the proximal-gradient algorithm.
#' @param svd_rank Optional integer specifying a truncated SVD rank to
#'   accelerate computation. If \code{NULL}, a full SVD is used.
#' @param verbose Logical; if \code{TRUE}, progress information is printed
#'   during cross-validation.
#'
#' @details
#' For each combination of \eqn{\tau} in \code{tau_grid} and \eqn{\lambda}
#' in \code{lambda_grid}, the data are split into \code{nfolds} folds.
#' The model is trained on \eqn{nfolds - 1} folds and evaluated on the
#' held-out fold using mean squared prediction error. The reported
#' cross-validation loss is the average error across folds.
#'
#' The optimal tuning parameters are selected as the pair
#' \eqn{(\tau, \lambda)} minimizing the cross-validated loss. A final
#' model is then refit on the full data using these parameters.
#'
#' @return A list with the following components:
#' \describe{
#'   \item{best_fit}{The fitted model object returned by
#'     \code{rr.robust} using the optimal tuning
#'     parameters.}
#'   \item{B_best}{Estimated coefficient matrix corresponding to the
#'     optimal model.}
#'   \item{tau_grid}{Vector of Huber loss parameters considered.}
#'   \item{lambda_grid}{Vector of regularization parameters considered.}
#'   \item{cv_loss}{Matrix of cross-validated losses, with rows
#'     corresponding to values in \code{tau_grid} and columns to values
#'     in \code{lambda_grid}.}
#'   \item{best_tau}{Selected value of the Huber loss parameter.}
#'   \item{best_lambda}{Selected value of the regularization parameter.}
#'   \item{best_tau_idx}{Index of \code{best_tau} in \code{tau_grid}.}
#'   \item{best_lambda_idx}{Index of \code{best_lambda} in
#'     \code{lambda_grid}.}
#' }
#'
#' @seealso \code{\link{rr.robust}}
#'
#' @references
#' Fan, J. and Li, R. (2001). Variable selection via nonconcave penalized
#' likelihood and its oracle properties. \emph{Journal of the American
#' Statistical Association}.
#'
#' Zhang, C.-H. (2010). Nearly unbiased variable selection under minimax
#' concave penalty. \emph{Annals of Statistics}.
#'
#' @examples
#' \dontrun{
#' ## Simulated robust reduced rank regression with three penalties
#' set.seed(123)
#'
#' n <- 200
#' p <- 12
#' q <- 7
#' r_true <- 2
#'
#' ## True low-rank coefficient matrix
#' U <- matrix(rnorm(p * r_true), p, r_true)
#' V <- matrix(rnorm(q * r_true), q, r_true)
#' Btrue <- U %*% t(V)
#'
#' ## Design matrix
#' X <- matrix(rnorm(n * p), n, p)
#'
#' ## Heavy-tailed noise
#' Y <- X %*% Btrue + 1.5 * rt(n * q, df = 3)
#'
#' ## Introduce missing values in Y
#' missing_rate <- 0.2
#' idx_na <- sample(seq_len(n * q), floor(missing_rate * n * q))
#' Y[idx_na] <- NA
#'
#' ## SCAD penalty
#' fit_scad <- cv.rr.robust(
#'   X = X,   Y = Y,
#'   penalty = "SCAD",   verbose = FALSE
#' )
#'
#' ## MCP penalty
#' fit_mcp <- cv.rr.robust(
#'   X = X,   Y = Y,
#'   penalty = "MCP",   verbose = FALSE
#' )
#'
#' ## Nuclear norm penalty
#' fit_nuclear <- cv.rr.robust(
#'   X = X,  Y = Y,
#'   penalty = "nuclear",  verbose = FALSE
#' )
#'
#' ## Frobenius norm estimation errors
#' c(
#'   SCAD    = sum((Btrue - fit_scad$B_best)^2),
#'   MCP     = sum((Btrue - fit_mcp$B_best)^2),
#'   Nuclear = sum((Btrue - fit_nuclear$B_best)^2)
#' )
#' }
#' @export
cv.rr.robust <-  function(
    X, Y,
    penalty = c("SCAD", "MCP", "nuclear"),
    lambda_grid = NULL,
    nfolds = 5,
    seed = 1,
    tau_grid = NULL,
    scad_a = 3.7,
    mcp_gamma = 3,
    alpha_in = NULL,
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
                                              alpha_in = alpha_in, max_iter = max_iter, tol = tol, verbose = FALSE, svd_rank = svd_rank  )

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






#' Proximal Gradient Solver for Robust Low-Rank Regression
#'
#' Solves a multivariate linear regression problem with Huber loss
#' and a spectral penalty using a proximal-gradient algorithm.
#'
#' The optimization problem is
#' \deqn{
#' \min_B \; \frac{1}{n} \sum_{i=1}^n \ell_\tau(Y_i - X_i B)
#' + \lambda \sum_j r(\sigma_j(B)),
#' }
#' where \eqn{\ell_\tau} is the Huber loss and \eqn{r} is a
#' nonconvex (SCAD, MCP) or convex (nuclear) penalty applied
#' to the singular values of \eqn{B}.
#'
#' @param X Numeric matrix of predictors of dimension \eqn{n \times p}.
#' @param Y Numeric matrix of responses of dimension \eqn{n \times q}.
#' @param tau Positive Huber threshold parameter.
#' @param lambda Nonnegative regularization parameter.
#' @param penalty Character string specifying the spectral penalty.
#'   One of \code{"SCAD"}, \code{"MCP"}, or \code{"nuclear"}.
#' @param scad_a SCAD shape parameter (ignored unless \code{penalty = "SCAD"}).
#' @param mcp_gamma MCP shape parameter (ignored unless \code{penalty = "MCP"}).
#' @param alpha_in Optional step size for the proximal-gradient algorithm.
#'   If \code{NULL}, the step size is chosen internally.
#' @param max_iter Maximum number of proximal-gradient iterations.
#' @param tol Convergence tolerance for relative change in the objective.
#' @param verbose Logical; if \code{TRUE}, progress information is printed.
#' @param svd_rank Optional integer specifying a truncated SVD rank
#'   for computational efficiency. If \code{NULL}, a full SVD is used.
#' @param warm_start Optional numeric matrix providing an initial value
#'   for \eqn{B}. Must have dimension \eqn{p \times q}.
#'
#' @return A list with components:
#' \describe{
#'   \item{B}{Estimated coefficient matrix of dimension \eqn{p \times q}.}
#'   \item{objective}{Final value of the objective function.}
#'   \item{iter}{Number of iterations performed.}
#'   \item{converged}{Logical indicating whether convergence was achieved.}
#' }
#'
#' @details
#' This function is a thin R wrapper around a compiled C++ routine
#' accessed via \code{.Call}. All heavy numerical computation is
#' performed in C++ for efficiency.
#'
#' Missing values in \code{Y} are allowed and are handled internally.
#'
#' @seealso \code{\link{cv.rr.robust}}
#' @examples
#' \dontrun{
#' ## Simulated robust reduced rank regression with three penalties
#' set.seed(123)
#'
#' n <- 200
#' p <- 12
#' q <- 7
#' r_true <- 2
#'
#' ## True low-rank coefficient matrix
#' U <- matrix(rnorm(p * r_true), p, r_true)
#' V <- matrix(rnorm(q * r_true), q, r_true)
#' Btrue <- U %*% t(V)
#'
#' ## Design matrix
#' X <- matrix(rnorm(n * p), n, p)
#'
#' ## Heavy-tailed noise
#' Y <- X %*% Btrue + 1.5 * rt(n * q, df = 3)
#'
#' ## Introduce missing values in Y
#' missing_rate <- 0.1
#' idx_na <- sample(seq_len(n * q), floor(missing_rate * n * q))
#' Y[idx_na] <- NA
#'
#' ## SCAD penalty
#' fit_scad <- rr.robust( lambda = 0.1 ,
#'   X = X,   Y = Y,
#'   penalty = "SCAD",   verbose = FALSE
#' )
#'
#' ## MCP penalty
#' fit_mcp <- rr.robust( lambda = 0.1 ,
#'   X = X,   Y = Y,
#'   penalty = "MCP",   verbose = FALSE
#' )
#'
#' ## Nuclear norm penalty
#' fit_nuclear <- rr.robust( lambda = 0.1 ,
#'   X = X,  Y = Y,
#'   penalty = "nuclear",  verbose = FALSE
#' )
#'
#' ## Frobenius norm estimation errors
#' c(
#'   SCAD    = sum((Btrue - fit_scad$B )^2),
#'   MCP     = sum((Btrue - fit_mcp$B )^2),
#'   Nuclear = sum((Btrue - fit_nuclear$B )^2)
#' )
#' }
#'
#' @useDynLib rrpackrobust
#' @importFrom Rcpp sourceCpp
#' @export
rr.robust <- function(
    X,
    Y,
    tau = 1,
    lambda = 1,
    penalty = c("SCAD", "MCP", "nuclear"),
    scad_a = 3.7,
    mcp_gamma = 3,
    alpha_in = NULL,
    max_iter = 2000L,
    tol = 1e-7,
    verbose = TRUE,
    svd_rank = NULL,
    warm_start = NULL
) {
  penalty <- match.arg(penalty)

  ## ---- basic validation ----
  if (!is.matrix(X) || !is.numeric(X))
    stop("X must be a numeric matrix.")

  if (!is.matrix(Y) || !is.numeric(Y))
    stop("Y must be a numeric matrix.")

  if (nrow(X) != nrow(Y))
    stop("X and Y must have the same number of rows.")

  if (!is.numeric(tau) || length(tau) != 1L || tau <= 0)
    stop("tau must be a positive scalar.")

  if (!is.numeric(lambda) || length(lambda) != 1L || lambda < 0)
    stop("lambda must be a nonnegative scalar.")

  if (!is.null(warm_start)) {
    if (!is.matrix(warm_start) || !is.numeric(warm_start))
      stop("warm_start must be a numeric matrix.")

    if (!all(dim(warm_start) == c(ncol(X), ncol(Y))))
      stop("warm_start must have dimension ncol(X) x ncol(Y).")
  }

  if (!is.null(svd_rank)) {
    if (!is.numeric(svd_rank) || svd_rank <= 0)
      stop("svd_rank must be a positive integer or NULL.")
    svd_rank <- as.integer(svd_rank)
  }

  ## ---- call C++ backend ----
  proximal_gradient_solver_cpp(
    X,
    Y,
    tau,
    lambda,
    penalty,
    scad_a,
    mcp_gamma,
    alpha_in,
    as.integer(max_iter),
    tol,
    verbose,
    svd_rank,
    warm_start
  )
}
