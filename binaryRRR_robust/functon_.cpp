#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::mat sigmoid_cpp(const arma::mat& Z) {
  arma::mat Zc = clamp(Z, -35.0, 35.0);
  return 1.0 / (1.0 + exp(-Zc));
}

// [[Rcpp::export]]
List logistic_rrr_cpp(const arma::mat& Y,
                      const arma::mat& X,
                      int rank,
                      double lambda = 0.0,
                      int maxit = 1000,
                      double tol = 1e-10,
                      double step = 1.0) {
  
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  
  // Design matrix with intercept
  arma::mat X0(n, p + 1, fill::ones);
  X0.cols(1, p) = X;
  
  arma::mat B(p + 1, q, fill::zeros);
  
  int iter;
  int r = 0;
  
  for (iter = 0; iter < maxit; ++iter) {
    
    arma::mat eta = X0 * B;
    arma::mat mu  = sigmoid_cpp(eta);
    
    // Gradient
    arma::mat G = X0.t() * (Y - mu);
    
    arma::mat B_new = B + step * G / n;
    
    // Reduced-rank projection (exclude intercept)
    arma::mat predictors = B_new.rows(1, p);
    
    arma::mat U, V;
    arma::vec d;
    svd(U, d, V, predictors);
    
    if (lambda > 0.0)
      d = clamp(d - lambda, 0.0, datum::inf);
    
    r = std::min(rank, (int)sum(d > 0));
    
    if (r > 0) {
      predictors =
        U.cols(0, r - 1) *
        diagmat(d.subvec(0, r - 1)) *
        V.cols(0, r - 1).t();
    } else {
      predictors.zeros();
    }
    
    B_new.rows(1, p) = predictors;
    
    double diff =
      accu(square(B_new - B)) / (accu(square(B)) + 1e-8);
    
    B = B_new;
    
    if (diff < tol)
      break;
  }
  
  arma::mat eta = X0 * B;
  arma::mat mu  = sigmoid_cpp(eta);
  
  // Log-likelihood
  arma::mat eps_mu = clamp(mu, 1e-8, 1.0 - 1e-8);
  double logLik =
    accu(Y % log(eps_mu) + (1.0 - Y) % log(1.0 - eps_mu));
  
  return List::create(
    _["coef"]      = B,
    _["fitted"]   = mu,
    _["rank"]     = r,
    _["iter"]     = iter + 1,
    _["converged"]= (iter < maxit - 1),
                   _["logLik"]   = logLik
  );
}




// [[Rcpp::export]]
List logistic_rrr_auc_fast_cpp(const arma::mat& Y,
                               const arma::mat& X,
                               int rank,
                               double lambda = 0.01,
                               int maxit = 1000,
                               double tol = 1e-10,
                               double step = 0.1) {
  
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  
  arma::mat X0(n, p + 1, fill::ones);
  X0.cols(1, p) = X;
  
  // Warm start
  List init = logistic_rrr_cpp(Y, X, rank, 0.0, 100, tol, 1.0);
  arma::mat B = as<arma::mat>(init["coef"]);
  
  int iter;
  int r = 0;
  
  for (iter = 0; iter < maxit; ++iter) {
    
    arma::mat B_old = B;
    arma::mat eta = X0 * B;
    arma::mat G(p + 1, q, fill::zeros);
    
    for (int j = 0; j < q; ++j) {
      
      arma::uvec pos = find(Y.col(j) == 1);
      arma::uvec neg = find(Y.col(j) == 0);
      
      int nP = pos.n_elem;
      int nN = neg.n_elem;
      if (nP == 0 || nN == 0) continue;
      
      arma::vec grad_eta(n, fill::zeros);
      
      for (int ii = 0; ii < nP; ++ii) {
        for (int kk = 0; kk < nN; ++kk) {
          double D = eta(pos[ii], j) - eta(neg[kk], j);
          double s = 1.0 / (1.0 + std::exp(D));
          double w = s * (1.0 - s);
          
          grad_eta[pos[ii]] -= w;
          grad_eta[neg[kk]] += w;
        }
      }
      
      G.col(j) = X0.t() * grad_eta / (double)(nP * nN);
    }
    
    // Gradient step
    arma::mat B_new = B - step * G;
    
    // Proximal reduced-rank step
    arma::rowvec intercept = B_new.row(0);
    arma::mat predictors   = B_new.rows(1, p);
    
    arma::mat U, V;
    arma::vec d;
    svd(U, d, V, predictors);
    
    if (lambda > 0.0)
      d = clamp(d - lambda * step, 0.0, datum::inf);
    
    r = std::min(rank, (int)sum(d > 1e-8));
    
    if (r > 0) {
      predictors =
        U.cols(0, r - 1) *
        diagmat(d.subvec(0, r - 1)) *
        V.cols(0, r - 1).t();
    } else {
      predictors.zeros();
    }
    
    B.row(0)        = intercept;
    B.rows(1, p)    = predictors;
    
    double diff =
      accu(square(B - B_old)) / (accu(square(B_old)) + 1e-8);
    
    if (diff < tol)
      break;
  }
  
  return List::create(
    _["coef"]      = B,
    _["iter"]     = iter + 1,
    _["converged"]= (iter < maxit - 1),
                   _["rank"]     = r
  );
}





inline arma::mat sigmoid_stable(const arma::mat& Z) {
  arma::mat Zc = clamp(Z, -35.0, 35.0);
  return 1.0 / (1.0 + exp(-Zc));
}

// [[Rcpp::export]]
List logistic_rrr_hybrid_cpp(const arma::mat& Y,
                             const arma::mat& X,
                             int rank,
                             double alpha = 0.2,      // AUC weight
                             double lambda = 0.01,
                             int maxit = 1000,
                             double tol = 1e-10,
                             double step = 0.1) {
  
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  
  // Design matrix
  arma::mat X0(n, p + 1, fill::ones);
  X0.cols(1, p) = X;
  
  // Warm start via likelihood
  arma::mat B(p + 1, q, fill::zeros);
  
  int iter, r = 0;
  
  for (iter = 0; iter < maxit; ++iter) {
    
    arma::mat B_old = B;
    arma::mat eta   = X0 * B;
    arma::mat mu    = sigmoid_stable(eta);
    
    arma::mat G_loglik(p + 1, q, fill::zeros);
    arma::mat G_auc(p + 1, q, fill::zeros);
    
    /* --------------------------------------------------
     1. Logistic log-likelihood gradient
     -------------------------------------------------- */
    G_loglik = X0.t() * (mu - Y) / n;
    
    /* --------------------------------------------------
     2. Pairwise AUC gradient
     -------------------------------------------------- */
    for (int j = 0; j < q; ++j) {
      
      arma::uvec pos = find(Y.col(j) == 1);
      arma::uvec neg = find(Y.col(j) == 0);
      
      int nP = pos.n_elem;
      int nN = neg.n_elem;
      if (nP == 0 || nN == 0) continue;
      
      arma::vec grad_eta(n, fill::zeros);
      
      for (int i = 0; i < nP; ++i) {
        for (int k = 0; k < nN; ++k) {
          
          double D = eta(pos[i], j) - eta(neg[k], j);
          double s = 1.0 / (1.0 + std::exp(D));
          double w = s * (1.0 - s);
          
          grad_eta(pos[i]) -= w;
          grad_eta(neg[k]) += w;
        }
      }
      
      G_auc.col(j) = X0.t() * grad_eta / (double)(nP * nN);
    }
    
    /* --------------------------------------------------
     3. Hybrid gradient
     -------------------------------------------------- */
    arma::mat G = (1.0 - alpha) * G_loglik + alpha * G_auc;
    
    /* --------------------------------------------------
     4. Gradient step
     -------------------------------------------------- */
    arma::mat B_new = B - step * G;
    
    /* --------------------------------------------------
     5. Reduced-rank proximal step (no intercept)
     -------------------------------------------------- */
    arma::rowvec intercept = B_new.row(0);
    arma::mat predictors   = B_new.rows(1, p);
    
    arma::mat U, V;
    arma::vec d;
    svd(U, d, V, predictors);
    
    if (lambda > 0.0)
      d = clamp(d - lambda * step, 0.0, datum::inf);
    
    r = std::min(rank, (int)sum(d > 1e-8));
    
    if (r > 0) {
      predictors =
        U.cols(0, r - 1) *
        diagmat(d.subvec(0, r - 1)) *
        V.cols(0, r - 1).t();
    } else {
      predictors.zeros();
    }
    
    B.row(0)     = intercept;
    B.rows(1, p) = predictors;
    
    /* --------------------------------------------------
     6. Convergence check
     -------------------------------------------------- */
    double diff =
    accu(square(B - B_old)) / (accu(square(B_old)) + 1e-8);
    
    if (diff < tol)
      break;
  }
  
  return List::create(
    _["coef"]       = B,
    _["rank"]       = r,
    _["iter"]       = iter + 1,
    _["converged"]  = (iter < maxit - 1)
  );
}



// [[Rcpp::export]]
List logistic_rrr_hybrid_adaptive_cpp(
    const arma::mat& Y,
    const arma::mat& X,
    int rank,
    double lambda = 0.01,
    int maxit = 1000,
    double tol = 1e-10,
    double step = 0.1,
    double alpha_max = 0.5,      // safety cap
    double alpha_smooth = 0.9,   // EMA smoothing
    int warmup = 10              // iterations before AUC kicks in
) {
  
  int n = Y.n_rows;
  int q = Y.n_cols;
  int p = X.n_cols;
  
  arma::mat X0(n, p + 1, fill::ones);
  X0.cols(1, p) = X;
  
  arma::mat B(p + 1, q, fill::zeros);
  
  double alpha = 0.0;
  int iter, r = 0;
  
  for (iter = 0; iter < maxit; ++iter) {
    
    arma::mat B_old = B;
    arma::mat eta = X0 * B;
    arma::mat mu  = sigmoid_stable(eta);
    
    /* -----------------------------
     Logistic gradient
     ----------------------------- */
    arma::mat G_loglik =
    X0.t() * (mu - Y) / n;
    
    /* -----------------------------
     AUC gradient
     ----------------------------- */
    arma::mat G_auc(p + 1, q, fill::zeros);
    
    if (iter >= warmup) {
      for (int j = 0; j < q; ++j) {
        
        arma::uvec pos = find(Y.col(j) == 1);
        arma::uvec neg = find(Y.col(j) == 0);
        int nP = pos.n_elem, nN = neg.n_elem;
        if (nP == 0 || nN == 0) continue;
        
        arma::vec grad_eta(n, fill::zeros);
        
        for (int i = 0; i < nP; ++i) {
          for (int k = 0; k < nN; ++k) {
            double D = eta(pos[i], j) - eta(neg[k], j);
            double s = 1.0 / (1.0 + std::exp(D));
            double w = s * (1.0 - s);
            
            grad_eta(pos[i]) -= w;
            grad_eta(neg[k]) += w;
          }
        }
        
        G_auc.col(j) =
          X0.t() * grad_eta / (double)(nP * nN);
      }
    }
    
    /* -----------------------------
     Adaptive alpha update
     ----------------------------- */
    if (iter >= warmup) {
      double n_log = norm(G_loglik, "fro");
      double n_auc = norm(G_auc, "fro");
      
      double alpha_raw =
        n_auc / (n_auc + n_log + 1e-12);
      
      alpha_raw = std::min(alpha_raw, alpha_max);
      
      alpha =
        alpha_smooth * alpha +
        (1.0 - alpha_smooth) * alpha_raw;
    }
    
    /* -----------------------------
     Hybrid gradient step
     ----------------------------- */
    arma::mat G =
    (1.0 - alpha) * G_loglik + alpha * G_auc;
    
    arma::mat B_new = B - step * G;
    
    /* -----------------------------
     Reduced-rank proximal step
     ----------------------------- */
    arma::rowvec intercept = B_new.row(0);
    arma::mat predictors   = B_new.rows(1, p);
    
    arma::mat U, V;
    arma::vec d;
    svd(U, d, V, predictors);
    
    d = clamp(d - lambda * step, 0.0, datum::inf);
    r = std::min(rank, (int)sum(d > 1e-8));
    
    if (r > 0) {
      predictors =
        U.cols(0, r - 1) *
        diagmat(d.subvec(0, r - 1)) *
        V.cols(0, r - 1).t();
    } else {
      predictors.zeros();
    }
    
    B.row(0)     = intercept;
    B.rows(1, p) = predictors;
    
    /* -----------------------------
     Convergence check
     ----------------------------- */
    double diff =
    accu(square(B - B_old)) /
      (accu(square(B_old)) + 1e-8);
    
    if (diff < tol)
      break;
  }
  
  return List::create(
    _["coef"]       = B,
    _["rank"]       = r,
    _["alpha"]      = alpha,
    _["iter"]       = iter + 1,
    _["converged"]  = (iter < maxit - 1)
  );
}
