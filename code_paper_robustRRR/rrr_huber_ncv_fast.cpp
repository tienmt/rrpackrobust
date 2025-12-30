#include <RcppArmadillo.h>
#include <string>
#include <algorithm>
#include <cctype>
#include <cmath>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//////////////////////////////////////////////////////////////
// Helper: lowercase a std::string
//////////////////////////////////////////////////////////////
static std::string to_lower_str(const std::string &s_in) {
  std::string s = s_in;
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c){ return std::tolower(c); });
  return s;
}

//////////////////////////////////////////////////////////////
// Huber loss with missing values ignored
//////////////////////////////////////////////////////////////
// [[Rcpp::export]]
double huber_loss_cpp(const arma::mat& R, double tau) {
  double loss_sum = 0.0;
  for (arma::uword i = 0; i < R.n_elem; ++i) {
    double val = R[i];
    if (Rcpp::NumericVector::is_na(val)) continue; // skip missing
    double absr = std::abs(val);
    if (absr <= tau)
      loss_sum += 0.5 * val * val;
    else
      loss_sum += tau * (absr - 0.5 * tau);
  }
  return loss_sum;
}

//////////////////////////////////////////////////////////////
// Huber gradient ignoring missing Y values
//////////////////////////////////////////////////////////////
// [[Rcpp::export]]
arma::mat huber_gradient_cpp(const arma::mat& Y, const arma::mat& X, 
                             const arma::mat& B, double tau) {
  int n = X.n_rows;
  arma::mat R = Y - X * B;
  arma::mat psi = arma::zeros<arma::mat>(R.n_rows, R.n_cols);
  
  for (arma::uword i = 0; i < R.n_rows; ++i) {
    for (arma::uword j = 0; j < R.n_cols; ++j) {
      double val = R(i, j);
      if (Rcpp::NumericVector::is_na(Y(i, j))) continue; 
      if (val > tau) psi(i, j) = tau;
      else if (val < -tau) psi(i, j) = -tau;
      else psi(i, j) = val;
    }
  }
  
  arma::mat grad = -(1.0 / n) * X.t() * psi;
  return grad;
}

////////////////////////////////////////////////////////////
// SCAD and MCP penalties
////////////////////////////////////////////////////////////
double scad_penalty_cpp(double t, double lambda, double a = 3.7) {
  if (t <= lambda) return lambda * t;
  if (t <= a * lambda)
    return (-t * t + 2 * a * lambda * t - lambda * lambda) / (2.0 * (a - 1));
  return (lambda * lambda * (a + 1)) / 2.0;
}

double mcp_penalty_cpp(double t, double lambda, double gamma = 3.0) {
  if (t <= gamma * lambda)
    return lambda * t - (t * t) / (2.0 * gamma);
  return 0.5 * gamma * lambda * lambda;
}

////////////////////////////////////////////////////////////
// Scalar prox by grid search + refinement
////////////////////////////////////////////////////////////
double scalar_prox_cpp(double s, double alpha, double lambda, 
                       const std::string &penalty_raw, double a_or_gamma) {
  std::string penalty = to_lower_str(penalty_raw);
  
  if (penalty == "nuclear" || penalty == "nn" || penalty == "nuc") {
    double thresh = alpha * lambda;
    double t = s - thresh;
    return (t > 0.0) ? t : 0.0;
  }
  
  auto obj = [&](double t) {
    if (t < 0) return 1e20;
    double val = 0.5 * std::pow(t - s, 2);
    if (penalty == "scad")
      val += alpha * scad_penalty_cpp(t, lambda, a_or_gamma);
    else 
      val += alpha * mcp_penalty_cpp(t, lambda, a_or_gamma);
    return val;
  };
  
  double ub = std::max({s + 5 * lambda, lambda * (a_or_gamma + 2), s * 1.5 + lambda});
  if (ub <= 0) ub = lambda + 1.0;
  int ngrid = 41;
  double step = ub / (ngrid - 1);
  double best_t = 0, best_val = 1e20;
  
  for (int i = 0; i < ngrid; i++) {
    double t = i * step;
    double v = obj(t);
    if (v < best_val) { best_val = v; best_t = t; }
  }
  
  double left = std::max(0.0, best_t - ub / 40.0);
  double right = std::min(ub, best_t + ub / 40.0);
  double best_ref = best_t;
  best_val = obj(best_t);
  for (int i = 0; i < 30; i++) {
    double t1 = left + (right - left) / 3.0;
    double t2 = right - (right - left) / 3.0;
    double f1 = obj(t1);
    double f2 = obj(t2);
    if (f1 < f2) right = t2; else left = t1;
  }
  double tmid = 0.5 * (left + right);
  if (obj(tmid) < best_val) best_ref = tmid;
  return best_ref;
}

////////////////////////////////////////////////////////////
// MANUAL THIN SVD IMPLEMENTATION
// Works on all Armadillo versions.
// Computes U, s, V such that X = U * diag(s) * V.t()
// Efficiently handles cases where n >> p or p >> n
////////////////////////////////////////////////////////////
void my_svd_econ(arma::mat &U, arma::vec &s, arma::mat &V, const arma::mat &X) {
  int n = X.n_rows;
  int p = X.n_cols;
  
  if (n >= p) {
    // Case 1: Tall matrix (n > p). Compute eigen of X'X (p x p matrix)
    // X'X = V * S^2 * V'
    arma::mat H = X.t() * X;
    arma::vec eigval;
    arma::mat eigvec;
    
    // eig_sym is standard and reliable
    arma::eig_sym(eigval, eigvec, H);
    
    // Sort descending (eig_sym returns ascending)
    arma::uvec indices = sort_index(eigval, "descend");
    eigval = eigval.elem(indices);
    V = eigvec.cols(indices);
    
    // Clamp negative eigenvalues due to precision noise
    eigval.elem(find(eigval < 0)).zeros();
    s = sqrt(eigval);
    
    // Recover U = X * V * S^-1
    // Filter out zero singular values to avoid division by zero
    arma::uvec nonzero = find(s > 1e-12);
    if (nonzero.n_elem < s.n_elem) {
      s = s.elem(nonzero);
      V = V.cols(nonzero);
    }
    
    U = X * V * diagmat(1.0 / s);
  } 
  else {
    // Case 2: Wide matrix (p > n). Compute eigen of XX' (n x n matrix)
    // XX' = U * S^2 * U'
    arma::mat H = X * X.t();
    arma::vec eigval;
    arma::mat eigvec;
    
    arma::eig_sym(eigval, eigvec, H);
    
    arma::uvec indices = sort_index(eigval, "descend");
    eigval = eigval.elem(indices);
    U = eigvec.cols(indices);
    
    eigval.elem(find(eigval < 0)).zeros();
    s = sqrt(eigval);
    
    arma::uvec nonzero = find(s > 1e-12);
    if (nonzero.n_elem < s.n_elem) {
      s = s.elem(nonzero);
      U = U.cols(nonzero);
    }
    
    // Recover V = X' * U * S^-1
    V = X.t() * U * diagmat(1.0 / s);
  }
}

////////////////////////////////////////////////////////////
// Spectral proximal operator (Uses manual thin SVD)
////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List spectral_prox_cpp(const arma::mat& G, double alpha, double lambda,
                       std::string penalty = "SCAD", 
                       double a = 3.7, double gamma = 3.0, 
                       Nullable<int> svd_rank = R_NilValue) {
  std::string pen = to_lower_str(penalty);
  
  arma::mat U, V;
  arma::vec s;
  
  // Use our manual compatible function instead of svd_econ
  my_svd_econ(U, s, V, G);
  
  // Truncate if rank limit is requested
  int k = s.n_elem;
  if (svd_rank.isNotNull()) {
    int r = as<int>(svd_rank);
    if (k > r) {
      k = r;
      s = s.head(k);
      U = U.cols(0, k - 1);
      V = V.cols(0, k - 1);
    }
  }
  
  arma::vec s_new(k);
  if (pen == "nuclear" || pen == "nn" || pen == "nuc") {
    double thresh = alpha * lambda;
    for (int j = 0; j < k; j++) {
      double t = s[j] - thresh;
      s_new[j] = (t > 0.0) ? t : 0.0;
    }
  } else {
    for (int j = 0; j < k; j++) {
      double param = (pen == "scad") ? a : gamma;
      s_new[j] = scalar_prox_cpp(s[j], alpha, lambda, penalty, param);
    }
  }
  
  arma::uvec keep = arma::find(s_new > 0.0);
  if (keep.n_elem == 0) {
    return List::create(_["B"] = arma::zeros<arma::mat>(G.n_rows, G.n_cols),
                        _["s"] = arma::vec());
  }
  
  arma::mat Uk = U.cols(keep);
  arma::mat Vk = V.cols(keep);
  arma::vec sk = s_new.elem(keep);
  arma::mat B_out = Uk * diagmat(sk) * Vk.t();
  
  return List::create(_["B"] = B_out, _["s"] = sk);
}

////////////////////////////////////////////////////////////
// Spectral norm squared (Power Method)
////////////////////////////////////////////////////////////
// [[Rcpp::export]]
double spectral_norm_sq_over_n_cpp(const arma::mat& X, int niter = 30) {
  int n = X.n_rows;
  int p = X.n_cols;
  arma::vec v = arma::randn(p);
  v /= arma::norm(v, 2);
  for (int i = 0; i < niter; i++) {
    arma::vec w = X.t() * (X * v);
    double normw = arma::norm(w, 2);
    if (normw < 1e-12) break;
    v = w / normw;
  }
  double r = arma::as_scalar(v.t() * (X.t() * (X * v)));
  return r / n;
}

////////////////////////////////////////////////////////////
// Helper: Calc Penalty (avoid SVD)
////////////////////////////////////////////////////////////
double calc_penalty_val(const arma::vec& s, double lambda, 
                        const std::string& pen, double scad_a, double mcp_gamma) {
  double pen_val = 0.0;
  if (pen == "nuclear" || pen == "nn" || pen == "nuc") {
    pen_val = lambda * arma::accu(s);
  } else {
    for (auto t : s) {
      if (pen == "scad") pen_val += scad_penalty_cpp(t, lambda, scad_a);
      else pen_val += mcp_penalty_cpp(t, lambda, mcp_gamma);
    }
  }
  return pen_val;
}

////////////////////////////////////////////////////////////
// MAIN SOLVER
////////////////////////////////////////////////////////////
// [[Rcpp::export]]
List proximal_gradient_solver_cpp(const arma::mat& X, const arma::mat& Y,
                                  double tau = 1.0, double lambda = 1.0,
                                  std::string penalty = "SCAD",
                                  double scad_a = 3.7, double mcp_gamma = 3.0,
                                  Nullable<double> alpha_in = R_NilValue,
                                  int max_iter = 2000, double tol = 1e-7,
                                  bool verbose = true,
                                  Nullable<int> svd_rank = R_NilValue,
                                  Nullable<arma::mat> warm_start = R_NilValue) {
  
  std::string pen = to_lower_str(penalty);
  int n = X.n_rows, p = X.n_cols, q = Y.n_cols;
  
  arma::mat Bk = warm_start.isNull() ? arma::zeros(p, q) : as<arma::mat>(warm_start);
  
  // Initial singular values
  arma::mat U_tmp, V_tmp; 
  arma::vec sk;
  my_svd_econ(U_tmp, sk, V_tmp, Bk);
  
  double alpha;
  if (alpha_in.isNull()) {
    double L = spectral_norm_sq_over_n_cpp(X);
    alpha = (L > 1e-12) ? (0.9 / L) : 1e-3; 
  } else {
    alpha = as<double>(alpha_in);
  }
  
  auto compute_obj = [&](const arma::mat& B_in, const arma::vec& s_val) {
    arma::mat R = Y - X * B_in;
    double hub = (1.0 / n) * huber_loss_cpp(R, tau);
    double pen_val = calc_penalty_val(s_val, lambda, pen, scad_a, mcp_gamma);
    return hub + pen_val;
  };
  
  double prev_obj = compute_obj(Bk, sk);
  
  if (verbose) Rcout << "Alpha: " << alpha << " Init Obj: " << prev_obj << "\n";
  
  for (int k = 0; k < max_iter; k++) {
    arma::mat grad = huber_gradient_cpp(Y, X, Bk, tau);
    arma::mat G = Bk - alpha * grad;
    
    // Reuse SVD result from prox!
    List prox_res = spectral_prox_cpp(G, alpha, lambda, penalty, scad_a, mcp_gamma, svd_rank);
    arma::mat Bnext = prox_res["B"];
    arma::vec snext = prox_res["s"];
    
    double next_obj = compute_obj(Bnext, snext);
    
    // Backtracking
    if (next_obj > prev_obj + 1e-12) {
      int bt = 0;
      while (next_obj > prev_obj + 1e-12 && bt < 8) {
        alpha *= 0.5;
        if (verbose) Rcout << " BT alpha=" << alpha << "\n";
        G = Bk - alpha * grad;
        prox_res = spectral_prox_cpp(G, alpha, lambda, penalty, scad_a, mcp_gamma, svd_rank);
        Bnext = Rcpp::as<arma::mat>(prox_res["B"]);
        snext = Rcpp::as<arma::vec>(prox_res["s"]);
        
        next_obj = compute_obj(Bnext, snext);
        bt++;
      }
    }
    
    double num = arma::norm(Bnext - Bk, "fro");
    double den = std::max(1.0, arma::norm(Bk, "fro"));
    
    if (num / den < tol) {
      if (verbose) Rcout << "Converged iter " << k << "\n";
      return List::create(_["B"] = Bnext, _["obj"] = next_obj, _["iter"] = k+1);
    }
    
    Bk = Bnext;
    sk = snext;
    prev_obj = next_obj;
    
    if (verbose && (k % 50 == 0)) Rcout << "iter " << k << " obj " << next_obj << "\n";
  }
  
  return List::create(_["B"] = Bk, _["obj"] = prev_obj, _["iter"] = max_iter);
}