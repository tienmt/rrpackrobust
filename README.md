# rrpack_robust
See the paper: "Robust reduced rank regression under heavy-tailed noise and missing data via non-convex penalization".

`rrpackrobust` is an R package for **robust multiple-response linear regression with low-rank structure**, combining  
Huber loss for robustness against heavy-tailed noise and outliers with spectral regularization to enforce reduced rank.

The package supports **nonconvex penalties (SCAD, MCP)** as well as the **nuclear norm**, and provides cross-validated tuning via an efficient proximal-gradient algorithm implemented in C++.

---

## Key Features

- Robust multivariate regression using **Huber loss**
- Reduced rank structure via **spectral penalties**
  - **SCAD**
  - **MCP**
  - **Nuclear norm**
- Efficient proximal-gradient solver implemented in C++
- K-fold cross-validation for tuning regularization parameters: for both **lambda and tau (in Huber loss)**
- Handles **missing values in the response matrix**
- Designed for high-dimensional and **heavy-tailed** data

---

# Installation From GitHub

```r
# install.packages("devtools")
devtools::install_github("tienmt/rrpackrobust")
````

You will need a working C++ toolchain:

* **Windows**: Rtools
* **macOS**: Xcode Command Line Tools
* **Linux**: `g++` / `clang`

---

## Main Functions

### `rr.robust()`

Fits a robust reduced rank regression model for fixed tuning parameters.

```r
rr.robust(
  X, Y,
  penalty = c("SCAD", "MCP", "nuclear"),
  tau = 1,
  lambda = 1,
  scad_a = 3.7,
  mcp_gamma = 3,
  max_iter = 2000,
  tol = 1e-5,
  svd_rank = NULL,
  verbose = TRUE
)
```

---

### `cv.rr.robust()`

Performs **K-fold cross-validation** to select tuning parameters and refits the final model.

```r
cv.rr.robust(
  X, Y,
  penalty = c("SCAD", "MCP", "nuclear"),
  lambda_grid = NULL,
  tau_grid = NULL,
  nfolds = 5,
  seed = 1,
  scad_a = 3.7,
  mcp_gamma = 3,
  max_iter = 2000,
  tol = 1e-5,
  svd_rank = NULL,
  verbose = TRUE
)
```

Returns the optimal coefficient matrix, CV loss surface, and selected parameters.

---

## Example

```r
 library(rrpackrobust)

set.seed(123)

n <- 200
p <- 12
q <- 7
r_true <- 2

## True low-rank coefficient matrix
U <- matrix(rnorm(p * r_true), p, r_true)
V <- matrix(rnorm(q * r_true), q, r_true)
Btrue <- U %*% t(V)

## Design matrix
X <- matrix(rnorm(n * p), n, p)

## Heavy-tailed noise
Y <- X %*% Btrue + matrix(rt(n * q, df = 3), n, q)

## Introduce missing values
missing_rate = 0.2
idx_na <- sample(seq_len(n * q), floor(missing_rate * n * q))
Y[idx_na] <- NA

## Cross-validated robust RRR
fit <- cv.rr.robust(
  X = X,
  Y = Y,
  penalty = "SCAD",
  verbose = FALSE
)

## Estimated coefficients
Bhat <- fit$B_best

## Estimation error
sum((Btrue - Bhat)^2)
```

---

## Supported Penalties

| Penalty | Type      | Description                            |
| ------- | --------- | -------------------------------------- |
| SCAD    | Nonconvex | Reduces bias for large singular values |
| MCP     | Nonconvex | Nearly unbiased low-rank estimation    |
| Nuclear | Convex    | Standard trace-norm regularization     |

---

## Implementation Details

* Optimization via **proximal gradient descent**
* Singular value thresholding with nonconvex penalties
* Optional truncated SVD for computational efficiency
* Core solver implemented in **C++ (Rcpp)**

---

## References

* Fan, J. and Li, R. (2001).
  *Variable selection via nonconcave penalized likelihood*.
  Journal of the American Statistical Association.

* Zhang, C.-H. (2010).
  *Nearly unbiased variable selection under minimax concave penalty*.
  Annals of Statistics.

* Bunea, F., She, Y., Wegkamp, M. (2011).
  *Optimal selection of reduced rank estimators*.
  Annals of Statistics.

---

## Status

* Actively developed
* Research-oriented
* Not yet submitted to CRAN

---

## License

GPL (>= 3)

---

## Contact

**Author:** The Tien Mai
For questions, issues, or contributions, please use GitHub Issues.





