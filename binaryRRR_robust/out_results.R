
out_result <- function(datain){out <- c()  # store results
for (j in c(1:2)) {
  m <- mean(sapply(datain, function(x) x[j]))
  s <- sd(sapply(datain, function(x) x[j]))
  out <- c(out, sprintf("%.2f (%.2f)", m, s)) }
out}

cat('AUC =',paste(out_result(auc_out), collapse = " & "), "\n")
cat('est. =',paste(out_result(est_out), collapse = " & "), "\n")
cat('prediction =',paste(out_result(accuracy_out), collapse = " & "), "\n")



