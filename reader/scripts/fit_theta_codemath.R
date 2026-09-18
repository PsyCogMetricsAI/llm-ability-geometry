# fit_theta_codemath.R — Re-fit 2PL IRT theta for code and math domains
# from shipped response matrices.
#
# Usage:
#   R_LIBS=/path/to/rlib-with-mirt \
#     Rscript scripts/fit_theta_codemath.R inputs/ outputs/theta_codemath/
#
# [IN]  inputs/response_matrix_code.csv, response_matrix_math.csv
#       (rows = 51 models, first column = model id, cells 0/1)
# [OUT] outputs/theta_codemath/theta_2pl_{code,math}.csv (model, theta, se)
#       outputs/theta_codemath/rel_{code,math}.txt (convergence, rel, n_items)
#       outputs/theta_codemath/versions.txt
suppressMessages(library(mirt))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript fit_theta_codemath.R <input_dir> <output_dir>")

ind  <- args[1]
outd <- args[2]
dir.create(outd, showWarnings = FALSE, recursive = TRUE)

writeLines(c(paste("R", R.version.string),
             paste("mirt", as.character(packageVersion("mirt")))),
           file.path(outd, "versions.txt"))

fit_2pl <- function(csv_path, tag) {
  X <- read.csv(csv_path, row.names = 1, check.names = FALSE)
  # mirt needs item columns with >0 variance; drop degenerate (constant/all-NA) columns
  keep <- vapply(X, function(c) { v <- var(c, na.rm = TRUE); is.finite(v) && v > 0 }, TRUE)
  X <- X[, keep, drop = FALSE]
  m <- mirt(X, 1, itemtype = "2PL", verbose = FALSE,
            technical = list(NCYCLES = 2000))
  conv <- extract.mirt(m, "converged")
  th <- fscores(m, method = "EAP", full.scores.SE = TRUE)
  rel <- empirical_rxx(th)
  out <- data.frame(model = rownames(X), theta = th[, 1], se = th[, 2])
  write.csv(out, file.path(outd, paste0("theta_2pl_", tag, ".csv")), row.names = FALSE)
  writeLines(c(paste("converged", conv), paste("empirical_rxx", rel),
               paste("n_items_kept", ncol(X))),
             file.path(outd, paste0("rel_", tag, ".txt")))
  cat(sprintf("[2PL %s] converged=%s rel=%.6f items_kept=%d/%d\n",
              tag, conv, rel, ncol(X), ncol(read.csv(csv_path, row.names = 1, check.names = FALSE))))
}

for (dom in c("code", "math")) {
  f <- file.path(ind, sprintf("response_matrix_%s.csv", dom))
  if (!file.exists(f)) stop(paste("Missing:", f))
  fit_2pl(f, dom)
}

cat("THETA-CODEMATH-DONE\n")
