# [IN]  theta_mirt_in/resp_{code,math}_{zero,na}.csv  (rows=32 core models, cols=items;
#        produced by prep_theta_csv.py from response_matrix_geom.json; 'zero'=empty-text->0
#        main coding, 'na'=empty-text->NA dual-coding sensitivity, prereg failure-coding row)
#        + theta_mirt_in/domains_pooled.csv (item -> domain map for the bifactor run)
# [OUT] theta_mirt_out/: theta_2pl_<dom>_<coding>.csv (EAP + SE), rel_<dom>_<coding>.txt
#        (empirical reliability), bifactor_F1_<coding>.csv (general-factor EAP profile),
#        bifactor_meta_<coding>.txt (convergence flag + item-fit sample), versions.txt.
# [POS] Imports/geometry/theta_mirt.R — A3 canonical theta-hat (user ruling D-4: install R,
#        write R analysis; R mirt = the only true bifactor engine, survey §1.4). Run via:
#        Rscript theta_mirt.R   (zero-GPU, CPU batch)
suppressMessages(library(mirt))

ind  <- "theta_mirt_in_panel51"
outd <- "theta_mirt_out_panel51"
dir.create(outd, showWarnings = FALSE)

writeLines(c(paste("R", R.version.string), paste("mirt", as.character(packageVersion("mirt")))),
           file.path(outd, "versions.txt"))

fit_2pl <- function(csv, tag) {
  X <- read.csv(csv, row.names = 1, check.names = FALSE)
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
  cat(sprintf("[2PL %s] converged=%s rel=%.4f items=%d\n", tag, conv, rel, ncol(X)))
}

for (dom in c("code", "math")) for (cod in c("zero", "na")) {
  f <- file.path(ind, sprintf("resp_%s_%s.csv", dom, cod))
  if (file.exists(f)) fit_2pl(f, sprintf("%s_%s", dom, cod))
}

# ---- bifactor F1 (general + code/math specifics) on the pooled matrix --------------
for (cod in c("zero", "na")) {
  f <- file.path(ind, sprintf("resp_pooled_%s.csv", cod))
  dmap <- read.csv(file.path(ind, "domains_pooled.csv"))
  if (!file.exists(f)) next
  X <- read.csv(f, row.names = 1, check.names = FALSE)
  keep <- vapply(X, function(c) { v <- var(c, na.rm = TRUE); is.finite(v) && v > 0 }, TRUE)
  X <- X[, keep, drop = FALSE]
  spec <- as.integer(factor(dmap$domain[match(colnames(X), dmap$item)]))
  m <- tryCatch(bfactor(X, spec, verbose = FALSE, technical = list(NCYCLES = 2000)),
                error = function(e) e)
  if (inherits(m, "error")) {
    writeLines(paste("bifactor ERROR:", conditionMessage(m)),
               file.path(outd, paste0("bifactor_meta_", cod, ".txt")))
    cat(sprintf("[bifactor %s] ERROR: %s\n", cod, conditionMessage(m)))
    next
  }
  conv <- extract.mirt(m, "converged")
  th <- fscores(m, method = "EAP", full.scores.SE = TRUE)   # col 1 = general factor F1
  rel_g <- empirical_rxx(th[, c(1, ncol(th) / 2 + 1), drop = FALSE])
  out <- data.frame(model = rownames(X), F1 = th[, 1])
  write.csv(out, file.path(outd, paste0("bifactor_F1_", cod, ".csv")), row.names = FALSE)
  writeLines(c(paste("converged", conv), paste("empirical_rxx_F1", rel_g),
               paste("n_items_kept", ncol(X))),
             file.path(outd, paste0("bifactor_meta_", cod, ".txt")))
  cat(sprintf("[bifactor %s] converged=%s\n", cod, conv))
}
cat("THETA-MIRT-DONE\n")
