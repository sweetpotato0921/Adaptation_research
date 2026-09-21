# 冗余分析（RDA）识别与环境多元轴强关联的位点
#
# 预测变量筛选：以 07 章在 10,042 像元上定的 7 个变量为候选集，在 18 个种群
# 的环境矩阵上再迭代剔除 VIF > 10 者。之所以不从全部 19 个起筛——19 个变量
# 只有 18 行，相关矩阵秩亏（秩 17/19），VIF 求逆出来是噪声，算不出来。
# 以 7 个为候选集既避开了秩亏，又让本章与适生区章的变量来源可追溯。
#
# 个体在行、位点在列（180 × 17,670），与 lfmm2 吃同一份补缺矩阵。
# 载荷取 axes 1-6，以均值 ± 3 个标准差为界取尾部；候选位点再与各预测变量
# 求相关，归因到 |r| 最大的那个。对标文献 4RDA.R 里这段列号按 19 个预测变量写死
# （bar[4:22]、bar[i,23:24]），预测变量个数一变就错位，这里按实际列数取。
#
# 模型显著性用种群间置换，不用 vegan 的自由置换：环境值只有 18 个不同取值、
# 每种群 10 个个体共用，个体不是独立单位，自由置换等价于给每个个体重抽一个
# 环境值，零分布是错的。不报逐轴 p：anova.cca(by="axis") 在 17,670 个响应
# 变量上要 16 GB 内存会崩，而且对标文献的 RDA 都只报 3SD 载荷，不报逐轴检验。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 04_rda.R

suppressPackageStartupMessages({
  library(data.table)
  library(vegan)
})

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
BASE <- dirname(HERE)
ROOT <- dirname(BASE)
RES <- file.path(BASE, "04_results")

VIF_MAX <- 10
N_AXES <- 6
SD_CUT <- 3

log <- function(...) cat(..., "\n", sep = "")

## ---- 预测变量：种群层面迭代 VIF ------------------------------------------
pop_env <- fread(file.path(RES, "pop_env.csv"))
cand <- c("bio2", "bio3", "bio8", "bio9", "bio15", "bio18", "bio19")

vif_tab <- function(D, cols) {
  R <- cor(D[, ..cols], method = "spearman")
  list(vif = diag(solve(R)), cond = kappa(R, exact = TRUE))
}

log("候选集在 18 个种群上的 Spearman VIF：")
vt <- vif_tab(pop_env, cand)
for (b in names(sort(vt$vif, decreasing = TRUE))) {
  log(sprintf("  %-6s %7.2f", b, vt$vif[[b]]))
}
log(sprintf("  条件数 %.1f", vt$cond))

dropped <- data.table()
cols <- cand
while (length(cols) > 1) {
  v <- vif_tab(pop_env, cols)
  if (max(v$vif) <= VIF_MAX) break
  victim <- names(which.max(v$vif))
  dropped <- rbind(dropped, data.table(dropped = victim,
                                       vif_at_drop = max(v$vif),
                                       n_remaining = length(cols) - 1))
  cols <- setdiff(cols, victim)
}
pred <- cols
log(sprintf("\n迭代 VIF > %d 剔除 %d 个 → RDA 预测变量 %d 个: %s",
            VIF_MAX, nrow(dropped), length(pred), paste(pred, collapse = ", ")))
if (nrow(dropped)) {
  log("剔除顺序: ", paste(sprintf("%s(VIF %.1f)", dropped$dropped,
                                 dropped$vif_at_drop), collapse = " → "))
  fwrite(dropped, file.path(RES, "rda_vars_dropped.csv"))
}
vf <- vif_tab(pop_env, pred)
log(sprintf("最终集合最大 VIF %.2f  条件数 %.1f", max(vf$vif), vf$cond))
fwrite(data.table(variable = pred, vif_spearman = vf$vif),
       file.path(RES, "rda_vars.csv"))

## ---- 读基因型与环境 ------------------------------------------------------
G <- as.matrix(fread(file.path(RES, "geno_imputed.csv")))
storage.mode(G) <- "double"
ind_env <- fread(file.path(RES, "ind_env.csv"))
stopifnot(nrow(G) == nrow(ind_env))
stopifnot(!anyNA(G))
log(sprintf("\n基因型 %d 个体 × %d 位点（无缺失）", nrow(G), ncol(G)))

X <- as.data.frame(ind_env[, ..pred])

## ---- RDA -----------------------------------------------------------------
fit <- rda(G ~ ., data = X, scale = TRUE)
ev <- fit$CCA$eig
log(sprintf("\n约束轴特征值占约束部分: %s",
            paste(sprintf("RDA%d=%.1f%%", seq_along(ev),
                          100 * ev / sum(ev)), collapse = "  ")))
log(sprintf("总解释量（约束/总）: %.2f%%  校正 R2: %.4f",
            100 * sum(ev) / fit$tot.chi, RsquareAdj(fit)$adj.r.squared))
## ---- 模型显著性：按种群整块置换环境值 ------------------------------------
# 手算 F，因为 permute 的 how() 在本例构造不出「整块搬」的设计（plots/blocks
# 各种组合都保持不住种群成组）。F 取约束方差与残差方差之比，已核对与
# vegan anova.cca 的 F 逐位相同（15.7089）。置换只重排 18 个种群的环境值，
# 种群内 10 个体跟着整块走，这样零假设才对应「环境与基因型无关」。
Yc <- scale(G)
Fstat <- function(Xd) {
  Xm <- cbind(1, as.matrix(Xd))
  XtX <- crossprod(Xm)
  B <- solve(XtX, crossprod(Xm, Yc))
  con <- colSums(B * (XtX %*% B))
  (sum(con) / (ncol(Xm) - 1)) /
    (sum(colSums(Yc^2) - con) / (nrow(Yc) - ncol(Xm)))
}
Xp <- as.matrix(ind_env[, ..pred])
lvl <- unique(ind_env$pop)
idx <- match(ind_env$pop, lvl)
F_obs <- Fstat(Xp)
set.seed(0)
N_PERM <- 999
F_perm <- vapply(seq_len(N_PERM), function(i) {
  Fstat(Xp[sample(seq_along(lvl))[idx], , drop = FALSE])
}, numeric(1))
pv_model <- (sum(F_perm >= F_obs) + 1) / (N_PERM + 1)
log(sprintf("\n种群间置换检验（%d 次，环境值在 18 个种群间整块重排）：",
            N_PERM))
log(sprintf("  观测 F = %.3f   p = %.4f", F_obs, pv_model))
log(sprintf("  置换 F：中位 %.3f  范围 %.2f–%.2f", median(F_perm),
            min(F_perm), max(F_perm)))
fwrite(data.table(F_obs = F_obs, p = pv_model, n_perm = N_PERM,
                  F_perm_median = median(F_perm),
                  F_perm_min = min(F_perm), F_perm_max = max(F_perm)),
       file.path(RES, "rda_model_test.csv"))

loadings <- scores(fit, choices = seq_len(N_AXES), display = "species")
log(sprintf("\n载荷矩阵 %d 位点 × %d 轴", nrow(loadings), ncol(loadings)))
fwrite(data.table(locus = rownames(loadings), as.data.table(loadings)),
       file.path(RES, "rda_loadings.csv"))

## ---- 3 个标准差以外的位点 ------------------------------------------------
outliers <- function(x, z) {
  lim <- mean(x) + c(-1, 1) * z * sd(x)
  x[x < lim[1] | x > lim[2]]
}
cand_list <- lapply(seq_len(N_AXES), function(a) outliers(loadings[, a], SD_CUT))
names(cand_list) <- sprintf("RDA%d", seq_len(N_AXES))
for (a in names(cand_list)) {
  log(sprintf("  %s 尾部位点 %d 个", a, length(cand_list[[a]])))
}

hit <- data.table(axis = rep(names(cand_list), lengths(cand_list)),
                  locus = unlist(lapply(cand_list, names)),
                  loading = unlist(lapply(cand_list, unname)))
hit <- unique(hit, by = "locus")
log(sprintf("  去重后 RDA 候选位点 %d 个（%.1f%%）",
            nrow(hit), 100 * nrow(hit) / ncol(G)))

## ---- 候选位点归因到气候变量 ----------------------------------------------
loci_all <- colnames(G)
idx <- match(hit$locus, loci_all)
stopifnot(!anyNA(idx))
cors <- cor(G[, idx, drop = FALSE], X)
hit[, predictor := pred[max.col(abs(cors), ties.method = "first")]]
hit[, cor_env := cors[cbind(seq_len(nrow(cors)), max.col(abs(cors),
                                                         ties.method = "first"))]]
fwrite(hit, file.path(RES, "rda_candidates.csv"))
log("\n候选位点归因到的气候变量：")
print(hit[, .N, by = predictor][order(-N)])

pdf(file.path(RES, "rda_loadings_hist.pdf"), width = 9, height = 6)
par(mfrow = c(3, 2))
for (a in seq_len(N_AXES)) {
  hist(loadings[, a], breaks = 60, main = sprintf("Loadings on RDA%d", a),
       xlab = "", col = "grey80", border = "grey40")
  abline(v = mean(loadings[, a]) + c(-1, 1) * SD_CUT * sd(loadings[, a]),
         col = "red", lty = 2)
}
dev.off()
log("\n载荷直方图 → ", file.path(RES, "rda_loadings_hist.pdf"))
