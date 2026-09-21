# LFMM2 逐 bio 找与环境关联的位点
#
# 19 个 bio 各跑一次（d=1），每次只进一个变量，所以变量间共线性与 LFMM 无关；
# 需要按 VIF 筛变量的是 RDA 那边（04_rda.R）。
#
# 环境值先标准化再入模。lfmm2 内部只中心化不标准化（Xs <- scale(X, scale=FALSE)），
# 而 19 个 bio 量纲差得远（bio1 是 0.1℃ 的累积值，bio15 是 mm），不标准化的话
# 同一个 lambda=1e-5 对不同变量的惩罚力度不在一个量级上。
#
# K 主报 4——本项目自己的结构分析（01_raw_data/K4_structure_labels.csv）落在 K=4，
# 不照搬对标文献的 3；另跑 3 与 5，看显著位点集对 K 稳不稳，稳不住的话主报 K=4 也站不住。
#
# p 值走 lfmm2.test()（默认 genomic.control=TRUE，即按 GIF 做膨胀校正，返回的 gif
# 一并记下来）。lfmm2 是岭回归、无 MCMC 抽样，同样输入必得同样输出，所以不像对标文献
# 那样跑 5 次再平均。FDR 用 qvalue 按 bio 分别控；qvalue 估计 pi0 失败时退回 BH。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 03_lfmm2.R

suppressPackageStartupMessages({
  library(data.table)
  library(LEA)
  library(qvalue)
})

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
BASE <- dirname(HERE)
RES <- file.path(BASE, "04_results")

K_MAIN <- 4
KS <- c(4, 3, 5)
LAMBDA <- 1e-5
FDR <- 0.05

log <- function(...) cat(..., "\n", sep = "")

BIO <- sprintf("bio%d", 1:19)

## ---- 读基因型与环境 ------------------------------------------------------
G <- as.matrix(fread(file.path(RES, "geno_imputed.csv")))
storage.mode(G) <- "double"
loci <- colnames(G)

kept <- fread(file.path(RES, "loci_kept.txt"), header = FALSE)[[1]]
stopifnot(identical(as.character(kept), loci))
log(sprintf("基因型 %d 个体 × %d 位点（无缺失），列序与 loci_kept.txt 一致",
            nrow(G), ncol(G)))

env <- fread(file.path(RES, "ind_env.csv"))
stopifnot(nrow(env) == nrow(G))
X <- scale(as.matrix(env[, ..BIO]))
stopifnot(!anyNA(X))
log(sprintf("环境矩阵 %d × %d，已按列标准化（均值 %s，标准差 %s）",
            nrow(X), ncol(X),
            paste(range(colMeans(X)), collapse = "~"),
            paste(range(apply(X, 2, sd)), collapse = "~")))

## ---- 逐 K、逐 bio 拟合 ---------------------------------------------------
q_of <- function(p) {
  p[is.na(p)] <- 1
  tryCatch(qvalue::qvalue(p)$qvalues,
           error = function(e) {
             log("    [qvalue 估计 pi0 失败，该 bio 退回 BH] ", conditionMessage(e))
             p.adjust(p, "BH")
           })
}

res <- list()
for (K in KS) {
  log(sprintf("\n--- K = %d ---", K))
  P <- matrix(NA_real_, nrow = ncol(G), ncol = length(BIO),
              dimnames = list(loci, BIO))
  Z <- P
  dg <- data.table(bio = BIO, K = K, gif = NA_real_, n_na = 0L,
                   n_p05 = 0L)
  for (j in seq_along(BIO)) {
    b <- BIO[j]
    mod <- LEA::lfmm2(G, X[, j, drop = FALSE], K = K, lambda = LAMBDA)
    tst <- LEA::lfmm2.test(mod, G, X[, j, drop = FALSE], linear = TRUE)
    P[, b] <- as.vector(tst$pvalues)
    Z[, b] <- as.vector(tst$zscores)
    dg[j, `:=`(gif = tst$gif, n_na = sum(is.na(tst$pvalues)),
               n_p05 = sum(tst$pvalues < 0.05, na.rm = TRUE))]
    log(sprintf("  %-5s gif=%.3f  缺失 p %d 个  p<0.05 %d 个",
                b, tst$gif, dg$n_na[j], dg$n_p05[j]))
  }
  Q <- apply(P, 2, q_of)
  res[[paste0("K", K)]] <- list(p = P, z = Z, q = Q, diag = dg)
}

D <- rbindlist(lapply(KS, function(K) res[[paste0("K", K)]]$diag))
fwrite(D, file.path(RES, "lfmm_runs.csv"))

## ---- 主结果（K = K_MAIN）-------------------------------------------------
main <- res[[paste0("K", K_MAIN)]]
for (nm in c("pvalues", "qvalues", "zscores")) {
  M <- switch(nm, pvalues = main$p, qvalues = main$q, zscores = main$z)
  fwrite(data.table(locus = loci, as.data.table(M)),
         file.path(RES, paste0("lfmm_", nm, ".csv")))
  log(sprintf("K=%d %s → %s", K_MAIN, nm, file.path(RES, paste0("lfmm_", nm, ".csv"))))
}

## ---- 各 K 的命中数与 K 间稳定性 ------------------------------------------
sigset <- matrix(FALSE, nrow = length(loci), ncol = length(KS),
                 dimnames = list(loci, sprintf("K%d", KS)))
for (K in KS) {
  Q <- res[[paste0("K", K)]]$q
  s <- Q < FDR
  sigset[, paste0("K", K)] <- rowSums(s) > 0
  log(sprintf("\nK=%d  q < %.2f：%d 个 bio×位点命中，落在 %d 个位点上（%.2f%%）",
              K, FDR, sum(s), sum(rowSums(s) > 0),
              100 * sum(rowSums(s) > 0) / length(loci)))
  print(data.table(bio = BIO, 命中 = colSums(s))[order(-命中)][命中 > 0])
}

log("\nK 之间显著位点集的吻合度：")
for (i in seq_len(length(KS) - 1)) {
  for (j in (i + 1):length(KS)) {
    a <- sigset[, i]; b <- sigset[, j]
    log(sprintf("  K%d ∩ K%d：共有 %d 个  Jaccard %.3f", KS[i], KS[j],
                sum(a & b), sum(a & b) / sum(a | b)))
  }
}

nK <- rowSums(sigset)
stab <- data.table(locus = loci, as.data.table(sigset), n_K = nK)
fwrite(stab[order(-n_K)], file.path(RES, "lfmm_sig_by_K.csv"))
log(sprintf("\n三个 K 都显著：%d 个位点；只在 1 个 K 显著：%d 个",
            sum(nK == length(KS)), sum(nK == 1)))
fwrite(data.table(locus = loci[nK == length(KS)]),
       file.path(RES, "lfmm_core_loci.txt"), col.names = FALSE)
