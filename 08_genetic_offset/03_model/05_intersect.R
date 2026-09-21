# 取 LFMM 与 RDA 的位点交集，作为核心受选择位点（GEA-SNPs）
#
# LFMM 判据：任一 bio 上 q < FDR。主报 5%，另跑 1%——一项已发表的同物种研究用的就是 FDR<0.01，
# 两个阈值都算出来，便于与其 3117 / 987 / 交集 410 对照。
# RDA  判据：在 axes 1-6 中至少一个轴上的载荷落在均值 ± 3 个标准差以外
# 交集即对标文献正文里说的 core adaptive variants。
#
# 同时报出重叠的富集情况：两法各自命中数固定时，随机重叠的期望是多少，
# 实际重叠是它的几倍。没有这一步，光报交集个数看不出两法是否相互印证。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 05_intersect.R

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
BASE <- dirname(HERE)
RES <- file.path(BASE, "04_results")

FDR_MAIN <- 0.05
FDRS <- c(FDR_MAIN, 0.01)
OUT_NAME <- c("gea_snps.csv", "gea_snps_q01.csv")   # 与 FDRS 一一对应

log <- function(...) cat(..., "\n", sep = "")

q <- fread(file.path(RES, "lfmm_qvalues.csv"))
p <- fread(file.path(RES, "lfmm_pvalues.csv"))
loci_all <- q$locus
N <- length(loci_all)
BIO <- setdiff(names(q), "locus")

rda_hit <- fread(file.path(RES, "rda_candidates.csv"))
log(sprintf("位点总数 %d   LFMM 变量 %d 个   RDA 候选 %d 个",
            N, length(BIO), nrow(rda_hit)))

## ---- LFMM 显著位点 -------------------------------------------------------
long <- rbindlist(lapply(BIO, function(b) data.table(
  locus = q$locus, bio = b, p = p[[b]], q = q[[b]])))
fwrite(long, file.path(RES, "lfmm_long.csv"))

rda_loci <- unique(rda_hit$locus)
summary_rows <- list()

for (k in seq_along(FDRS)) {
  FDR <- FDRS[k]
  sig <- long[q < FDR]
  log(sprintf("\nLFMM q < %.2f：%d 个命中（%.2f%%），落在 %d 个不同位点上",
              FDR, nrow(sig), 100 * nrow(sig) / (N * length(BIO)),
              uniqueN(sig$locus)))
  log("  各变量命中数：")
  print(sig[, .N, by = bio][order(-N)])

  lfmm_loci <- unique(sig$locus)
  inter <- intersect(lfmm_loci, rda_loci)

  ## ---- 交集与富集 --------------------------------------------------------
  exp_n <- length(lfmm_loci) * length(rda_loci) / N
  log(sprintf("\nLFMM 命中位点 %d   RDA 候选位点 %d", length(lfmm_loci),
              length(rda_loci)))
  log(sprintf("交集 %d 个；随机期望 %.1f 个，富集 %.2f 倍",
              length(inter), exp_n, length(inter) / exp_n))
  # 超几何检验：从 N 个位点里抽到 >= length(inter) 个重叠的概率
  pv <- phyper(length(inter) - 1, length(rda_loci), N - length(rda_loci),
               length(lfmm_loci), lower.tail = FALSE)
  log(sprintf("超几何检验 p = %.3g", pv))

  gea <- rda_hit[locus %in% inter]
  gea <- gea[order(-abs(loading))]

  # 交集位点里，RDA 归因的气候变量与把它检出的 LFMM 变量是否一致
  by_bio <- sig[locus %in% inter, .(lfmm_bio = paste(sort(unique(bio)),
                                                     collapse = ",")),
                by = locus]
  gea <- merge(gea, by_bio, by.x = "locus", by.y = "locus", all.x = TRUE)
  out <- file.path(RES, OUT_NAME[k])
  fwrite(gea, out)
  log(sprintf("\n核心 GEA 位点 %d 个 → %s", nrow(gea), out))
  log("  按 RDA 归因变量：")
  print(gea[, .N, by = predictor][order(-N)])
  agree <- sum(mapply(function(a, b) grepl(a, b, fixed = TRUE),
                      gea$predictor, gea$lfmm_bio))
  log(sprintf("  两法归因完全对上（LFMM 命中的变量里含 RDA 归因的那个）: %d / %d",
              agree, nrow(gea)))

  summary_rows[[k]] <- data.table(
    fdr = FDR, lfmm_loci = length(lfmm_loci), rda_loci = length(rda_loci),
    intersect_n = length(inter), expected = exp_n,
    enrichment = length(inter) / exp_n, hypergeom_p = pv,
    attribution_agree = agree, out_file = OUT_NAME[k])
}

summ <- rbindlist(summary_rows)
fwrite(summ, file.path(RES, "gea_intersect_summary.csv"))
log("\n两阈值汇总 → ", file.path(RES, "gea_intersect_summary.csv"))
print(summ)
