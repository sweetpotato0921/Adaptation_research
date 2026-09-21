# 建 GEA 的基因型输入：MAF 过滤 + 写成 LEA 的 lfmm 格式
#
# 输入 02_intermediate/dose_unfilled.csv（180 个体 × 17,728 位点，取值 0/1/2，缺失 NA）
# 输出 04_results/geno.lfmm          LEA 格式，缺失记 9
#      04_results/loci_kept.txt     保留的位点名，顺序与矩阵列一致
#      04_results/geno_qc.csv       位点级缺失率与 MAF
#      04_results/geno_imputed.csv  补缺后的 0/1/2 矩阵，无缺失，lfmm2 与 RDA 共用
#
# 缺失处理：只有 9 个位点完全无缺失，完整位点法不可行。LEA 的 impute() 只吃
# snmfProject，所以先 snmf 拿群体结构，再把每个缺失基因型按该个体所属 cluster
# 内对应位点的众数填上；K 取 4 与本项目自己的结构分析一致。补全结果 lfmm2 与
# RDA 共用，保证两法吃的是逐位点同一份数据。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 02_prep_geno.R

suppressPackageStartupMessages({
  library(data.table)
  library(LEA)
})

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
BASE <- dirname(HERE)
ROOT <- dirname(BASE)
RES <- file.path(BASE, "04_results")
dir.create(RES, showWarnings = FALSE)

MAF_MIN <- 0.05   # 对标文献用 0.1；本物种位点总数偏少，放宽到 0.05
MISS_MAX <- 1.0   # 位点缺失率上限，1.0 表示不额外过滤，见下方打印的分档计数

log <- function(...) cat(..., "\n", sep = "")

## ---- 读入 ----------------------------------------------------------------
dose <- fread(file.path(ROOT, "02_intermediate", "dose_unfilled.csv"))
samples <- dose[[1]]
loci <- names(dose)[-1]
M <- as.matrix(dose[, -1])
storage.mode(M) <- "double"
log("基因型矩阵 ", nrow(M), " 个体 × ", ncol(M), " 位点")

lab <- fread(file.path(ROOT, "02_intermediate", "labels.csv"))
stopifnot(identical(as.character(lab$sample), as.character(samples)))
log("个体顺序与 labels.csv 一致")

## ---- 位点级 QC -----------------------------------------------------------
miss <- colMeans(is.na(M))
p <- colMeans(M, na.rm = TRUE) / 2
maf <- pmin(p, 1 - p)
qc <- data.table(locus = loci, miss_rate = miss, maf = maf)

log("\n位点缺失率分位: ",
    paste(sprintf("%s=%.3f", c("中位", "90%", "99%", "最大"),
                  quantile(miss, c(.5, .9, .99, 1))), collapse = "  "))
for (t in c(0.1, 0.2, 0.3)) {
  log(sprintf("  缺失率 > %.0f%% 的位点: %d", t * 100, sum(miss > t)))
}

keep <- maf > MAF_MIN & miss <= MISS_MAX
log(sprintf("\nMAF > %.2f 且缺失率 <= %.2f → 保留 %d / %d 位点（%.1f%%）",
            MAF_MIN, MISS_MAX, sum(keep), length(keep), 100 * mean(keep)))
log(sprintf("  仅按 MAF>%.2f：%d 位点；再叠缺失率 <= %.2f：%d 位点",
            MAF_MIN, sum(maf > MAF_MIN), MISS_MAX, sum(keep)))
qc[, kept := keep]
fwrite(qc, file.path(RES, "geno_qc.csv"))
fwrite(data.table(locus = loci[keep]), file.path(RES, "loci_kept.txt"),
       col.names = FALSE)

## ---- 写 LEA 的 lfmm 文件 -------------------------------------------------
# LEA 的 lfmm/geno 格式：个体在行、位点在列，取值 0-9，9 表示缺失。
# 写完后用 read.lfmm() 读回来核对维度，避免行/列写反而不自知。
G <- M[, keep]
G[is.na(G)] <- 9L
lfmm_file <- file.path(RES, "geno.lfmm")
fwrite(as.data.table(G), lfmm_file, col.names = FALSE, sep = " ")
log("\n写出 ", lfmm_file, "（缺失记 9）")

chk <- LEA::read.lfmm(lfmm_file)
log("read.lfmm 读回维度: ", nrow(chk), " × ", ncol(chk),
    "  与预期一致: ", nrow(chk) == nrow(M) && ncol(chk) == ncol(G))

## ---- snmf 结构（impute 的入口）-------------------------------------------
# impute() 的 object 必须是 snmfProject，签名是
#   impute(object, input.file, method, K, run)，method 取 "mode" 或 "random"。
# 故先跑一次 snmf：K=4 与项目自己的结构分析（01_raw_data/K4_structure_labels.csv）
# 对齐，跑 10 次取交叉熵最小的那次用来补缺。
K_IMP <- 4
N_REP <- 10
# snmf 会在 lfmm 旁边派生 .geno、<stem>.snmfProject、<stem>.snmf/ 三样中转物。
# project="new" 遇到同名工程会报错，所以进 snmf 前先清一次。
stem <- sub("\\.lfmm$", "", lfmm_file)
inter <- paste0(stem, c(".geno", ".snmfProject", ".snmf"))
unlink(inter, recursive = TRUE)
proj <- LEA::snmf(lfmm_file, K = K_IMP, project = "new", repetitions = N_REP,
                  entropy = TRUE, CPU = 4, seed = 1)
ce <- LEA::cross.entropy(proj, K = K_IMP)
best <- which.min(ce)
log(sprintf("\nsnmf K=%d 跑 %d 次，交叉熵 %.3f–%.3f，取 run %d",
            K_IMP, N_REP, min(ce), max(ce), best))

## ---- 补缺（按 cluster 众数）----------------------------------------------
# impute() 只写 <input>_imputed.lfmm、不返回矩阵，所以从文件读回来。
# 众数填充只在已观测到的基因型里挑，不会造出新等位型，下面按位点验证。
Gimp_file <- paste0(lfmm_file, "_imputed.lfmm")
LEA::impute(proj, lfmm_file, method = "mode", K = K_IMP, run = best)
Gi <- LEA::read.lfmm(Gimp_file)
stopifnot(identical(dim(Gi), dim(G)), !anyNA(Gi), all(Gi %in% 0:2))

Mk <- M[, keep]
log(sprintf("\n补缺 %d 个缺失基因型（占矩阵 %.1f%%）",
            sum(is.na(Mk)), 100 * mean(is.na(Mk))))

miss_col <- which(colSums(is.na(Mk)) > 0)
novel <- vapply(miss_col, function(j) {
  any(!(Gi[, j] %in% unique(Mk[!is.na(Mk[, j]), j])))
}, logical(1))
log(sprintf("有缺失的位点 %d 个；补出该位点从未观测过的基因型的位点 %d 个",
            length(miss_col), sum(novel)))
stopifnot(!any(novel))

p_imp <- colMeans(Gi) / 2
d_maf <- abs(pmin(p_imp, 1 - p_imp) - maf[keep])
log(sprintf("补缺前后 MAF 变化：中位 %.4f  99%% %.4f  最大 %.4f",
            median(d_maf), quantile(d_maf, .99), max(d_maf)))

colnames(Gi) <- loci[keep]
fwrite(as.data.table(Gi), file.path(RES, "geno_imputed.csv"))
log(sprintf("补缺矩阵 %d × %d → %s", nrow(Gi), ncol(Gi),
            file.path(RES, "geno_imputed.csv")))
# _imputed.lfmm 与 geno_imputed.csv 内容等价，csv 带位点名，只留 csv；
# snmf 的三样中转物也一并清掉，04_results 里只留矩阵与清单。
unlink(c(inter, Gimp_file), recursive = TRUE)
log("snmf 中转物与 _imputed.lfmm 已删除，重跑本脚本会再生成")
