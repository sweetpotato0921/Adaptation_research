# 从梯度森林导出复现图所需的三个量：累积重要性曲线、训练点的变换环境、
# 以及适生区像元的变换环境。
#
# 07_gf_offset.R 只算偏移，不存 gf 对象也不存曲线。复现对标论文图3-13（重要性横条 +
# 6 条累积重要性曲线）与图3-14（变换空间的 PCA 与地理映射）需要：
#   cumimp(gf, b)$x / $y    —— 逐变量的累积重要性曲线
#   predict(gf, X)          —— 任意环境在该变换空间中的坐标
# 后者既用于训练点（gf$X），也用于适生区 42,063 个像元（图上要铺满整片区域）。
#
# 因此这里按与 07_gf_offset.R 完全相同的输入、种子与参数重拟合一次。
# 重拟合必须复现原模型，所以拟合后立刻与已落盘的 gf_importance_*.csv 逐格核对，
# 对不上就退出——不然导出的曲线与已发表的偏移量不是同一个模型。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 09_gf_curves.R

suppressPackageStartupMessages({
  library(data.table)
  library(gradientForest)
})

args <- commandArgs(trailingOnly = FALSE)
HERE <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
BASE <- dirname(HERE)
RES <- file.path(BASE, "04_results")
ENV <- file.path(BASE, "02_env_vars")

NTREE <- 500
SEED <- 1
log <- function(...) cat(..., "\n", sep = "")

## ---- 与 07_gf_offset.R 完全相同的输入 ------------------------------------
G <- as.matrix(fread(file.path(RES, "geno_imputed.csv")))
storage.mode(G) <- "double"
gea <- fread(file.path(RES, "gea_snps.csv"))
VARS <- fread(file.path(RES, "rda_vars.csv"))$variable
pop_env <- fread(file.path(RES, "pop_env.csv"))
ind_env <- fread(file.path(RES, "ind_env.csv"))
grid <- fread(file.path(ENV, "gf_grid_env.csv"))
sites <- fread(file.path(RES, "pop_env_sites.csv"))

loci <- gea$locus
Gg <- G[, loci, drop = FALSE]

pops <- as.character(pop_env$pop)
P <- matrix(NA_real_, length(pops), length(loci), dimnames = list(pops, loci))
for (i in seq_along(pops)) {
  P[i, ] <- colMeans(Gg[ind_env$pop == pops[i], , drop = FALSE]) / 2
}

Xpop <- as.data.frame(pop_env[, ..VARS])
Xind <- as.data.frame(ind_env[, ..VARS])
Ypop <- as.data.frame(P, check.names = FALSE)
Yind <- as.data.frame(Gg, check.names = FALSE)

## ---- 重拟合，并核对与已落盘模型一致 --------------------------------------
fit_gf <- function(X, Y, tag) {
  set.seed(SEED)
  gf <- gradientForest(data = cbind(X, Y), predictor.vars = colnames(X),
                       response.vars = colnames(Y), ntree = NTREE,
                       check.names = FALSE, trace = FALSE)
  w <- importance(gf, "Weighted", sort = FALSE)
  ref <- fread(file.path(RES, sprintf("gf_importance_%s.csv", tag)))
  # importance() 按 rownames(imp.rsq) 排序，ref 按名字取，一一对齐后核对
  d <- max(abs(as.vector(w[ref$variable]) - ref$weighted_R2))
  log(sprintf("[%s] 重拟合完成；与 gf_importance_%s.csv 的最大差 %.3g", tag, tag, d))
  if (d > 1e-9)
    stop(sprintf("[%s] 重拟合结果与已落盘模型不一致，导出的曲线不能用", tag))
  gf
}

log("--- 重拟合种群级模型 ---")
gf_pop <- fit_gf(Xpop, Ypop, "pop")
log("--- 重拟合个体级模型 ---")
gf_ind <- fit_gf(Xind, Yind, "ind")

env_of <- function(D, sc) {
  E <- as.data.frame(D[, paste0(VARS, "__", sc), with = FALSE])
  setnames(E, VARS)
  E
}

## ---- 1. 累积重要性曲线 ---------------------------------------------------
# 曲线的 y 上限即该变量的 weighted R²（getCU 里 height/sum(height)*Rsq），
# 导出的同时核一遍，确认画的曲线与重要性排序是同一个量。
curves <- rbindlist(lapply(c("pop", "ind"), function(nm) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  ref <- fread(file.path(RES, sprintf("gf_importance_%s.csv", nm)))
  rbindlist(lapply(VARS, function(b) {
    ci <- cumimp(gf, b)
    top <- max(ci$y)
    w <- ref[variable == b, weighted_R2]
    if (abs(top - w) > 1e-9)
      stop(sprintf("%s %s 曲线振幅 %.4f 与 weighted_R2 %.4f 不等", nm, b, top, w))
    data.table(model = nm, variable = b, x = ci$x, y = ci$y)
  }))
}))
fwrite(curves, file.path(RES, "gf_cumimp.csv"))
log(sprintf("\n累积重要性曲线 %d 行 → gf_cumimp.csv（%d 变量 × 2 模型）",
            nrow(curves), length(VARS)))
for (nm in c("pop", "ind")) {
  d <- curves[model == nm, .(amp = max(y), xmin = min(x), xmax = max(x)),
              by = variable][order(-amp)]
  log(sprintf("  [%s] %s", nm,
              paste(sprintf("%s=%.4f", d$variable, d$amp), collapse = "  ")))
}

## ---- 2. 训练点的变换坐标（gf$X 同口径）----------------------------------
# predict 给出的是各变量的 f_j(env)，与 gf$X 的列一致；这里用 predict 统一算，
# 保证训练点与像元两套坐标出自同一条路径，不会出现两处口径差。
for (nm in c("pop", "ind")) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  tr <- as.data.table(predict(gf, Xpop)); tr[, pop := pop_env$pop]
  setcolorder(tr, c("pop", VARS))
  fwrite(tr, file.path(RES, sprintf("gf_transformed_sites_%s.csv", nm)))
}
log("\n训练点变换坐标 → gf_transformed_sites_pop.csv / _ind.csv（18 行 × 6 列）")

## ---- 3. 适生区像元的变换坐标 --------------------------------------------
# 未来情景也要铺到整片适生区，图上 4 个情景才是同一个空间里的移动。
for (nm in c("pop", "ind")) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  cur <- as.data.table(predict(gf, env_of(grid, "current")))
  setnames(cur, VARS, paste0(VARS, "__current"))
  out <- cbind(grid[, .(lon, lat, row, col)], cur)
  for (sc in c("ssp245_2061-2080", "ssp245_2081-2100",
               "ssp585_2061-2080", "ssp585_2081-2100")) {
    f <- as.data.table(predict(gf, env_of(grid, sc)))
    setnames(f, VARS, paste0(VARS, "__", sc))
    out <- cbind(out, f)
  }
  fwrite(out, file.path(RES, sprintf("gf_transformed_grid_%s.csv", nm)))
  log(sprintf("像元变换坐标 [%s] %d 行 × %d 列 → gf_transformed_grid_%s.csv",
              nm, nrow(out), ncol(out), nm))
}

log("\n完成。绘图见 10_offset_figs.py")
