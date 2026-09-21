# 梯度森林（GF）拟合 + 遗传偏移
#
# 响应变量是 424 个核心 GEA 位点（05_intersect.R 的 gea_snps.csv）在 18 个种群上的
# 等位基因频率，预测变量是 VIF 筛过的那 6 个 bio（04_rda.R 的 rda_vars.csv）。
# 遗传偏移取「变换后环境空间」上当前与未来的欧氏距离，即
#   offset = sqrt( Σ_j ( f_j(未来环境) − f_j(当前环境) )^2 )
# f_j 是第 j 个 bio 的累积重要性曲线（cumimp）。每条曲线在总量上被缩放到该变量的
# Weighted 重要性（getCU 里 height/sum(height)*Rsq），所以平方和本身就按重要性加权，
# 不必再乘权重——对标文献 1GF.R 也是直接取欧氏距离。
#
# 训练层级：论文 160 段用 18 个种群的等位基因频率作响应、500 棵树、其余默认，这里
# 主报同一口径。但 18 行配 6 个预测变量、randomForest 默认 nodesize=5，每棵树只能
# 切一两刀，曲线很粗；个体级（180 行 × 基因型剂量）能有更多刀。两套都跑，看偏移的
# 地理排序是否一致，免得主报哪一套全凭口径。
#
# 一条硬约定：位点名形如 Cluster-219541.101134:2904，冒号会被 make.names() 改写，
# gradientForest 的 check.names=TRUE 直接报错，必须关掉。关掉只跳过列名校验，
# 响应变量是按字符串精确取的，不受影响。
#
# 外推：18 个种群的气候区间比适生区像元窄，逐变量有 3.4%–15.0% 的像元落在训练区间
# 外，predict 默认线性外推。另标出「当前 6 个变量全部落在训练区间内」的像元子集，
# 附一份只在该子集上的偏移，作为外推是否左右结论的对照。
#
# 种群偏移走 04_results/pop_env_sites.csv（06_grid_env.py 单独取的 18 个驻点），
# 不从适生区像元表里按 row/col 捞——掩膜是 MaxEnt 10% 阈值以上，采样点未必过阈值，
# FX 就落在掩膜外，内连接会静默少一个种群。
#
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript 07_gf_offset.R

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
SCEN_FUT <- c("ssp245_2061-2080", "ssp245_2081-2100",
              "ssp585_2061-2080", "ssp585_2081-2100")

log <- function(...) cat(..., "\n", sep = "")

## ---- 输入 ----------------------------------------------------------------
G <- as.matrix(fread(file.path(RES, "geno_imputed.csv")))
storage.mode(G) <- "double"
gea <- fread(file.path(RES, "gea_snps.csv"))
VARS <- fread(file.path(RES, "rda_vars.csv"))$variable
pop_env <- fread(file.path(RES, "pop_env.csv"))
ind_env <- fread(file.path(RES, "ind_env.csv"))
rng <- fread(file.path(RES, "gf_grid_range.csv"))
grid <- fread(file.path(ENV, "gf_grid_env.csv"))
sites <- fread(file.path(RES, "pop_env_sites.csv"))

loci <- gea$locus
stopifnot(all(loci %in% colnames(G)), nrow(ind_env) == nrow(G))
Gg <- G[, loci, drop = FALSE]

pops <- as.character(pop_env$pop)
P <- matrix(NA_real_, length(pops), length(loci), dimnames = list(pops, loci))
for (i in seq_along(pops)) {
  P[i, ] <- colMeans(Gg[ind_env$pop == pops[i], , drop = FALSE]) / 2
}
log(sprintf("GEA 位点 %d 个  种群 %d 个  个体 %d 个  预测变量 %d 个: %s",
            length(loci), length(pops), nrow(G), length(VARS),
            paste(VARS, collapse = ", ")))
log(sprintf("种群等位基因频率 %.3f–%.3f（中位 %.3f）",
            min(P), max(P), median(P)))
# 每种群只有 10 个个体，MAF 0.05 以上的位点在多数种群上仍是 0；randomForest 回归
# 在响应只有少数几个不同取值时会警告。这不是 bug，但它决定 GF 曲线能画多细，先报出来。
nu <- apply(P, 2, function(x) length(unique(x)))
maf <- pmin(colMeans(Gg) / 2, 1 - colMeans(Gg) / 2)
log(sprintf("响应矩阵 %d 个格子为 0（%.1f%%）；逐位点在 18 个种群上的不同取值数：中位 %d（%d–%d）",
            sum(P == 0), 100 * mean(P == 0), median(nu), min(nu), max(nu)))
log(sprintf("424 个位点在个体层面的 MAF：中位 %.3f  最小 %.3f  10%% 分位 %.3f",
            median(maf), min(maf), quantile(maf, .1)))

## ---- 两套训练矩阵 --------------------------------------------------------
Xpop <- as.data.frame(pop_env[, ..VARS])
Xind <- as.data.frame(ind_env[, ..VARS])
Ypop <- as.data.frame(P, check.names = FALSE)
Yind <- as.data.frame(Gg, check.names = FALSE)
stopifnot(identical(colnames(Ypop), loci), identical(colnames(Yind), loci))

fit_gf <- function(X, Y, tag) {
  set.seed(SEED)
  gf <- gradientForest(data = cbind(X, Y), predictor.vars = colnames(X),
                       response.vars = colnames(Y), ntree = NTREE,
                       check.names = FALSE, trace = TRUE)
  log("")
  nkept <- length(gf$result)
  log(sprintf("[%s] 训练 %d 行；响应 %d 个位点，R² > 0 被保留 %d 个（%.1f%%）",
              tag, nrow(X), ncol(Y), nkept, 100 * nkept / ncol(Y)))
  log(sprintf("[%s] 保留位点 R²：中位 %.4f  最大 %.4f", tag,
              median(gf$result), max(gf$result)))
  # 三个量的原始顺序不同：overall.imp / overall.imp2 按 names(gf$overall.imp)，
  # 而 importance(..., "Weighted") 返回的是按 rownames(gf$imp.rsq) 排的 weighted 向量。
  # 直接 as.vector() 并排会把标签错配（bio19 的 0.131 会被安到 bio9 头上），
  # 一律按名字取。原顺序对不上就报出来。
  stopifnot(setequal(names(gf$overall.imp), rownames(gf$imp.rsq)))
  if (!identical(names(gf$overall.imp), rownames(gf$imp.rsq)))
    log(sprintf("[%s] 注意：overall.imp 与 imp.rsq 的行序不同（%s vs %s），已按名称对齐",
                tag, paste(names(gf$overall.imp), collapse = ","),
                paste(rownames(gf$imp.rsq), collapse = ",")))
  vars <- names(gf$overall.imp)
  w <- importance(gf, "Weighted", sort = FALSE)
  imp <- data.table(variable = vars,
                    IncMSE = as.vector(gf$overall.imp[vars]),
                    IncNodePurity = as.vector(gf$overall.imp2[vars]),
                    weighted_R2 = as.vector(w[vars]))
  setorder(imp, -weighted_R2)
  log(sprintf("[%s] 变量重要性（按 weighted_R2 排序）：", tag))
  log("  weighted_R2(p) = (1/J) Σ_j ( w(p,j) × R²_j )，w(p,j) 为变量 p 在第 j 个位点随机森林内的归一化重要性，J 为参与拟合的位点数；6 个变量之和即保留位点 R² 的均值。")
  print(imp)
  # weighted_R2 同时是 cumimp 的振幅上限，predict 出来的 |Δf| 不可能超过它，
  # 拿这条自查一次，防止再一次把变量对错。
  for (b in vars) stopifnot(max(cumimp(gf, b)$y) <= imp[variable == b, weighted_R2] + 1e-9)
  tag_f <- if (tag == "种群级") "pop" else "ind"
  fwrite(imp, file.path(RES, sprintf("gf_importance_%s.csv", tag_f)))
  fwrite(data.table(locus = names(gf$result), rsq = as.vector(gf$result)),
         file.path(RES, sprintf("gf_snp_rsq_%s.csv", tag_f)))
  gf
}

log("\n--- 主模型：18 个种群 × 等位基因频率 ---")
gf_pop <- fit_gf(Xpop, Ypop, "种群级")
log("\n--- 对照模型：180 个个体 × 基因型剂量 ---")
gf_ind <- fit_gf(Xind, Yind, "个体级")

## ---- 像元环境 ------------------------------------------------------------
log(sprintf("\n适生区像元 %d 个", nrow(grid)))
inside <- rep(TRUE, nrow(grid))
for (b in VARS) {
  v <- grid[[paste0(b, "__current")]]
  inside <- inside & v >= rng[bio == b, train_min] & v <= rng[bio == b, train_max]
}
log(sprintf("当前 6 个变量全部落在训练区间内的像元 %d 个（%.1f%%）",
            sum(inside), 100 * mean(inside)))

env_of <- function(D, sc) {
  E <- as.data.frame(D[, paste0(VARS, "__", sc), with = FALSE])
  setnames(E, VARS)
  E
}

## ---- 偏移 ----------------------------------------------------------------
offset <- function(gf, D, sc) {
  p1 <- as.matrix(predict(gf, env_of(D, "current")))
  p2 <- as.matrix(predict(gf, env_of(D, sc)))
  sqrt(rowSums((p2 - p1)^2))
}

out <- data.table(lon = grid$lon, lat = grid$lat, row = grid$row,
                  col = grid$col, inside = inside)
summ <- list()
for (nm in c("pop", "ind")) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  for (sc in SCEN_FUT) {
    v <- offset(gf, grid, sc)
    set(out, j = paste0(nm, "__", sc), value = v)
    summ[[paste(nm, sc)]] <- data.table(
      model = nm, scenario = sc, n_cells = length(v),
      mean = mean(v), median = median(v), p90 = quantile(v, .9), max = max(v),
      median_inside = median(v[inside]),
      mean_inside = mean(v[inside]))
  }
}
summ <- rbindlist(summ)
fwrite(out, file.path(RES, "gf_offset_grid.csv"))
fwrite(summ, file.path(RES, "gf_offset_summary.csv"))

log("\n--- 遗传偏移（变换空间上的欧氏距离）---")
print(summ[, .(model, scenario, median = round(median, 4),
               p90 = round(p90, 4), max = round(max, 4),
               median_inside = round(median_inside, 4))])

log("\n各模型内 4 个场景相对该模型 SSP245/2061-2080 中位的倍数：")
for (nm in c("pop", "ind")) {
  s <- summ[model == nm]
  base <- s[scenario == SCEN_FUT[1], median]
  log(sprintf("  %-4s %s", nm,
              paste(sprintf("%s=%.2f", s$scenario, s$median / base),
                    collapse = "  ")))
}

## ---- 18 个种群各自驻点的偏移（局部偏移）----------------------------------
cn <- function(nm, sc) paste0(nm, "__", sc)
pop_off <- data.table(pop = sites$pop, lon = sites$lon, lat = sites$lat,
                      row = sites$row, col = sites$col, in_mask = sites$in_mask)
for (nm in c("pop", "ind")) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  for (sc in SCEN_FUT) {
    set(pop_off, j = cn(nm, sc), value = offset(gf, sites, sc))
  }
}
stopifnot(nrow(pop_off) == nrow(pop_env), !anyNA(pop_off))

# 驻点偏移的大头是否只是「未来气候跑到训练区间外、线性外推拉出来的」？
# 逐种群数一遍 6 个变量里有几个在 SSP585/2081-2100 下落到 18 种群的训练区间外，
# 以及越界幅度相对训练跨度是多少，跟着偏移一起报。
sc_last <- "ssp585_2081-2100"
dcols <- paste0("d_", VARS)
xd <- as.data.table(lapply(VARS, function(b) {
  (sites[[paste0(b, "__", sc_last)]] - sites[[paste0(b, "__current")]]) /
    rng[bio == b, train_max - train_min]
}))
setnames(xd, dcols)
xd[, `:=`(pop = sites$pop,
          n_var_moved_over_span = rowSums(abs(.SD) > 1),
          max_move_span = apply(abs(.SD), 1, max)), .SDcols = dcols]
pop_off <- merge(pop_off, xd[, c("pop", "n_var_moved_over_span",
                                 "max_move_span"), with = FALSE], by = "pop")
setcolorder(pop_off, c("pop", "lon", "lat", "row", "col", "in_mask"))
fwrite(pop_off, file.path(RES, "gf_offset_pops.csv"))
log("\n18 个种群驻点偏移（按 SSP585/2081-2100 的种群级偏移降序）；")
log("n_var_moved_over_span = 该种群未来气候相对当前的变化超过训练跨度 1 倍的变量个数")
print(data.table(
  pop = pop_off$pop,
  in_mask = pop_off$in_mask,
  pop_245_6180 = round(pop_off[[cn("pop", "ssp245_2061-2080")]], 3),
  pop_585_8180 = round(pop_off[[cn("pop", "ssp585_2081-2100")]], 3),
  ind_585_8180 = round(pop_off[[cn("ind", "ssp585_2081-2100")]], 3),
  n_var_moved_over_span = pop_off$n_var_moved_over_span,
  max_move_span = round(pop_off$max_move_span, 2)
)[order(-pop_585_8180)])
log(sprintf("\n种群级偏移 vs 气候变化幅度（倍数训练跨度）的 Spearman：%.2f",
            cor(pop_off[[cn("pop", sc_last)]], pop_off$max_move_span,
                method = "spearman")))
o <- pop_off$in_mask == 0
log(sprintf("掩膜外驻点 %d 个：偏移中位 %.3f；掩膜内 %d 个：%.3f",
            sum(o), median(pop_off[[cn("pop", sc_last)]][o]),
            sum(!o), median(pop_off[[cn("pop", sc_last)]][!o])))

log("\n--- 两套模型的像元偏移是否给出一致的地理排序 ---")
for (sc in SCEN_FUT) {
  a <- out[[paste0("pop__", sc)]]; b <- out[[paste0("ind__", sc)]]
  log(sprintf("  %-18s Spearman rho = %.3f", sc, cor(a, b, method = "spearman")))
}

## ---- 偏移按变量拆开 ------------------------------------------------------
# 偏移是 6 个变量变换量平方和的根号，拆开看每个种群的大偏移主要由哪个变量贡献，
# 免得只报一个总数、说不上是谁在推。
varwise <- rbindlist(lapply(c("pop", "ind"), function(nm) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  p0 <- as.matrix(predict(gf, env_of(sites, "current")))
  rbindlist(lapply(SCEN_FUT, function(sc) {
    p1 <- as.matrix(predict(gf, env_of(sites, sc)))
    d <- abs(p1 - p0)
    colnames(d) <- VARS
    data.table(model = nm, scenario = sc, pop = sites$pop, as.data.table(d),
               dominant = VARS[max.col(d, ties.method = "first")],
               offset = sqrt(rowSums((p1 - p0)^2)))
  }))
}))
fwrite(varwise, file.path(RES, "gf_offset_varwise.csv"))
log("\n各变量对种群级偏移的贡献（SSP585/2081-2100，按偏移降序取前 6 个种群）：")
vw <- varwise[model == "pop" & scenario == sc_last]
print(vw[order(-offset)][1:6, .SD,
                        .SDcols = c("pop", VARS, "offset", "dominant")])
for (nm in c("pop", "ind")) {
  v <- varwise[model == nm & scenario == sc_last]
  log(sprintf("  %s 主导变量计数：%s", nm,
              paste(sprintf("%s=%d", names(table(v$dominant)),
                            as.integer(table(v$dominant))), collapse = "  ")))
}

## ---- 图 ------------------------------------------------------------------
# plot.type="S" 内部对密度做 integrate()，个体级模型上会报 roundoff error，
# 一个 panel 崩掉不该带倒整个 pdf，逐个 try。
safe <- function(what, expr) {
  r <- try(expr, silent = TRUE)
  if (inherits(r, "try-error"))
    log("  [跳过] ", what, "：", trimws(as.character(r)))
}
for (nm in c("pop", "ind")) {
  gf <- if (nm == "pop") gf_pop else gf_ind
  pdf(file.path(RES, sprintf("gf_plots_%s.pdf", nm)), width = 10, height = 7)
  safe("overall importance", plot(gf, plot.type = "O"))
  safe(paste(nm, "split density"),
       plot(gf, plot.type = "S", imp.vars = VARS, leg.posn = "topright"))
  safe(paste(nm, "cumulative importance"),
       plot(gf, plot.type = "C", imp.vars = VARS, show.species = FALSE,
            common.scale = TRUE))
  safe(paste(nm, "performance"),
       plot(gf, plot.type = "P", show.names = FALSE))
  dev.off()
}
log("\n图 → gf_plots_pop.pdf / gf_plots_ind.pdf")
log("偏移栅格 → gf_offset_grid.csv   汇总 → gf_offset_summary.csv")
log("种群驻点偏移 → gf_offset_pops.csv")
