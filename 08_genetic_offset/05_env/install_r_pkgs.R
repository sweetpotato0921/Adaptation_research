# 装入本章所需的 R 包：LEA（LFMM2）、vegan（RDA）、qvalue（FDR）、psych、data.table
# 目标库为包内 03_r_env/R_libs，与已有的 assignPOP 那批并列，不碰系统库。
# 用法: R_LIBS_USER=<包>/03_r_env/R_libs Rscript install_r_pkgs.R

lib <- Sys.getenv("R_LIBS_USER")
if (!nzchar(lib)) stop("请通过 R_LIBS_USER 指定包内库路径")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
# 不设 options(repos=)：那会把 Bioconductor 的仓库一并覆盖掉，BiocManager
# 找不到 LEA。CRAN 仓库只在 install.packages 调用处显式传。
options(timeout = 3600)
CRAN <- "https://cloud.r-project.org"
# bioconductor.org 会重定向到 mghp.osn.xsede.org，那个镜像极慢且会截断
# （PACKAGES 30 秒只给 425 个包，完整是 2236 个），BiocManager 会挂死在这。
# Dortmund 镜像 3 秒给全，且 arm64 有 LEA 预编译二进制，无需本地编译。
options(BioC_mirror = "https://bioconductor.statistik.tu-dortmund.de")

cat("R:", R.version.string, "\n目标库:", lib, "\n\n")

cran <- c("BiocManager", "vegan", "psych", "data.table")
need <- cran[!vapply(cran, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) {
  cat("[CRAN] 安装:", paste(need, collapse = ", "), "\n")
  install.packages(need, lib = lib, repos = CRAN, Ncpus = 4)
} else {
  cat("[CRAN] 已齐\n")
}

# LEA 与 qvalue 在 Bioconductor；LEA 要从源码编译，是本轮最可能失败的环节。
bioc <- c("LEA", "qvalue")
need <- bioc[!vapply(bioc, requireNamespace, logical(1), quietly = TRUE)]
if (length(need)) {
  cat("[Bioc] 安装:", paste(need, collapse = ", "), "\n")
  BiocManager::install(need, lib = lib, ask = FALSE, update = FALSE, Ncpus = 4)
} else {
  cat("[Bioc] 已齐\n")
}

cat("\n=== 结果 ===\n")
for (p in c(cran, bioc)) {
  ok <- requireNamespace(p, quietly = TRUE)
  v <- if (ok) as.character(utils::packageVersion(p)) else "-"
  cat(sprintf("%-12s %-6s %s\n", p, ifelse(ok, "OK", "缺失"), v))
}
