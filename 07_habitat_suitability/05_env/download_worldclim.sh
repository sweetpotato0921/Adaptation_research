#!/usr/bin/env bash
# 下载适生区建模所需的 WorldClim 2.1 气候数据（2.5 arc-min）
#   - 当前气候 (1970-2000) BIO1-BIO19
#   - CMIP6 BCC-CSM2-MR 四景：ssp245/ssp585 × 2061-2080/2081-2100
# 合计约 2.5 GB。curl -C - 支持断点续传，可重复执行。
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
ENV_DIR="$BASE/02_env_vars"
CUR="$ENV_DIR/current"
FUT="$ENV_DIR/future"
mkdir -p "$CUR" "$FUT/ssp245" "$FUT/ssp585"

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

get() {  # get <url> <outpath>
  local url="$1" out="$2"
  if [ -s "$out" ]; then
    echo "[跳过] $(basename "$out") 已存在 ($(du -h "$out" | cut -f1))"
    return 0
  fi
  echo "[下载] $(basename "$out")"
  curl -L -C - --fail --retry 3 --retry-delay 5 -m 7200 \
       -H "User-Agent: $UA" -o "$out" "$url" || { echo "[失败] $url"; return 1; }
  echo "[完成] $(basename "$out")  $(du -h "$out" | cut -f1)"
}

WC="https://geodata.ucdavis.edu"
CMIP="$WC/cmip6/2.5m/BCC-CSM2-MR"

# 1) 当前气候，19 个生物气候变量打包在一个 zip 里
get "$WC/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip" "$CUR/wc2.1_2.5m_bio.zip"

# 2) 未来气候，每景一个多波段 GeoTIFF
for ssp in ssp245 ssp585; do
  for period in 2061-2080 2081-2100; do
    get "$CMIP/$ssp/wc2.1_2.5m_bioc_BCC-CSM2-MR_${ssp}_${period}.tif" \
        "$FUT/$ssp/wc2.1_2.5m_bioc_BCC-CSM2-MR_${ssp}_${period}.tif"
  done
done

# 3) 解包当前气候 zip
if [ -s "$CUR/wc2.1_2.5m_bio.zip" ] && [ ! -f "$CUR/wc2.1_2.5m_bio_1.tif" ]; then
  echo "[解包] wc2.1_2.5m_bio.zip"
  unzip -o -q "$CUR/wc2.1_2.5m_bio.zip" -d "$CUR"
fi

echo
echo "=== 下载结果 ==="
ls -lh "$CUR" | tail -n +2
for ssp in ssp245 ssp585; do ls -lh "$FUT/$ssp" | tail -n +2; done
