#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 07 章裁好的 6 个生物气候栅格按当前适生区掩膜展平成 R 端能直接吃的表，
供 07_gf_offset.R 预测遗传偏移。

栅格不重新下载也不重新裁剪，直接复用 07_habitat_suitability/02_env_vars/crop/ 下的成品，
与 01_prep_env.py 取种群环境值用的是同一套文件，这样种群训练值与偏移预测值
逐像元一致。当前与未来共用同一网格（1488×864，2.5′，EPSG:4326），不需要重采样。

掩膜取 04_results/bin_current.tif 中值为 1 的像元，即 MaxEnt 10% 训练存在阈值
（0.4346）以上的区域。偏移只在种群真实分布范围内解释，铺到整个研究区
（73–135°E / 18–54°N，绝大多数像元是海洋与非分布区）没有意义。

列名用 <bio>__<scenario>，双下划线分隔，R 端 paste0() 拼列名不会歧义。

种群驻点单独取一遍。种群偏移是 18 个具体点位的偏移，不能从适生区掩膜里捞——
掩膜是 MaxEnt 10% 阈值以上的像元，而采样点的存在不等于预测概率过阈值，
按 row/col 与掩膜内连接会静默丢点（实测 FX 丢）。这里按种群坐标直接从同一批
crop 栅格取值，与 01_prep_env.py 取法一致，训练值与偏移预测值逐点同一。

输出
    02_env_vars/gf_grid_env.csv   42,063 行：lon, lat, row, col + 6 bio × 5 场景
    04_results/pop_env_sites.csv     18 行：pop, lon, lat, row, col, in_mask + 6 bio × 5 场景
    04_results/gf_grid_range.csv     每个 bio 的训练范围、预测范围和落在范围外的像元比例

用法:
    ../07_habitat_suitability/05_env/.venv/bin/python 06_grid_env.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio

BASE = pathlib.Path(__file__).resolve().parent.parent      # 08_genetic_offset
ROOT = BASE.parent                                          # 转移包根
CH07 = ROOT / "07_habitat_suitability"
CROP = CH07 / "02_env_vars" / "crop"
BIN = CH07 / "04_results" / "bin_current.tif"
RES = BASE / "04_results"
ENVOUT = BASE / "02_env_vars"

BIO = ["bio2", "bio3", "bio8", "bio9", "bio18", "bio19"]
SCEN = ["current", "ssp245_2061-2080", "ssp245_2081-2100",
        "ssp585_2061-2080", "ssp585_2081-2100"]


def log(msg):
    print(msg, flush=True)


def read_band(path):
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
        return np.where(a == src.nodata, np.nan, a), src


def main():
    ENVOUT.mkdir(parents=True, exist_ok=True)

    with rasterio.open(BIN) as src:
        mask = src.read(1) == 1
        log(f"栅格 {src.width}×{src.height}  分辨率 {abs(src.transform.a):.4f}°  "
            f"{src.bounds.left:.1f}–{src.bounds.right:.1f}E "
            f"{src.bounds.bottom:.1f}–{src.bounds.top:.1f}N")
        log(f"当前适生区掩膜（bin_current==1）：{mask.sum():,} 个像元")
        rows, cols = np.nonzero(mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        transform, crs = src.transform, src.crs

    out = {"lon": np.array(xs), "lat": np.array(ys),
           "row": rows.astype("int32"), "col": cols.astype("int32")}
    per_band = {}
    for sc in SCEN:
        for b in BIO:
            a, _ = read_band(CROP / sc / f"{b}.tif")
            v = a[rows, cols]
            out[f"{b}__{sc}"] = v
            per_band[(b, sc)] = v
        log(f"  {sc:<18} 6 个 bio 已取")

    grid = pd.DataFrame(out)
    bad = grid[[f"{b}__{sc}" for sc in SCEN for b in BIO]].isna().any(axis=1)
    if bad.any():
        sys.exit(f"{bad.sum()} 个适生区像元在某个场景下取到 nodata，掩膜与栅格不匹配")
    log(f"\n取值完成：{len(grid):,} 行 × {len(BIO)*len(SCEN)} 个环境列，无缺失")

    # 外推诊断：GF 只用 18 个种群训练，预测范围远超训练范围时变换曲线是线性外推，
    # 偏移量不可靠，所以逐变量报出有多少像元落在训练区间之外。
    train = pd.read_csv(RES / "pop_env.csv")
    rng = []
    for b in BIO:
        lo, hi = train[b].min(), train[b].max()
        g = grid[f"{b}__current"]
        n_out = int(((g < lo) | (g > hi)).sum())
        rng.append(dict(bio=b, train_min=lo, train_max=hi,
                        grid_min=g.min(), grid_max=g.max(),
                        n_cells=len(g), n_outside=n_out,
                        pct_outside=100 * n_out / len(g)))
        log(f"  {b:<6} 训练 {lo:8.2f}–{hi:8.2f}   预测 "
            f"{g.min():8.2f}–{g.max():8.2f}   范围外 {100*n_out/len(g):5.1f}%")
    pd.DataFrame(rng).to_csv(RES / "gf_grid_range.csv", index=False,
                             float_format="%.4f")

    grid.to_csv(ENVOUT / "gf_grid_env.csv", index=False, float_format="%.4f")
    log(f"\n像元环境矩阵 {grid.shape} → {ENVOUT / 'gf_grid_env.csv'}")
    log(f"范围对照 → {RES / 'gf_grid_range.csv'}")

    ## ---- 18 个种群驻点 ----------------------------------------------------
    pop = pd.read_csv(RES / "pop_env.csv")[["pop", "lon", "lat", "row", "col"]]
    mask_set = set(zip(rows.tolist(), cols.tolist()))
    sites = pop.copy()
    sites["in_mask"] = [int((r, c) in mask_set)
                        for r, c in zip(pop["row"], pop["col"])]
    for sc in SCEN:
        for b in BIO:
            with rasterio.open(CROP / sc / f"{b}.tif") as src:
                rr, cc = rasterio.transform.rowcol(src.transform,
                                                   pop["lon"].values,
                                                   pop["lat"].values)
                v = src.read(1)[rr, cc].astype(float)
                if not np.isfinite(v).all():
                    sys.exit(f"驻点在某场景的 {b} 取到 nodata（{sc}）")
            sites[f"{b}__{sc}"] = v
        log(f"  驻点 {sc:<18} 6 个 bio 已取")
    outside = sites.loc[sites["in_mask"] == 0, "pop"].tolist()
    log(f"\n18 个种群驻点；{len(outside)} 个不在当前适生区掩膜内"
        + (f": {', '.join(outside)}" if outside else "")
        + "（掩膜 = MaxEnt 10% 阈值以上，存在点未必过阈值）")
    sites.to_csv(RES / "pop_env_sites.csv", index=False, float_format="%.4f")
    log(f"驻点环境矩阵 {sites.shape} → {RES / 'pop_env_sites.csv'}")
    log(f"网格 {transform.a:.6f}° / {crs}，与 07 章同网格")


if __name__ == "__main__":
    sys.exit(main())
