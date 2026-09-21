#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 07 章的 WorldClim 栅格按种群坐标取 19 个生物气候变量。

栅格直接复用 07_habitat_suitability/02_env_vars/crop/current/ 下裁好的 19 个波段，
不重新下载也不重新裁剪——这样两章正文里的环境值逐像元同一，不会出现
「同一物种、同一批 WorldClim，两章数值对不上」的问题。

取值用所在像元（最近邻），不做双线性插值：种群坐标是 GPS 点，插值会
把相邻两个气候差异明显的像元混在一起，而我们只需要该点落在哪个气候格子里。

输出
    04_results/pop_env.csv   18 行，种群级环境值 + 坐标 + 像元行列
    04_results/ind_env.csv   180 行，按 labels.csv 的个体顺序展开

用法:
    ../07_habitat_suitability/05_env/.venv/bin/python 01_prep_env.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio

BASE = pathlib.Path(__file__).resolve().parent.parent      # 08_genetic_offset
ROOT = BASE.parent                                          # 转移包根
CROP = ROOT / "07_habitat_suitability" / "02_env_vars" / "crop" / "current"
SAMPLE = ROOT / "01_raw_data" / "SAMPLE.csv"
LABELS = ROOT / "02_intermediate" / "labels.csv"
RES = BASE / "04_results"

BIO = [f"bio{i}" for i in range(1, 20)]


def log(msg):
    print(msg, flush=True)


def extract(pop, lon, lat):
    """取该坐标所在像元的 19 个 bio 值与像元行列。"""
    vals, pos = [], (None, None)
    for b in BIO:
        with rasterio.open(CROP / f"{b}.tif") as src:
            if not (src.bounds.left <= lon <= src.bounds.right
                    and src.bounds.bottom <= lat <= src.bounds.top):
                sys.exit(f"{pop} ({lon}, {lat}) 落在栅格范围外")
            row, col = src.index(lon, lat)
            v = float(src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0])
        if not np.isfinite(v):
            sys.exit(f"{pop} 的 {b} 取到 nodata，检查该点是否在陆地外")
        vals.append(v)
        pos = (row, col)
    return vals, pos


def main():
    RES.mkdir(parents=True, exist_ok=True)
    samp = pd.read_csv(SAMPLE)
    labels = pd.read_csv(LABELS)
    log(f"种群 {len(samp)} 个，个体 {len(labels)} 个")

    with rasterio.open(CROP / "bio1.tif") as src:
        log(f"栅格 {src.width}×{src.height}  分辨率 {abs(src.transform.a):.4f}°  "
            f"范围 {src.bounds.left:.2f}–{src.bounds.right:.2f}E "
            f"{src.bounds.bottom:.2f}–{src.bounds.top:.2f}N\n")

    rows, cord = [], []
    for _, r in samp.iterrows():
        vals, (row, col) = extract(r.NAME, r.X, r.Y)
        rows.append(dict(pop=r.NAME, lon=r.X, lat=r.Y, row=row, col=col,
                         **dict(zip(BIO, vals))))
        cord.append((r.NAME, row, col))
        log(f"{r.NAME:<5} 像元({row:>4},{col:>4})  " +
            " ".join(f"{b}={v:.2f}" for b, v in zip(BIO[:3], vals[:3])) + " ...")
    pop_env = pd.DataFrame(rows)

    # 像元重复说明两个种群落在同一气候格子，环境无法区分，会削弱种群层面的
    # 共线性判断和 RDA 的解释力，必须报出来。
    dup = pop_env.groupby(["row", "col"])["pop"].apply(list)
    dup = {k: v for k, v in dup.items() if len(v) > 1}
    if dup:
        log(f"\n[警告] {len(dup)} 组种群共用同一像元: "
            + "; ".join(f"{v} 在 {k}" for k, v in dup.items()))
    else:
        log("\n18 个种群落在 18 个互不相同的像元上")

    pop_env.to_csv(RES / "pop_env.csv", index=False, encoding="utf-8",
                   float_format="%.6f")

    assert set(pop_env["pop"]) == set(labels["pop"]), "种群名与 labels.csv 对不上"
    ind_env = labels.merge(pop_env[["pop"] + BIO + ["lon", "lat"]], on="pop",
                           how="left")
    assert ind_env[BIO].notna().all().all(), "有个体没配上环境值"
    ind_env[["sample", "pop", "province", "class", "lon", "lat"] + BIO].to_csv(
        RES / "ind_env.csv", index=False, encoding="utf-8", float_format="%.6f")
    log(f"\n种群级 {pop_env.shape} → {RES / 'pop_env.csv'}")
    log(f"个体级 {ind_env.shape} → {RES / 'ind_env.csv'}")
    log("变量共线性在 04_rda.R 里按实际预测集现算，不在本脚本里判")


if __name__ == "__main__":
    sys.exit(main())
