#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18 个种群驻点的未来命运：把 07 章的适生栅格在驻点坐标上取值，与遗传偏移并成一张表。

偏移回答「基因组组成要挪多远」，适生性回答「原地还剩不剩得住」。两者合起来才是保护
优先级：一片高偏移但未来直接退出适生区的区域，原地保护没有对象；而偏移中等、未来
仍适生的区域才是可以下手的地方。只报偏移会把这两类混成一类。

用法: ../07_habitat_suitability/05_env/.venv/bin/python 11_pop_fate.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE.parent                                   # 08_genetic_offset
ROOT = BASE.parent                                   # 转移包根
RES = BASE / "04_results"
RES7 = ROOT / "07_habitat_suitability" / "04_results"

SCEN = ["ssp245_2061-2080", "ssp245_2081-2100",
        "ssp585_2061-2080", "ssp585_2081-2100"]


def log(msg):
    print(msg, flush=True)


def sample(path, lon, lat):
    """在经纬度上取栅格值，nodata 转 NaN。suit 与 bin 的 nodata 编码不同，都读原值。"""
    with rasterio.open(path) as src:
        rr, cc = rasterio.transform.rowcol(src.transform, lon, lat)
        a = src.read(1)
        v = a[np.asarray(rr), np.asarray(cc)].astype(float)
        v[v == src.nodata] = np.nan
        return v


def main():
    sites = pd.read_csv(RES / "pop_env_sites.csv")

    lon, lat = sites["lon"].values, sites["lat"].values
    out = pd.DataFrame({"pop": sites["pop"].values, "lon": lon, "lat": lat,
                        "in_mask": sites["in_mask"].values})
    out["suit_current"] = np.round(sample(RES7 / "suit_current.tif", lon, lat), 4)
    for sc in SCEN:
        out[f"suit_{sc}"] = np.round(sample(RES7 / f"suit_{sc}.tif", lon, lat), 4)
        out[f"bin_{sc}"] = sample(RES7 / f"bin_{sc}.tif", lon, lat).astype(int)

    # 未来仍适生 = 4 个情景下 bin 是否都为 1。驻点在当前适生区内的前提由 in_mask 给出，
    # 掩膜外的点（FX）本来就不过阈值，单独看。
    out["n_scen_suitable"] = out[[f"bin_{sc}" for sc in SCEN]].sum(axis=1)
    out["lost_all"] = out["n_scen_suitable"] == 0

    sc_last = SCEN[-1]
    off = pd.read_csv(RES / "gf_offset_pops.csv")
    out = out.merge(off[["pop", f"pop__{sc_last}", "max_move_span"]], on="pop")
    out = out.sort_values(f"pop__{sc_last}", ascending=False)
    out.to_csv(RES / "pop_fate.csv", index=False, encoding="utf-8",
               float_format="%.4f")

    log("18 个驻点的适生性与遗传偏移 → pop_fate.csv\n")
    log(f"{'种群':<5}{'偏移':>8}{'最坏情景适生值':>16}{'未来适生情景数':>16}")
    # 列名含连字符（suit_ssp585_2081-2100），itertuples 会把它改成位置名，只能按名字取
    for _, r in out.iterrows():
        log(f"{r['pop']:<5}{r[f'pop__{sc_last}']:>8.4f}"
            f"{r[f'suit_{sc_last}']:>16.3f}"
            f"{int(r['n_scen_suitable']):>16d}")

    log(f"\n4 个情景下全部退出适生区的驻点（{int(out['lost_all'].sum())} 个）："
        f"{', '.join(out[out['lost_all']]['pop'])}")
    log(f"4 个情景全部保留适生的驻点"
        f"（{int((out['n_scen_suitable'] == 4).sum())} 个）："
        f"{', '.join(out[out['n_scen_suitable'] == 4]['pop'])}")
    log(f"只有部分情景仍然适生的驻点"
        f"（{int(out['n_scen_suitable'].between(1, 3).sum())} 个）："
        f"{', '.join(out[out['n_scen_suitable'].between(1, 3)]['pop'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
