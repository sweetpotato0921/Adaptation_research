#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 07_gf_offset.R 算出的逐像元遗传偏移写回 07 章的栅格网格，出图并做面积统计。

栅格模板直接取 07_habitat_suitability/04_results/bin_current.tif：遗传偏移的预测环境值就是从
07 章的 crop 栅格取的，网格必须与之一致，否则叠到 MaxEnt 适生区图上会错位。

偏移只覆盖当前适生区（bin_current==1，42,063 个像元），掩膜外写 nodata 而不是 0：
写 0 在 GIS 里会被读成「偏移为零」，把研究区里未评价的区域算成安全区。

配色用分位数分级，不用连续色带：偏移的分布右偏得很厉害（中位 0.017，最大 0.22），
连续色带会把 90% 的像元压成一色，看不出地理格局。分级断点按模型各自 4 个情景
合并后的分位数取，这样同一模型内 4 个情景可比；两个模型的量纲差 3.4 倍
（个体级伪重复抬高），跨模型不可比，图上分别标断点。

面积统计按球面像元面积算，与 07 章 cell_areas 同一口径，便于两章数字相接。
另附偏移分级 × 未来适生/消失的交叉表：高偏移且未来仍适生的区域是保护优先级，
高偏移但未来直接消失的区域守着也没用，二者管理含义不同，只报偏移总数分不出来。

用法:
    ../../07_habitat_suitability/05_env/.venv/bin/python 08_offset_map.py
"""

import pathlib
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

# 默认字体 DejaVu Sans 没有汉字，标题会渲染成一片方框。按可用性挑一个中文字体，
# 全系统都没有就退回英文标签，而不是出一张画满豆腐块的图。
_CJK = ["Songti SC", "Heiti SC", "STSong", "Arial Unicode MS", "Hiragino Sans GB"]
_have = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in _CJK if f in _have), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

BASE = pathlib.Path(__file__).resolve().parent.parent      # 08_genetic_offset
ROOT = BASE.parent                                          # 转移包根
CH07 = ROOT / "07_habitat_suitability"
RES = BASE / "04_results"
RES7 = CH07 / "04_results"
BOUND = BASE / "06_sources" / "naturalearth_lowres" / "naturalearth_lowres.shp"

TEMPLATE = RES7 / "bin_current.tif"

MODELS = ["pop", "ind"]
SCEN = ["ssp245_2061-2080", "ssp245_2081-2100",
        "ssp585_2061-2080", "ssp585_2081-2100"]
N_CLASS = 8
NODATA = -9999.0
BIN_NODATA = 255

# 浅黄 → 深红，低偏移到高偏移；07 章适生区用的是绿色系，两者叠图时不会混
CMAP = ListedColormap(["#ffffcc", "#ffeda0", "#fed976", "#feb24c",
                       "#fd8d3c", "#fc4e2a", "#e31a1c", "#800026"])


def log(msg):
    print(msg, flush=True)


def T(zh, en):
    """图上标签：有中文字体就用中文，否则退回英文。"""
    return zh if _font else en


def cell_areas(transform, height):
    """逐行球面像元面积（km²/像元），与 07 章 maxent_run.cell_areas 同一算法。"""
    R = 6371.0088
    dlam = abs(transform.a) * np.pi / 180.0
    rows = np.arange(height)
    lat_top = transform.f + rows * transform.e
    lat_bot = lat_top + transform.e
    return R * R * dlam * np.abs(np.sin(np.radians(lat_top))
                                 - np.sin(np.radians(lat_bot)))


def write_tif(lon, lat, vals, name, transform, shape):
    """按 row/col 放回整幅网格，掩膜外 nodata。"""
    with rasterio.open(TEMPLATE) as src:
        profile = src.profile.copy()
    out = np.full(shape, NODATA, dtype="float32")
    rr, cc = rasterio.transform.rowcol(transform, lon, lat)
    out[np.asarray(rr), np.asarray(cc)] = vals.astype("float32")
    profile.update(dtype="float32", nodata=NODATA, compress="deflate")
    with rasterio.open(RES / name, "w", **profile) as dst:
        dst.write(out, 1)
    return out


def breaks_of(v):
    """分位数分级断点。首尾取到数据范围，故两端类是闭区间。"""
    q = np.linspace(0, 100, N_CLASS + 1)
    b = np.unique(np.percentile(v, q))
    if len(b) < 3:                    # 取值太集中，退化成等距
        b = np.linspace(v.min(), v.max(), N_CLASS + 1)
    return b


def main():
    grid = pd.read_csv(RES / "gf_offset_grid.csv")
    with rasterio.open(TEMPLATE) as src:
        transform, shape = src.transform, (src.height, src.width)
        bounds = src.bounds
    log(f"栅格模板 {shape[1]}×{shape[0]}  分辨率 {abs(transform.a):.4f}°  "
        f"{bounds.left:.1f}–{bounds.right:.1f}E {bounds.bottom:.1f}–{bounds.top:.1f}N")
    log(f"适生区像元 {len(grid):,} 个（占整幅 {100*len(grid)/np.prod(shape):.2f}%）")

    area = cell_areas(transform, shape[0])
    rr, cc = rasterio.transform.rowcol(transform, grid["lon"].values,
                                       grid["lat"].values)
    cell_km2 = area[np.asarray(rr)]

    # 未来适生/消失：07 章的 bin_<scenario>.tif，1=适生。像元已限定在当前适生区内，
    # 所以只要看未来是否为 1，就能分成「留下」和「消失」两类。
    persist = {}
    for sc in SCEN:
        with rasterio.open(RES7 / f"bin_{sc}.tif") as src:
            a = src.read(1)
        persist[sc] = a[np.asarray(rr), np.asarray(cc)] == 1
        log(f"  {sc:<18} 当前适生区未来仍适生 {100*persist[sc].mean():5.1f}%  "
            f"消失 {100*(~persist[sc]).mean():5.1f}%")

    ## ---- 单波段 GeoTIFF ---------------------------------------------------
    for m in MODELS:
        for sc in SCEN:
            write_tif(grid["lon"].values, grid["lat"].values,
                      grid[f"{m}__{sc}"].values, f"offset_{m}_{sc}.tif",
                      transform, shape)
    # 多波段版本，波段顺序与 SCEN 一致，给 GIS 叠图用
    for m in MODELS:
        with rasterio.open(TEMPLATE) as src:
            profile = src.profile.copy()
        profile.update(dtype="float32", nodata=NODATA, compress="deflate",
                       count=len(SCEN))
        with rasterio.open(RES / f"offset_{m}.tif", "w", **profile) as dst:
            for k, sc in enumerate(SCEN, start=1):
                band = np.full(shape, NODATA, dtype="float32")
                band[np.asarray(rr), np.asarray(cc)] = grid[f"{m}__{sc}"].values
                dst.write(band, k)
                dst.set_band_description(k, sc)
    log(f"\n单波段 {len(MODELS)*len(SCEN)} 个 + 多波段 {len(MODELS)} 个 "
        f"→ offset_<模型>_<情景>.tif / offset_<模型>.tif")

    ## ---- 面积统计 ---------------------------------------------------------
    rows = []
    for m in MODELS:
        for sc in SCEN:
            v = grid[f"{m}__{sc}"].values
            b = breaks_of(v)
            cls = np.clip(np.digitize(v, b[1:-1], right=False), 0, len(b) - 2)
            p = persist[sc]
            for k in range(len(b) - 1):
                s = cls == k
                if not s.any():
                    continue
                rows.append(dict(
                    model=m, scenario=sc, cls=k + 1,
                    lo=b[k], hi=b[k + 1],
                    n_cells=int(s.sum()),
                    km2=float(cell_km2[s].sum()),
                    km2_persist=float(cell_km2[s & p].sum()),
                    km2_lost=float(cell_km2[s & ~p].sum())))
    cls_tab = pd.DataFrame(rows)
    cls_tab["pct_of_suitable"] = cls_tab.groupby(["model", "scenario"])["km2"] \
        .transform(lambda x: 100 * x / x.sum())
    cls_tab.to_csv(RES / "offset_classes.csv", index=False, encoding="utf-8",
                   float_format="%.4f")
    log(f"\n分级面积表 → offset_classes.csv（{N_CLASS} 级 × "
        f"{len(MODELS)} 模型 × {len(SCEN)} 情景）")

    # 顶部两级（最高 25%）的绝对面积，这是保护上真正需要点名的量
    top2 = cls_tab[cls_tab["cls"] >= N_CLASS - 1].groupby(
        ["model", "scenario"]).agg(km2=("km2", "sum"),
                                   km2_persist=("km2_persist", "sum")).reset_index()
    log("\n最高两级（分位 75% 以上）面积：")
    for r in top2.itertuples():
        log(f"  {r.model:<4} {r.scenario:<18} {r.km2:>10,.0f} km²   "
            f"其中未来仍适生 {r.km2_persist:>10,.0f} km² "
            f"({100*r.km2_persist/r.km2:.0f}%)")

    ## ---- 高偏移像元的地理位置 ---------------------------------------------
    geo = []
    for m in MODELS:
        for sc in SCEN:
            v = grid[f"{m}__{sc}"].values
            thr = np.percentile(v, 90)
            hi = v >= thr
            geo.append(dict(
                model=m, scenario=sc, q90=thr,
                lon_med_all=float(np.median(grid["lon"])),
                lat_med_all=float(np.median(grid["lat"])),
                lon_med_hi=float(np.median(grid["lon"][hi])),
                lat_med_hi=float(np.median(grid["lat"][hi])),
                lon_range_hi=f"{grid['lon'][hi].min():.1f}–{grid['lon'][hi].max():.1f}",
                lat_range_hi=f"{grid['lat'][hi].min():.1f}–{grid['lat'][hi].max():.1f}",
                pct_hi_inside=100 * float(grid["inside"].values[hi].mean()),
                km2_hi=float(cell_km2[hi].sum())))
    geo_tab = pd.DataFrame(geo)
    geo_tab.to_csv(RES / "offset_geography.csv", index=False, encoding="utf-8",
                   float_format="%.4f")
    log("\n高偏移（各情景前 10%）像元的地理位置：")
    for r in geo_tab[geo_tab.model == "pop"].itertuples():
        log(f"  {r.scenario:<18} q90={r.q90:.4f}  {r.km2_hi:>9,.0f} km²  "
            f"中位 {r.lon_med_hi:.1f}E {r.lat_med_hi:.1f}N  "
            f"经度 {r.lon_range_hi}  纬度 {r.lat_range_hi}  "
            f"训练区间内占 {r.pct_hi_inside:.0f}%")

    ## ---- 图 ---------------------------------------------------------------
    # 用 imshow 铺回整幅网格而不是散点：像元本身是 0.0417° 的方格，
    # 散点会留缝、也看不出连续的地带。掩膜外设 NaN 让它透明，底图露出来。
    bnd = gpd.read_file(BOUND)
    panels = {}
    for m in MODELS:
        for sc in SCEN:
            a = np.full(shape, np.nan, dtype="float32")
            a[np.asarray(rr), np.asarray(cc)] = grid[f"{m}__{sc}"].values
            panels[(m, sc)] = np.ma.masked_invalid(a)
    ext = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    # 视窗取 0.2–99.8 分位再放宽 1°，而不是直接取极值：适生区几个孤立像元
    # 落在极东极西，取极值会把主体压成一小块。
    xlim = (np.percentile(grid["lon"], .2) - 1, np.percentile(grid["lon"], 99.8) + 1)
    ylim = (np.percentile(grid["lat"], .2) - 1, np.percentile(grid["lat"], 99.8) + 1)

    fig, axes = plt.subplots(len(MODELS), len(SCEN),
                             figsize=(3.6 * len(SCEN), 2.7 * len(MODELS)),
                             sharex=True, sharey=True, constrained_layout=True)
    pops = pd.read_csv(RES / "gf_offset_pops.csv")
    top = pops.nlargest(4, "pop__ssp585_2081-2100")
    for i, m in enumerate(MODELS):
        allv = np.concatenate([grid[f"{m}__{sc}"].values for sc in SCEN])
        b = breaks_of(allv)                      # 一行内共用分级，情景间可比
        norm = BoundaryNorm(b, CMAP.N)
        for j, sc in enumerate(SCEN):
            ax = axes[i, j]
            im = ax.imshow(panels[(m, sc)], extent=ext, origin="upper",
                           cmap=CMAP, norm=norm, interpolation="nearest")
            bnd.plot(ax=ax, facecolor="none", edgecolor="#8c8c8c", lw=0.5)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            lat_mid = 0.5 * (ylim[0] + ylim[1])
            ax.set_aspect(1 / np.cos(np.radians(lat_mid)))
            ax.set_title(("种群级" if m == "pop" else "个体级") + " · " + sc
                         if _font else
                         ("Population" if m == "pop" else "Individual")
                         + " · " + sc, fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(T("纬度 °N", "Latitude °N"), fontsize=8)
            ax.set_xlabel(T("经度 °E", "Longitude °E"), fontsize=8)
            ax.scatter(pops["lon"], pops["lat"], s=9, facecolor="none",
                       edgecolor="#111111", lw=0.6, zorder=5)
            for r in top.itertuples():
                ax.annotate(r.pop, (r.lon, r.lat), fontsize=6.5, zorder=6,
                            xytext=(3, 3), textcoords="offset points")
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), orientation="horizontal",
                          fraction=0.06, pad=0.02, aspect=50)
        cb.set_label(T(f"{m} 模型偏移（4 情景合并分位分级）",
                       f"Genetic offset, {m} model (pooled-quantile breaks)"),
                     fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks(b)
    fig.suptitle(T("适生区逐像元遗传偏移（空白为当前适生区以外，不入统计）",
                   "Cell-wise genetic offset within the current suitable area "
                   "(blank = outside it, not summarized)"), fontsize=11)
    fig.savefig(RES / "offset_maps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("\n图 → offset_maps.png")

    ## ---- 种群偏移柱状图 ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, m in zip(axes, MODELS):
        d = pops.sort_values(f"pop__{SCEN[-1]}", ascending=False)
        x = np.arange(len(d))
        for sc, c in zip(SCEN, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
            ax.bar(x, d[f"{m}__{sc}"], color=c, width=0.8, label=sc)
        ax.set_xticks(x)
        ax.set_xticklabels(d["pop"], rotation=90, fontsize=7)
        ax.set_ylabel(T("驻点遗传偏移", "Genetic offset at site"), fontsize=8)
        ax.set_title(T("种群级模型" if m == "pop" else "个体级模型",
                       "Population-level" if m == "pop"
                       else "Individual-level"), fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=6.5, title=T("情景", "Scenario"), title_fontsize=7)
        ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(T("18 个种群驻点的遗传偏移（按 SSP585/2081-2100 降序；"
                   "两图纵轴量纲不同）",
                   "Genetic offset at the 18 populations (sorted by "
                   "SSP585/2081-2100; y-scales differ between panels)"),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RES / "offset_pops_bar.png", dpi=200)
    plt.close(fig)
    log("图 → offset_pops_bar.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
