#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
出论文配图。

版式复现对标论文的图3-13（重要性横条 + 累积重要性曲线）、图3-14（变换环境空间
的 PCA 与地理映射）与图3-15（遗传偏移地图）；偏移地图的配色与分级直接取对标文献
Sang et al. 2022 的 13 级离散 blue→red（他们 2genetic-offset-plot.R 里的
offset_color 与断点），不另造色带。

偏移地图只出种群级。个体级模型每 10 个个体共用同一个环境值，像元偏移实际由 18 个
种群的曲线决定，铺成地图只是把种群级结果按 3.4 倍的量纲重画一遍，不提供新的地理
信息；个体级只在数值与表格里出现。

图3-14 的 PCA 在适生区像元上拟合，不是只用 18 个驻点：像元是整个环境空间的采样，
载荷箭头与同一基底算出的 PC1 地理图才自洽。PCA 出三个面板，因为「标准化与否」在
本数据上直接决定 PC1 是什么——见 fig_gf_pca 的说明。18 个驻点的得分不画进载荷图，
PB/MG/HK 的标准化 PC1 达 4.6–5.1 而像元 p99 只有 2.5–3.0，画进来会把箭头压成一点。

用法: ../../07_habitat_suitability/05_env/.venv/bin/python 10_offset_figs.py
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
from scipy.ndimage import uniform_filter  # noqa: E402

# 默认字体 DejaVu Sans 没有汉字，标题会渲染成方框；按可用性挑一个中文字体，
# 全系统都没有就退回英文标签。
_CJK = ["Songti SC", "Heiti SC", "STSong", "Arial Unicode MS", "Hiragino Sans GB"]
_have = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in _CJK if f in _have), None)
if _font:
    plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def T(zh, en):
    return zh if _font else en


BASE = pathlib.Path(__file__).resolve().parent.parent       # 08_genetic_offset
ROOT = BASE.parent
CH07 = ROOT / "07_habitat_suitability"
RES = BASE / "04_results"
RES7 = CH07 / "04_results"
FIGS = RES / "figs"
BOUND = BASE / "06_sources" / "naturalearth_lowres" / "naturalearth_lowres.shp"

SCEN = ["ssp245_2061-2080", "ssp245_2081-2100",
        "ssp585_2061-2080", "ssp585_2081-2100"]
SCEN_LAB = {"ssp245_2061-2080": "SSP245 2061–2080", "ssp245_2081-2100": "SSP245 2081–2100",
            "ssp585_2061-2080": "SSP585 2061–2080", "ssp585_2081-2100": "SSP585 2081–2100"}
# 温度 / 降水二分，与对标论文图3-13 的橙蓝分色一致
TEMP = ["bio2", "bio3", "bio8", "bio9"]
PREC = ["bio18", "bio19"]
C_TEMP, C_PREC = "#E8743B", "#3B7DD8"

# 对标文献 2genetic-offset-plot.R 的 offset_color 与断点，原样照搬
XY_COLORS = ["#2892C7", "#57A0BA", "#78ADAC", "#97BD9E", "#B5CF8F", "#CFDB8A",
             "#E8DE82", "#f7cb79", "#F7B76D", "#f59e5f", "#F58653", "#F7754D", "#FF3333"]
XY_BREAKS = [0, 0.025, 0.0275, 0.030, 0.0325, 0.035, 0.040,
             0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.25]
_TOP = XY_BREAKS[-1]


def log(msg):
    print(msg, flush=True)


def map_window(grid):
    """视窗取 0.2–99.8 分位再放宽 1°。取极值会把主体压成一小块——适生区有几个
    孤立像元落在极东极西。"""
    return ((np.percentile(grid["lon"], .2) - 1, np.percentile(grid["lon"], 99.8) + 1),
            (np.percentile(grid["lat"], .2) - 1, np.percentile(grid["lat"], 99.8) + 1))


def scatter_on_grid(ax, lon, lat, vals, transform, shape, cmap, norm,
                    extent, xlim, ylim, bnd):
    """按 row/col 把像元值铺回整幅网格再 imshow。

    不用散点：像元是 0.0417° 的方格，散点会留缝、也看不出连续的地带。
    掩膜外设 NaN 让其透明。"""
    a = np.full(shape, np.nan, dtype="float32")
    rr, cc = rasterio.transform.rowcol(transform, np.asarray(lon), np.asarray(lat))
    a[np.asarray(rr), np.asarray(cc)] = vals
    im = ax.imshow(np.ma.masked_invalid(a), extent=extent, origin="upper",
                   cmap=cmap, norm=norm, interpolation="nearest")
    bnd.plot(ax=ax, facecolor="none", edgecolor="#8c8c8c", lw=0.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (ylim[0] + ylim[1]))))
    return im


# ---------------------------------------------------------------- 图1 采样点
def fig_sampling(grid, pops, transform, shape, extent, xlim, ylim, bnd):
    with rasterio.open(RES7 / "suit_current.tif") as src:
        suit = src.read(1)
    suit = np.where(suit < 0, np.nan, suit)
    occ = pd.read_csv(ROOT / "01_raw_data" / "SAMPLE.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(np.ma.masked_invalid(suit), extent=extent, origin="upper",
              cmap="YlGn", vmin=0, vmax=1, interpolation="nearest")
    bnd.plot(ax=ax, facecolor="none", edgecolor="#8c8c8c", lw=0.5)
    ax.scatter(occ["X"], occ["Y"], s=6, facecolor="none", edgecolor="#444444",
               lw=0.6, label=T("分布记录点", "Occurrence records"))
    ax.scatter(pops["lon"], pops["lat"], s=26, c="#c0392b", edgecolor="white",
               lw=0.6, zorder=5, label=T("采样种群（18）", "Sampled populations (18)"))
    for r in pops.itertuples():
        ax.annotate(r.pop, (r.lon, r.lat), fontsize=6.5, zorder=6,
                    xytext=(3.5, 3), textcoords="offset points")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (ylim[0] + ylim[1]))))
    ax.set_xlabel(T("经度 °E", "Longitude °E"), fontsize=8)
    ax.set_ylabel(T("纬度 °N", "Latitude °N"), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(T("研究区当前气候适生性与采样点分布（底色为 MaxEnt 适生概率）",
                   "Current climatic suitability and sampling sites "
                   "(background = MaxEnt suitability)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_sampling.png", dpi=200)
    plt.close(fig)
    log("图1 → figs/fig1_sampling.png")


# ------------------------------------------------- 图2 重要性 + 累积重要性曲线
def fig_importance(cum, imp_pop):
    order = imp_pop.sort_values("weighted_R2")["variable"].tolist()
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.1], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(order))
    for i, b in enumerate(order):
        v = imp_pop.set_index("variable").loc[b, "weighted_R2"]
        ax.barh(i, v, color=C_TEMP if b in TEMP else C_PREC, height=0.62)
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0, max(imp_pop["weighted_R2"]) * 1.18)
    ax.set_xlabel(T("加权 R²（累积重要性曲线振幅）", "Weighted R² (curve amplitude)"),
                  fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(T("(a) 变量重要性", "(a) Variable importance"), fontsize=9)
    ax.grid(axis="x", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=C_TEMP),
         plt.Rectangle((0, 0), 1, 1, color=C_PREC)]
    ax.legend(h, [T("温度变量", "Temperature"), T("降水变量", "Precipitation")],
              fontsize=6.5, loc="lower right")

    gs2 = gs[0, 1].subgridspec(2, 3, hspace=0.55, wspace=0.32)
    turn = {}
    for k, b in enumerate(order):
        ax = fig.add_subplot(gs2[k // 3, k % 3])
        d = cum[(cum.model == "pop") & (cum.variable == b)].sort_values("x")
        ax.plot(d["x"], d["y"], "-", color=C_TEMP if b in TEMP else C_PREC, lw=1.2)
        dy = np.diff(d["y"].values)
        turn[b] = float(d["x"].values[np.argmax(dy) + 1])
        ax.axvline(turn[b], color="#666666", ls=":", lw=0.8)
        ax.set_title(f"{b}   {T('转折点', 'turning')} {turn[b]:.3g}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.grid(ls=":", lw=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        if k % 3 == 0:
            ax.set_ylabel(T("累积重要性", "Cumulative importance"), fontsize=6.5)
        ax.set_xlabel(T("环境值", "Environmental value"), fontsize=6.5)
    fig.suptitle(T("(b) 6 个变量的累积重要性曲线（虚线为曲线转折点）",
                   "(b) Cumulative importance curves (dotted = turning point)"),
                 fontsize=9)
    fig.savefig(FIGS / "fig2_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log("图2 → figs/fig2_importance.png")
    log("  种群级曲线转折点：" +
        "  ".join(f"{b}={turn[b]:.4g}" for b in order))
    return turn


# ------------------------------------------------------- 图3 变换环境空间
def fig_gf_pca(tg, sites, VARS, amp, transform, shape, extent, xlim, ylim, bnd):
    """复现对标论文图3-14：遗传变异在分布范围内的主成分分析。

    两块面板：
      (a) 载荷箭头图——方向=变量作用方向，长度=相对贡献强度，即其图3-14a；
      (b) PC1 的地理分布，颜色勾出遗传变异沿分布范围的梯度，即其图3-14b。

    做 PCA 前先把 6 个 f_j 各自标准化。不标准化也能算，但 f_j 本身已按
    weighted R² 缩放（GF 的设计），而本数据 bio19 的振幅是第二位的 3.4 倍，
    不标准化时 PC1 会独占九成以上、退化成 bio19 单条轴，载荷箭头图也就
    失去意义。标准化后 PC1 才是温度与降水的合成梯度。
    """
    X = tg[VARS].values.astype(float)
    Z = (X - X.mean(0)) / X.std(0)

    # PC1 得分的正负号是任意的（SVD 符号不定）。这里统一约定：绝对值最大的
    # 载荷取正号，否则同一份数据两次运行可能给出方向相反的色阶。
    u, sv, vt = np.linalg.svd(Z, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    k = int(np.argmax(np.abs(vt[0])))
    if vt[0, k] < 0:
        u, vt = -u, -vt
    pc1 = u[:, 0] * sv[0]

    fig = plt.figure(figsize=(9.4, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])

    # ---- (a) 载荷箭头
    ax = fig.add_subplot(gs[0, 0])
    for i, b in enumerate(VARS):
        x, y = vt[0, i], vt[1, i]
        c = C_TEMP if b in TEMP else C_PREC
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6,
                                    shrinkA=0, shrinkB=0))
        ax.text(x * 1.08, y * 1.08, b, fontsize=8.5, color=c,
                ha="center", va="center")
    # 像元得分做背景。只画 ±3.2 窗口内的点：PB/MG/HK 的 PC1 达 4.6–5.1，
    # 而像元 p99 只有 2.5–3.0，画进来会把箭头挤成一点
    p2 = Z @ vt[1]
    m = (np.abs(pc1) < 3.2) & (np.abs(p2) < 3.2)
    ax.scatter(pc1[m], p2[m], s=1.2, c="#9e9e9e", alpha=.25, lw=0)
    ax.axhline(0, color="#d0d0d0", lw=.6)
    ax.axvline(0, color="#d0d0d0", lw=.6)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(T(f"PC1（{100*ev[0]:.1f}%）", f"PC1 ({100*ev[0]:.1f}%)"), fontsize=8)
    ax.set_ylabel(T(f"PC2（{100*ev[1]:.1f}%）", f"PC2 ({100*ev[1]:.1f}%)"), fontsize=8)
    ax.set_title(T("(a) 变量载荷：方向与相对贡献", "(a) Variable loadings"),
                 fontsize=9)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")

    # ---- (b) PC1 的地理分布
    axb = fig.add_subplot(gs[0, 1])
    lo, hi = np.percentile(pc1, [2, 98])
    im = scatter_on_grid(axb, tg["lon"], tg["lat"], pc1, transform, shape,
                         "RdYlBu_r", None, extent, xlim, ylim, bnd)
    im.set_clim(lo, hi)
    axb.scatter(sites["lon"], sites["lat"], s=8, facecolor="none",
                edgecolor="#111111", lw=0.6, zorder=5)
    axb.set_title(T(f"(b) PC1 的地理分布（{100*ev[0]:.1f}%）",
                    f"(b) Geographic pattern of PC1"), fontsize=9)
    axb.tick_params(labelsize=7)
    axb.set_xlabel(T("经度 °E", "Longitude °E"), fontsize=7.5)
    axb.set_ylabel(T("纬度 °N", "Latitude °N"), fontsize=7.5)
    cb = fig.colorbar(im, ax=axb, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label(T("PC1 得分", "PC1 score"), fontsize=6)

    fig.suptitle(T("遗传变异在分布范围内的主成分分析（当前适生区）",
                   "PCA of climate-associated genetic variation"), fontsize=10)
    fig.savefig(FIGS / "fig3_gf_pca.png", dpi=200)
    plt.close(fig)
    log(f"图3 → figs/fig3_gf_pca.png；PC1 {100*ev[0]:.1f}%、PC2 {100*ev[1]:.1f}%，"
        f"PC1 载荷 " + "  ".join(f"{b}={vt[0,i]:+.3f}" for i, b in enumerate(VARS)))
    return dict(pc1=100 * ev[0], pc2=100 * ev[1],
                load_pc1=dict(zip(VARS, np.round(vt[0], 3).tolist())))


def fig_transformed_maps(tg, sites, VARS, rng, amp, transform, shape, extent,
                         xlim, ylim, bnd):
    """逐变量的地理图。

    与 PCA 图互补：PCA 把 6 条轴压成 1–2 个合成量，回答「总的变化往哪个方向走」；
    这里逐轴铺开，回答「哪条环境轴在什么位置变化」。两者都要，PCA 单独用时
    PC1 会把 bio19 之外的信息全部掩掉。
    """
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.6), sharex=True, sharey=True,
                             constrained_layout=True)
    info = rng.set_index("bio")
    for k, b in enumerate(VARS):
        ax = axes[k // 3, k % 3]
        v = tg[b].values
        cmap = "YlOrBr" if b in TEMP else "PuBu"
        lo, hi = np.percentile(v, [1, 99])
        im = scatter_on_grid(ax, tg["lon"], tg["lat"], v, transform, shape,
                             cmap, None, extent, xlim, ylim, bnd)
        im.set_clim(lo, hi)
        ax.scatter(sites["lon"], sites["lat"], s=8, facecolor="none",
                   edgecolor="#111111", lw=0.6, zorder=5)
        ax.set_title(f"{b}  {T('振幅', 'amp')} {amp[b]:.3f}"
                     f"  {T('外推', 'extrap')} {info.loc[b, 'pct_outside']:.1f}%", fontsize=8)
        ax.tick_params(labelsize=6)
        if k % 3 == 0:
            ax.set_ylabel(T("纬度 °N", "Latitude °N"), fontsize=7.5)
        if k // 3 == 1:
            ax.set_xlabel(T("经度 °E", "Longitude °E"), fontsize=7.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label(T(f"f({b})", f"f({b})"), fontsize=6)
    fig.suptitle(T("变换环境空间的地理格局：6 个变量各自的 f_j(当前环境)，"
                   "实心色为训练区间内、饱和度到端头即外推",
                   "Geographic pattern of the transformed space: f_j(current) "
                   "for each variable"), fontsize=10)
    fig.savefig(FIGS / "fig4_transformed_maps.png", dpi=200)
    plt.close(fig)
    log("图4 → figs/fig4_transformed_maps.png")


def inside_fraction(grid, transform, shape, win=13):
    """滑动窗口内「6 个变量全在训练区间内」的像元占比，用来画外推区的边界。

    直接对逐像元的 inside 取等值线会画成一团麻：内外像元在像元尺度上交错的，
    39% 的外推像元散在整个适生区里，0/1 场在 0.5 上有几千个碎环。窗口内求占比
    再取 0.5 等值线得到的是一条连贯的界，13 像元约 0.54°（≈55 km）。"""
    rr, cc = rasterio.transform.rowcol(transform, grid["lon"].values, grid["lat"].values)
    suit = np.zeros(shape, dtype=float)
    suit[np.asarray(rr), np.asarray(cc)] = 1.0
    ins = np.zeros(shape, dtype=float)
    ins[np.asarray(rr), np.asarray(cc)] = grid["inside"].values
    frac = uniform_filter(ins, win) / np.maximum(uniform_filter(suit, win), 1e-9)
    return np.where(suit > 0, frac, np.nan)


# --------------------------------------------------------- 图4 遗传偏移地图
def fig_offset_map(grid, pops, transform, shape, extent, xlim, ylim, bnd):
    """2×2 种群级偏移地图。

    图上叠一圈虚线勾出「当前 6 个变量全部落在 18 个种群训练区间内」的范围。这不是
    装饰：线外是 predict 的线性外推，东段日本/朝鲜那块脱离分布区的高偏移像元正是
    外推拉出来的，不圈出来会把外推当成本地适应信号读。
    """
    cmap = ListedColormap(XY_COLORS)
    norm = BoundaryNorm(XY_BREAKS, cmap.N, clip=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    frac = inside_fraction(grid, transform, shape)
    top = pops.nlargest(4, "pop__ssp585_2081-2100")
    for i, sc in enumerate(SCEN):
        ax = axes[i // 2, i % 2]
        v = grid[f"pop__{sc}"].values
        im = scatter_on_grid(ax, grid["lon"], grid["lat"], v, transform, shape,
                             cmap, norm, extent, xlim, ylim, bnd)
        ax.contour(frac, levels=[0.5], extent=extent, origin="upper",
                   colors="#111111", linewidths=0.9, linestyles="--", zorder=4)
        ax.scatter(pops["lon"], pops["lat"], s=9, facecolor="none",
                   edgecolor="#111111", lw=0.6, zorder=5)
        for r in top.itertuples():
            ax.annotate(r.pop, (r.lon, r.lat), fontsize=6, zorder=6,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_title(SCEN_LAB[sc], fontsize=9)
        ax.tick_params(labelsize=7)
        if i % 2 == 0:
            ax.set_ylabel(T("纬度 °N", "Latitude °N"), fontsize=8)
        if i // 2 == 1:
            ax.set_xlabel(T("经度 °E", "Longitude °E"), fontsize=8)
    cb = fig.colorbar(im, ax=list(axes.ravel()), orientation="horizontal",
                      fraction=0.05, pad=0.02, aspect=55, ticks=XY_BREAKS[:-1])
    cb.ax.set_xticklabels([f"{a:g}–{b:g}" for a, b in
                           zip(XY_BREAKS[:-1], XY_BREAKS[1:-1])] + [f">{XY_BREAKS[-2]:g}"],
                          fontsize=6, rotation=32, ha="right")
    cb.set_label(T("种群级模型遗传偏移（变换空间欧氏距离；分级与配色同 Sang et al. 2022）",
                   "Population-level genetic offset (Sang et al. 2022 breaks/colours)"),
                 fontsize=7)
    fig.suptitle(T("当前适生区内的逐像元遗传偏移（空白为当前适生区以外，不入统计）；"
                   "虚线为 55 km 窗口内过半像元 6 个变量全在训练区间内的边界，"
                   "线外以线性外推为主",
                   "Cell-wise genetic offset within the current suitable area "
                   "(dashed = boundary of the ≥50% within-training-range region; "
                   "outside it, values are mostly linearly extrapolated)"), fontsize=9.5)
    fig.savefig(FIGS / "fig5_offset_map.png", dpi=200)
    plt.close(fig)
    log("图5 → figs/fig5_offset_map.png")


# ------------------------------------------------- 图5 种群偏移与变量分解
def fig_pop_offset(pops, vw, VARS):
    order = pops.sort_values("pop__ssp585_2081-2100", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    x = np.arange(len(order))
    for sc, c in zip(SCEN, ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]):
        ax.bar(x, order[f"pop__{sc}"], color=c, width=0.8, label=SCEN_LAB[sc])
    ax.set_xticks(x)
    ax.set_xticklabels(order["pop"], rotation=90, fontsize=7)
    ax.set_ylabel(T("驻点遗传偏移", "Genetic offset at site"), fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, title=T("情景", "Scenario"), title_fontsize=7)
    ax.grid(axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(T("(a) 18 个种群驻点偏移（按 SSP585 2081–2100 降序）",
                   "(a) Offsets at the 18 populations (sorted by SSP585 2081–2100)"),
                 fontsize=9)

    last = SCEN[-1]
    d = vw[(vw.model == "pop") & (vw.scenario == last)].set_index("pop").loc[order["pop"]]
    share = (d[VARS] ** 2).div(d["offset"] ** 2, axis=0) * 100
    ax = axes[1]
    bottom = np.zeros(len(share))
    cs = {"bio2": "#FDBF6F", "bio3": "#E8743B", "bio8": "#8C510A", "bio9": "#C7A17A",
          "bio18": "#9ECAE1", "bio19": "#2B6CB0"}
    for b in VARS:
        ax.bar(x, share[b], bottom=bottom, color=cs[b], width=0.8, label=b)
        bottom += share[b].values
    ax.set_xticks(x)
    ax.set_xticklabels(share.index, rotation=90, fontsize=7)
    ax.set_ylabel(T("各变量对偏移平方和的贡献 (%)", "Share of squared offset (%)"),
                  fontsize=8)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6.5, ncol=3, title=T("变量", "Variable"), title_fontsize=7)
    ax.set_title(T("(b) 偏移的变量分解（SSP585 2081–2100）",
                   "(b) Variable decomposition (SSP585 2081–2100)"), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_pop_offset.png", dpi=200)
    plt.close(fig)
    log("图6 → figs/fig6_pop_offset.png")


def cell_areas(transform, height):
    """逐行球面像元面积（km²/像元），与 07 章 maxent_run.cell_areas 同一算法。"""
    R = 6371.0088
    dlam = abs(transform.a) * np.pi / 180.0
    rows = np.arange(height)
    lat_top = transform.f + rows * transform.e
    lat_bot = lat_top + transform.e
    return R * R * dlam * np.abs(np.sin(np.radians(lat_top))
                                 - np.sin(np.radians(lat_bot)))


def main():
    FIGS.mkdir(exist_ok=True)
    VARS = pd.read_csv(RES / "rda_vars.csv")["variable"].tolist()
    grid = pd.read_csv(RES / "gf_offset_grid.csv")
    pops = pd.read_csv(RES / "gf_offset_pops.csv")
    sites = pd.read_csv(RES / "pop_env_sites.csv")
    cum = pd.read_csv(RES / "gf_cumimp.csv")
    imp_pop = pd.read_csv(RES / "gf_importance_pop.csv")
    # 导出表里每个变量带情景后缀；这里取出当前情景并改回裸变量名
    tg_pop = pd.read_csv(RES / "gf_transformed_grid_pop.csv").rename(
        columns={f"{b}__current": b for b in VARS})
    # 变换坐标表只有种群名与 6 个坐标，经纬度从驻点表并进来（同序）
    ts_pop = pd.read_csv(RES / "gf_transformed_sites_pop.csv").merge(
        sites[["pop", "lon", "lat"]], on="pop")
    vw = pd.read_csv(RES / "gf_offset_varwise.csv")

    with rasterio.open(RES7 / "bin_current.tif") as src:
        transform, shape = src.transform, (src.height, src.width)
        bounds = src.bounds
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    xlim, ylim = map_window(grid)
    bnd = gpd.read_file(BOUND)

    area = cell_areas(transform, shape[0])
    rr, _ = rasterio.transform.rowcol(transform, grid["lon"].values, grid["lat"].values)
    km2 = area[np.asarray(rr)].sum()
    log(f"栅格 {shape[1]}×{shape[0]}  适生区像元 {len(grid):,}  {km2:,.0f} km²  "
        f"视窗 {xlim[0]:.1f}–{xlim[1]:.1f}E {ylim[0]:.1f}–{ylim[1]:.1f}N")

    fig_sampling(grid, pops, transform, shape, extent, xlim, ylim, bnd)
    fig_importance(cum, imp_pop)
    pca = fig_gf_pca(tg_pop, ts_pop, VARS, None,
                     transform, shape, extent, xlim, ylim, bnd)
    fig_transformed_maps(tg_pop, ts_pop, VARS,
                         pd.read_csv(RES / "gf_grid_range.csv"),
                         dict(zip(imp_pop.variable, imp_pop.weighted_R2)),
                         transform, shape, extent, xlim, ylim, bnd)
    fig_offset_map(grid, pops, transform, shape, extent, xlim, ylim, bnd)
    fig_pop_offset(pops, vw, VARS)
    # PCA 的解释率与载荷写进 CSV：报告正文要引，手抄会与图脱节
    pd.DataFrame([pca]).to_csv(RES / "gf_pca_summary.csv", index=False,
                               encoding="utf-8")
    log(f"\n完成，6 张图 → {FIGS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
