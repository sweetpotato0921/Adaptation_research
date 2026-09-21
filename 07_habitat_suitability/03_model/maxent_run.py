#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
适生区建模全流程。

    裁剪环境变量到研究区 → 背景点与存在点取值 → 相关性/重要性诊断
    → MaxEnt 拟合与交叉验证 → 当前与四景未来投影 → 适生面积与质心变化

研究区是 73–135°E × 18–54°N 的纯矩形，不做行政边界或生态区掩膜。
WorldClim 2.1 栅格只在陆地有值，海洋为 nodata，因此背景点天然落在陆地。

变量筛选用 Spearman 秩相关矩阵上的 VIF 迭代剔除：先算出 19 个变量各自的
$\mathrm{VIF}_j=[\mathbf{R}^{-1}]_{jj}$ 与相关矩阵条件数，再逐个删掉 VIF 最大者，
每删一个都在剩余集合上重算，直到全部不超过 VIF_MAX。不能一次删掉所有超标变量——
超标是相关簇内部互相造成的，一次全删会把整个簇清空。

用法:
    python maxent_run.py                     # 全流程
    python maxent_run.py --vif 5             # 收紧 VIF 上限
    python maxent_run.py --vif 0             # 不筛选，用全部 19 个变量
    python maxent_run.py --stage crop        # 只跑某一阶段: crop/vars/fit/project
"""

import argparse
import json
import math
import pathlib
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from sklearn.metrics import roc_auc_score

import elapid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# 路径与常量
# ---------------------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent          # 03_model
BASE = HERE.parent                                      # 07_habitat_suitability
ENV_CUR = BASE / "02_env_vars" / "current"
ENV_FUT = BASE / "02_env_vars" / "future"
CROP = BASE / "02_env_vars" / "crop"
RES = BASE / "04_results"
OCC = BASE / "01_occurrence" / "occ_points.csv"

# 研究区矩形，与建模约定一致
BBOX = dict(lon_min=73.0, lon_max=135.0, lat_min=18.0, lat_max=54.0)

BIO = [f"bio{i}" for i in range(1, 20)]

SCENARIOS = [
    ("current", None),
    ("ssp245_2061-2080", ("ssp245", "2061-2080")),
    ("ssp245_2081-2100", ("ssp245", "2081-2100")),
    ("ssp585_2061-2080", ("ssp585", "2061-2080")),
    ("ssp585_2081-2100", ("ssp585", "2081-2100")),
]

N_BACKGROUND = 10000
N_FOLDS = 10
SEED = 0

# 变量筛选：Spearman 秩相关矩阵上迭代剔除 VIF 超限者，见 select_by_vif。
CORR_REPORT = 0.7    # 只用于诊断输出，不作筛选依据
VIF_MAX = 10.0

# MaxEnt 官方默认特征组合随样本量变化：n>15 时为 linear+quadratic+hinge。
# 本数据 42 个存在点，elapid 默认即 LQH，两者在此规模下一致。
FEATURE_TYPES = ["linear", "hinge", "product"]


def log(msg):
    print(msg, flush=True)


def scenario_tif(tag, pair):
    """返回某情景的多波段 GeoTIFF 路径。current 为 19 个单波段文件，返回 None。"""
    if pair is None:
        return None
    ssp, period = pair
    return ENV_FUT / ssp / f"wc2.1_2.5m_bioc_BCC-CSM2-MR_{ssp}_{period}.tif"


# ---------------------------------------------------------------------------
# 阶段 1：裁剪到研究区
# ---------------------------------------------------------------------------
def crop_window(src):
    """研究区在源栅格里的窗口，取整到像元边界以免重采样导致网格错位。"""
    win = from_bounds(BBOX["lon_min"], BBOX["lat_min"],
                      BBOX["lon_max"], BBOX["lat_max"], src.transform)
    return win.round_offsets().round_lengths()


NODATA_OUT = -9999.0


def write_band(src, band, bio_idx, out_path):
    """读 src 第 band 波段的窗口写出 bio{bio_idx}.tif，返回值域信息。

    band 是波段序号，bio_idx 是变量序号。当前气候是 19 个单波段文件（band 恒为 1，
    bio_idx 取自文件名），未来情景是单个 19 波段文件（两者相等）。

    未来情景的 nodata 是 NaN 而当前气候是 -3.4e38，统一改写为 NODATA_OUT，
    这样下游的背景采样与取值只需认一种约定。

    WorldClim 1.4 的温度变量以 10 倍整数存储，2.1 的 2.5′ 数据已全部是物理值
    （bio4 定义即月均温标准差×100，中位数约 1047），此处不做任何缩放。
    """
    win = crop_window(src)
    data = src.read(band, window=win).astype("float32")
    # 截断的文件可能被 rasterio 按头部声明的尺寸打开却少读行，必须核对实际读出的大小
    if data.shape != (int(win.height), int(win.width)):
        raise ValueError(f"读出 {data.shape}，窗口为 ({int(win.height)},{int(win.width)})")

    in_nodata = src.nodata
    valid_mask = np.isfinite(data)
    if in_nodata is not None and np.isfinite(in_nodata):
        valid_mask &= (data != in_nodata)
    valid = data[valid_mask]

    data[~valid_mask] = NODATA_OUT
    profile = dict(driver="GTiff", dtype="float32", count=1,
                   height=data.shape[0], width=data.shape[1],
                   crs=src.crs, transform=src.window_transform(win),
                   nodata=NODATA_OUT, compress="deflate")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)

    return dict(var=f"bio{bio_idx}", src_nodata=str(in_nodata),
                vmin=float(valid.min()) if valid.size else None,
                vmax=float(valid.max()) if valid.size else None)


def stage_crop():
    CROP.mkdir(parents=True, exist_ok=True)
    report = []

    cur_out = CROP / "current"
    cur_out.mkdir(exist_ok=True)
    for i in range(1, 20):
        src_path = ENV_CUR / f"wc2.1_2.5m_bio_{i}.tif"
        if not src_path.exists():
            sys.exit(f"缺少当前气候文件: {src_path}")
        with rasterio.open(src_path) as src:
            report.append(write_band(src, 1, i, cur_out / f"bio{i}.tif"))
    log(f"[裁剪] 当前气候 19 个变量 → {cur_out.relative_to(BASE)}")

    for tag, pair in SCENARIOS:
        if pair is None:
            continue
        src_path = scenario_tif(tag, pair)
        if not src_path.exists():
            log(f"[跳过] 未来情景缺失: {src_path.name}")
            continue
        out = CROP / tag
        out.mkdir(exist_ok=True)
        # 下载未完成时文件会以完整路径存在但内容截断，这里跳过而非中断整轮
        try:
            with rasterio.open(src_path) as src:
                if src.count != 19:
                    raise ValueError(f"波段数为 {src.count}，预期 19")
                desc = [d or "" for d in src.descriptions]
                if any(desc) and not all("bio" in d.lower() or d.strip().isdigit() for d in desc):
                    log(f"[警告] {src_path.name} 波段描述异常: {desc[:3]}")
                for i in range(1, 20):
                    write_band(src, i, i, out / f"bio{i}.tif")
        except Exception as exc:
            log(f"[跳过] {tag} 读取失败，疑似未下完: {type(exc).__name__}: {exc}")
            for f in out.glob("bio*.tif"):
                f.unlink()
            continue
        log(f"[裁剪] {tag} → {out.relative_to(BASE)}")

    (RES / "crop_report.json").parent.mkdir(parents=True, exist_ok=True)
    (RES / "crop_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with rasterio.open(cur_out / "bio1.tif") as src:
        log(f"[裁剪] 网格 {src.width}×{src.height}  像元 {abs(src.transform.a):.6f}°  "
            f"范围 {src.bounds}")
    return cur_out


# ---------------------------------------------------------------------------
# 阶段 2：背景点与存在点取值 + 变量诊断
# ---------------------------------------------------------------------------
def load_points():
    df = pd.read_csv(OCC, encoding="utf-8")
    gs = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    return df, gs


def stage_vars(cur_dir):
    RES.mkdir(parents=True, exist_ok=True)
    bio_paths = [str(cur_dir / f"{b}.tif") for b in BIO]
    df, gs = load_points()

    with rasterio.open(bio_paths[0]) as src:
        nodata = src.nodata
    # elapid.sample_raster 走 numpy 全局随机数发生器，不播种则每次背景点都不同
    np.random.seed(SEED)
    bg = elapid.sample_raster(str(cur_dir / "bio1.tif"), count=N_BACKGROUND,
                              nodata=nodata)
    log(f"[采样] 背景点 {len(bg)}（从研究区陆地像元随机抽）")

    # drop_na=False 保持行序与 occ_points.csv 一一对应，便于回溯异常点
    pres_env = elapid.annotate(gs, bio_paths, labels=BIO, quiet=True, drop_na=False)
    bg_env = elapid.annotate(bg, bio_paths, labels=BIO, quiet=True)
    pres_env["label"] = df["label"].values
    pres_env["source"] = df["source"].values
    pres_env["lon"] = df["lon"].values
    pres_env["lat"] = df["lat"].values
    bg_x = bg_env.drop(columns="geometry")
    bg_x.to_csv(RES / "background_env.csv", index=False, encoding="utf-8")

    na = pres_env[BIO].isna().any(axis=1)
    if na.any():
        log(f"[警告] {int(na.sum())} 个存在点落在环境 nodata 上，已剔除:")
        for r in pres_env[na].itertuples():
            log(f"        {r.label}  {r.lon:.4f}E {r.lat:.4f}N")
        pres_env = pres_env[~na].reset_index(drop=True)
    pres_env.drop(columns="geometry").to_csv(RES / "presence_env.csv",
                                             index=False, encoding="utf-8")
    log(f"[取值] 存在点可用 {len(pres_env)}/{len(df)}   背景点 {len(bg_x)}")

    # 相关性诊断在研究区整体环境空间上做，背景点占绝大多数权重
    x = pd.concat([pres_env[BIO], bg_x[BIO]], ignore_index=True).astype(float)
    corr = x.corr(method="spearman")
    pairs = []
    for a in range(len(BIO)):
        for b in range(a + 1, len(BIO)):
            r = corr.iloc[a, b]
            if abs(r) >= CORR_REPORT:
                pairs.append((BIO[a], BIO[b], float(r)))
    pairs.sort(key=lambda t: -abs(t[2]))
    pd.DataFrame(pairs, columns=["var_a", "var_b", "spearman_r"]).to_csv(
        RES / "variable_correlation.csv", index=False, encoding="utf-8")
    log(f"[相关性] Spearman |r|≥{CORR_REPORT} 的变量对 {len(pairs)} 组"
        + (f"，最强 {pairs[0][0]}~{pairs[0][1]} r={pairs[0][2]:.3f}" if pairs else ""))
    return pres_env, bg_x, x


def vif_table(x):
    r"""$\mathrm{VIF}_j = [\mathbf{R}^{-1}]_{jj}$，R 为 Spearman 秩相关矩阵。

    用 Spearman 而非 Pearson：研究区跨 18–54°N，最冷月温一类的分布重尾，
    Pearson 矩阵的条件数达 $10^{15}$ 量级（float64 精度极限），求逆结果是噪声。
    """
    r = x.corr(method="spearman").to_numpy()
    return pd.Series(np.diag(np.linalg.inv(r)), index=x.columns), float(np.linalg.cond(r))


def select_by_vif(x, thr):
    """逐个删掉 VIF 最大者直到全部达标。

    不能一次性删掉所有 VIF 超标的变量：超标是相关簇内部互相造成的，
    一次全删会只剩一两个变量。每删一个都要在剩余集合上重算 VIF。
    """
    keep = list(x.columns)
    dropped = []
    while len(keep) > 1:
        v, cond = vif_table(x[keep])
        if v.max() <= thr:
            break
        victim = v.idxmax()
        keep.remove(victim)
        dropped.append(dict(dropped=victim, vif_at_drop=float(v.max()),
                            cond_at_drop=cond, n_remaining=len(keep)))
    final, cond = vif_table(x[keep])
    log(f"[VIF] Spearman VIF > {thr:g} 迭代剔除 {len(dropped)} 个 → 保留 {len(keep)}: "
        + ", ".join(keep))
    log(f"[VIF] 最终集合最大 VIF {final.max():.2f}  条件数 {cond:,.1f}")
    return keep, dropped, final


def random_cv_aucs(pres_x, bg_use):
    """10 折随机交叉验证 AUC。存在点与背景点各自独立分折。"""
    rng = np.random.default_rng(SEED)
    pres_perm = rng.permutation(len(pres_x))
    bg_perm = rng.permutation(len(bg_use))
    aucs = []
    for k in range(N_FOLDS):
        te_p, te_b = np.array_split(pres_perm, N_FOLDS)[k], np.array_split(bg_perm, N_FOLDS)[k]
        tr_p = np.setdiff1d(pres_perm, te_p)
        tr_b = np.setdiff1d(bg_perm, te_b)
        xtr = pd.concat([pres_x.iloc[tr_p], bg_use.iloc[tr_b]], ignore_index=True)
        ytr = np.r_[np.ones(len(tr_p)), np.zeros(len(tr_b))]
        xte = pd.concat([pres_x.iloc[te_p], bg_use.iloc[te_b]], ignore_index=True)
        yte = np.r_[np.ones(len(te_p)), np.zeros(len(te_b))]
        m = elapid.MaxentModel(feature_types=FEATURE_TYPES, random_state=SEED)
        m.fit(xtr, ytr)
        aucs.append(float(roc_auc_score(yte, m.predict(xte))))
    return aucs


def geographic_cv_aucs(pres_x, bg_use, lon, lat):
    """地理分块交叉验证：KMeans 把存在点按空间聚类分折，避免空间自相关抬高 AUC。"""
    gs = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    gkf = elapid.GeographicKFold(n_splits=4, random_state=SEED)
    rng = np.random.default_rng(SEED)
    # 保持与 random_cv_aucs 相同的取数次序：存在点排列先消耗一次随机数，
    # 背景点排列才是第二次。次序变了地理折的背景子集会变、AUC 对不上旧结果。
    rng.permutation(len(pres_x))
    bg_perm = rng.permutation(len(bg_use))
    aucs = []
    for tr_idx, te_idx in gkf.split(gs):
        te_p, tr_p = np.asarray(te_idx), np.asarray(tr_idx)
        te_b = bg_perm[:len(bg_use) // 4]
        tr_b = bg_perm[len(bg_use) // 4:]
        xtr = pd.concat([pres_x.iloc[tr_p], bg_use.iloc[tr_b]], ignore_index=True)
        ytr = np.r_[np.ones(len(tr_p)), np.zeros(len(tr_b))]
        xte = pd.concat([pres_x.iloc[te_p], bg_use.iloc[te_b]], ignore_index=True)
        yte = np.r_[np.ones(len(te_p)), np.zeros(len(te_b))]
        m = elapid.MaxentModel(feature_types=FEATURE_TYPES, random_state=SEED)
        m.fit(xtr, ytr)
        aucs.append(float(roc_auc_score(yte, m.predict(xte))))
    return aucs


def stage_fit(pres_env, bg_x, keep_vars, dropped):
    """拟合最终模型并做两种交叉验证。返回模型、阈值、重要性表。"""
    pres_x = pres_env[keep_vars].astype(float).reset_index(drop=True)
    bg_use = bg_x[keep_vars].astype(float).reset_index(drop=True)
    x_all = pd.concat([pres_x, bg_use], ignore_index=True)
    y_all = np.r_[np.ones(len(pres_x)), np.zeros(len(bg_use))]

    model = elapid.MaxentModel(feature_types=FEATURE_TYPES, random_state=SEED)
    model.fit(x_all, y_all)
    log(f"[拟合] 存在 {len(pres_x)}  背景 {len(bg_use)}  变量 {len(keep_vars)}  "
        f"特征 {FEATURE_TYPES}")

    imp = np.asarray(model.permutation_importance_scores(x_all, y_all, n_repeats=10))
    if imp.ndim == 2:                      # (n_features, n_repeats)
        imp = imp.mean(axis=1)
    imp_df = pd.DataFrame(dict(variable=keep_vars, permutation_importance=imp))
    imp_df = imp_df.sort_values("permutation_importance", ascending=False)
    imp_df.to_csv(RES / "variable_importance.csv", index=False, encoding="utf-8")
    top = imp_df.head(6)
    log("[重要性] " + "  ".join(f"{r.variable}={r.permutation_importance:.3f}"
                               for r in top.itertuples()))

    aucs = random_cv_aucs(pres_x, bg_use)
    geo_aucs = geographic_cv_aucs(pres_x, bg_use, pres_env["lon"], pres_env["lat"])

    cv = pd.DataFrame(dict(
        scheme=["random_10fold"] * len(aucs) + ["geographic_4fold"] * len(geo_aucs),
        fold=list(range(1, len(aucs) + 1)) + list(range(1, len(geo_aucs) + 1)),
        auc=aucs + geo_aucs))
    cv.to_csv(RES / "cv_scores.csv", index=False, encoding="utf-8")
    log(f"[交叉验证] 随机10折 AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}   "
        f"地理4折 AUC={np.mean(geo_aucs):.3f}±{np.std(geo_aucs):.3f}")

    # 10 百分位训练存在阈值，MaxEnt 常用阈值规则
    pres_suit = np.asarray(model.predict(pres_x))
    thr = float(np.percentile(pres_suit, 10))
    meta = dict(feature_types=FEATURE_TYPES, n_presence=int(len(pres_x)),
                n_background=int(len(bg_use)), variables=keep_vars,
                threshold_10pct=thr, auc_random_10fold=float(np.mean(aucs)),
                auc_geographic_4fold=float(np.mean(geo_aucs)),
                bbox=BBOX, seed=SEED)
    (RES / "model_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[阈值] 10 百分位训练存在 = {thr:.4f}")

    plot_response(model, pres_x, keep_vars)
    return model, thr


def plot_response(model, pres_x, keep_vars):
    n = len(keep_vars)
    ncol = 4
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    for ax, var in zip(axes.ravel(), keep_vars):
        pct = np.percentile(pres_x[var], [0, 100])
        grid = np.linspace(pct[0], pct[1], 100)
        rows = pd.DataFrame({v: np.full(100, pres_x[v].median()) for v in keep_vars})
        rows[var] = grid
        ax.plot(grid, np.asarray(model.predict(rows)), color="#1f6f4a", lw=1.4)
        ax.set_title(var, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Marginal response curves (others at median)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "response_curves.png", dpi=180)
    plt.close(fig)
    log(f"[曲线] 边缘响应 → response_curves.png")


# ---------------------------------------------------------------------------
# 阶段 3：投影到当前与四景未来
# ---------------------------------------------------------------------------
def cell_areas(transform, height, width):
    """逐行球面像元面积，km²。纬度方向用 sin 差算球带面积。"""
    R = 6371.0088
    dlam = abs(transform.a) * math.pi / 180.0
    rows = np.arange(height)
    lat_top = transform.f + rows * transform.e
    lat_bot = lat_top + transform.e
    return R * R * dlam * np.abs(np.sin(np.radians(lat_top))
                                 - np.sin(np.radians(lat_bot))), lat_top + transform.e / 2


def study_area_km2(template_path):
    """研究区内有环境值的陆地像元总面积。"""
    with rasterio.open(template_path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = np.isfinite(data) if nodata is None else (data != nodata) & np.isfinite(data)
        area, _ = cell_areas(src.transform, src.height, src.width)
        return float((area[:, None] * valid).sum())


BIN_NODATA = 255


def write_binary(src_path, out_path, thr):
    """按阈值二值化。1=适生，0=非适生，255=海洋等无环境值处。

    海洋必须保留为 nodata 而不是写 0，否则在 GIS 里会被当成非适生陆地，
    适生占比与适生面积都会被算错。
    """
    with rasterio.open(src_path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = data != nodata if nodata is not None else np.ones_like(data, bool)
        valid &= np.isfinite(data)
        out = np.where(~valid, BIN_NODATA,
                       np.where(data >= thr, 1, 0)).astype("uint8")
        profile = src.profile.copy()
        profile.update(dtype="uint8", nodata=BIN_NODATA, compress="deflate")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)
        area, lats = cell_areas(src.transform, src.height, src.width)
        cell = area[:, None]
        mask = (out == 1)
        total = float((cell * mask).sum())
        if mask.any():
            w = cell * mask
            clat = float((w * lats[:, None]).sum() / w.sum())
            cols = src.xy(0, 0)[0] + np.arange(src.width) * src.transform.a
            clon = float((w * np.broadcast_to(cols, w.shape)).sum() / w.sum())
        else:
            clat = clon = float("nan")
        return total, clat, clon


def stage_project(model, thr, keep_vars):
    """投影到各情景。变量顺序须与拟合时一致，否则模型按错列取值。"""
    rows = []
    res = []
    for tag, pair in SCENARIOS:
        d = CROP / tag
        if not d.exists():
            log(f"[跳过] 未裁剪: {tag}")
            continue
        paths = [str(d / f"{b}.tif") for b in keep_vars]
        suit = RES / f"suit_{tag}.tif"
        elapid.apply_model_to_rasters(model, paths, str(suit), quiet=True)
        area, clat, clon = write_binary(suit, RES / f"bin_{tag}.tif", thr)
        rows.append(dict(scenario=tag, suitable_km2=area, centroid_lat=clat,
                         centroid_lon=clon))
        res.append((tag, area, clat, clon))
        log(f"[投影] {tag:<18} 适生 {area:>12,.0f} km²  质心 {clat:.3f}N {clon:.3f}E")

    df = pd.DataFrame(rows)
    base = df.loc[df.scenario == "current"].iloc[0]
    study_area = study_area_km2(CROP / "current" / "bio1.tif")
    df["pct_of_study_area"] = df["suitable_km2"] / study_area * 100
    df["pct_of_present"] = df["suitable_km2"] / base["suitable_km2"] * 100
    df["centroid_shift_km"] = [
        haversine((base.centroid_lat, base.centroid_lon), (r.centroid_lat, r.centroid_lon))
        for r in df.itertuples()]
    df.to_csv(RES / "area_change.csv", index=False, encoding="utf-8")

    log("")
    log(f"研究区陆地面积 {study_area:,.0f} km²")
    log(f"{'情景':<20}{'适生面积 km²':>16}{'占研究区%':>11}{'相对现状%':>11}{'质心位移 km':>13}")
    for r in df.itertuples():
        log(f"{r.scenario:<20}{r.suitable_km2:>16,.0f}{r.pct_of_study_area:>11.2f}"
            f"{r.pct_of_present:>11.1f}{r.centroid_shift_km:>13.0f}")
    log(f"\n结果写入 {RES}")
    return df


def write_vif_report(x, keep, dropped):
    """全 19 变量的 Spearman VIF 表与迭代剔除日志。

    只写 Spearman 一列。Pearson 版的数值在本数据上是求逆噪声（相关矩阵条件数
    达 $10^{15}$），留在 variable_vif_compare.csv 里作对照，不进生产结果。
    """
    spear, _ = vif_table(x)
    max_r = (x.corr(method="spearman").abs() - np.eye(x.shape[1])).max(axis=0)
    tab = pd.DataFrame(dict(variable=x.columns, spearman_vif=spear,
                            max_abs_spearman_r=max_r,
                            kept=[c in keep for c in x.columns]))
    tab.sort_values("spearman_vif", ascending=False).to_csv(
        RES / "variable_vif.csv", index=False, encoding="utf-8", float_format="%.4f")
    if dropped:
        pd.DataFrame(dropped).to_csv(RES / "variable_dropped.csv", index=False,
                                     encoding="utf-8")
    log(f"[VIF] Pearson 版不进结果：条件数 "
        f"{np.linalg.cond(x.corr(method='pearson').to_numpy()):,.0f}，"
        f"该量级下求逆结果不可用，对照见 variable_vif_compare.csv")


def haversine(a, b):
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "crop", "vars", "fit", "project"])
    ap.add_argument("--vif", type=float, default=VIF_MAX,
                    help="Spearman VIF 迭代剔除的上限，设 0 表示不筛选、保留全部 19 个")
    args = ap.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    cur_dir = CROP / "current"
    # 全局源栅格已清理，all 在裁剪产物齐全时直接复用；显式 crop 仍强制重裁
    if args.stage == "crop" or (args.stage == "all" and not (cur_dir / "bio19.tif").exists()):
        stage_crop()

    model = thr = keep_vars = None
    if args.stage in ("all", "vars", "fit"):
        pres_env, bg_x, x = stage_vars(cur_dir)
        if args.vif > 0:
            keep_vars, dropped, _ = select_by_vif(x, args.vif)
        else:
            keep_vars, dropped = list(BIO), []
            log("[VIF] 已关闭变量筛选，使用全部 19 个变量")
        write_vif_report(x, keep_vars, dropped)
        model, thr = stage_fit(pres_env, bg_x, keep_vars, dropped)
        elapid.save_object(model, str(RES / "maxent_model.pkl"))
    elif args.stage == "project":
        model = elapid.load_object(str(RES / "maxent_model.pkl"))
        meta = json.loads((RES / "model_meta.json").read_text(encoding="utf-8"))
        thr, keep_vars = meta["threshold_10pct"], meta["variables"]
        log(f"[载入] maxent_model.pkl  阈值 {thr:.4f}  变量 {len(keep_vars)} 个")

    if model is not None and args.stage in ("all", "project"):
        stage_project(model, thr, keep_vars)


if __name__ == "__main__":
    main()
