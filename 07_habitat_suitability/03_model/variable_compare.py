#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
不同变量筛选规则下的 MaxEnt 表现与投影对照。

对每个候选变量集做同一套事：拟合 → 随机 10 折与地理 4 折 AUC →
取 10 百分位训练存在阈值 → 五个情景各自的适生面积与质心 →
与「全部 19 个变量」基准比较（当前情景适生度 Spearman 相关、二值掩膜 Jaccard）。

变量集来自 variable_screen.py 的对照结果，只读其 json，不在本脚本里重算筛选。
本脚本只写 variable_compare.csv，不碰 maxent_run.py 产出的任何文件。
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from maxent_run import (BIO, CROP, RES, SCENARIOS, SEED, cell_areas, log,  # noqa: E402
                        geographic_cv_aucs, haversine, random_cv_aucs)
import elapid  # noqa: E402


def scenario_grid(tag, keep_vars):
    """读某情景下 keep_vars 对应波段的像元矩阵与有效掩膜。"""
    bands = []
    for b in keep_vars:
        with rasterio.open(CROP / tag / f"{b}.tif") as src:
            bands.append(src.read(1).astype("float64"))
            transform, nodata = src.transform, src.nodata
    arr = np.stack(bands, axis=-1)
    valid = np.isfinite(arr).all(axis=-1)
    if nodata is not None:
        valid &= (arr != nodata).all(axis=-1)
    return arr, valid, transform


def predict_layer(model, keep_vars, tag):
    """在整幅研究区上预测适生度，返回 (适生度数组, 有效掩膜, transform)。"""
    arr, valid, transform = scenario_grid(tag, keep_vars)
    suit = np.full(valid.size, np.nan)
    idx = np.flatnonzero(valid)
    suit[idx] = np.asarray(model.predict(
        pd.DataFrame(arr.reshape(-1, len(keep_vars))[idx], columns=keep_vars)))
    return suit.reshape(valid.shape), valid, transform


def area_and_centroid(suit, valid, transform, thr):
    mask = np.zeros(valid.shape, bool)
    mask[valid] = suit[valid] >= thr
    area, lats = cell_areas(transform, valid.shape[0], valid.shape[1])
    weight = area[:, None] * mask
    total = float(weight.sum())
    if not mask.any():
        return total, float("nan"), float("nan")
    cols = transform.c + np.arange(valid.shape[1]) * transform.a
    return (total,
            float((weight * lats[:, None]).sum() / weight.sum()),
            float((weight * np.broadcast_to(cols, weight.shape)).sum() / weight.sum()))


def project_scenarios(model, keep_vars, thr):
    """五个情景各自的适生面积与质心。返回 {tag: (km2, 质心lat, 质心lon)}。"""
    out = {}
    for tag, _ in SCENARIOS:
        suit, valid, transform = predict_layer(model, keep_vars, tag)
        out[tag] = area_and_centroid(suit, valid, transform, thr) + (suit, valid)
    return out


def scenario_columns(name, keep_vars, proj, base, extra):
    """把五情景结果摊平成一行，面积用相对现状的百分比表示。"""
    cur_km2, clat, clon = proj["current"][:3]
    row = dict(extra)
    row.update(rule=name, n_vars=len(keep_vars), variables=",".join(keep_vars),
               current_km2=cur_km2, pct_of_all19=cur_km2 / base["current_km2"] * 100)
    for tag, _ in SCENARIOS:
        km2, lat, lon = proj[tag][:3]
        row[f"{tag}_km2"] = km2
        row[f"{tag}_pct_of_current"] = km2 / cur_km2 * 100
        row[f"{tag}_centroid_shift_km"] = haversine((clat, clon), (lat, lon))
    return row


def log_block(name, keep_vars, proj, extra_note):
    log(f"{name:<20} {keep_vars:>2} 变量  {extra_note}")
    for tag, _ in SCENARIOS:
        km2, lat, lon = proj[tag][:3]
        if tag == "current":
            log(f"{'':<20}   现状              {km2:>9,.0f} km²")
            continue
        c0, lat0, lon0 = proj["current"][:3]
        log(f"{'':<20}   {tag:<18} {km2:>9,.0f} km²  "
            f"({km2 / c0 * 100:>5.1f}% 相对现状)  质心位移 "
            f"{haversine((lat0, lon0), (lat, lon)):>4.0f} km")


def evaluate(name, keep_vars, pres_env, bg_x, base):
    pres_x = pres_env[keep_vars].astype(float).reset_index(drop=True)
    bg_use = bg_x[keep_vars].astype(float).reset_index(drop=True)

    aucs = random_cv_aucs(pres_x, bg_use)
    geo = geographic_cv_aucs(pres_x, bg_use, pres_env["lon"], pres_env["lat"])

    model = elapid.MaxentModel(random_state=SEED)
    model.fit(pd.concat([pres_x, bg_use], ignore_index=True),
              np.r_[np.ones(len(pres_x)), np.zeros(len(bg_use))])
    thr = float(np.percentile(np.asarray(model.predict(pres_x)), 10))

    proj = project_scenarios(model, keep_vars, thr)
    suit_cur, valid = proj["current"][3], proj["current"][4]

    rng = np.random.default_rng(SEED)
    sub = rng.choice(np.flatnonzero(valid), size=20000, replace=False)
    sup = np.zeros(valid.shape, bool)
    sup[valid] = suit_cur[valid] >= thr
    rho = float(spearmanr(base["suit"].reshape(-1)[sub],
                          suit_cur.reshape(-1)[sub]).statistic)
    jac = int((sup & base["mask"]).sum()) / int((sup | base["mask"]).sum())

    note = (f"AUC {np.mean(aucs):.4f}/{np.mean(geo):.4f}  "
            f"ρ {rho:.3f}  J {jac:.3f}  vs 全19现状 {proj['current'][0] / base['current_km2'] * 100:.1f}%")
    log_block(name, len(keep_vars), proj, note)
    return scenario_columns(name, keep_vars, proj, base, dict(
        auc_random=float(np.mean(aucs)), auc_random_sd=float(np.std(aucs)),
        auc_geographic=float(np.mean(geo)), threshold=thr,
        spearman_vs_all19=rho, jaccard_vs_all19=jac))


def main():
    screen = json.loads((RES / "variable_screen.json").read_text(encoding="utf-8"))
    pres_env = pd.read_csv(RES / "presence_env.csv", encoding="utf-8")
    bg_x = pd.read_csv(RES / "background_env.csv", encoding="utf-8")

    log("基准：全部 19 个变量")
    base_model = elapid.MaxentModel(random_state=SEED)
    base_model.fit(pd.concat([pres_env[BIO].astype(float), bg_x[BIO].astype(float)],
                             ignore_index=True),
                   np.r_[np.ones(len(pres_env)), np.zeros(len(bg_x))])
    base_thr = float(np.percentile(np.asarray(base_model.predict(
        pres_env[BIO].astype(float))), 10))
    base_proj = project_scenarios(base_model, BIO, base_thr)
    base_suit, base_valid = base_proj["current"][3], base_proj["current"][4]
    base_mask = np.zeros(base_valid.shape, bool)
    base_mask[base_valid] = base_suit[base_valid] >= base_thr
    base = dict(suit=base_suit, mask=base_mask, current_km2=base_proj["current"][0])
    with rasterio.open(CROP / "current" / "bio1.tif") as src:
        a, _l = cell_areas(src.transform, src.height, src.width)
    log(f"  现状适生 {base['current_km2']:,.0f} km²  阈值 {base_thr:.4f}  "
        f"研究区陆地 {float((a[:, None] * base_valid).sum()):,.0f} km²\n")
    log_block("all19", len(BIO), base_proj, "基准")

    rows = [scenario_columns("all19", BIO, base_proj, base, dict(
        auc_random=np.nan, auc_random_sd=np.nan, auc_geographic=np.nan,
        threshold=base_thr, spearman_vs_all19=1.0, jaccard_vs_all19=1.0))]
    for name, spec in screen.items():
        if name.startswith("_"):
            continue
        rows.append(evaluate(name, spec["keep"], pres_env, bg_x, base))

    df = pd.DataFrame(rows)
    df.to_csv(RES / "variable_compare.csv", index=False, encoding="utf-8",
              float_format="%.4f")
    log(f"\n对照表写入 {RES / 'variable_compare.csv'}")


if __name__ == "__main__":
    sys.exit(main())
