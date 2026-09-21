#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
变量共线性筛查的对照诊断：Pearson / Spearman / VIF 三套标准各会留下什么。

输入直接复用 maxent_run.py 已落盘的 04_results/presence_env.csv 与
background_env.csv，不重新采样背景点，因此三套标准作用在完全相同的矩阵上。

VIF 由相关矩阵求逆得到：$\\mathrm{VIF}_j = [\\mathbf{R}^{-1}]_{jj}$，
等价于把第 $j$ 个变量对其余变量做回归所得的 $1/(1-R_j^2)$。
Pearson 矩阵和 Spearman 矩阵各算一次，看度量选择是否改变结论。
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CHAP = HERE.parent
RES = CHAP / "04_results"
BIO = [f"bio{i}" for i in range(1, 20)]

CORR_THRESHOLD = 0.7        # 对标文献 ENM 章节用的阈值，本项目对齐
VIF_THRESHOLDS = (10.0, 5.0, 3.0)


def load_matrix():
    pres = pd.read_csv(RES / "presence_env.csv", encoding="utf-8")
    bg = pd.read_csv(RES / "background_env.csv", encoding="utf-8")
    x = pd.concat([pres[BIO], bg[BIO]], ignore_index=True).astype(float)
    return x, len(pres), len(bg)


def vif_from_corr(corr):
    """VIF = diag(R^-1)。同时回报条件数，用来判断求逆结果是否还数得出来。"""
    r = corr.to_numpy()
    cond = float(np.linalg.cond(r))
    inv = np.linalg.inv(r)
    return pd.Series(np.diag(inv), index=corr.index), cond


def pairs_above(corr, thr):
    out = []
    n = len(corr)
    for a in range(n):
        for b in range(a + 1, n):
            r = float(corr.iloc[a, b])
            if abs(r) >= thr:
                out.append((corr.index[a], corr.index[b], r))
    out.sort(key=lambda t: -abs(t[2]))
    return out


def greedy_by_corr(corr, thr):
    """迭代删掉 |r| 超阈值的那一对里「与其余变量平均 |r| 更高」的一方。"""
    keep = list(corr.index)
    dropped = []
    while True:
        sub = corr.loc[keep, keep]
        p = pairs_above(sub, thr)
        if not p:
            break
        a, b, r = p[0]                       # 全局最相关的一对
        mean_a = float(sub.loc[a].drop(a).abs().mean())
        mean_b = float(sub.loc[b].drop(b).abs().mean())
        victim, survivor = (a, b) if mean_a >= mean_b else (b, a)
        keep.remove(victim)
        dropped.append(dict(dropped=victim, kept=survivor, r=r,
                            mean_abs_r_victim=mean_a if victim == a else mean_b))
    return keep, dropped


def greedy_by_vif(corr, thr):
    """迭代删掉 VIF 最大者，直到所有变量 VIF 都不超过 thr。"""
    keep = list(corr.index)
    dropped = []
    while len(keep) > 1:
        v, cond = vif_from_corr(corr.loc[keep, keep])
        if v.max() <= thr:
            break
        victim = v.idxmax()
        keep.remove(victim)
        dropped.append(dict(dropped=victim, vif_at_drop=float(v.max()),
                            cond_at_drop=cond))
    return keep, dropped


def main():
    x, n_pres, n_bg = load_matrix()
    print(f"合并矩阵 {x.shape[0]} 行（存在 {n_pres} + 背景 {n_bg}）× {x.shape[1]} 变量\n")

    pear = x.corr(method="pearson")
    spear = x.corr(method="spearman")

    # --- 三套 VIF ---------------------------------------------------------
    vif_p, cond_p = vif_from_corr(pear)
    vif_s, cond_s = vif_from_corr(spear)
    tab = pd.DataFrame({
        "vif_pearson": vif_p, "vif_spearman": vif_s,
        "max_abs_r_pearson": (pear.abs() - np.eye(19)).max(),
        "max_abs_r_spearman": (spear.abs() - np.eye(19)).max(),
    }).sort_values("vif_spearman", ascending=False)
    tab.to_csv(RES / "variable_vif_compare.csv", encoding="utf-8",
               float_format="%.4f")

    print(f"相关矩阵条件数  Pearson {cond_p:,.0f}   Spearman {cond_s:,.0f}")
    print(f"{'变量':<8}{'VIF(P)':>12}{'VIF(S)':>12}{'最大|r|(P)':>12}{'最大|r|(S)':>12}")
    for v, r in tab.iterrows():
        print(f"{v:<8}{r.vif_pearson:>12,.1f}{r.vif_spearman:>12,.1f}"
              f"{r.max_abs_r_pearson:>12.3f}{r.max_abs_r_spearman:>12.3f}")

    for thr in VIF_THRESHOLDS:
        np_ = int((vif_p > thr).sum())
        ns_ = int((vif_s > thr).sum())
        print(f"\nVIF > {thr:>4.0f}  Pearson 命中 {np_:>2d} 个   Spearman 命中 {ns_:>2d} 个")

    # --- 相关性对 ---------------------------------------------------------
    print()
    for name, c in (("Pearson", pear), ("Spearman", spear)):
        p = pairs_above(c, CORR_THRESHOLD)
        print(f"|r| ≥ {CORR_THRESHOLD} 的变量对  {name:<8} {len(p):>3d} 组"
              + (f"   最强 {p[0][0]}~{p[0][1]} r={p[0][2]:.3f}" if p else ""))

    # --- 四种剔除规则各留下什么 -------------------------------------------
    rules = {}
    for name, c in (("Pearson", pear), ("Spearman", spear)):
        keep, dropped = greedy_by_corr(c, CORR_THRESHOLD)
        rules[f"{name}|r|>{CORR_THRESHOLD}"] = (keep, dropped)
    for thr in VIF_THRESHOLDS:
        keep, dropped = greedy_by_vif(spear, thr)
        rules[f"SpearmanVIF>{thr:g}"] = (keep, dropped)

    print(f"\n{'规则':<22}{'保留数':>7}   保留变量")
    for name, (keep, _) in rules.items():
        print(f"{name:<22}{len(keep):>7}   {', '.join(keep)}")

    common = set(BIO)
    for keep, _ in rules.values():
        common &= set(keep)
    print(f"\n四套规则都保留的变量（{len(common)} 个）: "
          + ", ".join(b for b in BIO if b in common))
    always_dropped = [b for b in BIO
                      if all(b not in k for k, _ in rules.values())]
    print(f"四套规则都剔除的变量（{len(always_dropped)} 个）: "
          + ", ".join(always_dropped))

    out = {name: dict(keep=keep, dropped=dropped)
           for name, (keep, dropped) in rules.items()}
    out["_vif_pearson"] = vif_p.round(4).to_dict()
    out["_vif_spearman"] = vif_s.round(4).to_dict()
    out["_cond"] = dict(pearson=cond_p, spearman=cond_s)
    (RES / "variable_screen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细写入 {RES / 'variable_vif_compare.csv'} 与 variable_screen.json")


if __name__ == "__main__":
    sys.exit(main())
