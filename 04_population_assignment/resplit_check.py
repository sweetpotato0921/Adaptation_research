#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结面板的换划分重测：面板固定不动，只换外层折划分，看成绩怎么动。

对标文献（Liang et al. 2025, BMC Genomics 26:1119）的对应步骤是
「用五个新生成的随机划分数据集验证同一套 768 位点面板」。
这里做同样的事：面板取 05_results/frozen_panel.csv（150 个位点，由 seed=0 的
十折 top-500 交集冻结而来），折划分换成 seed=1..5，其余协议完全不变
（外层 10 折、内层 5 折调参、众数填补折内进行）。

该成绩仍是乐观估计：面板由 seed=0 全部十折的选择结果汇总而来，
每个个体都参与过面板构建。换划分只改变成绩的方差，不消除这层乐观。

用法: python3 04_population_assignment/resplit_check.py
输出: 05_results/resplit_check.csv
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

spec = importlib.util.spec_from_file_location(
    "pipeline", os.path.join(ROOT, "04_population_assignment", "pipeline.py"))
pipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipe)

Xraw, miss, y, ids, loci, cls_names, prov = pipe.read_inputs(
    "02_intermediate/dose_unfilled.csv", "02_intermediate/labels.csv")
ncls = len(cls_names)
print(f"数据 {Xraw.shape[0]} 个体 × {Xraw.shape[1]} 位点，{ncls} 类")

panel_ids = pd.read_csv("05_results/frozen_panel.csv")["locus"].tolist()
idx = {l: i for i, l in enumerate(loci)}
panel = [idx[l] for l in panel_ids]
print(f"冻结面板 {len(panel)} 个位点\n")

NEW_SEEDS = [1, 2, 3, 4, 5]
rows, wrong_detail = [], []
for s in NEW_SEEDS:
    folds = list(StratifiedKFold(10, shuffle=True, random_state=s).split(Xraw, y))
    pred, score, fold_acc, _ = pipe.eval_combo(
        Xraw, miss, y, folds, lambda f: panel, "SVM", ncls, seed=s)
    ps = pipe.summarize(y, pred, score, ncls)
    n_wrong = int((pred != y).sum())
    rows.append({"seed": s, "ACC": ps["ACC"], "AUC": ps["AUC"],
                 "macro_recall": ps["macro_recall"], "n_wrong": n_wrong,
                 "worst_fold": min(fold_acc)})
    print(f"  seed={s}  ACC {ps['ACC']:.4f}  AUC {ps['AUC']:.4f}  "
          f"错分 {n_wrong}/180  最差折 {min(fold_acc):.3f}")
    for b in np.where(pred != y)[0]:
        wrong_detail.append({"seed": s, "id": ids[b],
                             "true": cls_names[y[b]], "pred": cls_names[pred[b]]})

df = pd.DataFrame(rows)
print(f"\n5 个新划分平均 ACC {df.ACC.mean():.4f}"
      f"（范围 {df.ACC.min():.4f}–{df.ACC.max():.4f}，"
      f"错分 {df.n_wrong.min()}–{df.n_wrong.max()} 个体）")
print(f"平均 AUC {df.AUC.mean():.4f}   平均宏召回 {df.macro_recall.mean():.4f}")
print(f"逐划分完全正确（180/180）的有 {int((df.n_wrong == 0).sum())} 个")
df.to_csv("05_results/resplit_check.csv", index=False)

if wrong_detail:
    wd = pd.DataFrame(wrong_detail)
    wd.to_csv("05_results/resplit_check_errors.csv", index=False)
    print(f"\n错分个体（{len(wd)} 例）写在 05_results/resplit_check_errors.csv")
else:
    print("\n无错分")
