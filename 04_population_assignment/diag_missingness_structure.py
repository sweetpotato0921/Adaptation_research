"""缺失结构诊断：只描述数据，不产出归属成绩。

设计方案里三处引用了本脚本的输出，此处集中复现，避免数字只存在于某次临时运行：

  §2 / §3.4  各采集种群、各省的缺失率
  §3.4       只喂缺失剖面（17,728 维 0/1）时的归属准确率
  §6.3       各位点的缺失个体数分布，用于交代 FST 的有效样本量
  §12        同一 contig 内缺失的成片程度、每个缺失格可用的同 contig 邻居数

用法：python3 diag_missingness_structure.py [--dose ...] [--labels ...] [--bim ...]
"""
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

PKG = pathlib.Path(__file__).resolve().parent.parent
ALL = ["云南", "江西", "湖南", "湖北", "贵州", "川渝"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", default=f"{PKG}/02_intermediate/dose_unfilled.csv")
    ap.add_argument("--labels", default=f"{PKG}/02_intermediate/labels.csv")
    ap.add_argument("--bim", default=f"{PKG}/02_intermediate/snp.bim")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dose = pd.read_csv(a.dose, index_col=0)
    lab = pd.read_csv(a.labels).set_index("sample").loc[dose.index]
    V = dose.values.astype(np.float32)
    miss = np.isnan(V)
    n, p = V.shape
    print(f"{n} 个体 × {p} 位点，整格缺失率 {miss.mean():.4f}（{miss.sum()} 格）\n")

    # ---- §2 / §3.4：种群与省份的缺失率 ----
    per = pd.DataFrame({"pop": lab["pop"].values,
                        "prov": lab["province"].values,
                        "rate": miss.mean(1)})
    print("各采集种群缺失率（升序）：")
    g = per.groupby("pop")["rate"].mean().sort_values()
    for k, v in g.items():
        print(f"  {k:>4} {v:.4f}")
    print("\n各省缺失率：")
    for k in ALL:
        v = per.loc[per["prov"] == k, "rate"].mean()
        print(f"  {k:>4} {v:.4f}")

    # ---- §3.4：缺失剖面本身的归属能力 ----
    y = np.array([ALL.index(k) for k in lab["province"]])
    cv = StratifiedKFold(5, shuffle=True, random_state=a.seed)
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=a.seed)
    s = cross_val_score(rf, miss.astype(np.float32), y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"\n只喂缺失剖面（{p} 维 0/1，不含任何基因型）5 折 ACC = {s.mean():.3f}"
          f"  ({np.round(s, 3)})")
    rf.fit(miss.astype(np.float32), y)
    top = np.argsort(-rf.feature_importances_, kind="stable")[:500]
    s2 = cross_val_score(rf, miss[:, top].astype(np.float32), y, cv=cv,
                         scoring="accuracy", n_jobs=1)
    print(f"  取 RF 重要性前 500 位时           5 折 ACC = {s2.mean():.3f}"
          f"  ({np.round(s2, 3)})")

    # ---- §6.3：位点层面的缺失规模，决定 FST 的有效样本量 ----
    lm = miss.sum(0)
    print(f"\n各位点的缺失个体数：最小 {lm.min()}，中位 {np.median(lm):.0f}，"
          f"均值 {lm.mean():.1f}，最大 {lm.max()}")
    print(f"  训练折（90% 个体）下，各位点平均 {lm.mean() * 0.9:.1f} 个个体缺失，"
          f"即约 {int(n * 0.9) - lm.mean() * 0.9:.0f} 人参与 FST 估计")

    # ---- §12：缺失在 contig 内的成片程度 ----
    contig = pd.read_csv(a.bim, sep="\t", header=None)[0].values
    if len(contig) != p:
        print(f"\n[bim 位点数 {len(contig)} 与表达矩阵 {p} 不符，跳过 contig 分析]")
        return
    cid = pd.factorize(contig)[0]
    K = cid.max() + 1
    size = np.bincount(cid, minlength=K)
    typed = np.zeros((n, K), dtype=np.int32)
    for k in range(K):
        typed[:, k] = (~miss[:, cid == k]).sum(1)
    print(f"\n{len(np.unique(contig))} 条 contig；平均每条 {p / len(np.unique(contig)):.2f} 个位点，"
          f"单个位点所在的 contig 平均含 {size[cid].mean():.2f} 个位点")

    # 定义：同 contig 内任一其它位点也缺失的条件概率，用 (个体, contig) 上的缺失计数算
    #   分子 = 该个体在该 contig 上缺失位点的有序对 (a≠b)
    #   分母 = 该个体在该 contig 上缺失位点 × 该 contig 其余位点
    nm1 = miss.sum(1)  # 占位，下面按 contig 重算
    num = den = 0
    for k in range(K):
        c = miss[:, cid == k].sum(1).astype(np.int64)
        L = int(size[k])
        if L < 2:
            continue
        num += int((c * (c - 1)).sum())
        den += int((c * (L - 1)).sum())
    r = num / den
    print(f"  同 contig 内另一位点也缺失 | 本位点缺失：{r:.4f}"
          f"（整格缺失率 {miss.mean():.4f} 的 {r / miss.mean():.1f} 倍）")
    for lo in (2, 3, 5, 10, 20):
        nu = de = 0
        for k in range(K):
            L = int(size[k])
            if L < lo:
                continue
            c = miss[:, cid == k].sum(1).astype(np.int64)
            nu += int((c * (c - 1)).sum())
            de += int((c * (L - 1)).sum())
        print(f"    位点数 >= {lo:>2} 的 contig：{nu / de:.4f}（{nu / de / miss.mean():.1f} 倍）")

    # 跨 contig：缺失格在其它 contig 上的缺失比例
    tot = miss.sum(1)
    mi, mj = np.where(miss)
    cnt = np.zeros((n, K), dtype=np.int64)
    for k in range(K):
        cnt[:, k] = miss[:, cid == k].sum(1)
    cross_rate = np.mean((tot[mi] - cnt[mi, cid[mj]]) / (p - size[cid[mj]]))
    print(f"  跨 contig：{cross_rate:.4f}（{cross_rate / miss.mean():.2f} 倍）")

    # 每个缺失格可用的同 contig 邻居数
    avail = typed[mi, cid[mj]]
    print(f"  每个缺失格可用的同 contig 已检出邻居：均值 {avail.mean():.2f}，"
          f"一个都没有的占 {(avail == 0).mean():.2%}")
    avail_raw = size[cid[mj]] - 1
    print(f"  （同 contig 邻居总数均值 {avail_raw.mean():.2f}）")


if __name__ == "__main__":
    main()
