"""FST 结构诊断：各省内部与各省之间的位点 FST。全量诊断，不是归属成绩。

用途见设计方案 §6.3：

  各省之间 —— 定两段式排名的分段边界。排名 A 用六省、排名 B 用除云南外的五省，
               依据是云南与其余五省的分化明显高一个量级。
  各省内部 —— 一个省内的 3 个采集种群之间分化多大。省内若很同质，
               两段式里就不必为它再切一段。云南在排名 A 中被整体排除，
               更需单独核对：它省内分化若很高，这个排除就说不过去。

做法：省内的算法是 --keep 只留该省的 30 个个体，--within 把该省的 3 个采集种群
当作 3 个组；省间的算法是 --keep 留该两省的 60 个个体与同样的 --within。

用法：python3 diag_within_province_fst.py [--bfile ...] [--labels ...] [--out ...]
"""
import argparse
import itertools
import os
import pathlib
import subprocess

import numpy as np
import pandas as pd

PKG = pathlib.Path(__file__).resolve().parent.parent
ALL = ["云南", "江西", "湖南", "湖北", "贵州", "川渝"]
OTHERS = [k for k in ALL if k != "云南"]


def _run(plink, bfile, keep, inner, tag):
    """跑一次 PLINK 并把 .fst 读成 (位点 ID, FST) 两个数组。

    PLINK 对单态位点写的是字面量 nan，这里直接剔掉再返回。
    """
    subprocess.run([plink, "--bfile", bfile, "--allow-extra-chr", "--allow-no-sex",
                    "--keep", keep, "--fst", "--within", inner, "--out", tag],
                   capture_output=True)
    snps, fst = [], []
    with open(f"{tag}.fst") as fh:
        next(fh)
        for line in fh:
            c = line.split()
            if len(c) >= 5:
                snps.append(c[1]); fst.append(float(c[4]))
    v = np.array(fst)
    ok = ~np.isnan(v)
    return np.array(snps)[ok], v[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfile", default=f"{PKG}/02_intermediate/snp")
    ap.add_argument("--labels", default=f"{PKG}/02_intermediate/labels.csv")
    ap.add_argument("--out", default=f"{PKG}/05_results/diagnostics/FST")
    ap.add_argument("--plink", default="plink")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lab = pd.read_csv(a.labels).set_index("sample")
    iids = [l.split()[1] for l in open(f"{a.bfile}.fam")]
    pop = [lab.loc[i, "pop"] for i in iids]
    prov = [lab.loc[i, "province"] for i in iids]

    # PLINK 的 --within 期望三列：表型占位、个体 ID、组名。两张分组表口径不同，
    # 分开建：算省间 FST 时组名取省（每省一个组），算省内 FST 时组名取采集种群。
    inner_prov = f"{a.out}/_inner_prov.txt"
    with open(inner_prov, "w") as f:
        for i, p in zip(iids, prov):
            f.write(f"0\t{i}\t{p}\n")
    inner_pop = f"{a.out}/_inner_pop.txt"
    with open(inner_pop, "w") as f:
        for i, p in zip(iids, pop):
            f.write(f"0\t{i}\t{p}\n")

    def keep_file(name, wanted):
        path = f"{a.out}/_keep_{name}.txt"
        with open(path, "w") as f:
            for i, v in zip(iids, prov):
                if v in wanted:
                    f.write(f"0\t{i}\n")
        return path

    # ---------------------------------------------------------- 各省之间
    print("各省之间（两两，位点 FST；分组按省，每省一个组）")
    print(f"  {'省对':>12} {'位点数':>7} {'均值':>7} {'中位':>7}")
    pair = {}
    for x, y in itertools.combinations(ALL, 2):
        _, v = _run(a.plink, a.bfile, keep_file(f"{x}_{y}", {x, y}), inner_prov,
                    f"{a.out}/_pair_{x}_{y}")
        pair[(x, y)] = v
        print(f"  {x}-{y:>4} {len(v):>7} {v.mean():>7.3f} {np.median(v):>7.3f}")

    def rng(vs, stat):
        return min(stat(v) for v in vs), max(stat(v) for v in vs)

    y = "云南"
    yv = [pair[(tuple(sorted((y, k))))] for k in OTHERS]
    ov = [pair[(x, y2)] for x, y2 in itertools.combinations(OTHERS, 2)]
    lm, hm = rng(yv, lambda v: v.mean()); ln, hn = rng(yv, np.median)
    print(f"\n  云南 vs 其余五省：均值 {lm:.3f}–{hm:.3f}，中位 {ln:.3f}–{hn:.3f}")
    om, hm = rng(ov, lambda v: v.mean()); on, hn = rng(ov, np.median)
    print(f"  其余五省彼此　：均值 {om:.3f}–{hm:.3f}，中位 {on:.3f}–{hn:.3f}")
    # 两组区间不重叠，故「相差几倍」有一个确定的最小值：
    # 云南一线里最低的那个中位 FST，除以五省彼此里最高的那个中位 FST。
    print(f"  中位口径的分离度：云南一线的最低值 {ln:.3f} 是五省彼此最高值 {hn:.3f} 的 "
          f"{ln / hn:.1f} 倍")

    # ---------------------------------------------------------- 省内
    print(f"\n各省内部（该省 3 个采集种群之间；分组按采集种群）")
    print(f"  {'省':>4} {'所含种群':>16} {'位点数':>7} {'均值':>7} {'中位':>7} "
          f"{'P95':>7} {'最高':>7} {'FST>0.2':>8}")
    res = {}
    for k in ALL:
        snps, v = _run(a.plink, a.bfile, keep_file(k, {k}), inner_pop,
                       f"{a.out}/_fst_inside_{k}")
        res[k] = (snps, v)
        np.savez(f"{a.out}/within_{k}.npz", locus=snps, fst=v)
        pops = sorted({p for p, q in zip(pop, prov) if q == k})
        print(f"{k:>4} {'/'.join(pops):>16} {len(v):>7} {v.mean():>7.3f} "
              f"{np.median(v):>7.3f} {np.percentile(v, 95):>7.3f} {v.max():>7.3f} "
              f"{(v > 0.2).sum():>8}", flush=True)

    print("\n按省内平均 FST 排序：")
    ms = {k: v.mean() for k, (_, v) in res.items()}
    for k in sorted(ms, key=ms.get, reverse=True):
        print(f"  {k:>4} {ms[k]:.3f}")


if __name__ == "__main__":
    main()
