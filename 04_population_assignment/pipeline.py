"""产地溯源：SNP 特征选择 × 分类器组合的完整评估流程。

本文件逐行标注，标注说明「这一行做什么」与「为什么这么做」。

输入
    --dose    未填补的单列剂量矩阵 CSV，缺失写 NA，表头为位点 ID、首列为样本 ID
    --labels  样本标签 CSV，列 sample, pop, province, class
    --bfile   PLINK 二进制文件前缀（FST 用，读原始基因型而非填补后的矩阵）

流程
    1. 外层 K 折分层交叉验证。每折只在该折的训练集上做两件事，测试折一概不参与：
         (a) 众数填补：众数由训练个体算出，同一套众数填训练折与测试折
         (b) 位点排名，三套：
               mRMR     —— mrmr-selection 原包，全列参与，K = kmax（默认 2000）
               Relief-F —— CORElearn::attrEval 的 ReliefFbestK，邻居数由库内部自搜
               FST      —— PLINK 两段式（六省之间 / 除云南外五省之间），--keep 只喂训练个体
       排名按折缓存在 out/rankings/，已存在则跳过，支持断点续跑
    2. 网格：3 种选择方法 × len(k) 个 k × 4 个分类器。每格在训练集内部用内层 5 折调参，
       用最佳参数在全训练集上重训，预测测试折。折外预测拼合成一条向量后一次算指标
    3. 按折外 ACC（AUC 作次选）挑最佳组合
    4. 在该组合下把各折选出的位点取交集，冻结成面板，用面板重跑一遍并单独报告

输出
    控制台成绩表；CSV 明细（网格、逐折、逐类召回、混淆矩阵、冻结面板、选中的位点）

注意
    k 由外围网格扫描后按成绩挑选（与水牛文献一致），不是由内层交叉验证搜索得到。
    Relief-F 走 R 的 CORElearn 包，须以 Rscript --max-ppsize=500000 启动：
    17,728 项的公式会顶爆 R 默认的指针保护栈（实测报 protect()：防护堆叠上溢）。
    CORElearn 已从 CRAN 归档下架，本包内的副本装在 03_r_env/R_libs，
    由 --rlib 指定；从源码装它只差 plotrix 与 rpart.plot 两个画图依赖。
"""

# ---------------------------------------------------------------- 标准库
import argparse          # 命令行参数解析，让脚本的每个可调项都能从外部覆盖
import os                # 路径拼接与目录创建
import subprocess        # 调 PLINK（外部可执行文件），FST 由 PLINK 计算而非自己实现
import time              # 计时，用于打印各阶段的耗时
import warnings          # 关掉 sklearn 在迭代中的收敛/除零等噪声警告
from collections import Counter   # 统计类别计数，用于日志与打印

import numpy as np       # 数值矩阵运算
import pandas as pd      # 读 CSV、写 CSV、以及 mRMR 包要求的 DataFrame 输入

# sklearn 在若干处会发 FutureWarning 与收敛警告（如 GaussianNB 的方差下溢），
# 它们在正常使用下不指示问题，且本流程要连续打印几十行进度，警告会把输出冲散。
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier       # 既是分类器，也是「森林重要性」的来源
from sklearn.metrics import (accuracy_score,              # 准确率
                             confusion_matrix,            # 混淆矩阵
                             recall_score,                # 逐类召回
                             roc_auc_score)               # 单类 AUC（宏平均由本文件手算）
from sklearn.model_selection import StratifiedKFold       # 分层 K 折，保证每折类别构成一致
from sklearn.naive_bayes import GaussianNB                # 高斯朴素贝叶斯
from sklearn.neighbors import KNeighborsClassifier        # K 近邻
from sklearn.svm import SVC                               # 支持向量机

# 记录脚本启动时刻，之后所有日志都打印相对用时，便于判断哪一步慢
t_start = time.time()


def log(msg):
    """统一格式的进度打印。

    flush=True 是必须的：本脚本要跑数小时，若不强制刷新，
    输出会滞留在缓冲区里，看不到进度、也无法判断卡在哪一步。
    """
    print(f"[{time.time()-t_start:7.1f}s] {msg}", flush=True)


# ==================================================================== 读入

def read_inputs(dose_csv, labels_csv):
    """读剂量矩阵与标签，返回后续全程要用的六样东西。

    剂量矩阵一行一个个体、一列一个位点，取值 {0, 1, 2, NA}：
    0 为纯合参考，2 为纯合替代，1 为杂合，NA 为该格没有基因型。
    """
    # index_col=0 把首列（样本 ID）当行索引。缺失在 CSV 里写作 NA，
    # pandas 读到 NA 会自动转成 NaN，所以下面的 np.isnan 就是缺失掩膜。
    df = pd.read_csv(dose_csv, index_col=0)

    # 转成 float32 的 ndarray：有 NaN 就必须是浮点型（整数型装不下 NaN）；
    # 用 float32 而非默认 float64，矩阵 180×17728 由 25.6 MB 降到 12.8 MB。
    X = df.to_numpy(np.float32)

    # 缺失掩膜：True 表示这一格没有基因型，后面填补只动这些格。
    # 掩膜由数据本身（NA）决定，与测序深度无关——深度只在生成这份 CSV 的上游参与过判定。
    miss = np.isnan(X)

    # set_index("sample") 让标签表以样本 ID 为索引；.loc[df.index] 按剂量矩阵的行的顺序
    # 重排标签。这行是必须的：两张表若行序不同，标签就会错位到别的个体上，
    # 而错位不会报错、只会让成绩变成噪声，属最难发现的 bug。
    lab = pd.read_csv(labels_csv).set_index("sample").loc[df.index]

    # 分类目标：省份的整数编码 0..5
    y = lab["class"].to_numpy()

    # 样本 ID 与位点 ID 转成字符串列表。FST 要按位点 ID 与 PLINK 的输出对齐，
    # 要把 ID 写进 PLINK 的文本文件，所以必须是字符串而非 numpy 标量。
    ids = [str(s) for s in df.index]
    loci = [str(c) for c in df.columns]

    # 类别编号到省份名的对照。drop_duplicates("class") 每个类别留一行，
    # 再按 class 排序，于是列表下标即类别编号，可直接用 y 的下标去索引省份名。
    cls_names = [str(c) for c in
                 lab.drop_duplicates("class").sort_values("class")["province"]]

    # 每个个体的省份名（用于 FST 分组）。这是 6 个类别之上的原样标签。
    prov = lab["province"].to_numpy()

    log(f"剂量矩阵 {X.shape}，缺失 {miss.sum()} 格（{miss.mean()*100:.2f}%）")
    log(f"类别 {cls_names}，计数 {dict(Counter(y))}")
    return X, miss, y, ids, loci, cls_names, prov


def locus_modes(X, miss):
    """算出每位点的众数基因型，返回长度 nloc 的向量。

    逐个位点独立做，不用个体层面的信息：
    每位点的取值只有 0/1/2 三档，众数即出现最多的那一档，
    等价于「用该位点最常见的基因型代表所有人」。
    这是最朴素的填补，不引入任何跨位点的模型假设。

    众数由传进来的这批个体决定，所以调用方是谁很重要：
    在训练折内部调用，众数就只代表训练折的人。

    不需要处理「整列全缺」：位点集在上游已按「单点缺失率 ≤ 0.5」筛过，
    故众数必存在。本数据里缺失率最高的位点也只有 0.406，
    检出率最低的位点仍有 107/180 个个体有基因型。
    若换个输入文件而使该前提不成立，此处会在 np.argmax 上对空数组报错，
    这比静默填一个臆测值要好——静默填错不会留下任何痕迹。
    """
    modes = np.empty(X.shape[1], dtype=X.dtype)
    for j in range(X.shape[1]):          # 逐列（逐位点）
        # np.unique 返回排序后的取值与各自的计数；argmax 给出计数最大者的下标，
        # 即众数。计数并列时 np.argmax 返回最小的下标，即取较小那一档，行为是确定的。
        # ~miss[:, j] 是逻辑取反，取出该位点所有非缺失的取值。
        v, c = np.unique(X[~miss[:, j], j], return_counts=True)
        modes[j] = v[np.argmax(c)]
    return modes


def mode_impute(X, miss, modes=None):
    """把缺失格填成众数基因型。

    modes 省略时在传入的这批个体上现算，此时填训练折个体是对的
    （决定填什么值的人里没有测试折个体）；
    modes 给定时用传入的那一套，跨折填测试折个体必须走这条路——
    两边各算各的众数，等于让测试集通过「填成什么值」间接参与了训练。
    """
    if modes is None:
        modes = locus_modes(X, miss)
    Xi = X.copy()          # 在副本上改，保留原始矩阵供核对
    for j in range(X.shape[1]):          # 逐列（逐位点）
        m = miss[:, j]                   # 该位点的缺失掩膜，长度等于个体数
        if m.any():                      # 整列无缺失就什么都不用做
            Xi[m, j] = modes[j]
    return Xi


# ==================================================================== 三套排名

def rank_mrmr(Xtr, ytr, nloc, kmax, n_jobs):
    """mRMR 排名。输入该折训练集的全部列，输出长度 nloc 的列下标排序。

    mRMR 的择优准则是「最大相关、最小冗余」：第 t 步从剩余候选里挑
        J(f) = Relevance(f) − Redundancy(f)
    最大者，其中相关项取该位点对标签的 F 统计量，
        F = [SS_between/(k−1)] / [SS_within/(n−k)]
    k 为类别数、n 为个体数；冗余项取该位点与已选集合 S 中各位点的平均
    Pearson 相关（带符号，负相关会在下面被当负冗余而加分）：
        Redundancy(f) = (1/|S|) Σ_{s∈S} r(f, s)
    选完第 t 个后 S 扩张，下一步重新评分，故它是逐步贪心、且解依赖于前几步的选择。
    """
    from mrmr import mrmr_classif       # 局部导入：只有用到 mRMR 时才要求装了该包
    # 该包要求 DataFrame 且列名唯一。列名用 L0..L17727，是为了把下标编进名字里，
    # 返回后再解析回下标——包只回传列名，不回传位置。
    df = pd.DataFrame(Xtr, columns=[f"L{j}" for j in range(nloc)])
    # K=kmax 取「排名保留长度」，而不是网格里的 k。跑一次长排序，网格里各个 k
    # 都从这份排序里截取，于是 mRMR 每折只需跑一次。这是全流程的瓶颈：
    # kmax=2000 时单折实测 1538–1889 秒、峰值内存 3.4 GB，十折合计约 4.7 小时。
    # n_jobs 取 1：该包的冗余矩阵与多进程 worker 是内存大户，而 n_jobs 对耗时几乎无影响。
    # relevance='f'、redundancy='c'、denominator='mean' 三者是该包默认值，此处显式写出以免误解。
    sel = mrmr_classif(X=df, y=pd.Series(ytr), K=kmax, n_jobs=n_jobs,
                       relevance="f", redundancy="c", denominator="mean",
                       show_progress=False)
    order = [int(s[1:]) for s in sel]    # "L123" → 123，即列下标
    # 包只返回入选的 kmax 个位点，其余未入选的位点在网格里不会被用到，
    # 但下面要返回长度恰为 nloc 的完整排序（FST 那条路也是完整长度），
    # 故把剩下的按原顺序接在后面，保持三种方法的返回结构一致。
    chosen = set(order)                  # 先建好集合，否则每次判断都要重建一遍
    rest = [j for j in range(nloc) if j not in chosen]
    return np.array(order + rest)


# ---------------------------------------------------------------- R 侧脚本
# Relief-F 的邻居数要由库内部自搜，Python 侧没有等价实现（skrebate 只收一个固定的
# n_neighbors），只能调 R 的 CORElearn。这段脚本运行时写到临时目录再由 Rscript 执行，
# 目的是保持本文件自包含——与 FST 借 PLINK 是同一个模式，都是调外部程序完成一段计算。
RELIEF_R = r'''
args <- commandArgs(trailingOnly = TRUE)
lib <- args[3]
if (nzchar(lib)) .libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages(library(CORElearn))
d <- read.csv(args[1], check.names = FALSE)
d$.class <- factor(d$.class)
sc <- attrEval(.class ~ ., data = d, estimator = "ReliefFbestK")
write.csv(data.frame(locus = names(sc), score = as.numeric(sc)),
          args[2], row.names = FALSE)
'''


def rank_relief(Xtr, ytr, loci, tmpdir, rlib):
    """Relief-F 排名。调 R 的 CORElearn，estimator 取 ReliefFbestK。

    ReliefFbestK 不是一个固定大小的近邻集合，而是「所有可能的邻居数都试一遍、
    每个位点取它在各邻居数下的最高分」，邻居数由库内部搜。
    故这里没有 n_neighbors 这个参数可调，也就没有「选几个邻居」这一层网格。

    这个口径有一个须交代的取向：邻居数越小，单个邻居的 diff 权重 1/k 越大、
    分数的抽样方差越大，取 max 于是系统性偏向小邻居数。
    这是该 estimator 的定义决定的，不是实现问题，与水牛文献所用的完全相同。
    后果是 Relief-F 的排名与「某个固定邻居数下」的排名并不一致，头部尤其不同。

    RELIEF_R 里的 read.csv 必须带 check.names = FALSE：
    位点 ID 形如 Cluster-219541.64309:2761，默认的参数名检查会把它改写成
    Cluster.219541.64309.2761，返回的位点名就对不上输入的列名了。
    """
    # 值只有 0/1/2（众数填补后无缺失），转 int8 写出去，CSV 体积比按 float 写小一个量级。
    Xd = pd.DataFrame(Xtr.astype(np.int8), columns=loci)
    Xd[".class"] = ytr                   # R 侧 read.csv 后按列名取用
    in_csv = os.path.join(tmpdir, "relief_in.csv")
    out_csv = os.path.join(tmpdir, "relief_out.csv")
    rfile = os.path.join(tmpdir, "relieff_bestk.R")
    Xd.to_csv(in_csv, index=False)
    with open(rfile, "w") as fh:
        fh.write(RELIEF_R)
    # --max-ppsize 是必须的：17,728 项的公式会让 R 默认的指针保护栈溢出，
    # 报错是「protect()：防护堆叠上溢」，从报错信息看不出跟栈大小有关。
    # 不写 check=True：R 的报错在 stderr 里，要接住它才能给出可读的原因。
    p = subprocess.run(["Rscript", "--max-ppsize=500000", rfile,
                        in_csv, out_csv, rlib], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("Rscript 调 CORElearn 失败：\n" + p.stderr[-3000:])
    r = pd.read_csv(out_csv)
    if len(r) != len(loci):
        raise RuntimeError(f"CORElearn 返回 {len(r)} 个位点，输入是 {len(loci)} 个")
    # 先按位点 ID 对齐回输入的列顺序再排序。R 写出的行序不保证与输入列序一致，
    # 直接按返回的行序 argsort，列下标就会错位到别的位点上——错位不报错，
    # 只让成绩变成噪声，属最难倒查的一类 bug。
    sc = r.set_index("locus")["score"].reindex(loci).to_numpy(float)
    if np.isnan(sc).any():
        raise RuntimeError(f"CORElearn 有 {int(np.isnan(sc).sum())} 个位点没给出分数")
    # argsort 默认升序，取负号即降序；kind="stable" 保证分数并列时按列下标原序，
    # 使排序结果可复现（否则并列的顺序依赖底层实现）。
    return np.argsort(-sc, kind="stable")


def _plink_fst(bfile, ids, prov, train_idx, groups, outprefix):
    """调 PLINK 算指定位点集合上的位点 FST，返回 (位点 ID 列表, FST 值列表)。

    PLINK 的 --fst 实现的是 Weir & Cockerham (1984) 的 FST 估计量：
    对每个位点，按分组算出等位基因频率的标准化方差，
        FST = σ²_p / [p̄(1−p̄)]
    其中 σ²_p 是各群体等位基因频率的方差、p̄ 是总平均频率。
    单态位点（p̄ 为 0 或 1）分母为 0，PLINK 不给值，故该位点没有 FST 估计。
    """
    keep_f = outprefix + ".keep"
    within_f = outprefix + ".within"
    # --keep 文件：两列 FID IID。PLINK 要求两列，只写一列会被静默忽略（不报错也不筛人）。
    # 只写训练折里属于 groups 的个体，这是防泄漏的关键一步：
    # FST 由等位基因频率算出，测试折个体的基因型绝不能进入频率估计。
    with open(keep_f, "w") as f:
        for i in train_idx:
            if prov[i] in groups:
                f.write(f"0\t{ids[i]}\n")
    # --within 文件：三列 FID IID CLUSTER。第三列即分组标签，PLINK 按它分群体算 FST。
    with open(within_f, "w") as f:
        for i in train_idx:
            if prov[i] in groups:
                f.write(f"0\t{ids[i]}\t{prov[i]}\n")
    # capture_output=True 把 PLINK 的输出接住，不让它刷屏；
    # check=True 让 PLINK 退出码非零时直接抛异常——否则 PLINK 出错时会静默产出一个空结果，
    # 后面解析不到任何位点，表现为「这一折 FST 全空」，很难倒查。
    subprocess.run(["plink", "--bfile", bfile,
                    "--allow-extra-chr",    # 本数据的位点 ID 是 Cluster-219541.x:y 形式，
                                            # 不是标准染色体编号，不加这个 PLINK 直接报错退出
                    "--allow-no-sex",       # fam 文件没有性别列，不加会被当成错误
                    "--keep", keep_f,
                    "--fst",                # 打开 FST 计算
                    "--within", within_f,   # 声明分组；--fst 必须配 --within
                    "--out", outprefix], capture_output=True, check=True)
    snp, val = [], []
    # .fst 输出为题头一行 + 每行一个位点，五列：CHR SNP POS NMISS FST
    with open(outprefix + ".fst") as fh:
        next(fh)                            # 跳过题头
        for line in fh:
            c = line.split()
            if len(c) >= 5:                 # 列数不足说明该行不完整，丢弃
                snp.append(c[1])            # 第 2 列是位点 ID，用于与我们的列对齐
                val.append(float(c[4]))     # 第 5 列是 FST 值
    # NMISS 列（第 4 列）这里不用：本数据的缺失格在 PLINK 里就是没有基因型，
    # PLINK 会自行按有基因型的个体算频率，不需要额外处理。
    return snp, val


def rank_fst(bfile, ids, prov, train_idx, loci, groups, outprefix):
    """把 PLINK 的位点级 FST 结果铺成与列顺序一致的完整排名。

    PLINK 只对「有估计值」的位点输出行（单态位点没有），故结果是不定长的，
    要按位点 ID 映射回 17728 个列位置，缺的记 NaN，再统一降序。
    """
    snp, val = _plink_fst(bfile, ids, prov, train_idx, groups, outprefix)
    pos = {s: j for j, s in enumerate(loci)}    # 位点 ID → 列下标，字典查找是 O(1)
    full = np.full(len(loci), np.nan)           # 先全填 NaN，再逐个覆盖有值的
    hit = 0                                     # 记录有多少个位点拿到了估计值，供日志核对
    for s, v in zip(snp, val):
        j = pos.get(s)
        if j is not None:                       # 位点 ID 理论上必在表内，防御性判断
            full[j] = v
            hit += 1
    # 降序排列。NaN 没有大小，直接 argsort 会把它排到末尾（升序尾部），反转后跑到最前，
    # 所以先把 NaN 换成 -inf 再排序：升序时它们沉底，反转后仍在最后。
    # kind="stable" 让并列的位点按列下标原序，保证可复现。
    order = np.argsort(np.where(np.isnan(full), -np.inf, full), kind="stable")[::-1]
    return order, int(hit)                      # [::-1] 把升序翻成降序


# ==================================================================== 分类器

def make_clf(name, params):
    """按名字与参数字典构造分类器。params 里缺的键用默认值补。"""
    if name == "SVM":
        # 线性核：水牛文献用 e1071::svm 的线性分类器，此处对齐。
        # random_state 对线性 SVC 无实际影响（无随机成分），写上是为了统一口径。
        return SVC(kernel="linear", C=params.get("C", 1.0), random_state=0)
    if name == "RF":
        return RandomForestClassifier(n_estimators=params.get("n_estimators", 300),
                                      max_features=params.get("max_features", "sqrt"),
                                      n_jobs=-1,              # 用满所有核，单次拟合极快
                                      random_state=0)
    if name == "KNN":
        # n_neighbors 是要调的参数；距离度量用默认的欧氏距离、邻域权重用默认的均匀权重。
        # 不做标准化：各位点都是同一套 0/1/2 剂量编码，量纲相同，
        # 而标准化会把「位点的多态程度」也一并抹掉，反而改变距离的含义。
        return KNeighborsClassifier(n_neighbors=params.get("n_neighbors", 5))
    if name == "NB":
        # GaussianNB 假定每个位点在各类内服从正态分布，与水牛所用 e1071::naiveBayes
        # 对数值型预测变量的处理一致。var_smoothing 是方差平滑项，
        # 加在各类方差上防止某个位点在某类内方差为 0 导致概率坍缩，作用相当于拉普拉斯平滑。
        return GaussianNB(var_smoothing=params.get("var_smoothing", 1e-9))
    raise ValueError(name)      # 名字写错时立刻报错，而不是静默返回 None


# 四个分类器的调参网格。取值范围的依据：
#   SVM 的 C 跨 4 个数量级，覆盖「几乎不正则化」到「强正则化」；
#   RF 的树数与每次分裂特征数（对应 R 里 tuneRF 调的 ntree 与 mtry）；
#   KNN 的近邻数取奇数，避免二选一投票时的平票；
#   NB 的方差平滑同样跨数量级。
GRIDS = {
    "SVM": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "RF": [{"n_estimators": n, "max_features": m}
           for n in (300, 500) for m in ("sqrt", "log2", None)],
    "KNN": [{"n_neighbors": k} for k in (1, 3, 5, 7, 9, 11)],
    "NB": [{"var_smoothing": s} for s in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5)],
}


def scores_of(clf, X):
    """取分类器的连续打分矩阵，形状 (个体数, 类别数)，用于算 AUC。

    AUC 只需要「打分能排序」，不需要是概率，所以优先用 predict_proba（有则自然可用），
    没有就退回 decision_function。SVC 默认 probability=False，没有 predict_proba，
    走的是 decision_function；RF、KNN、NB 走 predict_proba。
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return clf.decision_function(X)


def tune_and_fit(name, Xtr, ytr, inner_folds):
    """在训练集内部调参，返回 (用最佳参数在整个训练集上重训的模型, 最佳参数, 内层成绩)。

    inner_folds 是训练集内部划出的 5 折。每个参数组合都在内层折上「训练—验证」一遍，
    用内层平均准确率评价。最佳参数再在全部训练集上重训一次供预测用——
    这一步是必要的：内层评价时每个模型只见过 4/5 的训练集，
    直接用那批模型预测测试折是浪费数据；用最佳参数在全部训练集上重训，才能用满信息。
    """
    best, best_acc = None, -1.0
    for params in GRIDS[name]:                # 遍历该分类器的全部参数组合
        accs = []
        for itr, ite in inner_folds:          # 内层 5 折
            # make_clf(...).fit(...) 每次新建模型，不复用上一组参数的拟合结果，
            # 避免参数之间串味。
            clf = make_clf(name, params).fit(Xtr[itr], ytr[itr])
            accs.append(accuracy_score(ytr[ite], clf.predict(Xtr[ite])))
        a = float(np.mean(accs))              # 该参数组合的内层成绩
        # 严格大于才替换，于是并列时取网格里靠前的组合，结果确定。
        if a > best_acc:
            best, best_acc = params, a
    return make_clf(name, best).fit(Xtr, ytr), best, best_acc


# ==================================================================== 网格

def eval_combo(Xraw, miss, y, folds, cols_of_fold, name, ncls, seed=0):
    """跑一个「选择方法 × k × 分类器」组合，返回折外预测、打分、每折 ACC、选中参数。

    cols_of_fold 是一个函数：传折号、回该折要用的列下标。这样把「三套方法各自怎么选列」
    的差异挡在外面，网格这层代码对三种方法完全一样。

    传进来的是未填补的原始矩阵 Xraw 与它的缺失掩膜 miss，
    填补在每折内部做，众数只由该折的训练个体算出。
    """
    n = len(y)
    pred = np.empty(n, dtype=int)         # 折外预测，长度 180
    score = np.zeros((n, ncls))           # 折外打分，180×6
    fold_acc, params_used = [], []
    for f, (tr, te) in enumerate(folds):  # tr/te 是训练/测试个体的下标
        cols = cols_of_fold(f)            # 该折的列（位点）下标
        # 众数只由训练折个体算，测试折的缺失格用同一套众数填。
        # 若两边各算各的，测试集就通过「填成什么值」间接参与了训练。
        modes = locus_modes(Xraw[tr], miss[tr])
        Xtr = mode_impute(Xraw[tr], miss[tr], modes)
        Xte = mode_impute(Xraw[te], miss[te], modes)
        # 内层 5 折。每折都重新构造一次，虽然随 seed 固定、结果与上一折相同，
        # 但避免把生成器存起来复用，也让「内层折只由训练集决定」这件事在代码上一目了然。
        inner = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(Xtr, y[tr]))
        # 切列是在已切好的行子集上再做一次列索引，仍是内存操作，没有磁盘读写。
        clf, params, _ = tune_and_fit(name, Xtr[:, cols], y[tr], inner)
        # 注意：预测时用的是该折自己选出的列，各折的列可以完全不同。
        pred[te] = clf.predict(Xte[:, cols])
        score[te] = scores_of(clf, Xte[:, cols])
        fold_acc.append(accuracy_score(y[te], pred[te]))
        params_used.append(params)        # 记下该折实际选中的参数，供核对调参是否稳定
    # 收尾时 pred/score 已按 te 填满。每个个体恰好出现在一个测试折里，
    # 故填充不重叠、不漏——这就是「折外预测拼合」的做法，
    # 拼完是一条 180 长的向量，可以当成「每个个体都被一个没见过它的模型预测过」的结果。
    return pred, score, fold_acc, params_used


def macro_ovr_auc(y, score):
    """逐类算「该类对其余」的 AUC 再宏平均。

    不用 sklearn 的 multi_class='ovr'：那个入口要求打分是概率（各类和为 1），
    而 SVM 给的是 decision_function，不满足该前提会直接报错。
    逐类手算对任何连续打分都成立，且口径就是宏平均 OVR AUC：
        AUC_c = P(随机取一个真实为 c 的个体，其 c 类打分 > 随机取一个非 c 的个体的 c 类打分)
    数值上等于把两组打分混在一起排序后的秩和统计量，再对 6 个类取平均。

    score.shape[1] 即类别数，逐列（逐类）计算。
    """
    aucs = [roc_auc_score((y == c).astype(int), score[:, c]) for c in range(score.shape[1])]
    return float(np.mean(aucs))


def clopper_pearson(c, n, alpha=0.05):
    """精确二项区间（Clopper-Pearson）。c 为判对个数，n 为个体总数。

    参数名用 c 而非 k，避免与网格里的位点数 k 混淆。

    区间由 Beta 分布的分位点给出，不用 bootstrap：对个体做有放回重抽样造不出新的
    错误类型，接近满分时区间会被压至零宽，窄区间反映的是重抽样机制而非性能的稳定性。

    局限须与结果一并写明：该区间只刻画「180 个个体里抽到的这一批」的二项抽样误差
    （相当于把个体当作独立同分布的抽样单元），既不含特征选择与调参带来的乐观，
    也不反映跨采集种群的泛化能力，不能当作成绩的置信区间来读。
    """
    from scipy.stats import beta
    lo = 0.0 if c == 0 else float(beta.ppf(alpha / 2, c, n - c + 1))
    hi = 1.0 if c == n else float(beta.ppf(1 - alpha / 2, c + 1, n - c))
    return lo, hi


def summarize(y, pred, score, ncls):
    """在拼合后的折外结果上一次算出全部指标。"""
    acc = accuracy_score(y, pred)                             # 准确率：判对的比例
    lo, hi = clopper_pearson(int(round(acc * len(y))), len(y))
    out = {"ACC": acc,
           "ACC_lo": lo, "ACC_hi": hi,                        # ACC 的 Clopper-Pearson 区间
           "AUC": macro_ovr_auc(y, score),                    # 宏平均 OVR AUC，不依赖判定阈值
           "macro_recall": recall_score(y, pred, average="macro")}   # 逐类召回的平均
    return out


def fold_stability(lists, nfolds):
    """折间一致性指标，回答「同一个 k 下各折选出的位点有多像」。

    设计方案第十一节要求的内容全部由这几个数派生：
        inter       各折集合的交集大小（= 冻结面板的规模）
        union       各折集合的并集大小
        jaccard     两两折之间 Jaccard 系数的平均，|A∩B| / |A∪B|
        core_80     入选率 >= 80% 的位点数，即「多数折都选中」那一档的规模
    交集随折数增加迅速塌缩，只看交集会低估可复现性；并集与 Jaccard 不随折数
    单调塌缩，三者并列才能看清。core_80 是介于两者之间的一个稳健口径。
    """
    inter = set.intersection(*lists) if lists else set()
    union = set.union(*lists) if lists else set()
    jac = []
    for i in range(len(lists)):
        for j in range(i + 1, len(lists)):
            u = len(lists[i] | lists[j])
            jac.append(len(lists[i] & lists[j]) / u if u else 0.0)
    # 逐位点统计入选折数：先并入一个全集，再数每个位点出现在几个折里
    cnt = {}
    for s in lists:
        for x in s:
            cnt[x] = cnt.get(x, 0) + 1
    core = sum(1 for v in cnt.values() if v >= np.ceil(0.8 * nfolds))
    return {"inter": len(inter), "union": len(union),
            "jaccard": float(np.mean(jac)) if jac else 0.0, "core_80": core}


# ==================================================================== 主流程

def main():
    ap = argparse.ArgumentParser()
    # 三个输入都在包内，默认值即正式用法。
    ap.add_argument("--dose", default="02_intermediate/dose_unfilled.csv")
    ap.add_argument("--labels", default="02_intermediate/labels.csv")
    ap.add_argument("--bfile", default="02_intermediate/snp")
    # CORElearn 已从 CRAN 归档下架，副本装在包内的 R 库；传给 Rscript 后由 RELIEF_R 追加到 .libPaths()。
    # 留空则用系统默认库路径（仅在 CORElearn 装在默认库里时才成立）。
    ap.add_argument("--rlib", default="03_r_env/R_libs")
    ap.add_argument("--out", default="05_results")
    ap.add_argument("--folds", type=int, default=10)          # 外层折数
    # k 网格取水牛 9 档里的 6 档，去掉 3000/5000/10000：本数据只有 17728 个位点，
    # 而那篇有全基因组 WGS 打底，取到 10000 仍有富余。
    ap.add_argument("--k", default="100,250,500,800,1000,2000")
    ap.add_argument("--methods", default="mRMR,Relief-F,FST")
    ap.add_argument("--classifiers", default="SVM,RF,KNN,NB")
    # kmax 必须 ≥ 最大的 k，否则截取时会越界（实际是取不到足够的位点）。
    # 它只影响 mRMR 的耗时：其余两种方法本来就返回完整长度的排名。
    ap.add_argument("--kmax", type=int, default=2000, help="排名保留的长度")
    ap.add_argument("--jobs", type=int, default=8, help="mRMR 的 n_jobs")
    # FST 折内候选池的取法：half 每段取 k/2，并集约 k，使三套方法的面板规模可比；
    # full 每段取 k，并集约 2k，即水牛原样的做法，但此时同名 k 下 FST 实际多喂一倍位点。
    ap.add_argument("--fst-split", choices=("half", "full"), default="half",
                    help="half: 每段取 k/2，使面板规模与其他方法可比；full: 每段取 k")
    ap.add_argument("--no-panel", action="store_true", help="跳过冻结面板与面板重跑")
    ap.add_argument("--seed", type=int, default=0)            # 全程统一随机种子
    args = ap.parse_args()

    # 逗号分隔的字符串转成列表
    ks = [int(s) for s in args.k.split(",")]
    methods = args.methods.split(",")
    classifiers = args.classifiers.split(",")
    os.makedirs(args.out, exist_ok=True)                      # 输出目录，已存在不报错
    rdir = os.path.join(args.out, "rankings")                 # 排名缓存目录
    tmpdir = os.path.join(args.out, "_tmp")                   # PLINK 的临时文件目录
    os.makedirs(rdir, exist_ok=True)
    os.makedirs(tmpdir, exist_ok=True)

    Xraw, miss, y, ids, loci, cls_names, prov = read_inputs(args.dose, args.labels)
    nloc = Xraw.shape[1]             # 17728，位点数；后面多个函数要按它生成完整长度的排名
    ncls = len(cls_names)            # 6
    # 这里不预先做一次全局填补。填补一律挪进折内，众数只由该折的训练个体算出，
    # 否则测试折个体会通过「众数落在哪一档」影响填进训练集的值。
    log(f"众数填补改为折内进行，共 {args.folds} 折各填一次")

    # FST 两段式的分组。排名 A 用全部省份（六个省之间），排名 B 用除云南外的省份。
    # 依据是实测的两两 FST：云南与其余五省 0.53–0.58，其余五省彼此 0.08–0.16，
    # 相差一个数量级。不分段的话「云南对其余」这类位点会把总排名的前列占满，
    # 「区分五省之间」的位点被整体挤掉。
    provinces = sorted(set(prov))
    others = [p for p in provinces if p != "云南"]
    if "云南" not in provinces:
        # 分组依据不成立时直接停，不要带着错的假设一路跑完再发现结论没法解释。
        raise SystemExit("标签里没有云南，FST 两段式的分段依据不成立，请改分组")

    # 外层划分。StratifiedKFold 保证每折的类别构成与全体一致：
    # 6 类各 30 个，10 折后每折测试集 18 个、每类恰好 3 个。
    # shuffle=True 才有随机性，random_state=seed 让它可复现。
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                 .split(Xraw, y))
    log(f"{args.folds} 折分层划分完成，各折测试集大小 "
        f"{sorted(Counter([len(te) for _, te in folds]).items())}")

    # ---------------------------------------------------------- 排名（按折缓存）
    # 结构：ranks[方法名][折号] = {"A": 完整排名数组, "B": ...}。FST 有两个排名，其余只有一个。
    ranks = {m: [None] * args.folds for m in methods}
    for f, (tr, _) in enumerate(folds):        # 只需要训练折的下标
        need = [m for m in methods if ranks[m][f] is None]
        if not need:
            continue
        for m in need:
            cache = os.path.join(rdir, f"fold{f}_{m}.npy")
            pair = os.path.join(rdir, f"fold{f}_{m}_B.npy")
            # 缓存命中则直接读入。这一步是为断点续跑服务的：
            # 单折 mRMR 约 970 秒、十折近 3 小时，中途中断后不该从头再来。
            # 判 FST 时要两个文件都在，因为它有两段排名。
            if os.path.exists(cache) and (m != "FST" or os.path.exists(pair)):
                ranks[m][f] = {"A": np.load(cache),
                               **({"B": np.load(pair)} if m == "FST" else {})}
                log(f"折{f} {m} 从缓存读入")
                continue
            t = time.time()                    # 记录该次排名的耗时
            if m == "mRMR":
                # mode_impute 只传训练折个体，众数便只由这批人算出；
                # y[tr] 同理——标签与基因型必须取自同一批人。
                o = rank_mrmr(mode_impute(Xraw[tr], miss[tr]), y[tr], nloc, args.kmax, args.jobs)
                ranks[m][f] = {"A": o}
                np.save(cache, o)
            elif m == "Relief-F":
                # 同样只传训练折：喂进 attrEval 的是该折众数填补后的矩阵。
                # 三种选择方法必须看到同一份输入（都是折内填补后的矩阵），
                # 否则「谁选的位点更好」分不清是方法差异还是输入差异。
                o = rank_relief(mode_impute(Xraw[tr], miss[tr]), y[tr],
                                loci, tmpdir, args.rlib)
                ranks[m][f] = {"A": o}
                np.save(cache, o)
            elif m == "FST":
                # 排名 A：六个省之间。排名 B：除云南外五省之间。
                # 两次都传 tr（训练折个体），测试折个体不进 FST 的频率估计。
                oA, hA = rank_fst(args.bfile, ids, prov, tr, loci, provinces,
                                  os.path.join(tmpdir, f"fstA_f{f}"))
                oB, hB = rank_fst(args.bfile, ids, prov, tr, loci, others,
                                  os.path.join(tmpdir, f"fstB_f{f}"))
                ranks[m][f] = {"A": oA, "B": oB}
                np.save(cache, oA)
                np.save(pair, oB)
                # 打印有多少位点拿到了 FST 估计值。单态位点无估计值，
                # 这个数明显偏低说明分组间分化太小，属需要留意的信号。
                log(f"折{f} FST 排名 A {hA}/{nloc} 个位点有估计值，B {hB}/{nloc}")
            else:
                raise ValueError(m)
            log(f"折{f} {m} 排名完成，用时 {time.time()-t:.1f}s")

    def cols_of(method, f, k):
        """给定方法、折号、k，返回该组合实际使用的列下标列表。"""
        r = ranks[method][f]
        if method == "FST":
            # 每段取 per 个位点再取并集。half 时每段 k/2，并集约 k；
            # 两段会有少量重合，所以并集通常略小于 k。
            per = k // 2 if args.fst_split == "half" else k
            # dict.fromkeys 去掉两段的重合项，同时保持顺序（先 A 后 B），
            # 于是「FST 的 top-k」有确定含义。
            return list(dict.fromkeys(list(r["A"][:per]) + list(r["B"][:per])))
        return list(r["A"][:k])     # mRMR 与 Relief-F 只有一个排名，直接截前 k 个

    # ---------------------------------------------------------- 网格
    rows, fold_rows = [], []       # 汇总成两张表，最后写成 CSV
    detail = {}                    # 每个组合的完整结果，供挑出最佳后取用
    log("开始网格：" + " × ".join([
        f"{len(methods)} 种选择方法", f"{len(ks)} 个 k", f"{len(classifiers)} 个分类器"]))
    for method in methods:
        for k in ks:
            # 记录该组合在 10 折上的实际面板规模。FST 的并集略小于 k，故这里取平均后打印，
            # 让「面板列」反映真实位点数，而不是名义的 k。
            sizes = [len(cols_of(method, f, k)) for f in range(args.folds)]
            # 折间一致性只由「各折选了哪些位点」决定，与分类器无关，
            # 故在 (method, k) 这一层算一次，写进该 k 下每一行。
            stab = fold_stability([set(cols_of(method, f, k)) for f in range(args.folds)],
                                  args.folds)
            for name in classifiers:
                t = time.time()
                # lambda 把 (method, k) 固定住，传进去的仍是「一个折号 → 一组列」的函数。
                pred, score, facc, pused = eval_combo(
                    Xraw, miss, y, folds, lambda f: cols_of(method, f, k), name, ncls, args.seed)
                s = summarize(y, pred, score, ncls)
                rows.append({"method": method, "k": k, "classifier": name,
                             "panel_size": int(np.mean(sizes)),
                             # 第十一节的稳定性指标，随每个组合一并落盘
                             "fold_inter": stab["inter"], "fold_union": stab["union"],
                             "fold_jaccard": round(stab["jaccard"], 4),
                             "core_80": stab["core_80"],
                             "ACC": s["ACC"], "ACC_lo": s["ACC_lo"], "ACC_hi": s["ACC_hi"],
                             "AUC": s["AUC"],
                             "macro_recall": s["macro_recall"],
                             # 只记第一折的参数做示例；逐折参数完整记在下面的 fold_rows 之外的
                             # 内存结构 detail 里，控制台上也会打印最佳组合的逐折参数。
                             "params": str(pused[0])})
                for f, a in enumerate(facc):
                    fold_rows.append({"method": method, "k": k, "classifier": name,
                                      "fold": f, "ACC": a})
                # 存下完整结果，后面挑出最佳组合时不必重跑
                detail[(method, k, name)] = (pred, score, facc, pused)
                log(f"{method:>14} k={k:<5} {name:>4}  面板 {int(np.mean(sizes)):>5}  "
                    f"ACC {s['ACC']:.3f}  AUC {s['AUC']:.3f}  用时 {time.time()-t:.1f}s")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "grid_results.csv"), index=False)
    pd.DataFrame(fold_rows).to_csv(os.path.join(args.out, "per_fold_accuracy.csv"),
                                   index=False)

    # 控制台成绩表。按选择方法分块，行是 k，列是分类器，格子里是 ACC/AUC。
    print("\n" + "=" * 78)
    print(f"网格成绩（{args.folds} 折折外预测拼合后计算）")
    print("=" * 78)
    for method in methods:
        sub = res[res.method == method]
        print(f"\n{method}")
        print(f"  {'k':>5} {'面板':>6} " + " ".join(f"{c:>13}" for c in classifiers))
        for k in ks:
            row = [f"  {k:>5}"]
            sz = sub[(sub.k == k)].panel_size.iloc[0] if len(sub[sub.k == k]) else 0
            row.append(f"{sz:>6}")
            for name in classifiers:
                r = sub[(sub.k == k) & (sub.classifier == name)]
                row.append(f"{r.ACC.iloc[0]:>6.3f}/{r.AUC.iloc[0]:<6.3f}" if len(r)
                           else f"{'--':>13}")
            print(" ".join(row))
    print("\n  每格为 ACC/AUC。面板列为该组合实际使用的平均位点数。")

    # 稳定性表（第十一节）。只按 (方法, k) 看，与分类器无关。
    # 交集随折数增加会迅速塌缩，单看它会低估可复现性，故并列给出并集与 Jaccard。
    print("\n" + "=" * 78)
    print(f"折间一致性（{args.folds} 折，与分类器无关）")
    print("=" * 78)
    print(f"  {'方法':>14} {'k':>5} {'折内规模':>8} {'交集':>6} {'并集':>6} "
          f"{'Jaccard':>8} {'入选≥80%':>9}")
    for method in methods:
        for k in ks:
            r = res[(res.method == method) & (res.k == k)].iloc[0]
            print(f"  {method:>14} {k:>5} {r.panel_size:>8} {r.fold_inter:>6} "
                  f"{r.fold_union:>6} {r.fold_jaccard:>8.3f} {r.core_80:>9}")

    # ---------------------------------------------------------- 最佳组合
    # 按 ACC 降序、AUC 作次选，取第一行。sort_values 是稳定排序，
    # 于是 ACC 与 AUC 都相同时保留网格中的原顺序。
    best = res.sort_values(["ACC", "AUC"], ascending=False).iloc[0]
    bmethod, bk, bclf = best.method, int(best.k), best.classifier
    print("\n" + "=" * 78)
    print(f"最佳组合 {bmethod} + k={bk} + {bclf}   ACC {best.ACC:.3f}  AUC {best.AUC:.3f}")
    print(f"  ACC 的 Clopper-Pearson 95% 区间 [{best.ACC_lo:.3f}, {best.ACC_hi:.3f}]"
          f"（只含个体抽样误差，不含选位点与调参的乐观）")
    print("=" * 78)
    # 从 detail 里取回该组合已算好的结果，不重跑
    pred, _, facc, pused = detail[(bmethod, bk, bclf)]
    # 混淆矩阵：第 i 行第 j 列是「真实为第 i 类、被判为第 j 类」的个体数
    cm = confusion_matrix(y, pred)
    print(f"\n逐折 ACC: " + " ".join(f"{a:.3f}" for a in facc)
          + f"   均值 {np.mean(facc):.3f}  最低 {np.min(facc):.3f}")
    # 召回率 = 该类判对的人数 / 该类总人数，即对角线除以行和。
    # 类别均衡时（每类 30 个）召回率与精确率的分母相同，故只报召回即可。
    print(f"逐类召回: " + "  ".join(
        f"{n} {r:.3f}" for n, r in zip(cls_names, cm.diagonal() / cm.sum(1))))
    print(f"\n混淆矩阵（行=真实，列=预测）")
    print("        " + "".join(f"{n:>7}" for n in cls_names))
    for i, n in enumerate(cls_names):
        print(f"{n:>6}  " + "".join(f"{v:>7}" for v in cm[i]))
    # 打印逐折选中的参数：若各折差异很大，说明调参结果不稳定，
    # 该组合的成绩对参数敏感，报告时值得说明。
    print(f"\n各折选中参数: {[str(p) for p in pused]}")

    pd.DataFrame(cm, index=cls_names, columns=cls_names).to_csv(
        os.path.join(args.out, "confusion_matrix_best.csv"))
    pd.DataFrame({"class": cls_names,
                  "recall": cm.diagonal() / cm.sum(1),
                  "support": cm.sum(1)}).to_csv(
        os.path.join(args.out, "per_class_recall_best.csv"), index=False)

    if args.no_panel:
        log("已跳过冻结面板")
        return

    # ---------------------------------------------------------- 冻结面板
    print("\n" + "=" * 78)
    print(f"冻结面板：{bmethod} + k={bk}，各折交集")
    print("=" * 78)
    # 把各折选出的位点集合取交集。取交集而非并集的理由：
    # 并集会被单折的偶然选择放大，交集只保留所有折共同认可的位点，
    # 与「跨折可复现」这一目标一致。
    lists = [set(cols_of(bmethod, f, bk)) for f in range(args.folds)]
    panel = sorted(set.intersection(*lists))     # 排序仅为让输出稳定
    print(f"各折规模 {[len(s) for s in lists]}  →  交集 {len(panel)} 个位点")
    # 把面板的位点 ID（不是下标）写成 CSV，这份文件可以直接拿去做检测设计
    pd.DataFrame({"locus": [loci[j] for j in panel]}).to_csv(
        os.path.join(args.out, "frozen_panel.csv"), index=False)
    if len(panel) == 0:
        log("交集为空，无法冻结面板")
        return
    # 同时落盘各折各自的选中清单，供检查折间一致性
    for f in range(args.folds):
        pd.DataFrame({"locus": [loci[j] for j in sorted(lists[f])]}).to_csv(
            os.path.join(args.out, f"selected_fold{f}.csv"), index=False)

    # 用冻结的面板重跑一遍。此处 cols_of_fold 对任何折都返回同一个 panel，
    # 即所有折用完全相同的列——这才是「固定面板」的含义。
    t = time.time()
    ppred, pscore, pfacc, _ = eval_combo(
        Xraw, miss, y, folds, lambda f: panel, bclf, ncls, args.seed)
    ps = summarize(y, ppred, pscore, ncls)
    log(f"面板重跑完成，用时 {time.time()-t:.1f}s")
    print(f"\n面板 {len(panel)} 个位点 + {bclf}：ACC {ps['ACC']:.3f}  AUC {ps['AUC']:.3f}  "
          f"macro 召回 {ps['macro_recall']:.3f}")
    print(f"逐折 ACC: " + " ".join(f"{a:.3f}" for a in pfacc))
    # 这句必须印出来：面板由全部折的选择结果汇总而来，每个个体都曾参与面板构建，
    # 故该成绩是乐观估计，不能当作独立验证的上界。
    print("（该成绩为乐观估计：面板由全部折的选择结果汇总而来，"
          "每个个体都曾参与面板构建）")
    pd.DataFrame([{"panel_size": len(panel), "method": bmethod, "k": bk,
                   "classifier": bclf, **ps}]).to_csv(
        os.path.join(args.out, "panel_result.csv"), index=False)
    pd.DataFrame({"fold": range(args.folds), "ACC": pfacc}).to_csv(
        os.path.join(args.out, "panel_per_fold.csv"), index=False)

    log(f"全部完成，结果写在 {args.out}/")


# 只有直接运行本文件时才执行 main()；被 import 时不会自动跑，
# 便于把里面的函数（如 locus_modes、macro_ovr_auc）单独拿来测试。
if __name__ == "__main__":
    main()
