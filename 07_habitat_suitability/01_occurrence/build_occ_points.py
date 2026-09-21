#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建适生区建模用的 occurrence 点表。

三个来源合并后按 10 km 做 haversine 贪心稀疏：
  1. 自有 18 个采样种群 (01_raw_data/SAMPLE.csv)          —— 实测，6 位小数
  2. 文献种群表 (Sci Rep 6:25031 Table 1, ISSR 研究)       —— 弧分取整
  3. GBIF 凭证记录 (taxonKey 见脚本常量 GBIF_TAXON_KEY, country=CN)          —— 标本精确坐标

已排除的源：CVH/NPSRC (633 条记录但保护物种坐标屏蔽, 只有省)、
教学标本平台 (mnh.scu.edu.cn, 0.1° 县质心约 11 km 误差)。

稀疏时按优先级保留：own > literature > GBIF，即同一 10 km 半径内
若有实测点则优先保留实测点。
"""

import csv
import json
import math
import pathlib
import re
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent.parent                       # package root
SAMPLE_CSV = PKG / "01_raw_data" / "SAMPLE.csv"
GBIF_CACHE = HERE / "gbif_raw.json"
OUT_CSV = HERE / "occ_points.csv"

GBIF_TAXON_KEY = 2685300                       # GBIF backbone taxon key of the study species
THIN_KM = 10.0
# 中国境内矩形，与建模研究区一致
BBOX = dict(lat_min=18.0, lat_max=54.0, lon_min=73.0, lon_max=135.0)

# 优先级：数字越小越优先保留
PRIORITY = {"own": 1, "literature": 2, "gbif": 3}

# 迁地保护 / 园林栽培个体不是野生分布点，必须剔除。
# 已剔除：GBIF 4450874850 (locality="Wuhan Botanical Garden", 2015, BOLD wh197)。
GARDEN_KW = re.compile(
    r"植物园|树木园|公园|校园|栽培|迁地|引种|"
    r"botanic|arboretum|garden|park|campus|glasshouse|greenhouse|horticult",
    re.I)

# ---------------------------------------------------------------------------
# 文献种群表：Sci Rep 6:25031 (2016) Table 1，ISSR 研究，22 个种群
# 字段：(缩写, 省, 地点, 纬度度分, 经度度分, 海拔m)
# published=True 表示该行经纬度自洽可信；False 表示原表串行错误、已剔除
# ---------------------------------------------------------------------------
LITERATURE = [
    ("CQjfs", "Chongqing", "Jin Fo Shan",        29,  1, 107,  5,  650, True),
    ("CQzs",  "Chongqing", "Liang Ping Zhu Shan",30, 39, 107, 32,  530, True),
    ("YNdws", "Yunnan",    "Da Wei Shan",        27, 37, 113, 52, 2109, False),  # 113.9E 在湘赣，且 27.6N 非云南
    ("SCems", "Sichuan",   "E Mei Shan",         29, 33, 103, 23,  980, True),
    ("JXaf",  "Jiangxi",   "An Fu",              27, 13, 114, 11,  420, True),
    ("JXyf",  "Jiangxi",   "Yi Feng",            28, 37, 114, 54,  741, True),
    ("JXxs",  "Jiangxi",   "Xiu Shui",           28, 46, 114, 46,  307, True),
    ("HNdh",  "Hunan",     "De Hang",            28, 21, 109, 35,  429, True),
    ("HNhps", "Hunan",     "Hu Ping Shan",       29, 57, 110, 38,  495, True),
    ("HNhl",  "Hunan",     "Hui Long",           28, 54, 110, 10,  399, True),
    ("HNym",  "Hunan",     "Yong Mao",           28, 58, 110, 18,  534, True),
    ("HNhng", "Hunan",     "Ha Ni Gong",         28, 56, 109, 57,  305, True),
    ("GZwyh", "Guizhou",   "Wu Yang He",         27,  3, 108, 18,  550, True),
    ("GZdsh", "Guizhou",   "Da Sha He",          29,  4, 107, 24,  700, True),
    ("GZfjs", "Guizhou",   "Fan Jing Shan",      27, 49, 108, 36,  860, True),
    ("GDdxs", "Guangdong", "Dan Xia Shan",       25,  3, 113, 45,  800, True),
    ("HBcy",  "Hubei",     "Chang Yang",         30, 43, 110, 54,  420, True),
    ("HBld",  "Hubei",     "Long Dong",          34, 40, 111,  2,  342, False),  # 34.67N 超出湖北北界(约33.3N)
    ("HBlmx", "Hubei",     "La Mei Xia",         30, 39, 111,  3,  307, True),
    ("HBcbx", "Hubei",     "Chai Bu Xi",         30, 11, 111,  1,  248, True),
    ("HBhh",  "Hubei",     "Hou He",             30,  5, 110, 40,  440, True),
    ("Hbzg",  "Hubei",     "Zi Gui Si Xi",       30, 43, 111, 54,  248, True),
]


def haversine_km(a, b):
    """两 (lat, lon) 之间的大圆距离，km。"""
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def dm_to_dd(deg, minute):
    """度分转十进制度。"""
    return deg + minute / 60.0


def in_bbox(lat, lon):
    return (BBOX["lat_min"] <= lat <= BBOX["lat_max"]
            and BBOX["lon_min"] <= lon <= BBOX["lon_max"])


# ---------------------------------------------------------------------------
# 源 1：自有 18 个采样种群
# ---------------------------------------------------------------------------
def load_own():
    pts = []
    with open(SAMPLE_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            name = (row["NAME"] or "").strip()
            if not name:
                continue
            lon, lat = float(row["X"]), float(row["Y"])
            pts.append(dict(label=f"OWN:{name}", lat=lat, lon=lon,
                            source="own", detail=name))
    return pts


# ---------------------------------------------------------------------------
# 源 2：文献种群表
# ---------------------------------------------------------------------------
def load_literature():
    pts, dropped = [], []
    for abbr, prov, place, latd, latm, lond, lonm, elev, ok in LITERATURE:
        if not ok:
            dropped.append(abbr)
            continue
        pts.append(dict(label=f"LIT:{abbr}", lat=dm_to_dd(latd, latm),
                        lon=dm_to_dd(lond, lonm), source="literature",
                        detail=f"{prov} / {place} / {elev} m"))
    return pts, dropped


# ---------------------------------------------------------------------------
# 源 3：GBIF
# ---------------------------------------------------------------------------
def fetch_gbif():
    """按 speciesKey + country=CN 拉全部带坐标记录，结果缓存到本地。"""
    if GBIF_CACHE.exists():
        return json.loads(GBIF_CACHE.read_text()), True
    base = "https://api.gbif.org/v1/occurrence/search"
    params = dict(taxonKey=GBIF_TAXON_KEY, country="CN",
                  hasCoordinate="true", limit=300, offset=0)
    recs = []
    while True:
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "occ-points/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.loads(r.read())
        recs += page.get("results", [])
        if page.get("endOfRecords") or not page.get("results"):
            break
        params["offset"] += params["limit"]
    GBIF_CACHE.write_text(json.dumps(recs, ensure_ascii=False))
    return recs, False


def load_gbif():
    recs, cached = fetch_gbif()
    seen, pts = set(), []
    stats = dict(raw=len(recs), dup=0, bbox=0, unc=0, garden=0)
    for r in recs:
        lat, lon = r.get("decimalLatitude"), r.get("decimalLongitude")
        if lat is None or lon is None:
            continue
        unc = r.get("coordinateUncertaintyInMeters")
        if unc is not None and unc > 10000:
            stats["unc"] += 1
            continue
        if not in_bbox(lat, lon):
            stats["bbox"] += 1
            continue
        loc = " ".join(str(r.get(f) or "") for f in ("locality", "habitat", "occurrenceRemarks"))
        if GARDEN_KW.search(loc):
            stats["garden"] += 1
            continue
        key = (round(lat, 4), round(lon, 4))
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        pts.append(dict(label=f"GBIF:{r.get('key')}", lat=lat, lon=lon,
                        source="gbif",
                        detail=f"{r.get('year') or '?'} / {r.get('institutionCode') or '?'}"))
    return pts, stats, cached


# ---------------------------------------------------------------------------
# 稀疏
# ---------------------------------------------------------------------------
def thin(points, radius_km):
    # 高优先级在前；同级按 (lat, lon) 排序保证可复现
    ordered = sorted(points, key=lambda p: (PRIORITY[p["source"]], p["lat"], p["lon"]))
    kept = []
    for p in ordered:
        if all(haversine_km((p["lat"], p["lon"]), (k["lat"], k["lon"])) > radius_km
               for k in kept):
            kept.append(p)
    return kept


def main():
    own = load_own()
    lit, dropped = load_literature()
    gbif, gstats, cached = load_gbif()

    print(f"自有采样点      : {len(own)}")
    print(f"文献种群表      : {len(lit)} 可用（剔除 {len(dropped)}: {', '.join(dropped)}）")
    print(f"GBIF            : 原始 {gstats['raw']} → 去重 -{gstats['dup']} "
          f"精度 -{gstats['unc']} 越界 -{gstats['bbox']} 栽培 -{gstats['garden']} "
          f"→ {len(gbif)}{'（读缓存）' if cached else '（本次抓取）'}")

    for km in (5.0, 10.0):
        print(f"  稀疏 {km:4.0f} km → {len(thin(own + lit + gbif, km))} 点")

    kept = thin(own + lit + gbif, THIN_KM)
    kept.sort(key=lambda p: (PRIORITY[p["source"]], p["lat"]))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "lat", "lon", "source", "detail"])
        for p in kept:
            w.writerow([p["label"], f"{p['lat']:.6f}", f"{p['lon']:.6f}",
                        p["source"], p["detail"]])

    by_src = {}
    for p in kept:
        by_src[p["source"]] = by_src.get(p["source"], 0) + 1
    print(f"\n最终 {len(kept)} 点 → {OUT_CSV.name}  {by_src}")


if __name__ == "__main__":
    main()
