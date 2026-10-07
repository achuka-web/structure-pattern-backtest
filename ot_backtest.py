"""
OT (0-1-2) бүтцийн статистик шалгалт
====================================
Дүрэм (өсөлтийн тал; уналтад толин тусгалаар):
  L0 = үндсэн хөдөлгөөний эхлэл (үндсэн 0)
  H1 = эхний орой
  HL = шилжүүлэгчийн өмнөх pullback-ийн төгсгөл (шилжүүлэгчийн 0), HL > L0
  HH = шилжүүлэгчийн орой (шилжүүлэгчийн 2), HH > H1
  M  = (HL + HH) / 2            -> 1-ийн түвшин
  T  = 2*M - L0 = HL + HH - L0  -> үндсэн хэмжээний 2 (зорилт)
  P  = шилжүүлэгчийн дараах pullback-ийн төгсгөл
  Загвар: r = (P - HL) / (HH - HL);  A: r > 0.55,  B: 0.45..0.55,  C: r < 0.45

Ирээдүйг харахгүй байх зарчим:
  Swing цэг бүр үнэ k*ATR-ээр эргэсэн мөчид л "баталгаажна".
  Арилжаа баталгаажсан лааны ДАРААГИЙН лааны нээлтээр орно.
"""
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. Өгөгдлийн файлыг олох
#    Скриптийг хаанаас ч ажиллуулсан энэ файлын хажуу дахь "data" хавтаснаас хайна.
# ----------------------------------------------------------------------------
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:                                   # Windows терминалд кирилл үсэг зөв гарахын тулд
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
SEARCH_DIRS = [HERE / "data", HERE / "data" / "raw", HERE, Path.cwd() / "data", Path.cwd() / "data" / "raw"]


def find_data(*patterns, what=""):
    """patterns-ийн аль нэгтэй таарах эхний CSV файлын замыг буцаана (том жижиг үсэг ялгахгүй)."""
    for d in SEARCH_DIRS:
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.csv"))
        for pat in patterns:
            for f in files:
                if pat.lower() in f.name.lower():
                    return str(f)
    print()
    print("ӨГӨГДӨЛ ОЛДСОНГҮЙ:", what or patterns[0])
    print("  Нэрэндээ", " эсвэл ".join(repr(p) for p in patterns), "гэсэн хэсэгтэй CSV файлыг")
    print("  энэ хавтсанд хуулна уу:", HERE / "data")
    sys.exit(1)


# ----------------------------------------------------------------------------
# 1. Өгөгдөл
# ----------------------------------------------------------------------------
def load_h1(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    df.columns = [c.strip().lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y.%m.%d %H:%M")
    df = df[["datetime", "open", "high", "low", "close"]].dropna()
    df = df.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    return df.astype(float)


def resample(df, rule):
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return out.dropna()


def atr_wilder(df, n=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.empty_like(tr)
    atr[:n] = np.nan
    atr[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        atr[i] = (atr[i - 1] * (n - 1) + tr[i]) / n
    return atr


# ----------------------------------------------------------------------------
# 2. Swing цэг илрүүлэгч (causal zigzag)
#    Буцаах утга: [(idx, price, type, confirm_idx)], type=+1 орой, -1 ёроол
# ----------------------------------------------------------------------------
def zigzag(h, l, atr, k, start=20):
    piv = []
    d = 0
    hi_i = lo_i = start
    n = len(h)
    for i in range(start + 1, n):
        thr = k * atr[i]
        if d == 0:
            if h[i] > h[hi_i]:
                hi_i = i
            if l[i] < l[lo_i]:
                lo_i = i
            if hi_i < i and h[hi_i] - l[i] >= thr:
                piv.append((hi_i, h[hi_i], +1, i)); d = -1
                lo_i = hi_i + 1 + int(np.argmin(l[hi_i + 1:i + 1]))
            elif lo_i < i and h[i] - l[lo_i] >= thr:
                piv.append((lo_i, l[lo_i], -1, i)); d = +1
                hi_i = lo_i + 1 + int(np.argmax(h[lo_i + 1:i + 1]))
        elif d == +1:                       # өсөлтийн хөл: оройг дагана
            if h[i] > h[hi_i]:
                hi_i = i
            elif h[hi_i] - l[i] >= thr:     # орой баталгаажлаа
                piv.append((hi_i, h[hi_i], +1, i)); d = -1
                lo_i = hi_i + 1 + int(np.argmin(l[hi_i + 1:i + 1]))
        else:                               # уналтын хөл: ёроолыг дагана
            if l[i] < l[lo_i]:
                lo_i = i
            elif h[i] - l[lo_i] >= thr:     # ёроол баталгаажлаа
                piv.append((lo_i, l[lo_i], -1, i)); d = +1
                hi_i = lo_i + 1 + int(np.argmax(h[lo_i + 1:i + 1]))
    return piv


# ----------------------------------------------------------------------------
# 3. Гарах цэгийг олох (stop эсвэл target, аль түрүүлж хүрсэн нь)
# ----------------------------------------------------------------------------
def simulate(o, h, l, c, e, side, S, T, max_hold):
    """side=+1 long, -1 short. Нэг лаан дотор хоёуланд нь хүрвэл STOP гэж тооцно."""
    end = min(len(o), e + max_hold)
    for i in range(e, end):
        if side > 0:
            if o[i] <= S: return i, o[i], "stop"
            if o[i] >= T: return i, o[i], "target"
            if l[i] <= S: return i, S, "stop"
            if h[i] >= T: return i, T, "target"
        else:
            if o[i] >= S: return i, o[i], "stop"
            if o[i] <= T: return i, o[i], "target"
            if h[i] >= S: return i, S, "stop"
            if l[i] <= T: return i, T, "target"
    return end - 1, c[end - 1], "time"


# ----------------------------------------------------------------------------
# 4. Бүтэц таних + арилжаа үүсгэх
# ----------------------------------------------------------------------------
def classify(r):
    return "A" if r > 0.55 else ("B" if r >= 0.45 else "C")


def find_trades(df, k=3.0, cost_frac=0.0, stop_mode="P", buf_atr=0.1, max_hold=3000):
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    atr = atr_wilder(df)
    piv = zigzag(h, l, atr, k)
    n = len(o)
    rows = []
    for j in range(4, len(piv)):
        p0, p1, p2, p3, p4 = piv[j - 4:j + 1]
        side = -p4[2]                      # P ёроол (-1) бол long (+1)
        s = side
        L0, H1, HL, HH, P = p0[1], p1[1], p2[1], p3[1], p4[1]
        # бүтцийн нөхцөл (s=+1 үед өсөлт, s=-1 үед уналт)
        if not (s * (HL - L0) > 0 and s * (HH - H1) > 0 and s * (P - HL) > 0):
            continue
        M = (HL + HH) / 2.0
        T = 2 * M - L0
        r = (P - HL) / (HH - HL)
        e = p4[3] + 1                      # баталгаажсан лааны дараагийн лаа
        if e >= n:
            continue
        E = o[e]
        a = atr[p4[3]]
        S = (P if stop_mode == "P" else HL) - s * buf_atr * a
        # орохоос өмнө зорилтод аль хэдийн хүрсэн эсэх
        seg_h, seg_l = h[p4[0]:e], l[p4[0]:e]
        reached = (seg_h.max() >= T) if s > 0 else (seg_l.min() <= T)
        if reached or s * (T - E) <= 0 or s * (E - S) <= 0:
            continue
        x, px, why = simulate(o, h, l, c, e, s, S, T, max_hold)
        risk = abs(E - S)
        gross = s * (px - E) / risk
        cost_r = cost_frac * E / risk
        rows.append(dict(
            entry_time=df.index[e], exit_time=df.index[x], side=s, model=classify(r), r=r,
            L0=L0, H1=H1, HL=HL, HH=HH, P=P, M=M, T=T, E=E, S=S, exit=px, why=why,
            rr=abs(T - E) / risk, p_null=risk / abs(T - S),
            R_gross=gross, R_net=gross - cost_r, cost_R=cost_r,
            e=e, x=x, dS_atr=risk / a, dT_atr=abs(T - E) / a,
            i_L0=p0[0], i_H1=p1[0], i_HL=p2[0], i_HH=p3[0], i_P=p4[0]))
    return pd.DataFrame(rows), atr


# ----------------------------------------------------------------------------
# 4b. Хувилбар: 1-ийн түвшин дээр limit захиалга (B загварын санаа)
#     Шилжүүлэгч баталгаажмагц M дээр хүлээнэ, stop = шилжүүлэгчийн 0-ийн цаана,
#     зорилт = үндсэн 2. Үнэ M-д хүрэлгүй T-д хүрвэл арилжаа үүсэхгүй.
# ----------------------------------------------------------------------------
def find_trades_mlimit(df, k=3.0, cost_frac=0.0, buf_atr=0.1, max_hold=3000):
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    atr = atr_wilder(df)
    piv = zigzag(h, l, atr, k)
    n = len(o)
    rows = []
    for j in range(3, len(piv)):
        p0, p1, p2, p3 = piv[j - 3:j + 1]
        s = p3[2]
        L0, H1, HL, HH = p0[1], p1[1], p2[1], p3[1]
        if not (s * (HL - L0) > 0 and s * (HH - H1) > 0):
            continue
        M = (HL + HH) / 2.0
        T = 2 * M - L0
        a = atr[p3[3]]
        S = HL - s * buf_atr * a
        e0 = p3[3] + 1
        if e0 >= n or s * (o[e0] - M) <= 0:
            continue                       # аль хэдийн M-ээс цаана нээгдсэн
        # M-д хүрэх эсэхийг хүлээх
        e = None
        for i in range(e0, min(n, e0 + max_hold)):
            if (s > 0 and l[i] <= M) or (s < 0 and h[i] >= M):
                e = i; break
            if (s > 0 and h[i] >= T) or (s < 0 and l[i] <= T):
                break                      # M-д хүрэлгүй зорилтод хүрсэн -> арилжаагүй
        if e is None:
            continue
        E = M
        risk = abs(E - S)
        # орсон лаан дээр зөвхөн stop-ийг шалгана (хатуу таамаглал)
        if (s > 0 and l[e] <= S) or (s < 0 and h[e] >= S):
            x, px, why = e, S, "stop"
        elif e + 1 >= n:
            continue
        else:
            x, px, why = simulate(o, h, l, c, e + 1, s, S, T, max_hold)
        gross = s * (px - E) / risk
        cost_r = cost_frac * E / risk
        rows.append(dict(entry_time=df.index[e], exit_time=df.index[x], side=s, model="M",
                         E=E, S=S, T=T, exit=px, why=why, rr=abs(T - E) / risk,
                         p_null=risk / abs(T - S), R_gross=gross, R_net=gross - cost_r,
                         cost_R=cost_r, e=e, x=x, dS_atr=risk / a, dT_atr=abs(T - E) / a))
    return pd.DataFrame(rows), atr


# ----------------------------------------------------------------------------
# 5. Гол таамаглал: шилжүүлэгч баталгаажсанаас хойш үнэ 2-т хүрэх үү,
#    эсвэл шилжүүлэгчийн 0-ийг түрүүлж эвдэх үү?
# ----------------------------------------------------------------------------
def core_claim(df, k=3.0, max_hold=3000):
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    atr = atr_wilder(df)
    piv = zigzag(h, l, atr, k)
    n = len(o)
    rows = []
    for j in range(3, len(piv)):
        p0, p1, p2, p3 = piv[j - 3:j + 1]
        s = p3[2]                          # HH орой (+1) бол өсөлтийн бүтэц
        L0, H1, HL, HH = p0[1], p1[1], p2[1], p3[1]
        if not (s * (HL - L0) > 0 and s * (HH - H1) > 0):
            continue
        T = HL + HH - L0
        e = p3[3] + 1
        if e >= n:
            continue
        E = o[e]
        if s * (E - HL) <= 0 or s * (T - E) <= 0:
            continue
        x, px, why = simulate(o, h, l, c, e, s, HL, T, max_hold)
        rows.append(dict(time=df.index[e], side=s, why=why,
                         p_null=abs(E - HL) / abs(T - HL),
                         R=s * (px - E) / abs(E - HL)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 6. Статистик
# ----------------------------------------------------------------------------
def hit_test(d):
    """Бодит хүрэлтийн хувь vs чиглэлгүй санамсаргүй алхалтын үеийн хүлээгдэх хувь."""
    d = d[d["why"] != "time"]
    n = len(d)
    if n == 0:
        return dict(n=0, hit=np.nan, null=np.nan, z=np.nan)
    hits = (d["why"] == "target").sum()
    exp = d["p_null"].sum()
    var = (d["p_null"] * (1 - d["p_null"])).sum()
    return dict(n=n, hit=hits / n, null=exp / n, z=(hits - exp) / np.sqrt(var))


def boot_ci(x, months, B=5000, seed=1):
    """Сараар бүлэглэсэн bootstrap: давхцсан арилжаануудын хамаарлыг тооцно."""
    rng = np.random.default_rng(seed)
    g = pd.Series(x).groupby(np.asarray(months))
    sums, cnts = g.sum().values, g.count().values
    m = len(sums)
    idx = rng.integers(0, m, size=(B, m))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(means, [2.5, 97.5]), (means <= 0).mean()


def placebo(df, atr, trades, n_rep=40, cost_frac=0.0, max_hold=3000, seed=7):
    """Санамсаргүй цагт, ижил чиглэл, ATR-ээр хэмжсэн ижил stop/target зайтай арилжаа."""
    rng = np.random.default_rng(seed)
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    n = len(o)
    out = np.empty((n_rep, len(trades)))
    yrs = df.index.year.values
    by_year = {y: np.where((yrs == y) & (np.arange(n) > 30) & (np.arange(n) < n - 5))[0]
               for y in np.unique(yrs)}
    for t_i, t in enumerate(trades.itertuples()):
        pool = by_year[df.index[t.e].year]
        for rep in range(n_rep):
            e = int(rng.choice(pool))
            a = atr[e - 1]
            E = o[e]
            S = E - t.side * t.dS_atr * a
            T = E + t.side * t.dT_atr * a
            x, px, why = simulate(o, h, l, c, e, t.side, S, T, max_hold)
            risk = abs(E - S)
            out[rep, t_i] = t.side * (px - E) / risk - cost_frac * E / risk
    return out


def summarize(tr, label):
    if len(tr) == 0:
        return dict(label=label, n=0)
    months = tr["entry_time"].dt.to_period("M").astype(str).values
    ht = hit_test(tr)
    ci_g, _ = boot_ci(tr["R_gross"].values, months)
    ci_n, p_le0 = boot_ci(tr["R_net"].values, months)
    return dict(label=label, n=len(tr), hit=ht["hit"], null=ht["null"], z=ht["z"],
                rr=tr["rr"].median(), R_gross=tr["R_gross"].mean(), g_lo=ci_g[0], g_hi=ci_g[1],
                R_net=tr["R_net"].mean(), n_lo=ci_n[0], n_hi=ci_n[1], cost_R=tr["cost_R"].mean())


if __name__ == "__main__":
    # Энэ файл нь функцүүдийн сан. Шууд ажиллуулбал цагийн графикийн үндсэн тохиргоог шалгана.
    # Бүрэн шалгалт: ot_analysis.py
    COST = {"XAUUSD": 0.00015, "EURUSD": 0.00009}   # нэг арилжааны нийт зардал (үнийн хувиар), ТААМАГЛАЛ
    for sym in ["XAUUSD", "EURUSD"]:
        h1 = load_h1(find_data(f"{sym}H1", what=f"{sym} цагийн өгөгдөл"))
        tr, atr = find_trades(h1, k=3.0, cost_frac=COST[sym])
        r = summarize(tr, "H1 k=3")
        print(f"{sym}: арилжаа {r['n']}, зорилтод хүрсэн {r['hit']:.1%}, санамсаргүй үед {r['null']:.1%}, "
              f"дундаж (зардлын дараа) {r['R_net']:+.2f}R  [{r['n_lo']:+.2f} .. {r['n_hi']:+.2f}]")
