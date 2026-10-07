"""
OT 0-1-2: pullback доторх жижиг бүтцээр орох ("Ойлгомжтой 01") дүрмийн шалгалт
==============================================================================
Өсөлтийн тал (уналтад толин тусгалаар):
  ТОМ бүтэц (swing босго K*ATR):  L0 -> H1 -> HL -> HH,  HL > L0, HH > H1
      M = (HL + HH)/2,  зорилт T = HL + HH - L0
  HH-ийн дараах pullback дотор ЖИЖИГ эсрэг бүтэц (swing босго ks*ATR):
      h0 = HH -> l1 -> lh -> ll,  lh < h0, ll < l1
      жижиг зорилт t = lh + ll - h0   (pullback "хэмжээгээ гүйцээх" цэг)
  Орох хувилбарууд:
      touch   : t дээр limit захиалга, stop = HL-ийн цаана (том бүтцийн шилжүүлэгчийн 0)
      confirm : үнэ t-д хүрсний дараа ks*ATR-ээр эргэхийг хүлээж дараагийн лааны нээлтээр
                орно, stop = pullback-ийн туйлын цэгийн цаана (зургууд дээрхтэй адил)
  Зорилт хоёуланд нь T.

Үнэ: MT5-ийн лаа нь BID үнэ. Long нь ASK-аар (bid+spread) орж BID-ээр гарна,
short нь BID-ээр орж ASK-аар гарна. spread=0 өгвөл зардалгүй (gross) үр дүн.
"""
import numpy as np
import pandas as pd
from ot_backtest import atr_wilder, zigzag, find_data, HERE


def load_mt5(path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S")
    df = (df.drop(columns=["date", "time"]).sort_values("datetime")
            .drop_duplicates("datetime").set_index("datetime"))
    return df.astype(float)


def sim_exit(o, h, l, c, sp, e, s, S, T, max_hold, skip_target_first=False):
    """Гарах цэг. Нэг лаан дотор stop, target хоёуланд хүрвэл STOP гэж тооцно."""
    end = min(len(o), e + max_hold)
    for i in range(e, end):
        if s > 0:                                   # long: BID-ээр гарна
            if o[i] <= S: return i, o[i], "stop"
            if o[i] >= T and not (skip_target_first and i == e): return i, o[i], "target"
            if l[i] <= S: return i, S, "stop"
            if h[i] >= T and not (skip_target_first and i == e): return i, T, "target"
        else:                                       # short: ASK-аар гарна
            a_o, a_h, a_l = o[i] + sp[i], h[i] + sp[i], l[i] + sp[i]
            if a_o >= S: return i, a_o, "stop"
            if a_o <= T and not (skip_target_first and i == e): return i, a_o, "target"
            if a_h >= S: return i, S, "stop"
            if a_l <= T and not (skip_target_first and i == e): return i, T, "target"
    i = end - 1
    return i, (c[i] if s > 0 else c[i] + sp[i]), "time"


def find_nested(df, K=6.0, ks=2.0, variant="confirm", use_spread=True,
                buf_atr=0.1, max_wait=3000, max_hold=5000, default_spread=0.18):
    o, h, l, c = (df[x].values for x in ["open", "high", "low", "close"])
    n = len(o)
    if use_spread:
        sp = df["spread"].values * 0.01
        sp = np.where(sp <= 0, default_spread, sp)
    else:
        sp = np.zeros(n)
    atr = atr_wilder(df)
    big = zigzag(h, l, atr, K)
    small = zigzag(h, l, atr, ks)
    s_idx = np.array([p[0] for p in small])
    rows = []
    stats = dict(setups=0, small_found=0, t_valid=0, filled=0)
    for j in range(3, len(big)):
        p0, p1, p2, p3 = big[j - 3:j + 1]
        s = p3[2]                                   # HH орой бол өсөлт (+1)
        L0, H1, HL, HH = p0[1], p1[1], p2[1], p3[1]
        if not (s * (HL - L0) > 0 and s * (HH - H1) > 0):
            continue
        stats["setups"] += 1
        M = (HL + HH) / 2.0
        T = HL + HH - L0
        # жижиг zigzag дээр HH-тай давхцах цэгийг олох
        k0 = np.searchsorted(s_idx, p3[0])
        if k0 >= len(small) or small[k0][0] != p3[0] or small[k0][2] != s or k0 + 3 >= len(small):
            continue
        q1, q2, q3 = small[k0 + 1], small[k0 + 2], small[k0 + 3]
        h0, l1, lh, ll = HH, q1[1], q2[1], q3[1]
        if not (s * (h0 - lh) > 0 and s * (l1 - ll) > 0):
            continue
        stats["small_found"] += 1
        t = lh + ll - h0
        if s * (t - HL) <= 0:                       # жижиг зорилт том бүтцийг эвдэх түвшинд
            continue
        stats["t_valid"] += 1
        start = max(q3[3], p3[3]) + 1               # хоёр бүтэц хоёулаа мэдэгдсэн мөч
        if start >= n:
            continue
        # --- орох цэгийг хайх ---
        e = None; E = None; S = None
        touched = False; ext = None; ext_i = None
        for i in range(start, min(n, start + max_wait)):
            ask_l = l[i] + sp[i]
            if not touched:
                # хүчингүй болох нөхцөл: жижиг бүтцийн lh эвдэгдэх / зорилтод хүрэх
                if (s > 0 and (h[i] >= lh or h[i] >= T)) or (s < 0 and (l[i] <= lh or l[i] <= T)):
                    break
                hit = (ask_l <= t) if s > 0 else (h[i] >= t)
                if hit:
                    touched = True
                    if variant == "touch":
                        e = i
                        E = min(t, o[i] + sp[i]) if s > 0 else max(t, o[i])
                        S = HL - s * buf_atr * atr[i - 1]
                        break
                    ext, ext_i = (l[i], i) if s > 0 else (h[i], i)
                    if (s > 0 and l[i] <= HL) or (s < 0 and h[i] >= HL):
                        break
                    continue
            else:                                   # confirm: эргэлтийг хүлээнэ
                if (s > 0 and l[i] <= HL) or (s < 0 and h[i] >= HL):
                    break                           # том бүтэц эвдэрсэн
                if (s > 0 and l[i] < ext) or (s < 0 and h[i] > ext):
                    ext, ext_i = (l[i], i) if s > 0 else (h[i], i)
                    continue
                turned = (h[i] - ext >= ks * atr[i]) if s > 0 else (ext - l[i] >= ks * atr[i])
                if turned:
                    if i + 1 >= n:
                        break
                    e = i + 1
                    E = o[e] + sp[e] if s > 0 else o[e]
                    S = ext - s * buf_atr * atr[i]
                    # орохоос өмнө зорилтод хүрсэн бол алгасна
                    if (s > 0 and h[ext_i:e].max() >= T) or (s < 0 and l[ext_i:e].min() <= T):
                        e = None
                    break
        if e is None:
            continue
        if s * (T - E) <= 0 or s * (E - S) <= 0:
            continue
        stats["filled"] += 1
        risk = abs(E - S)
        if variant == "touch":
            # орсон лаан дээр зөвхөн stop-ийг шалгана
            x, px, why = sim_exit(o, h, l, c, sp, e, s, S, T, max_hold, skip_target_first=True)
        else:
            x, px, why = sim_exit(o, h, l, c, sp, e, s, S, T, max_hold)
        R = s * (px - E) / risk
        r_pos = (t - HL) / (HH - HL)
        rows.append(dict(entry_time=df.index[e], exit_time=df.index[x], side=s,
                         model=("A" if r_pos > 0.55 else "B" if r_pos >= 0.45 else "C"),
                         L0=L0, H1=H1, HL=HL, HH=HH, M=M, Tgt=T, l1=l1, lh=lh, ll=ll, t=t,
                         E=E, S=S, exit=px, why=why, risk=risk, rr=abs(T - E) / risk,
                         p_null=risk / abs(T - S), R=R, e=e, x=x, spread=sp[e],
                         i_L0=p0[0], i_H1=p1[0], i_HL=p2[0], i_HH=p3[0],
                         i_l1=q1[0], i_lh=q2[0], i_ll=q3[0]))
    return pd.DataFrame(rows), stats


def hit_stats(d):
    d = d[d["why"] != "time"]
    n = len(d)
    if n == 0:
        return dict(n=0, hit=np.nan, null=np.nan, z=np.nan)
    hits = (d["why"] == "target").sum()
    exp = d["p_null"].sum()
    var = (d["p_null"] * (1 - d["p_null"])).sum()
    return dict(n=n, hit=hits / n, null=exp / n, z=(hits - exp) / np.sqrt(var))


def boot(x, groups, B=5000, seed=1):
    """Өдрөөр бүлэглэсэн bootstrap (нэг өдрийн арилжаанууд хамааралтай байж болно)."""
    rng = np.random.default_rng(seed)
    g = pd.Series(np.asarray(x, float)).groupby(np.asarray(groups))
    sums, cnts = g.sum().values, g.count().values
    m = len(sums)
    idx = rng.integers(0, m, size=(B, m))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


def summary(tr):
    if len(tr) < 5:
        return dict(n=len(tr))
    days = tr["entry_time"].dt.strftime("%Y-%m-%d").values
    lo, hi = boot(tr["R"].values, days)
    hs = hit_stats(tr)
    return dict(n=len(tr), hit=hs["hit"], null=hs["null"], z=hs["z"], rr=tr["rr"].median(),
                R=tr["R"].mean(), lo=lo, hi=hi, risk_med=tr["risk"].median(),
                spread_over_risk=(tr["spread"] / tr["risk"]).median())


if __name__ == "__main__":
    # Энэ файл нь функцүүдийн сан. Шууд ажиллуулбал нэг тохиргоог жишээ болгон шалгана.
    # Бүрэн шалгалт: ot_nested_run.py
    import warnings
    warnings.filterwarnings("ignore")
    df = load_mt5(find_data("_M1", "M1", what="XAUUSD 1 минутын өгөгдөл (MT5-аас экспортолсон)"))
    for var in ["confirm", "touch"]:
        tr, st = find_nested(df, K=4, ks=1.5, variant=var, use_spread=True)
        s = summary(tr)
        print(f"M1, K=4, ks=1.5, {var}: арилжаа {s['n']}, зорилтод хүрсэн {s['hit']:.1%}, "
              f"санамсаргүй үед {s['null']:.1%}, дундаж (spread хассан) {s['R']:+.2f}R  [{s['lo']:+.2f} .. {s['hi']:+.2f}]")
