"""Жижиг бүтцээр орох дүрмийн бүрэн шалгалт (алт, 1 ба 5 минут).
Шаардлагатай өгөгдөл: data хавтсанд MT5-аас экспортолсон M1 ба M5 файл
(нэрэндээ "M1", "M5" гэсэн хэсэгтэй, жишээ нь XAUUSD_M1.csv).
Ажиллуулах:  python ot_nested_run.py
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from ot_nested import load_mt5, find_nested, summary, find_data, HERE

m1 = load_mt5(find_data("_M1", "M1", what="XAUUSD 1 минутын өгөгдөл (MT5-аас экспортолсон)"))
m5 = load_mt5(find_data("_M5", "M5", what="XAUUSD 5 минутын өгөгдөл (MT5-аас экспортолсон)"))
rows = []
for tf, df, grid in [("M1", m1, [(3, 1), (4, 1), (4, 1.5), (5, 1.5), (6, 2), (8, 2)]),
                     ("M5", m5, [(3, 1), (4, 1), (4, 1.5), (6, 2)])]:
    for K, ks in grid:
        for var in ["confirm", "touch"]:
            gross, _ = find_nested(df, K=K, ks=ks, variant=var, use_spread=False)
            net, st = find_nested(df, K=K, ks=ks, variant=var, use_spread=True)
            s, sg = summary(net), summary(gross)
            rows.append(dict(tf=tf, K=K, ks=ks, variant=var, setups=st["setups"], n=s.get("n"),
                             hit=s.get("hit"), chance=s.get("null"), z=s.get("z"), rr=s.get("rr"),
                             R_gross=sg.get("R"), R_net=s.get("R"), lo=s.get("lo"), hi=s.get("hi"),
                             stop_usd=s.get("risk_med")))
R = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(R.round(3).to_string(index=False))
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
R.to_csv(OUT / "ot_nested_results.csv", index=False)
print("\nҮр дүнгийн хүснэгт хадгалагдлаа:", OUT)
