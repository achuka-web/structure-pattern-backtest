"""0-1-2 бүтцийн бүрэн шалгалт (цагийн өгөгдөл: H1, H4, D1).
Шаардлагатай өгөгдөл: data/XAUUSDH1.csv, data/EURUSDH1.csv
Ажиллуулах:  python ot_analysis.py
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from ot_backtest import *
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
# Нэг арилжааны нийт зардал (spread+комисс), үнийн хувиар. ТААМАГЛАЛ: MOT-ийн бодит тоогоор солино.
COST={"XAUUSD":0.00015,"EURUSD":0.00009}
TF={"H1":None,"H4":"4h","D1":"1D"}
res=[]; core=[]; keep={}
for sym in ["XAUUSD","EURUSD"]:
    h1=load_h1(find_data(f"{sym}H1", what=f"{sym} цагийн өгөгдөл"))
    for tf,rule in TF.items():
        df=h1 if rule is None else resample(h1,rule)
        for k in [2.0,3.0,4.0]:
            for sm in ["P","HL"]:
                tr,atr=find_trades(df,k=k,cost_frac=COST[sym],stop_mode=sm)
                r=summarize(tr,f"{sym} {tf} k={k:g} stop={sm}"); r.update(sym=sym,tf=tf,k=k,variant="confirm/"+sm); res.append(r)
                keep[(sym,tf,k,sm)]=(df,atr,tr)
            tr,atr=find_trades_mlimit(df,k=k,cost_frac=COST[sym])
            r=summarize(tr,f"{sym} {tf} k={k:g} M-limit"); r.update(sym=sym,tf=tf,k=k,variant="M-limit"); res.append(r)
            keep[(sym,tf,k,"M")]=(df,atr,tr)
            cc=core_claim(df,k=k); ht=hit_test(cc); ht.update(sym=sym,tf=tf,k=k); core.append(ht)
R=pd.DataFrame(res); C=pd.DataFrame(core)
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
cols=["sym","tf","k","variant","n","hit","null","z","rr","R_gross","g_lo","g_hi","R_net","n_lo","n_hi"]
print("=== ARILJAANII HUVILBARUUD ==="); print(R[cols].round(3).to_string(index=False))
print("\n=== GOL TAAMAGLAL: shiljuulegch batalgaajsnaas hoish T vs shilj.0 ==="); print(C.round(3).to_string(index=False))
R.to_csv(OUT / "ot_grid.csv",index=False); C.to_csv(OUT / "ot_core_claim.csv",index=False)

# --- Baseline (H1, k=3, stop=P): zagvar, tal, on ---
print("\n=== BASELINE H1 k=3 stop=P: zadargaa ===")
allb=[]
for sym in ["XAUUSD","EURUSD"]:
    df,atr,tr=keep[(sym,"H1",3.0,"P")]
    tr=tr.assign(sym=sym); allb.append(tr)
    for name,sub in [("buh",tr),("A",tr[tr.model=="A"]),("B",tr[tr.model=="B"]),("C",tr[tr.model=="C"]),
                     ("long",tr[tr.side>0]),("short",tr[tr.side<0]),
                     ("2020-2023",tr[tr.entry_time<"2024-01-01"]),("2024-2026",tr[tr.entry_time>="2024-01-01"])]:
        s=summarize(sub,name)
        print(sym.ljust(7),name.ljust(10),{k:(round(float(v),3) if isinstance(v,(float,np.floating)) else v) for k,v in s.items() if k in ["n","hit","null","z","rr","R_gross","R_net","n_lo","n_hi"]})
pool=pd.concat(allb)
# pooled by model across all timeframes & k (stop=P)
print("\n=== ZAGVARAAR (buh tf, buh k, stop=P negtgesen) ===")
big=pd.concat([keep[(s,t,k,"P")][2].assign(sym=s,tf=t,k=k) for s in ["XAUUSD","EURUSD"] for t in TF for k in [2.0,3.0,4.0]])
for m in ["A","B","C"]:
    sub=big[big.model==m]; ht=hit_test(sub)
    print(m,{k:round(float(v),3) for k,v in ht.items()},"R_gross",round(sub.R_gross.mean(),3),"R_net",round(sub.R_net.mean(),3))

# --- Placebo for baseline ---
print("\n=== PLACEBO (sanamsargui tsag, ijil chiglel, ijil stop/target zai) ===")
for sym in ["XAUUSD","EURUSD"]:
    df,atr,tr=keep[(sym,"H1",3.0,"P")]
    pl=placebo(df,atr,tr,n_rep=40,cost_frac=COST[sym])
    pm=pl.mean(1)
    print(sym,"bodit R_net",round(tr.R_net.mean(),3),"| placebo dundaj",round(pm.mean(),3),"placebo 2.5-97.5%",np.round(np.percentile(pm,[2.5,97.5]),3),"| placebo>=bodit huvi",round((pm>=tr.R_net.mean()).mean(),3))
pool.to_csv(OUT / "ot_baseline_trades.csv",index=False)

print("\nҮр дүнгийн хүснэгтүүд хадгалагдлаа:", OUT)
