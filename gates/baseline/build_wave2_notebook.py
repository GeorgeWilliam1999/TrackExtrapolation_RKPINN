#!/usr/bin/env python3
"""Assemble the Wave-2 exploration notebook (nbformat) and execute it.

Reads the Wave-2 artifacts from TE_LAB (corpus meta, training histories, three-arm
jsons) and produces a self-contained executed .ipynb with the audit reproduction,
the restratified-corpus composition, the residual-loss design, the accuracy<->size
curve, the lambda sweep, and the three-arm comparison vs the incumbent.

Usage: TE_LAB=... python build_wave2_notebook.py
Output: TE_LAB/paper_p0/explore_wave2.ipynb  (executed)
"""
from __future__ import annotations
import os
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

LAB = os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3")

nb = new_notebook()
C = []


def md(s): C.append(new_markdown_cell(s))
def code(s): C.append(new_code_cell(s))


md("""# Wave-2 retraining + gen-4 data-appropriateness audit — exploration

**Thesis:** wave-1 gen-4 models failed UT→T (median 175–224 mm vs the production
polynomial's 11 µm) not from network capacity but from **data weighting + an
untamed 9.9-decade target range + under-training**. This notebook reproduces the
audit, documents the fixes (residual/kick head, range-aware loss, restratified
deployment-weighted corpus, tuned schedule, capacity ladder), and re-judges with
the three-arm evaluation.""")

code(f"""import json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
LAB = Path("{LAB}")
DATA, TM, REF = LAB/"data", LAB/"trained_models", LAB/"paper_p0"
plt.rcParams['figure.dpi']=110""")

md("## 1 · Gen-4 appropriateness audit (the *why*)")
code("""g = np.load(DATA/"train_10M_gen4.npz")
Xg, Yg = g["X"], g["Y"]
z0,dz = Xg[:,5], Xg[:,6]; zf=z0+dz
utt = (z0>=2300)&(z0<=3000)&(zf>=7600)&(zf<=9500)&(dz>0)
p = 0.299792458/np.abs(Xg[:,4])
rx = np.abs(Yg[:,0]-(Xg[:,0]+Xg[:,2]*dz))     # x-bend vs straight line [mm]
print(f"UT->T fraction      : {100*utt.mean():.3f}%   ({utt.sum():,} tracks)")
print(f"|dz|<1m / >6m       : {100*(np.abs(dz)<1000).mean():.1f}% / {100*(np.abs(dz)>6000).mean():.1f}%")
print(f"x-bend |rx| decades : {np.log10(np.quantile(rx[rx>0],0.999)/max(np.quantile(rx[rx>0],1e-3),1e-6)):.1f}"
      f"  (median {1e3*np.median(rx):.0f} um, p99.9 {np.quantile(rx,0.999):.0f} mm)")
fig,ax=plt.subplots(1,2,figsize=(11,3.4))
ax[0].hist(np.log10(np.abs(dz)),bins=60,color='steelblue'); ax[0].set_title("gen-4 |dz| (log10 mm)")
ax[0].axvline(np.log10(5161),color='r',ls='--',label='UT->T leg'); ax[0].legend()
ax[1].hist(np.log10(np.clip(rx,1e-4,None)),bins=80,color='indianred')
ax[1].set_title("gen-4 target dynamic range: log10 |x-bend| [mm]"); plt.tight_layout()""")

md("**Verdict:** correct but mis-weighted — UT→T is 0.145 % of rows and the "
   "target spans ~10 decades; a plain endpoint regressor parks at the straight line.")

md("## 2 · Restratified deployment-weighted corpus (the *fix*, data side)")
code("""meta = json.load(open(DATA/"train_wave2_deploy.meta.json"))
print(json.dumps(meta, indent=1))
w = np.load(DATA/"train_wave2_deploy.npz")
Xw=w["X"]; z0=Xw[:,5]; zf=z0+Xw[:,6]
uttw=(z0>=2300)&(z0<=3000)&(zf>=7600)&(zf<=9500)&(Xw[:,6]>0)
pw=0.299792458/np.abs(Xw[:,4])
fig,ax=plt.subplots(1,3,figsize=(13,3.3))
ax[0].bar(["gen-4","wave-2"],[0.145, 100*uttw.mean()],color=['gray','seagreen'])
ax[0].set_ylabel("UT->T %"); ax[0].set_title("UT->T fraction (target >=10%)"); ax[0].axhline(10,ls='--',c='r')
ax[1].hist(np.log10(p),bins=60,alpha=.6,density=True,label='gen-4')
ax[1].hist(np.log10(pw),bins=60,alpha=.6,density=True,label='wave-2'); ax[1].legend()
ax[1].set_title("log10 p [GeV] (wave-2 low-p weighted)")
ax[2].hist(Xw[uttw,5],bins=40,color='seagreen'); ax[2].set_title("wave-2 UT->T z0 [mm]"); plt.tight_layout()""")

md("## 3 · Range-aware residual loss design")
md("""The legacy log-cosh divides by the **endpoint** std (x ≈ 1.2 m), so a 100 µm
error registers as ~1e-7 — only metre-scale tail tracks produce gradient. Wave-2
scales by a per-track **residual** scale `sqrt(floor² + (α·bend)²)` (floor 20 µm /
20 µrad, α=0.25), measured per component → x,y(mm) balanced against tx,ty(rad).""")
code("""# endpoint std (legacy scale) vs residual std (what matters)
rb = np.stack([Yg[:,i]-(Xg[:,0]+Xg[:,2]*dz if i==0 else Xg[:,1]+Xg[:,3]*dz if i==1 else Xg[:,i]) for i in range(4)],1)
import numpy as np
endpoint_std = Yg[:,:4].std(0); resid_std = rb.std(0)
print("component         x[mm]    y[mm]   tx[rad]  ty[rad]")
print("endpoint std :", np.round(endpoint_std,4))
print("residual std :", np.round(resid_std,4))
print("ratio        :", np.round(endpoint_std/resid_std,1), " <- legacy over-scales x the most")""")

md("## 4 · Wave-2 training: capacity ladder + λ sweep")
code("""def load_hist(name):
    h=TM/name/"history.json"
    return json.load(open(h)) if h.exists() else None
sizes=[("wave2_resid_h%d"%h, h) for h in (32,64,96,128,256,384)]
rows=[]
for name,h in sizes:
    H=load_hist(name)
    if not H: continue
    tf=H.get("test_final",{}); bf=H.get("best_val_full",{})
    npar=None
    try: npar=json.load(open(TM/name/"config.json")) and None
    except: pass
    rows.append((name,h,H.get("best_epoch"),tf.get("utt_median_dx_um"),tf.get("median_dx_mm"),len(H.get("val",[]))))
print(f"{'run':<20}{'h':>5}{'best_ep':>8}{'utt_med_um':>12}{'bulk_med_mm':>12}{'epochs':>8}")
for r in rows: print(f"{r[0]:<20}{r[1]:>5}{str(r[2]):>8}{str(round(r[3],1) if r[3]==r[3] else r[3]):>12}{str(round(r[4],4)):>12}{r[5]:>8}")
# training curves
fig,ax=plt.subplots(figsize=(7,4))
for name,h in sizes:
    H=load_hist(name)
    if not H: continue
    u=[v.get("utt_median_dx_um",np.nan) for v in H["val"]]
    ax.plot(u,label="h%d"%h)
ax.set_yscale('log'); ax.set_xlabel("epoch"); ax.set_ylabel("val UT->T median |dx| [um]")
ax.legend(); ax.set_title("Wave-2 training curves (UT->T)"); plt.tight_layout()""")

md("## 5 · Three-arm evaluation vs the incumbent (plane ref)")
md("Incumbent profile to beat: **median 11 µm · low-p quartile 475 µm · p95 1.6 mm**.")
code("""res=json.load(open(REF/"wave2_three_arm.json"))
print(f"{'arm':<26}{'med um':>9}{'p95 um':>10}{'specMed':>9}  byQ hi->lo p [um]")
for k,m in res.items():
    bq="/".join(f"{v:.0f}" for v in m["median_dx_um_by_qop_quartile_hi2lo_p"])
    print(f"{k:<26}{m['median_dx_um']:>9.1f}{m['p95_dx_um']:>10.1f}{m['spec_weighted_median_dx_um']:>9.1f}  [{bq}]")""")
code("""# error vs p: incumbent vs best wave-2 NN vs straight line
A=np.load(REF/"wave2_three_arm_arrays.npz")
p=A["p_GeV"]; order=np.argsort(p)
nn = [k for k in A.files if k.startswith("wave2_")]
best = min(nn, key=lambda k: np.median(A[k])) if nn else None
def binned(p,dx,nb=18):
    e=np.quantile(p,np.linspace(0,1,nb+1)); c=0.5*(e[1:]+e[:-1])
    m=[np.median(dx[(p>=e[i])&(p<e[i+1])]) for i in range(nb)]; return c,m
fig,ax=plt.subplots(figsize=(7.5,4.3))
for key,lab,st in [("extrapUTT","extrapUTT (incumbent)",'k-o'),("straight_line","straight line",'r--'),(best,best,'g-s')]:
    if key is None or key not in A.files: continue
    c,m=binned(p,A[key]); ax.plot(c,m,st,label=lab,ms=4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel("p [GeV]"); ax.set_ylabel("median |dx| [um]")
ax.axhline(475,ls=':',c='orange',label='incumbent low-p 475um'); ax.legend(); ax.set_title("UT->T error vs momentum"); plt.tight_layout()""")

md("## 6 · Accuracy ↔ size curve (the capacity question)")
code("""# UT->T median on the plane ref vs parameter count, with the 64KB Allen budget line
res=json.load(open(REF/"wave2_three_arm.json"))
pts=[]
for k,m in res.items():
    if k.startswith("wave2_resid_h") and "params" in m:
        pts.append((m["params"], m["median_dx_um"], m["spec_weighted_median_dx_um"], k))
pts.sort()
if pts:
    par=[p[0] for p in pts]; med=[p[1] for p in pts]; sm=[p[2] for p in pts]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(par,med,'o-',label='flat median'); ax.plot(par,sm,'s--',label='spec-weighted median')
    ax.axvline(16384,c='r',ls='--',label='64KB budget (16384 f32)')
    ax.axhline(11,c='k',ls=':',label='incumbent 11um')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel("params"); ax.set_ylabel("UT->T median |dx| [um]")
    ax.legend(); ax.set_title("Accuracy vs capacity"); plt.tight_layout()
    for p_,m_,s_,k_ in pts: print(f"{k_:<20} params={p_:>7}  med={m_:8.1f}um  specMed={s_:8.1f}um")""")

md("## 7 · Frozen UT→T pool (sanity gate: beat straight line ≥10×)")
code("""fp=json.load(open(REF/"wave2_frozen_pool.json"))
sl=fp["straight_line"]["median_dx_um"]
print(f"straight-line median = {sl:.0f} um")
print(f"{'arm':<22}{'med um':>10}{'x vs SL':>10}")
for k,m in fp.items():
    if k=="straight_line": continue
    print(f"{k:<22}{m['median_dx_um']:>10.1f}{sl/m['median_dx_um']:>9.1f}x")""")

md("## 8 · Verdict")
md("*(filled from the numbers above — see the Notion write-up for the decision.)*")

nb["cells"] = C
out = Path(LAB)/"paper_p0"/"explore_wave2.ipynb"
nbf.write(nb, out)
print("wrote", out)
