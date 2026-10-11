"""Summarise NG-RF experiments -> tokens.json (every number reported in the paper) + figures.
Usage (Kaggle or locally):  python ngrf_report.py /kaggle/working/ngrf_out
"""
import os, sys, json, math, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DS = ["cic17", "unsw", "cse18"]
DS_NAME = {"cic17": "CIC-IDS2017", "unsw": "UNSW-NB15", "cse18": "CSE-CIC-IDS2018"}
METHODS = ["RF", "ExtraTrees", "XGBoost", "LightGBM", "CNN", "BiLSTM", "CNN-LSTM",
           "WRF-Gini", "WRF-Perm", "WRF-MI", "WRF-SHAP", "CNN-FS+RF", "NG-RF"]
WRF = ["WRF-Gini", "WRF-Perm", "WRF-MI", "WRF-SHAP"]
UNSW_CLASSES = ["Benign", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                "Reconnaissance", "Shellcode", "Worms"]
PCT = ["acc", "prec", "rec", "f1", "fpr"]
UNIT = ["auc", "prauc", "mcc"]

def corrected_t(a, b, n_tr, n_te):
    d = np.asarray(a, float) - np.asarray(b, float); J = len(d)
    if J < 2: return float("nan")
    v = np.var(d, ddof=1)
    if v == 0: return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / math.sqrt((1 / J + n_te / n_tr) * v)
    return float(2 * stats.t.sf(abs(t), J - 1))

def holm(p):
    keys = sorted(p, key=lambda k: p[k]); m = len(keys); out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k])); out[k] = run
    return out

def ms(v, dec=2):
    v = np.asarray(v, float)
    return f"{v.mean():.{dec}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{dec}f}"

def fp(p):
    if p is None or (isinstance(p, float) and math.isnan(p)): return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"

def phrase(delta, p, higher_is_better=True):
    if p is None or math.isnan(p) or p >= 0.05: return "not significantly different from"
    better = (delta > 0) == higher_is_better
    return "significantly higher than" if delta > 0 else "significantly lower than"

def main(out_dir):
    RAW = pickle.load(open(os.path.join(out_dir, "results_raw.pkl"), "rb"))
    T = {}
    env = RAW.get("env", {})
    T["ENV.cpu"] = str(env.get("cpu", "n/a")); T["ENV.cores"] = str(env.get("cores", "n/a"))
    T["ENV.gpu"] = str(env.get("gpu", "n/a")); T["ENV.ram"] = str(env.get("ram_gb", "n/a"))
    for k in ("python", "torch", "sklearn", "xgboost", "lightgbm"): T[f"ENV.{k}"] = str(env.get(k, "n/a"))
    cfg = env.get("cfg", {})
    T["CFG.R"] = str(cfg.get("REPEATS", "n/a")); T["CFG.RA"] = str(cfg.get("ABL_REPEATS", "n/a"))
    T["CFG.MAXROWS"] = f"{cfg.get('MAX_ROWS', 0):,}"
    T["CFG.EPOCHS"] = str(cfg.get("EPOCHS", "n/a")); T["CFG.PATIENCE"] = str(cfg.get("PATIENCE", "n/a"))
    summary = {}
    for ds in DS:
        if ds not in RAW or "bin" not in RAW[ds]: continue
        info, B, M = RAW[ds]["info"], RAW[ds]["bin"], RAW[ds]["multi"]
        tun = RAW[ds]["tuning"]
        # ---------- dataset table
        T[f"DS.{ds}.n_raw"] = f"{info['n_raw']:,}"; T[f"DS.{ds}.n_dedup"] = f"{info['n_dedup']:,}"
        T[f"DS.{ds}.n_used"] = f"{info['n_used']:,}"; T[f"DS.{ds}.d"] = str(info["d"])
        T[f"DS.{ds}.n_cls"] = str(len(info["class_counts"])); T[f"DS.{ds}.attack"] = f"{100 * info['attack_frac']:.1f}"
        T[f"DS.{ds}.dup_pct"] = f"{100 * (1 - info['n_dedup'] / max(info['n_clean'], 1)):.1f}"
        T[f"DS.{ds}.n_tr"] = f"{B[0]['n_tr']:,}"; T[f"DS.{ds}.n_va"] = f"{B[0]['n_va']:,}"; T[f"DS.{ds}.n_te"] = f"{B[0]['n_te']:,}"
        # ---------- tuned hyper-parameters
        T[f"HP.{ds}.tau"] = f"{tun['tau']:g}"; T[f"HP.{ds}.alpha"] = f"{tun['alpha']:g}"
        T[f"HP.{ds}.m"] = str(tun["m"]); T[f"HP.{ds}.msl"] = str(tun["msl"])
        T[f"HP.{ds}.xgb"] = str(tun["xgb_depth"]); T[f"HP.{ds}.lgbm"] = str(tun["lgbm_leaves"])
        T[f"HP.{ds}.topk"] = f"{100 * tun['topk_frac']:.0f}%"
        for w in WRF: T[f"HP.{ds}.tau.{w}"] = f"{tun['tau_w'][w]:g}"
        # ---------- binary results
        n_tr, n_te = B[0]["n_tr"], B[0]["n_te"]
        pv = {}
        for mth in METHODS:
            if mth not in B[0]["metrics"]:
                for k in PCT + UNIT + ["p"]: T[f"B.{ds}.{mth}.{k}"] = "n/a"
                continue
            for k in PCT: T[f"B.{ds}.{mth}.{k}"] = ms([r["metrics"][mth][k] for r in B])
            for k in UNIT: T[f"B.{ds}.{mth}.{k}"] = ms([r["metrics"][mth][k] for r in B], 3)
            if mth != "NG-RF":
                pv[mth] = corrected_t([r["metrics"]["NG-RF"]["f1"] for r in B], [r["metrics"][mth]["f1"] for r in B], n_tr, n_te)
        ph = holm(pv)
        for mth in METHODS:
            if mth in ph: T[f"B.{ds}.{mth}.p"] = fp(ph[mth])
        T[f"B.{ds}.NG-RF.p"] = "—"
        mean = lambda mth, k, S=B: float(np.mean([r["metrics"][mth][k] for r in S]))
        for mth in ph:
            dlt = mean("NG-RF", "f1") - mean(mth, "f1")
            T[f"CMP.{ds}.{mth}"] = phrase(dlt, ph[mth]); T[f"DF1.{ds}.{mth}"] = f"{dlt:+.2f}"
        best_wrf = max(WRF, key=lambda w: mean(w, "f1")) if all(w in B[0]["metrics"] for w in WRF) else "WRF-Gini"
        best_all = max([m for m in ["RF", "ExtraTrees", "XGBoost", "LightGBM", "CNN", "BiLSTM", "CNN-LSTM", "CNN-FS+RF"] if m in B[0]["metrics"]], key=lambda w: mean(w, "f1"))
        T[f"BESTWRF.{ds}"] = best_wrf; T[f"BESTBASE.{ds}"] = best_all
        T[f"CMP.{ds}.BESTWRF"] = T[f"CMP.{ds}.{best_wrf}"]; T[f"DF1.{ds}.BESTWRF"] = T[f"DF1.{ds}.{best_wrf}"]
        T[f"P.{ds}.BESTWRF"] = T[f"B.{ds}.{best_wrf}.p"]
        T[f"CMP.{ds}.BESTBASE"] = T[f"CMP.{ds}.{best_all}"]; T[f"DF1.{ds}.BESTBASE"] = T[f"DF1.{ds}.{best_all}"]
        T[f"P.{ds}.BESTBASE"] = T[f"B.{ds}.{best_all}.p"]; T[f"P.{ds}.RF"] = T[f"B.{ds}.RF.p"]
        T[f"NG.{ds}.f1"] = f"{mean('NG-RF', 'f1'):.2f}"; T[f"RF.{ds}.f1"] = f"{mean('RF', 'f1'):.2f}"
        T[f"NG.{ds}.fpr"] = f"{mean('NG-RF', 'fpr'):.2f}"; T[f"RF.{ds}.fpr"] = f"{mean('RF', 'fpr'):.2f}"
        rf_fpr, ng_fpr = mean("RF", "fpr"), mean("NG-RF", "fpr")
        T[f"FPRCHG.{ds}"] = f"{100 * (ng_fpr - rf_fpr) / max(rf_fpr, 1e-9):+.1f}%"
        # ---------- efficiency
        for mth in ["RF", "XGBoost", "LightGBM", "CNN", "BiLSTM", "CNN-LSTM", "NG-RF", "WRF-Gini"]:
            if mth not in B[0]["metrics"]:
                for k in ("tr", "te", "sz"): T[f"E.{ds}.{mth}.{k}"] = "n/a"
                continue
            T[f"E.{ds}.{mth}.tr"] = ms([r["metrics"][mth]["train_min"] for r in B])
            T[f"E.{ds}.{mth}.te"] = ms([r["metrics"][mth]["test_ms"] for r in B], 4)
            key = "size_" + mth
            T[f"E.{ds}.{mth}.sz"] = ms([r["extra"][key] for r in B], 1) if key in B[0]["extra"] else "n/a"
        T[f"NG.{ds}.te"] = f"{mean('NG-RF', 'test_ms'):.4f}"; T[f"RF.{ds}.te"] = f"{mean('RF', 'test_ms'):.4f}"
        T[f"NG.{ds}.thr"] = f"{1000 / max(mean('NG-RF', 'test_ms'), 1e-9):,.0f}"
        T[f"CNN.{ds}.params"] = f"{B[0]['extra']['cnn_params']:,}"
        T[f"CNN.{ds}.epochs"] = ms([r["extra"]["cnn_epochs"] for r in B], 1)
        T[f"CNN.{ds}.min"] = ms([r["extra"]["cnn_min"] for r in B])
        T[f"RHO.{ds}"] = ms([r["extra"]["perm_rho"] for r in B], 3)
        T[f"RHOM.{ds}"] = f"{np.mean([r['extra']['perm_rho'] for r in B]):.2f}"
        T[f"TRATIO.{ds}"] = f"{mean('NG-RF', 'train_min') / max(mean('RF', 'train_min'), 1e-9):.0f}"
        T[f"DIV.{ds}.RF"] = ms([r["extra"]["div_RF"] for r in B], 3); T[f"DIV.{ds}.NG-RF"] = ms([r["extra"]["div_NG-RF"] for r in B], 3)
        # ---------- multi-class
        n_tr, n_te = M[0]["n_tr"], M[0]["n_te"]; pv = {}
        for mth in METHODS:
            if mth not in M[0]["metrics"]: T[f"M.{ds}.{mth}.mf1"] = T[f"M.{ds}.{mth}.acc"] = T[f"M.{ds}.{mth}.p"] = "n/a"; continue
            T[f"M.{ds}.{mth}.mf1"] = ms([r["metrics"][mth]["macro_f1"] for r in M])
            T[f"M.{ds}.{mth}.acc"] = ms([r["metrics"][mth]["acc"] for r in M])
            if mth != "NG-RF":
                pv[mth] = corrected_t([r["metrics"]["NG-RF"]["macro_f1"] for r in M], [r["metrics"][mth]["macro_f1"] for r in M], n_tr, n_te)
        for k, v in holm(pv).items(): T[f"M.{ds}.{k}.p"] = fp(v)
        T[f"M.{ds}.NG-RF.p"] = "—"
        mm = lambda mth: float(np.mean([r["metrics"][mth]["macro_f1"] for r in M]))
        T[f"MNG.{ds}"] = f"{mm('NG-RF'):.2f}"; T[f"MRF.{ds}"] = f"{mm('RF'):.2f}"
        T[f"DMF1.{ds}"] = f"{mm('NG-RF') - mm('RF'):.1f}"
        T[f"MCMP.{ds}.RF"] = phrase(mm("NG-RF") - mm("RF"), holm(pv).get("RF", float("nan")))
        if ds == "unsw":
            gains = []
            for c in UNSW_CLASSES:
                for mth, tag in (("RF", "RF"), ("NG-RF", "NG")):
                    for k in ("p", "r", "f1"):
                        vals = [r["metrics"][mth]["per_class"][c][k] for r in M if c in r["metrics"][mth]["per_class"]]
                        T[f"PC.{c}.{tag}.{k}"] = f"{np.mean(vals):.2f}" if vals else "n/a"
                n = [r["metrics"]["RF"]["per_class"][c]["n"] for r in M if c in r["metrics"]["RF"]["per_class"]]
                T[f"PC.{c}.n"] = f"{int(np.mean(n)):,}" if n else "n/a"
                if T[f"PC.{c}.NG.f1"] != "n/a":
                    gains.append((float(T[f"PC.{c}.NG.f1"]) - float(T[f"PC.{c}.RF.f1"]), c))
            gains.sort(reverse=True)
            for i, (g, c) in enumerate(gains[:3]): T[f"PCG{i + 1}.name"] = c; T[f"PCG{i + 1}.val"] = f"{g:+.2f}"
            T["PCW.name"] = gains[-1][1]; T["PCW.val"] = f"{gains[-1][0]:+.2f}"
            T["PC.n_improved"] = str(sum(g > 0 for g, _ in gains)); T["PC.n_total"] = str(len(gains))
            T["PC.macro.RF"] = T["MRF.unsw"]; T["PC.macro.NG"] = T["MNG.unsw"]
        # ---------- ablations
        if "abl" in RAW[ds]:
            A = RAW[ds]["abl"]
            for a in A[0]["alpha"]:
                T[f"A.{ds}.alpha.{a:g}.f1"] = ms([x["alpha"][a]["f1"] for x in A]); T[f"A.{ds}.alpha.{a:g}.fpr"] = ms([x["alpha"][a]["fpr"] for x in A])
            for t in A[0]["tau"]:
                tk = "inf" if math.isinf(t) else f"{t:g}"
                T[f"A.{ds}.tau.{tk}.f1"] = ms([x["tau"][t]["f1"] for x in A]); T[f"A.{ds}.tau.{tk}.div"] = ms([x["tau"][t]["div"] for x in A], 3)
            for k in A[0]["K"]:
                T[f"A.{ds}.K.{k}.f1"] = ms([x["K"][k]["f1"] for x in A]); T[f"A.{ds}.K.{k}.tr"] = ms([x["K"][k]["train_min"] for x in A])
                T[f"A.{ds}.K.{k}.sz"] = ms([x["K"][k]["size"] for x in A], 1)
            for i, nm in enumerate(A[0]["depth"]):
                T[f"A.{ds}.depth.{i}.f1"] = ms([x["depth"][nm]["f1"] for x in A]); T[f"A.{ds}.depth.{i}.params"] = f"{A[0]['depth'][nm]['params']:,}"
                T[f"A.{ds}.depth.{i}.min"] = ms([x["depth"][nm]["cnn_min"] for x in A])
            T[f"A.{ds}.perm.1.f1"] = ms([x["perm"][1]["f1"] for x in A])
            kp = [k for k in A[0]["perm"] if k != 1][0]
            T[f"A.{ds}.perm.P.f1"] = ms([x["perm"][kp]["f1"] for x in A if x["perm"][kp]]) if A[0]["perm"][kp] else "n/a"
            T[f"A.{ds}.rho"] = ms([x["perm_rho"] for x in A], 3)
        summary[ds] = dict(B=B, M=M, info=info)

    json.dump(T, open(os.path.join(out_dir, "tokens.json"), "w"), indent=1, ensure_ascii=False)
    make_figures(RAW, out_dir)
    # summary of the main comparison
    print("\n=== SUMMARY (binary F1, corrected resampled t-test with Holm adjustment) ===")
    for ds in DS:
        if f"NG.{ds}.f1" in T:
            print(f"{DS_NAME[ds]}: NG-RF {T[f'NG.{ds}.f1']} | RF {T[f'RF.{ds}.f1']} (p={T[f'P.{ds}.RF']}) | "
                  f"best WRF {T[f'BESTWRF.{ds}']} ΔF1 {T[f'DF1.{ds}.BESTWRF']} p={T[f'P.{ds}.BESTWRF']} | "
                  f"best baseline {T[f'BESTBASE.{ds}']} ΔF1 {T[f'DF1.{ds}.BESTBASE']} p={T[f'P.{ds}.BESTBASE']}")
    print("tokens:", len(T), "->", os.path.join(out_dir, "tokens.json"))

# ------------------------------------------------------------------ FIGURES
C = {"RF": "#9aa5b1", "XGBoost": "#5b8db8", "LightGBM": "#7fb3a6", "WRF-Gini": "#d9a441", "WRF-SHAP": "#c77c48", "NG-RF": "#b03a48"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

def make_figures(RAW, out):
    ds_ok = [d for d in DS if d in RAW and "bin" in RAW[d]]
    # Fig 2: F1 and FPR per dataset
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.8), dpi=300)
    meths = [m for m in C if m in RAW[ds_ok[0]]["bin"][0]["metrics"]]
    w = 0.8 / len(meths); x = np.arange(len(ds_ok))
    for j, (k, lab) in enumerate((("f1", "F1-score (%)"), ("fpr", "False positive rate (%)"))):
        for i, m in enumerate(meths):
            v = [[r["metrics"][m][k] for r in RAW[d]["bin"]] for d in ds_ok]
            ax[j].bar(x + (i - len(meths) / 2 + 0.5) * w, [np.mean(a) for a in v], w, yerr=[np.std(a, ddof=1) if len(a) > 1 else 0 for a in v],
                      color=C[m], label=m, capsize=1.5, error_kw=dict(lw=0.6))
        ax[j].set_xticks(x); ax[j].set_xticklabels([DS_NAME[d] for d in ds_ok], fontsize=7.5); ax[j].set_ylabel(lab)
        if k == "f1":
            lo = min(np.mean([r["metrics"][m]["f1"] for r in RAW[d]["bin"]]) for d in ds_ok for m in meths)
            ax[j].set_ylim(max(0, lo - 3), 100)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(meths), fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(os.path.join(out, "fig2_binary.png")); plt.close(fig)
    # Fig 3: per-class F1 (UNSW-NB15, multi-class)
    fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)
    if "unsw" in RAW and "multi" in RAW["unsw"]:
        M = RAW["unsw"]["multi"]; cls = [c for c in UNSW_CLASSES if c in M[0]["metrics"]["RF"]["per_class"]]
        x = np.arange(len(cls))
        for i, (m, col) in enumerate((("RF", C["RF"]), ("NG-RF", C["NG-RF"]))):
            v = [[r["metrics"][m]["per_class"][c]["f1"] for r in M] for c in cls]
            ax.bar(x + (i - 0.5) * 0.38, [np.mean(a) for a in v], 0.38, yerr=[np.std(a, ddof=1) if len(a) > 1 else 0 for a in v],
                   color=col, label=m, capsize=1.5, error_kw=dict(lw=0.6))
        ax.set_xticks(x); ax.set_xticklabels(cls, rotation=30, ha="right", fontsize=7.5); ax.set_ylabel("Per-class F1 (%)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_perclass.png")); plt.close(fig)
    # Fig 4: sampling probability vs Gini importance, top-15 features (first dataset)
    d0 = ds_ok[0]; B = RAW[d0]["bin"]; feats = B[0]["features"]
    P = np.mean([r["extra"]["P"] for r in B], 0); Gi = np.mean([r["extra"]["W"]["WRF-Gini"] for r in B], 0)
    U_rf = np.mean([r["extra"]["usage_RF"] for r in B], 0); U_ng = np.mean([r["extra"]["usage_NG-RF"] for r in B], 0)
    top = np.argsort(-P)[:15][::-1]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.4), dpi=300, sharey=True)
    y = np.arange(len(top))
    ax[0].barh(y - 0.2, P[top] / P.max(), 0.4, color=C["NG-RF"], label="NG-RF sampling prob. $P_\\tau$")
    ax[0].barh(y + 0.2, Gi[top] / Gi.max(), 0.4, color=C["WRF-Gini"], label="RF Gini importance")
    ax[0].set_yticks(y); ax[0].set_yticklabels([feats[i][:28] for i in top], fontsize=6.5); ax[0].set_xlabel("Normalised score")
    ax[0].legend(fontsize=6, frameon=False, loc="lower right")
    ax[1].barh(y - 0.2, U_ng[top] * 100, 0.4, color=C["NG-RF"], label="NG-RF")
    ax[1].barh(y + 0.2, U_rf[top] * 100, 0.4, color=C["RF"], label="RF")
    ax[1].set_xlabel("Share of split nodes (%)"); ax[1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle(DS_NAME[d0], fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_importance.png")); plt.close(fig)
    # Fig 5: tau sensitivity
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), dpi=300)
    for d, mk in zip(ds_ok, "osd"):
        if "abl" not in RAW[d]: continue
        A = RAW[d]["abl"]; ts = list(A[0]["tau"]); xs = np.arange(len(ts))
        ax[0].errorbar(xs, [np.mean([a["tau"][t]["f1"] for a in A]) for t in ts], [np.std([a["tau"][t]["f1"] for a in A]) for t in ts], marker=mk, ms=3, lw=1, capsize=2, label=DS_NAME[d])
        ax[1].plot(xs, [np.mean([a["tau"][t]["div"] for a in A]) for t in ts], marker=mk, ms=3, lw=1, label=DS_NAME[d])
        lab = ["∞" if math.isinf(t) else f"{t:g}" for t in ts]
    for a_, yl in zip(ax, ("F1-score (%)", "Tree disagreement")):
        a_.set_xticks(xs); a_.set_xticklabels(lab); a_.set_xlabel("Temperature τ"); a_.set_ylabel(yl)
    ax[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_tau.png")); plt.close(fig)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/ngrf_out")
