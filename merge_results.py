"""Merge results_raw.pkl files from several Kaggle sessions (one dataset each) and rebuild tokens/figures.
Usage: python merge_results.py out_merged run_cic17/results_raw.pkl run_unsw/results_raw.pkl run_cse18/results_raw.pkl"""
import sys, os, pickle
out = sys.argv[1]; os.makedirs(out, exist_ok=True); M = {}
for p in sys.argv[2:]:
    R = pickle.load(open(p, "rb"))
    for k, v in R.items():
        if k == "env": M.setdefault("env", v)
        else: M[k] = v
pickle.dump(M, open(os.path.join(out, "results_raw.pkl"), "wb"))
import ngrf_report; ngrf_report.main(out)
