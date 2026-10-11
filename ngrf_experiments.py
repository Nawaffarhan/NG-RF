# %% [markdown]
# # NG-RF: Neural-Guided Random Forest - full experimental pipeline
#
# Produces every result, table value and figure reported in the paper.
#
# * Datasets: CIC-IDS2017, UNSW-NB15 (partitioned sets), CSE-CIC-IDS2018
# * Protocol: 5 repeated stratified 60/20/20 train/validation/test splits.
#   All hyper-parameters (tau, alpha, RF/XGB/LGBM settings) are chosen on the
#   validation partition of repeat 0 and then frozen. Test partitions are used once.
# * Output: OUT_DIR/{results_raw.pkl, tokens.json, fig*.png}; tokens.json holds every number in the paper.
#
# Every (dataset, repeat) is checkpointed in OUT_DIR/ckpt, so an interrupted run resumes where it stopped.
# To split the work, run one dataset per session (NGRF_DATASETS=unsw, ...) and merge with merge_results.py.
# %%
import os, sys, gc, json, time, glob, math, pickle, platform, warnings, itertools, subprocess
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix, precision_recall_fscore_support)
from scipy import stats

import torch
import torch.nn as nn

# %% ---------------------------------------------------------------- CONFIG
CFG = dict(
    # paths (recursive globs). Override with NGRF_CFG_JSON, e.g. '{"CIC17_GLOB": "data/cic17/*.csv"}'.
    # Defaults below are the settings used for every result in the paper.
    CIC17_GLOB = "/kaggle/input/**/*.pcap_ISCX.csv",
    UNSW_TRAIN = "/kaggle/input/**/UNSW_NB15_training-set.csv",
    UNSW_TEST  = "/kaggle/input/**/UNSW_NB15_testing-set.csv",
    CSE18_GLOB = "/kaggle/input/**/*TrafficForML_CICFlowMeter.csv",
    OUT_DIR    = "/kaggle/working/ngrf_out",

    DATASETS   = os.environ.get("NGRF_DATASETS", "cic17,unsw,cse18").split(","),
    REPEATS    = int(os.environ.get("NGRF_REPEATS", 5)),
    ABL_REPEATS= int(os.environ.get("NGRF_ABL_REPEATS", 3)),
    MAX_ROWS   = int(os.environ.get("NGRF_MAX_ROWS", 100_000)),   # after cleaning/dedup, stratified cap
    CSE18_FRAC_BENIGN = 0.05, CSE18_FRAC_ATTACK = 0.35,            # chunk sampling while reading 16M rows
    MIN_CLASS  = 50,                                                # smaller classes merged into "Other"

    K          = 100,           # trees in all forests (main results)
    POOL_MULT  = 3,             # replication pool size = POOL_MULT * d
    TAU_GRID   = [0.5, 1.0, 2.0, 5.0, 10.0],
    ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0],
    TUNE_ROWS  = 150_000,       # train rows used during validation tuning
    TUNE_K     = 50,

    CNN_CHANNELS = (64, 128, 256), CNN_FC = (512, 128), DROPOUT = 0.3,
    EPOCHS = 20, PATIENCE = 3, BATCH = 256, LR = 1e-3, WD = 1e-4,
    N_PERM = 3,                 # CNNs trained on random column orders (order invariance)
    N_IMP  = 20_000,            # training samples used to compute importances
    RUN_SEQ_BASELINES = False,  # BiLSTM & CNN-LSTM (not used in the paper; needs a GPU to be fast)
    RUN_ABLATIONS = True,
    SEED = 42,
)
CFG.update(json.loads(os.environ.get("NGRF_CFG_JSON", "{}")))   # optional overrides
os.makedirs(CFG["OUT_DIR"], exist_ok=True)
CKPT = os.path.join(CFG["OUT_DIR"], "ckpt"); os.makedirs(CKPT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_JOBS = -1
print("device:", DEV, "| cpu:", platform.processor() or platform.machine(), "| cores:", os.cpu_count())

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# %% ---------------------------------------------------------------- DATA LOADING
def _glob(p):
    f = sorted(glob.glob(p, recursive=True))
    if not f: raise FileNotFoundError(p)
    return f

DROP_ID_COLS = {"flow id", "src ip", "source ip", "dst ip", "destination ip", "src port", "source port",
                "dst port", "destination port", "timestamp", "protocol_name"}

CIC17_MAP = {"BENIGN": "Benign", "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS",
             "DoS Slowhttptest": "DoS", "DDoS": "DDoS", "PortScan": "PortScan", "FTP-Patator": "BruteForce",
             "SSH-Patator": "BruteForce", "Bot": "Bot", "Infiltration": "Infiltration", "Heartbleed": "Heartbleed"}
def map_cic17(l):
    l = str(l).strip()
    if l.startswith("Web Attack"): return "WebAttack"
    return CIC17_MAP.get(l, l)

def map_cse18(l):
    l = str(l).strip()
    if l == "Benign": return "Benign"
    if l.startswith("DDOS") or l.startswith("DDoS"): return "DDoS"
    if l.startswith("DoS"): return "DoS"
    if l in ("FTP-BruteForce", "SSH-Bruteforce"): return "BruteForce"
    if l.startswith("Brute Force") or l == "SQL Injection": return "WebAttack"
    if l == "Bot": return "Bot"
    if l == "Infilteration": return "Infiltration"
    return l

def clean_numeric(df, label_col):
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    y = df[label_col].astype(str)
    X = df.drop(columns=[c for c in df.columns if c == label_col or c.lower() in DROP_ID_COLS])
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1)
    return X[ok].astype("float32"), y[ok]

def load_cic17():
    Xs, ys, n_raw = [], [], 0
    for f in _glob(CFG["CIC17_GLOB"]):                      # clean file by file to bound memory
        d = pd.read_csv(f, low_memory=False, encoding="latin-1")
        n_raw += len(d)
        X, y = clean_numeric(d, "Label"); del d
        Xs.append(X); ys.append(y.map(map_cic17))
    cols = Xs[0].columns
    X = pd.concat([x[cols] for x in Xs], ignore_index=True); y = pd.concat(ys, ignore_index=True)
    return X, y, None, n_raw

def load_unsw():
    df = pd.concat([pd.read_csv(_glob(CFG["UNSW_TRAIN"])[0]), pd.read_csv(_glob(CFG["UNSW_TEST"])[0])], ignore_index=True)
    n_raw = len(df)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.drop(columns=[c for c in ["id", "label"] if c in df.columns])
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip().replace({"Normal": "Benign", "Backdoors": "Backdoor"})
    cats = ["proto", "service", "state"]
    y = df.pop("attack_cat")
    for c in df.columns:
        if c not in cats: df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df.drop(columns=cats).notna().all(axis=1)
    return df[ok], y[ok], cats, n_raw

def load_cse18():
    parts, n_raw = [], 0
    rng = np.random.default_rng(CFG["SEED"])
    for f in _glob(CFG["CSE18_GLOB"]):
        for ch in pd.read_csv(f, chunksize=500_000, low_memory=False):
            ch.columns = [c.strip() for c in ch.columns]
            ch = ch[ch["Label"].astype(str) != "Label"]          # repeated header rows
            n_raw += len(ch)
            ben = ch["Label"].astype(str).str.strip() == "Benign"
            frac = np.where(ben, CFG["CSE18_FRAC_BENIGN"], CFG["CSE18_FRAC_ATTACK"])
            keep = rng.random(len(ch)) < frac
            ch = ch[keep]
            X, y = clean_numeric(ch, "Label")
            X["__y"] = y.map(map_cse18).values
            parts.append(X)
        log("read", os.path.basename(f), "rows so far", n_raw)
    df = pd.concat(parts, ignore_index=True); del parts
    common = [c for c in df.columns if c != "__y"]
    df = df.dropna(subset=common)                                 # files with extra columns
    y = df.pop("__y")
    return df, y, None, n_raw

LOADERS = {"cic17": load_cic17, "unsw": load_unsw, "cse18": load_cse18}
DS_NAME = {"cic17": "CIC-IDS2017", "unsw": "UNSW-NB15", "cse18": "CSE-CIC-IDS2018"}

def prepare(ds):
    cache = os.path.join(CKPT, f"{ds}_data.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    log("loading", ds)
    X, y, cats, n_raw = LOADERS[ds]()
    X = X.reset_index(drop=True); y = y.reset_index(drop=True)
    n_clean = len(X)
    # duplicate removal (identical feature vector + label)
    dup = pd.concat([X, y.rename("__y")], axis=1).duplicated()
    X, y = X[~dup].reset_index(drop=True), y[~dup].reset_index(drop=True)
    n_dedup = len(X)
    # conflicting duplicates (same features, different labels) are removed as label noise
    conf = X.duplicated(keep=False)
    X, y = X[~conf].reset_index(drop=True), y[~conf].reset_index(drop=True)
    # drop constant columns
    num = [c for c in X.columns if cats is None or c not in cats]
    const = [c for c in num if X[c].nunique() <= 1]
    X = X.drop(columns=const)
    # merge tiny classes
    vc = y.value_counts()
    y = y.where(~y.isin(vc[vc < CFG["MIN_CLASS"]].index), "Other")
    # stratified cap
    n = len(X)
    if n > CFG["MAX_ROWS"]:
        frac = CFG["MAX_ROWS"] / n
        rng = np.random.default_rng(CFG["SEED"]); idx = []
        for c, g in y.groupby(y).groups.items():
            g = np.asarray(g); k = max(int(round(frac * len(g))), min(len(g), 5000))
            idx.append(rng.choice(g, size=min(k, len(g)), replace=False))
        idx = np.sort(np.concatenate(idx))
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    info = dict(n_raw=int(n_raw), n_clean=int(n_clean), n_dedup=int(n_dedup), n_conflict=int(conf.sum()),
                n_const=len(const), n_used=len(X), d=X.shape[1],
                class_counts={k: int(v) for k, v in y.value_counts().items()},
                attack_frac=float((y != "Benign").mean()))
    log(ds, info)
    out = (X, y, cats, info)
    pickle.dump(out, open(cache, "wb"), protocol=4)
    return out

def make_split(X, y, rep):
    s = CFG["SEED"] + rep
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
    tr, va = train_test_split(tr, test_size=0.25, stratify=y.iloc[tr], random_state=s)
    return tr, va, te

class Prep:
    """Frequency-encode categoricals, signed-log transform, standardise. Fitted on TRAIN only."""
    def __init__(self, cats): self.cats = cats or []
    def fit(self, X):
        self.freq = {c: X[c].astype(str).value_counts(normalize=True) for c in self.cats}
        Z = self._t(X); self.sc = StandardScaler().fit(Z); self.cols = list(X.columns); return self
    def _t(self, X):
        X = X.copy()
        for c in self.cats: X[c] = X[c].astype(str).map(self.freq[c]).fillna(0.0)
        Z = X.values.astype("float64")
        return np.sign(Z) * np.log1p(np.abs(Z))
    def transform(self, X):
        return np.clip(self.sc.transform(self._t(X)), -10, 10).astype("float32")

# %% ---------------------------------------------------------------- METRICS
def binary_metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=100 * accuracy_score(y, pred), prec=100 * precision_score(y, pred, zero_division=0),
                rec=100 * recall_score(y, pred), f1=100 * f1_score(y, pred), fpr=100 * fp / max(fp + tn, 1),
                auc=roc_auc_score(y, prob), prauc=average_precision_score(y, prob), mcc=matthews_corrcoef(y, pred))

def multiclass_metrics(y, pred, classes):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=np.arange(len(classes)), zero_division=0)
    return dict(acc=100 * accuracy_score(y, pred), macro_f1=100 * f.mean(),
                per_class={c: dict(p=100 * p[i], r=100 * r[i], f1=100 * f[i], n=int(s[i])) for i, c in enumerate(classes)})

# %% ---------------------------------------------------------------- CNN FEATURE LEARNER
class CNN1D(nn.Module):
    def __init__(self, d, n_cls, chans=(64, 128, 256), fc=(512, 128), p=0.3):
        super().__init__()
        blocks, cin, L = [], 1, d
        for c in chans:
            blocks.append(nn.ModuleDict(dict(conv=nn.Conv1d(cin, c, 3, padding=1), bn=nn.BatchNorm1d(c))))
            cin, L = c, math.ceil(L / 2)
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool1d(2, ceil_mode=True)
        layers, fin = [nn.Flatten(), nn.Dropout(p)], cin * L
        for h in fc: layers += [nn.Linear(fin, h), nn.ReLU(), nn.Dropout(p)]; fin = h
        layers.append(nn.Linear(fin, n_cls))
        self.head = nn.Sequential(*layers)
    def forward(self, x, return_act=False):
        h = x.unsqueeze(1); act = None
        for i, b in enumerate(self.blocks):
            h = torch.relu(b["bn"](b["conv"](h)))
            if i == 0: act = h                      # first-block responses, shape (n, c1, d)
            h = self.pool(h)
        out = self.head(h)
        return (out, act) if return_act else out

def n_params(m): return int(sum(p.numel() for p in m.parameters()))

def class_weights(y, n_cls):
    cnt = np.bincount(y, minlength=n_cls).astype("float64")
    w = 1.0 / np.sqrt(np.maximum(cnt, 1)); w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEV)

def train_net(net, Xtr, ytr, Xva, yva, n_cls, seed):
    seed_all(seed); net = net.to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=CFG["LR"], weight_decay=CFG["WD"])
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr, n_cls))
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    Xv, yv = torch.tensor(Xva).to(DEV), torch.tensor(yva, dtype=torch.long).to(DEV)
    best, best_state, bad, ep_run = 1e9, None, 0, 0
    t0 = time.time()
    for ep in range(CFG["EPOCHS"]):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), CFG["BATCH"]):
            b = perm[i:i + CFG["BATCH"]]
            xb, yb = Xt[b].to(DEV), yt[b].to(DEV)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
        net.eval(); vl = 0.0
        with torch.no_grad():
            for i in range(0, len(Xv), 8192): vl += lossf(net(Xv[i:i + 8192]), yv[i:i + 8192]).item() * len(Xv[i:i + 8192])
        vl /= len(Xv); ep_run = ep + 1
        if vl < best - 1e-5: best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["PATIENCE"]: break
    net.load_state_dict(best_state); net.eval()
    return net, time.time() - t0, ep_run

def net_proba(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            out.append(torch.softmax(net(torch.tensor(X[i:i + 8192]).to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)

def cnn_importance(net, X, y, n_cls):
    """Gradient importance (Eq. 16) and activation importance (Eq. 18) on a training subset."""
    lossf = nn.CrossEntropyLoss(weight=class_weights(y, n_cls), reduction="sum")
    g_sum = np.zeros(X.shape[1]); a_sum = np.zeros(X.shape[1])
    for i in range(0, len(X), 4096):
        xb = torch.tensor(X[i:i + 4096]).to(DEV).requires_grad_(True)
        yb = torch.tensor(y[i:i + 4096], dtype=torch.long).to(DEV)
        out, act = net(xb, return_act=True)
        lossf(out, yb).backward()
        g_sum += xb.grad.abs().sum(0).detach().cpu().numpy()
        a_sum += act.detach().norm(dim=1).sum(0).cpu().numpy()
    return g_sum / len(X), a_sum / len(X)

def learn_importances(Xtr, ytr, Xva, yva, n_cls, seed, n_perm=None, chans=None):
    """Train n_perm CNNs on random column orders; return averaged normalised importances."""
    n_perm = n_perm or CFG["N_PERM"]; chans = chans or CFG["CNN_CHANNELS"]
    d = Xtr.shape[1]; rng = np.random.default_rng(seed)
    sub = rng.choice(len(Xtr), size=min(CFG["N_IMP"], len(Xtr)), replace=False)
    G, A, nets, secs = [], [], [], 0.0
    for p in range(n_perm):
        pi = np.arange(d) if p == 0 else rng.permutation(d)
        net = CNN1D(d, n_cls, chans, CFG["CNN_FC"], CFG["DROPOUT"])
        net, t, ep = train_net(net, Xtr[:, pi], ytr, Xva[:, pi], yva, n_cls, seed + p)
        secs += t
        g, a = cnn_importance(net, Xtr[sub][:, pi], ytr[sub], n_cls)
        gi, ai = np.zeros(d), np.zeros(d); gi[pi] = g; ai[pi] = a      # map back to original order
        G.append(gi / gi.max()); A.append(ai / ai.max()); nets.append((net, pi, ep))
    rho = [stats.spearmanr(G[i] * 0.5 + A[i] * 0.5, G[j] * 0.5 + A[j] * 0.5)[0] for i, j in itertools.combinations(range(n_perm), 2)]
    return dict(G=np.mean(G, 0), A=np.mean(A, 0), G_list=G, A_list=A, nets=nets, cnn_secs=secs,
                perm_rho=float(np.mean(rho)) if rho else float("nan"))

def combine(G, A, alpha):
    I = alpha * G / G.max() + (1 - alpha) * A / A.max()
    return np.maximum(I, 1e-3)

def to_prob(I, tau):
    if tau is None or np.isinf(tau): return np.full(len(I), 1.0 / len(I))
    w = np.power(I / I.max(), 1.0 / tau); return w / w.sum()

# %% ---------------------------------------------------------------- WEIGHTED-SAMPLING FOREST
class WeightedRF:
    """Random forest whose per-node candidate features are drawn with probability P(j).
    Implemented exactly through column replication: feature j appears r_j = max(1, round(M*P(j)))
    times in a pool of M = POOL_MULT*d columns; scikit-learn draws m candidate columns per node
    uniformly from the pool, so feature j is drawn with probability r_j / sum(r)."""
    def __init__(self, P=None, K=100, m=None, min_samples_leaf=1, seed=0, forest="rf"):
        self.P, self.K, self.m, self.msl, self.seed, self.forest = P, K, m, min_samples_leaf, seed, forest
    def _idx(self, d):
        if self.P is None: return np.arange(d)
        M = CFG["POOL_MULT"] * d
        r = np.maximum(1, np.round(self.P * M)).astype(int)
        return np.repeat(np.arange(d), r)
    def fit(self, X, y):
        d = X.shape[1]; self.idx = self._idx(d)
        m = self.m or max(1, int(round(math.sqrt(d))))
        Cls = RandomForestClassifier if self.forest == "rf" else ExtraTreesClassifier
        self.model = Cls(n_estimators=self.K, max_features=min(m, len(self.idx)), min_samples_leaf=self.msl,
                         n_jobs=N_JOBS, random_state=self.seed)
        self.model.fit(X[:, self.idx], y); return self
    def predict_proba(self, X): return self.model.predict_proba(X[:, self.idx])
    def size_mb(self): return len(pickle.dumps(self.model, protocol=4)) / 2 ** 20
    def diversity(self, X, n_pairs=30, n=5000, seed=0):
        rng = np.random.default_rng(seed); Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)][:, self.idx]
        est = self.model.estimators_; preds = {}
        dis = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(est), 2, replace=False)
            for k in (i, j):
                if k not in preds: preds[k] = est[k].predict(Xs)
            dis.append((preds[i] != preds[j]).mean())
        return float(np.mean(dis))
    def feature_usage(self, d):
        cnt = np.zeros(d)
        for t in self.model.estimators_:
            f = t.tree_.feature; f = f[f >= 0]
            np.add.at(cnt, self.idx[f], 1)
        return cnt / cnt.sum()

def fit_eval_forest(P, Xtr, ytr, Xte, yte, K, m, msl, seed, forest="rf", binary=True, classes=None):
    t0 = time.time(); f = WeightedRF(P, K, m, msl, seed, forest).fit(Xtr, ytr); ttr = time.time() - t0
    t0 = time.time(); pr = f.predict_proba(Xte); tte = (time.time() - t0) / len(Xte) * 1e3
    res = binary_metrics(yte, pr[:, 1]) if binary else multiclass_metrics(yte, pr.argmax(1), classes)
    res.update(train_min=ttr / 60, test_ms=tte)
    return f, res

# %% ---------------------------------------------------------------- OTHER BASELINES
def run_xgb(Xtr, ytr, Xva, yva, Xte, n_cls, depth, seed):
    import xgboost as xgb
    kw = dict(n_estimators=1000, learning_rate=0.1, max_depth=depth, tree_method="hist", random_state=seed,
              n_jobs=N_JOBS, early_stopping_rounds=30, eval_metric="logloss" if n_cls == 2 else "mlogloss")
    if torch.cuda.is_available(): kw["device"] = "cuda"
    m = xgb.XGBClassifier(**kw)
    t0 = time.time(); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False); ttr = time.time() - t0
    t0 = time.time(); pr = m.predict_proba(Xte); tte = (time.time() - t0) / len(Xte) * 1e3
    return m, pr, ttr, tte

def run_lgbm(Xtr, ytr, Xva, yva, Xte, n_cls, leaves, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=leaves, random_state=seed,
                           n_jobs=N_JOBS, verbose=-1)
    t0 = time.time(); m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)]); ttr = time.time() - t0
    t0 = time.time(); pr = m.predict_proba(Xte); tte = (time.time() - t0) / len(Xte) * 1e3
    return m, pr, ttr, tte

class BiLSTM(nn.Module):
    def __init__(self, d, n_cls, h=64):
        super().__init__(); self.lstm = nn.LSTM(1, h, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, n_cls))
    def forward(self, x):
        o, _ = self.lstm(x.unsqueeze(-1)); return self.fc(o[:, -1])

class CNNLSTM(nn.Module):
    def __init__(self, d, n_cls, h=64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True))
        self.lstm = nn.LSTM(64, h, batch_first=True); self.fc = nn.Linear(h, n_cls)
    def forward(self, x):
        h = self.conv(x.unsqueeze(1)).transpose(1, 2); o, _ = self.lstm(h); return self.fc(o[:, -1])

def eval_net(net, Xte, yte, binary, classes):
    t0 = time.time(); pr = net_proba(net, Xte); tte = (time.time() - t0) / len(Xte) * 1e3
    res = binary_metrics(yte, pr[:, 1]) if binary else multiclass_metrics(yte, pr.argmax(1), classes)
    res["test_ms"] = tte; return res

def importance_baselines(rf_uniform, Xtr, ytr, Xva, yva, xgb_model, seed):
    """Non-neural feature weights for the closest-work weighted forests."""
    d = Xtr.shape[1]; rng = np.random.default_rng(seed)
    W = {}
    W["WRF-Gini"] = rf_uniform.model.feature_importances_.copy()
    sub = rng.choice(len(Xva), min(10000, len(Xva)), replace=False)
    pi = permutation_importance(rf_uniform.model, Xva[sub][:, rf_uniform.idx], yva[sub], n_repeats=3, random_state=seed, n_jobs=1,
                                scoring="f1_macro")
    W["WRF-Perm"] = np.maximum(pi.importances_mean, 0)
    sub = rng.choice(len(Xtr), min(50000, len(Xtr)), replace=False)
    W["WRF-MI"] = mutual_info_classif(Xtr[sub], ytr[sub], random_state=seed)
    import xgboost as xgb
    sub = rng.choice(len(Xva), min(5000, len(Xva)), replace=False)
    contrib = xgb_model.get_booster().predict(xgb.DMatrix(Xva[sub]), pred_contribs=True)
    contrib = np.abs(contrib).sum(1) if contrib.ndim == 3 else np.abs(contrib)   # multi-class: (n, C, d+1)
    W["WRF-SHAP"] = contrib[:, :d].mean(0) if contrib.ndim == 2 else contrib[:d]
    for k in W: W[k] = np.maximum(W[k] / max(W[k].max(), 1e-12), 1e-3)
    return W

# %% ---------------------------------------------------------------- TUNING (repeat 0, validation only)
def tune(ds, Xtr, ytr, Xva, yva, n_cls, imp, W):
    path = os.path.join(CKPT, f"{ds}_tuning.json")
    if os.path.exists(path): return json.load(open(path))
    rng = np.random.default_rng(0)
    s = rng.choice(len(Xtr), min(CFG["TUNE_ROWS"], len(Xtr)), replace=False)
    Xs, ys = Xtr[s], ytr[s]; d = Xtr.shape[1]; msq = max(1, int(round(math.sqrt(d))))
    def score(P, m, msl):
        f = WeightedRF(P, CFG["TUNE_K"], m, msl, 0).fit(Xs, ys)
        return f1_score(yva, f.predict_proba(Xva).argmax(1), average="macro")
    T = dict()
    # 1) RF tree hyper-parameters (shared by every forest variant -> fair comparison)
    grid = [(m, msl) for m in (msq, 2 * msq) for msl in (1, 3)]
    sc = {g: score(None, *g) for g in grid}; T["m"], T["msl"] = max(sc, key=sc.get)
    T["rf_grid"] = {f"m={k[0]},msl={k[1]}": v for k, v in sc.items()}
    # 2) NG-RF: tau at alpha=0.5, then alpha at best tau (coordinate search)
    sc = {t: score(to_prob(combine(imp["G"], imp["A"], 0.5), t), T["m"], T["msl"]) for t in CFG["TAU_GRID"]}
    T["tau"] = max(sc, key=sc.get)
    sc2 = {a: score(to_prob(combine(imp["G"], imp["A"], a), T["tau"]), T["m"], T["msl"]) for a in CFG["ALPHA_GRID"]}
    T["alpha"] = max(sc2, key=sc2.get); T["tau_val"] = sc; T["alpha_val"] = sc2
    # 3) every non-neural weighting gets its own tau (same budget as NG-RF)
    T["tau_w"] = {}
    for k, w in W.items():
        sc = {t: score(to_prob(w, t), T["m"], T["msl"]) for t in CFG["TAU_GRID"]}
        T["tau_w"][k] = max(sc, key=sc.get)
    # 4) CNN top-k hard selection: k chosen on validation
    I = combine(imp["G"], imp["A"], T["alpha"])
    sc = {}
    for frac in (0.25, 0.5, 0.75):
        k = max(2, int(round(frac * d))); top = np.argsort(-I)[:k]
        f = WeightedRF(None, CFG["TUNE_K"], min(T["m"], k), T["msl"], 0).fit(Xs[:, top], ys)
        sc[frac] = f1_score(yva, f.predict_proba(Xva[:, top]).argmax(1), average="macro")
    T["topk_frac"] = max(sc, key=sc.get)
    # 5) boosted trees
    sc = {}
    for dep in (6, 10):
        _, pr, _, _ = run_xgb(Xs, ys, Xva, yva, Xva, n_cls, dep, 0); sc[dep] = f1_score(yva, pr.argmax(1), average="macro")
    T["xgb_depth"] = max(sc, key=sc.get)
    sc = {}
    for lv in (63, 255):
        _, pr, _, _ = run_lgbm(Xs, ys, Xva, yva, Xva, n_cls, lv, 0); sc[lv] = f1_score(yva, pr.argmax(1), average="macro")
    T["lgbm_leaves"] = max(sc, key=sc.get)
    T = json.loads(json.dumps(T, default=float))
    json.dump(T, open(path, "w"), indent=1); log("tuning", ds, T)
    return T

# %% ---------------------------------------------------------------- ONE REPEAT
def run_repeat(ds, rep, X, y, cats, task):
    """task: 'bin' (benign vs attack) or 'multi' (attack families)."""
    path = os.path.join(CKPT, f"{ds}_{task}_r{rep}.pkl")
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    seed = CFG["SEED"] + rep; seed_all(seed)
    tr, va, te = make_split(X, y, rep)
    pp = Prep(cats).fit(X.iloc[tr])
    Xtr, Xva, Xte = pp.transform(X.iloc[tr]), pp.transform(X.iloc[va]), pp.transform(X.iloc[te])
    if task == "bin":
        classes = ["Benign", "Attack"]; yy = (y != "Benign").astype(int).values
    else:
        le = LabelEncoder().fit(y); classes = list(le.classes_); yy = le.transform(y)
    ytr, yva, yte = yy[tr], yy[va], yy[te]
    n_cls, binary = len(classes), task == "bin"
    R = dict(ds=ds, rep=rep, task=task, classes=classes, n_tr=len(tr), n_va=len(va), n_te=len(te),
             features=list(pp.cols), metrics={}, extra={})

    log(ds, task, "rep", rep, "CNN importances")
    imp = learn_importances(Xtr, ytr, Xva, yva, n_cls, seed)
    R["extra"]["G"], R["extra"]["A"], R["extra"]["perm_rho"] = imp["G"].tolist(), imp["A"].tolist(), imp["perm_rho"]
    R["extra"]["G_list"] = [g.tolist() for g in imp["G_list"]]; R["extra"]["A_list"] = [a.tolist() for a in imp["A_list"]]
    R["extra"]["cnn_params"] = n_params(imp["nets"][0][0]); R["extra"]["cnn_epochs"] = imp["nets"][0][2]
    R["extra"]["cnn_min"] = imp["cnn_secs"] / 60

    # uniform RF (needed for tuning weights) with provisional params
    d = Xtr.shape[1]; msq = max(1, int(round(math.sqrt(d))))
    tpath = os.path.join(CKPT, f"{ds}_tuning.json")
    if task == "bin" and not os.path.exists(tpath):
        rf0 = WeightedRF(None, CFG["TUNE_K"], msq, 1, seed).fit(Xtr[:min(len(Xtr), CFG["TUNE_ROWS"])], ytr[:min(len(Xtr), CFG["TUNE_ROWS"])])
        xgb0, _, _, _ = run_xgb(Xtr[:CFG["TUNE_ROWS"]], ytr[:CFG["TUNE_ROWS"]], Xva, yva, Xva, n_cls, 6, seed)
        W0 = importance_baselines(rf0, Xtr, ytr, Xva, yva, xgb0, seed)
        tune(ds, Xtr, ytr, Xva, yva, n_cls, imp, W0)
    T = json.load(open(tpath))
    K, m, msl = CFG["K"], T["m"], T["msl"]

    # ---- forests
    rf, R["metrics"]["RF"] = fit_eval_forest(None, Xtr, ytr, Xte, yte, K, m, msl, seed, binary=binary, classes=classes)
    R["extra"]["size_RF"] = rf.size_mb(); R["extra"]["div_RF"] = rf.diversity(Xva, seed=seed)
    R["extra"]["usage_RF"] = rf.feature_usage(d).tolist()
    _, R["metrics"]["ExtraTrees"] = fit_eval_forest(None, Xtr, ytr, Xte, yte, K, m, msl, seed, forest="et", binary=binary, classes=classes)

    I = combine(imp["G"], imp["A"], T["alpha"]); P = to_prob(I, T["tau"])
    ng, R["metrics"]["NG-RF"] = fit_eval_forest(P, Xtr, ytr, Xte, yte, K, m, msl, seed, binary=binary, classes=classes)
    R["metrics"]["NG-RF"]["train_min"] += imp["cnn_secs"] / 60           # include CNN cost
    R["extra"]["size_NG-RF"] = ng.size_mb(); R["extra"]["div_NG-RF"] = ng.diversity(Xva, seed=seed)
    R["extra"]["usage_NG-RF"] = ng.feature_usage(d).tolist(); R["extra"]["P"] = P.tolist()
    del ng; gc.collect()

    # ---- boosted trees
    xg, pr, ttr, tte = run_xgb(Xtr, ytr, Xva, yva, Xte, n_cls, T["xgb_depth"], seed)
    R["metrics"]["XGBoost"] = (binary_metrics(yte, pr[:, 1]) if binary else multiclass_metrics(yte, pr.argmax(1), classes))
    R["metrics"]["XGBoost"].update(train_min=ttr / 60, test_ms=tte)
    R["extra"]["size_XGBoost"] = len(xg.get_booster().save_raw()) / 2 ** 20
    lg, pr, ttr, tte = run_lgbm(Xtr, ytr, Xva, yva, Xte, n_cls, T["lgbm_leaves"], seed)
    R["metrics"]["LightGBM"] = (binary_metrics(yte, pr[:, 1]) if binary else multiclass_metrics(yte, pr.argmax(1), classes))
    R["metrics"]["LightGBM"].update(train_min=ttr / 60, test_ms=tte)
    R["extra"]["size_LightGBM"] = len(lg.booster_.model_to_string().encode()) / 2 ** 20; del lg

    # ---- closest work: forests weighted by non-neural importances
    W = importance_baselines(rf, Xtr, ytr, Xva, yva, xg, seed)
    for k, w in W.items():
        _, R["metrics"][k] = fit_eval_forest(to_prob(w, T["tau_w"][k]), Xtr, ytr, Xte, yte, K, m, msl, seed, binary=binary, classes=classes)
    R["extra"]["W"] = {k: v.tolist() for k, v in W.items()}
    kk = max(2, int(round(T["topk_frac"] * d))); top = np.argsort(-I)[:kk]
    _, R["metrics"]["CNN-FS+RF"] = fit_eval_forest(None, Xtr[:, top], ytr, Xte[:, top], yte, K, min(m, kk), msl, seed, binary=binary, classes=classes)
    del rf; gc.collect()

    # ---- neural baselines
    net, perm_, _ = imp["nets"][0]
    R["metrics"]["CNN"] = eval_net(net, Xte, yte, binary, classes); R["metrics"]["CNN"]["train_min"] = imp["cnn_secs"] / CFG["N_PERM"] / 60
    R["extra"]["size_CNN"] = n_params(net) * 4 / 2 ** 20
    if CFG["RUN_SEQ_BASELINES"] and (binary or CFG.get("SEQ_MULTI", True)):
        for name, Net in (("BiLSTM", BiLSTM), ("CNN-LSTM", CNNLSTM)):
            n2, t, _ = train_net(Net(d, n_cls), Xtr, ytr, Xva, yva, n_cls, seed)
            R["metrics"][name] = eval_net(n2, Xte, yte, binary, classes); R["metrics"][name]["train_min"] = t / 60
            R["extra"]["size_" + name] = n_params(n2) * 4 / 2 ** 20
            del n2
    del imp; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    pickle.dump(R, open(path, "wb")); log(ds, task, "rep", rep, "done", {k: round(v.get("f1", v.get("macro_f1", 0)), 3) for k, v in R["metrics"].items()})
    return R

# %% ---------------------------------------------------------------- ABLATIONS (binary task)
def run_ablation(ds, rep, X, y, cats):
    path = os.path.join(CKPT, f"{ds}_abl_r{rep}.pkl")
    if os.path.exists(path): return pickle.load(open(path, "rb"))
    seed = CFG["SEED"] + rep; seed_all(seed)
    tr, va, te = make_split(X, y, rep)
    pp = Prep(cats).fit(X.iloc[tr])
    Xtr, Xva, Xte = pp.transform(X.iloc[tr]), pp.transform(X.iloc[va]), pp.transform(X.iloc[te])
    yy = (y != "Benign").astype(int).values; ytr, yva, yte = yy[tr], yy[va], yy[te]
    T = json.load(open(os.path.join(CKPT, f"{ds}_tuning.json"))); m, msl = T["m"], T["msl"]
    A = dict(alpha={}, tau={}, K={}, depth={}, perm={})
    bpath = os.path.join(CKPT, f"{ds}_bin_r{rep}.pkl")      # same split and seed -> reuse the relevance of the main run
    Rb = pickle.load(open(bpath, "rb")) if os.path.exists(bpath) else None
    if Rb is not None and "G_list" in Rb["extra"]:
        ex = Rb["extra"]
        imp = dict(G=np.array(ex["G"]), A=np.array(ex["A"]), G_list=[np.array(g) for g in ex["G_list"]],
                   A_list=[np.array(a) for a in ex["A_list"]], perm_rho=ex["perm_rho"])
    else:
        imp = learn_importances(Xtr, ytr, Xva, yva, 2, seed)
    for a in CFG["ALPHA_GRID"]:
        _, A["alpha"][a] = fit_eval_forest(to_prob(combine(imp["G"], imp["A"], a), T["tau"]), Xtr, ytr, Xte, yte, CFG["K"], m, msl, seed)
    for t in CFG["TAU_GRID"] + [float("inf")]:
        f, r = fit_eval_forest(to_prob(combine(imp["G"], imp["A"], T["alpha"]), t), Xtr, ytr, Xte, yte, CFG["K"], m, msl, seed)
        r["div"] = f.diversity(Xva, seed=seed); A["tau"][t] = r; del f
    P = to_prob(combine(imp["G"], imp["A"], T["alpha"]), T["tau"])
    for k in (25, 50, 100, 200):
        f, r = fit_eval_forest(P, Xtr, ytr, Xte, yte, k, m, msl, seed); r["size"] = f.size_mb(); A["K"][k] = r; del f
    # order invariance: single identity-order CNN vs N_PERM-averaged
    P1 = to_prob(combine(imp["G_list"][0], imp["A_list"][0], T["alpha"]), T["tau"])
    _, A["perm"][1] = fit_eval_forest(P1, Xtr, ytr, Xte, yte, CFG["K"], m, msl, seed)
    A["perm"][CFG["N_PERM"]] = A["alpha"][T["alpha"]] if T["alpha"] in A["alpha"] else None
    A["perm_rho"] = imp["perm_rho"]
    for name, ch in (("Shallow (32-64)", (32, 64)), ("Medium (64-128-256)", (64, 128, 256)), ("Deep (64-128-256-512)", (64, 128, 256, 512))):
        im = learn_importances(Xtr, ytr, Xva, yva, 2, seed, n_perm=1, chans=ch)
        _, r = fit_eval_forest(to_prob(combine(im["G"], im["A"], T["alpha"]), T["tau"]), Xtr, ytr, Xte, yte, CFG["K"], m, msl, seed)
        r["params"] = n_params(im["nets"][0][0]); r["cnn_min"] = im["cnn_secs"] / 60; A["depth"][name] = r
    pickle.dump(A, open(path, "wb")); log(ds, "ablation rep", rep, "done")
    return A

# %% ---------------------------------------------------------------- STATISTICS
def corrected_t(a, b, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test, two-sided."""
    d = np.asarray(a) - np.asarray(b); J = len(d)
    if J < 2 or np.all(d == d[0]): return float("nan"), 1.0 if np.all(d == 0) else 0.0
    v = np.var(d, ddof=1); t = d.mean() / math.sqrt((1 / J + n_te / n_tr) * v) if v > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), J - 1); return float(t), float(p)

def holm(pvals):
    keys = sorted(pvals, key=pvals.get); m = len(keys); out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k])); out[k] = run
    return out

# %% ---------------------------------------------------------------- MAIN LOOP
if __name__ == "__main__":
    t_all = time.time()
    RAW = {}
    rawp = os.path.join(CFG["OUT_DIR"], "results_raw.pkl")
    if os.path.exists(rawp): RAW = pickle.load(open(rawp, "rb"))
    for ds in CFG["DATASETS"]:
        X, y, cats, info = prepare(ds)
        RAW.setdefault(ds, {})["info"] = info
        RAW[ds]["bin"] = [run_repeat(ds, r, X, y, cats, "bin") for r in range(CFG["REPEATS"])]
        RAW[ds]["multi"] = [run_repeat(ds, r, X, y, cats, "multi") for r in range(CFG["REPEATS"])]
        if CFG["RUN_ABLATIONS"]:
            RAW[ds]["abl"] = [run_ablation(ds, r, X, y, cats) for r in range(CFG["ABL_REPEATS"])]
        RAW[ds]["tuning"] = json.load(open(os.path.join(CKPT, f"{ds}_tuning.json")))
        RAW["env"] = dict(cpu=platform.processor() or platform.machine(), cores=os.cpu_count(),
                          gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                          ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30) if hasattr(os, "sysconf") else None,
                          python=platform.python_version(), torch=torch.__version__,
                          sklearn=__import__("sklearn").__version__, xgboost=__import__("xgboost").__version__,
                          lightgbm=__import__("lightgbm").__version__, cfg={k: v for k, v in CFG.items() if not k.endswith(("GLOB", "TRAIN", "TEST", "DIR"))})
        pickle.dump(RAW, open(rawp, "wb"))
        del X, y; gc.collect()
    log("all experiments finished in %.1f h" % ((time.time() - t_all) / 3600))
    # summarise -> tokens.json + figures (separate script so it can be re-run cheaply)
    import ngrf_report
    ngrf_report.main(CFG["OUT_DIR"])
