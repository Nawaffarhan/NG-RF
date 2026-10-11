# NG-RF: Informed Feature Sampling for Random Forest Intrusion Detection

Code and results for the paper
**"Informed Feature Sampling for Random Forest Intrusion Detection: A Controlled Comparison of Neural and Model-Based Relevance"**
(S. Abuowaida, A. Alsarhan, H. A. Mashagba, N. Alshdaifat, A. B. Abd Aziz, M. Bin Roslee; submitted to *Discover Computing*).

NG-RF learns feature relevance with a 1-D CNN, turns it into a per-node feature-sampling distribution, and grows a random forest that samples candidate features from that distribution (implemented exactly by column replication). The code compares it with uniform RF, Extra-Trees, XGBoost, LightGBM, a CNN, and weighted-subspace forests based on Gini, permutation, mutual-information and SHAP importance, on CIC-IDS2017, UNSW-NB15 and CSE-CIC-IDS2018 (binary and multi-class), with ablations.

## Files
| File | Content |
|---|---|
| `ngrf_experiments.py` | full pipeline: data cleaning, NG-RF, baselines, ablations, statistics |
| `ngrf_report.py` | builds `tokens.json` (every number in the paper) and the figures from `results_raw.pkl` |
| `NGRF_experiments.ipynb` | the same code as a notebook (Kaggle / Jupyter) |
| `merge_results.py` | merges results of separate runs (one dataset per session) |
| `results/` | results of the paper: `results_raw.pkl` (all repeats), `tokens.json`, `run.log`, figures |

## Data
- **CIC-IDS2017**: the eight `*.pcap_ISCX.csv` files (MachineLearningCVE), https://www.unb.ca/cic/datasets/ids-2017.html
- **UNSW-NB15**: `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`, https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CSE-CIC-IDS2018**: the ten `*_TrafficForML_CICFlowMeter.csv` files, https://www.unb.ca/cic/datasets/ids-2018.html

## Run
```bash
pip install -r requirements.txt
export NGRF_CFG_JSON='{"CIC17_GLOB": "data/cic17/*.csv",
                       "UNSW_TRAIN": "data/unsw/UNSW_NB15_training-set.csv",
                       "UNSW_TEST":  "data/unsw/UNSW_NB15_testing-set.csv",
                       "CSE18_GLOB": "data/cse18/*.csv",
                       "OUT_DIR": "ngrf_out"}'
python ngrf_experiments.py            # all datasets; or NGRF_DATASETS=unsw python ngrf_experiments.py
python ngrf_report.py ngrf_out        # tables (tokens.json) and figures only
```
On Kaggle, attach the datasets and run `NGRF_experiments.ipynb`; the default paths search `/kaggle/input` recursively.
Each (dataset, repeat) is checkpointed in `OUT_DIR/ckpt`, so an interrupted run resumes.

## Settings used in the paper
The defaults in `CONFIG` are those of the paper: 5 repeated stratified 60/20/20 splits (seed 42), 3 ablation repeats, 100,000-flow stratified cap after duplicate removal, K = 100 trees, replication pool 3d, CNN 20 epochs (patience 3) averaged over 3 random feature orders; hyper-parameters chosen on the validation partition of the first split. Significance: corrected resampled t-test with Holm adjustment.
Environment of the reported run: 2-core x86-64 CPU, no GPU, 8 GB RAM, Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0, LightGBM 4.7.0; total time about 11 h.

## Citation
Please cite the paper (details will be added after publication).

## License
MIT
