# benchmark-nn  

## Tabular-Regression Benchmarks: Accuracy ⚖️  Latency ⚡  Robustness 🛡️  

> End-to-end study of four popular regression packages (LightGBM, XGBoost, CatBoost, MLP/Keras)  
> evaluated across **three experimental phases**: **baseline hyper-parameter tuning**,  
> **multi-objective trade-offs**, and **robustness to data perturbations**.

---

## Table of Contents

1. Objective  
2. Prerequisites  
3. Quick Start  
4. How to Run Stand-Alone Pipelines  
   - Phase 1 – Baseline Tuning  
   - Phase 2 – Multi-Objective Trade-Off  
   - Phase 3 – Research Questions (RQ1 – RQ3)  
5. Reproducing Plots & Tables  
6. Directory Map  
7. Citation / References  
8. Contact / Maintainers  
9. Additional Notes  

---

##  Objective
This repository accompanies the **Machine Learning and Scientific Computing** course project **Benchmarking Tuned Tree-Based and Neural
Models: Trade-Offs in Latency, Footprint, Accuracy,
and Noise Robustness** (MLSC 2025) lectured by **Prof. Dr. Beate Rhein** , **Technische Hochschule Koeln**.  

We investigate:

* **RQ1 – Bayesian hyper-parameter tuning:** Which package reaches the best test RMSE / R²?  
* **RQ2 – Latency-Size-Accuracy trade-off:** How do inference latency, binary size, and accuracy interact?  
* **RQ3 – Robustness:** How gracefully do tuned models degrade under Gaussian noise & feature dropout?

---


## Prerequisites

* **Python ≥ 3.9** (project was developed on 3.12).
* **Virtual-environment** (venv or Conda) to keep packages isolated.
* **All Python libraries** listed in [`requirements.txt`](./requirements.txt).
* **Basic C/C++ tool-chain** (gcc / clang + make) – needed only if a binary
  wheel for LightGBM or XGBoost is not available for your platform.
* *(Optional)* a **Weights & Biases** account + API key if you want live
  experiment dashboards.
* **Dataset** – place `diamonds.csv` in `data/raw/`.  
  The Phase-1 driver will auto-create  
  `data/processed/{train,val,test}.pkl`.

---

### Quick install (A) – pip + venv

```bash
git clone https://github.com/sbarai927/benchmark-nn.git
cd benchmark-nn

# create & activate virtual-env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# upgrade pip & install deps
pip install --upgrade pip
pip install -r requirements.txt

```

### Quick install (B) – Conda / Mamba

```bash
conda create -n tabular-reg python=3.12
conda activate tabular-reg
pip install -r requirements.txt

## or convert to environment.yml

# create env with all exact versions & native libs
conda env create -f configs/environment.yml
conda activate benchmark-nn

```

## How to Run Standalone Code

Each phase can be reproduced from the command line with a single driver script
(located in **`pipelines/`**).  
All scripts accept `-h/--help` for the full list of options.

---

### Phase 1 – Baseline Hyper-Parameter Tuning

The script `exploration_pipeline_endpoints_phase1.py` loops over the **ten
notebooks** in `notebooks/Phase 1/`, performs grid + Bayesian optimisation for
LightGBM, XGBoost, CatBoost and Keras-MLP, and writes artefacts to
`results/phase1_*`.

**Usage:**
```bash
python pipelines/exploration_pipeline_endpoints_phase1.py \
       [--n_jobs J] [--save_dir DIR]
```

**Arguments:**
- --n_jobs J	        Parallel CPU workers used by Optuna / scikit-optimize	         1
- --save_dir DIR	Destination for phase1_results.csv, best-HP tables and plots	results/

**Example(4-core CPU):**
```bash
python pipelines/exploration_pipeline_endpoints_phase1.py \
       --n_jobs 4 --save_dir results/
```
### Phase 2 - Multi-objective Trade-off Analysis

`multi_objective_tuning_phase2.py` re-tunes the four Phase-1 winners on a
shared search space described in `configs/config.yml` and writes
`phase2_comparison_results.csv` + the latency-size-accuracy bubble plot.

**Usage:**
```bash
python pipelines/multi_objective_tuning_phase2.py \
       --config configs/config.yml      \
       --trials 80                      # Optuna trials / model
```

**Arguments:**
- --config	YAML file with upper/lower bounds for every HP	      required
- --trials	Number of Bayesian-Optuna evaluations per model	        80
- --timeout	Global time-budget in minutes (overrides --trials)	None

### Phase 3 - Robustness & Research Questions (RQ 1 ➜ RQ 2 ➜ RQ 3)
`research_questions_phase3.py` runs the five-seed hyper-parameter sweep with HPO engines (optuna/hyperopt/skopt)
for RQ 1, generates the latency–size–accuracy bubble for RQ 2, and executes
Gaussian-noise / feature-dropout experiments for RQ 3.

**Usage:**
```bash
python pipelines/research_questions_phase3.py \
       --seeds 0 42 7 123 999          \
       --save_dir results/             \
       --wandb_project "benchmark-nn"
```
**Arguments:**
- --seeds	    List of RNG seeds ↔ 5 repeated Optuna runs	  0 42 7 123 999
- --save_dir	    Destination for phase3_rq*_*.csv and PNGs	  results/
- --wandb_project   (Optional) log all trials to Weights & Biases  none

---

### Reproduce a Single Notebook Interactively

```bash
jupyter notebook notebooks/Phase\ 3/RQ\ 1/rq1_cb.ipynb
```
and execute the cells. Results will be written to the same folders used by
the pipeline scripts.


## Directory Map

The directory structure of this repository is as follows:

```bash
.
├── configs/                         # YAML search-spaces & settings for Phase 2
│   └── config.yml
│
├── data/                            # Raw & processed tabular data
│   ├── raw/
│   │   └── diamonds.csv
│   └── processed/
│       ├── train.pkl
│       ├── val.pkl
│       └── test.pkl
│
├── models/                          # Saved Keras binaries (Phase-1 winners)
│   ├── baseline_model.keras
│   ├── hyperopt_model.keras
│   ├── kt_model.keras               # Keras-Tuner best
│   ├── optuna_model.keras
│   └── skopt_model.keras
│ 
├── notebooks/                       # Interactive development & analysis
│   ├── Phase 1/                     # Baseline tuning (10 notebooks)
│   │   ├── 01-data-exploration.ipynb
│   │   ├── 02-baseline-model.ipynb
│   │   ├── 03-optuna.ipynb
│   │   ├── 04-hyperopt.ipynb
│   │   ├── 05-keras.ipynb
│   │   ├── 06-skopt.ipynb
│   │   ├── 07-lightgbm.ipynb
│   │   ├── 08-xgboost.ipynb
│   │   ├── 09-catboost.ipynb
│   │   └── 10-kfold.ipynb
│   ├── Phase 2/
│   │   ├── optuna_tuning_seeds_exp.ipynb      # Shared Optuna search-space
│   │   └── multi_objective_comparison.ipynb   # Latency-size-accuracy analysis
│   └── Phase 3/                               # Research-question notebooks
│       ├── RQ 1/
│       │   ├── rq1_cb.ipynb  ·  rq1_lgbm.ipynb  ·  rq1_xgb.ipynb  ·  rq1_mlp.ipynb
│       │   └── rq1_comparison.ipynb
│       ├── RQ 2/
│       │   └── rq2_plot.ipynb                  # Bubble-chart generator
│       └── RQ 3/
│           └── rq3_robustness.ipynb            # Noise & feature-drop curves
│
├── pipelines/                       # Head-less drivers (Papermill-ready †)
│   ├── preprocessing.py
│   ├── exploration_pipeline_endpoints_phase1.py
│   ├── multi_objective_tuning_phase2.py
│   └── research_questions_phase3.py
│       # † Each script saves CSVs to results/ and artefacts to plots and tables/
│
├── plots and tables/                # Final PNG / CSV artefacts copied to LaTeX
│   ├── Phase 1/
│   │   ├── catboost_phase1.png      ·  xgboost_phase1.png
│   │   ├── lgmb_phase1.png          ·  mlp_phase1.png
│   │   └── mlp_epochs.csv
│   ├── Phase 2/
│   │   └── rq2_plot.png             # Pareto front (latency-size-accuracy)
│   └── Phase 3/
│       ├── catboost_rq1_training.png
│       ├── rq1_*_tuning.csv / .png  # *_ = cb | lgbm | xgb | mlp
│       ├── rq1_package_comparison.csv
│       ├── rq3_robustness.csv
│       └── rq3_robustness_heatmap.png
│
├── results/                         # Re-generated result CSVs
│   ├── phase1_results.csv
│   ├── phase2_comparison_results.csv
│   ├── phase3_rq1_package_comparison.csv
│   └── phase3_rq3_robustness.csv
│
├── trials/                          # Full Optuna / Hyperopt / SK-opt logs
│   ├── optuna_summary.csv           ·  hyperopt_summary.csv
│   ├── skopt_tuning_results.csv     ·  lightgbm_results.csv
│   ├── xgboost_results.csv          ·  catboost_trials_phase1.csv
│   ├── catboost_training_phase2.json
│   └── kt_phase1.json               # Keras Tuner log (Phase 1)
│
├── report.pdf                       # Compiled LaTeX paper (for convenience)
├── requirements.txt                 # Exact package versions (Python 3.12)
└── README.md                        # You are here

```

## Citation / References

1. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). **LightGBM: A highly efficient gradient boosting decision tree**. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

2. Chen, T., & Guestrin, C. (2016). **XGBoost: A scalable tree boosting system**. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

3. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A., & Gulin, A. (2018). **CatBoost: Unbiased boosting with categorical features**. *Advances in Neural Information Processing Systems*, 31, 6638-6648.

4. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). **Algorithms for hyper-parameter optimization**. *Advances in Neural Information Processing Systems*, 24, 2546-2554.

5. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). **Optuna: A next-generation hyperparameter optimization framework**. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623-2631).

6. Loshchilov, I., & Hutter, F. (2019). **Decoupled weight decay regularization**. *International Conference on Learning Representations (ICLR)*.

7. Barai, S. (2025). **Phase 1 trial plots – W&B project dashboard**. *Weights & Biases Reports*. https://wandb.ai/suvendu-barai-th-k-ln/Phase1%20trial%20plots

8. Barai, S. (2025). **Phase 3 RQ1 hyper-parameter tuning – W&B project dashboard**. Accessed 20 June 2025. *Weights & Biases Reports*. https://wandb.ai/suvendu-barai-th-k-ln/Phase3%20rq1%20hyperparam%20tuning


## Contact / Maintainers

For any questions or issues, please contact:
- Suvendu Barai
- Email: suvendu.barai@smail.th-koeln.de

## Additional Notes

- All scripts support --help for extra CLI options.
- For W&B logging, set WANDB_API_KEY and WANDB_PROJECT in your shell.
- If you hit memory limits on M1/M2 MacBooks (8 GB), reduce n_estimators or use the CPU built-in mixed-precision flag in LightGBM/XGBoost.
