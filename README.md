# benchmark-nn  

## Tabular-Regression Benchmarks: Accuracy ⚖️ Latency ⚡ Robustness 🔧

> End-to-end study of four popular regression packages (LightGBM, XGBoost, CatBoost, MLP/Keras)  
> evaluated across **three experimental phases**: **baseline hyper-parameter tuning**,  
> **multi-objective trade-offs**, and **robustness to data perturbations**.



## 1  Repository Layout

configs/ YAML config(s) for all phases
data/ Raw & pre-processed tabular data 
models/ Saved model binaries (.keras)
notebooks/
├── Phase 1/ 10 baseline / tuning notebooks
├── Phase 2/
│ ├── optuna_tuning_seeds_exp.ipynb
│ └── multi_objective_comparison.ipynb
└── Phase 3/
├── RQ 1/ per-RQ notebooks (seeds 0, 42, 7, 123, 999)
├── RQ 2/
└── RQ 3/
pipelines/ Python drivers (papermill) to run phases head-less
plots and tables/
├── Phase 1/ Final .png / .csv artefacts
├── Phase 2/
└── Phase 3/
results/ Re-generated result CSVs
trials/ Full Optuna/Hyperopt/SK-opt logs (JSON/CSV)
README.md



## 2  Quick Start

```bash
# ① Clone and create environment
git clone https://github.com/sbarai927/benchmark-nn.git
cd benchmark-nn
conda env create -f configs/environment.yml   # Python 3.12
conda activate benchmark-nn

# ② Download / place data
#   Put raw `.csv` (or .parquet) in  data/  and adjust paths in configs if needed

# ③ Run phase pipelines (head-less)
## Phase-1  baseline tuning (grids / BO)
'python pipelines/exploration_pipeline_endpoints_phase1.py'

## Phase-2  multi-objective trade-off
'python pipelines/multi_objective_tuning_phase2.py' \
       --config configs/config.yml            # shared Optuna search-space

## Phase-3  research questions (runs each RQ in sequence)
'python pipelines/research_questions_phase3.py' \
       --seeds 0 42 7 123 999                 # reproducible seed list for RQ-1



## 3  Pipeline Overview

Phase	Driver Script	Key Outputs (saved under results/ & plots and tables/)
1 — Baseline tuning	exploration_pipeline_endpoints_phase1.py	phase1_results.csv, best-HP tables & tuning plots
2 — Trade-off analysis	multi_objective_tuning_phase2.py	phase2_comparison_results.csv, latency × size × accuracy bubble plot
3 — RQ-based robustness	research_questions_phase3.py	phase3_rq1_package_comparison.csv, phase3_rq3_robustness.csv, heat-maps & line charts
Config note Phase-2 reads its search-space and common Optuna options from
configs/config.yml, ensuring every package is tuned on an identical domain.


## 4  Reproducing Figures & Tables

Figure / Table	Regenerate with	Description
Phase-1 trial plots	notebooks/Phase 1/*_tuning.ipynb	Parallel-coordinates of lowest validation RMSE trials
RQ-2 bubble plot	part of Phase-2 pipeline	Pareto-front: latency vs size vs accuracy (test R² as colour)
RQ-3 robustness line chart	Phase-3 pipeline, rq3_robustness_heatmap.png	Gaussian noise & feature-drop curves for 4 finalists
All artefacts are automatically copied to plots and tables/ upon completion.

## 5  Citation

@misc{benchmark-nn-2025,
  title   = {benchmark-nn: Tabular Regression Benchmarks},
  author  = {S. Author},
  year    = {2025},
  howpublished = {\url{https://github.com/sbarai927/benchmark-nn}}
}

## 6 Contact & Contributions
Questions, bug-reports or pull-requests are welcome!
Author: Suvendu Barai · suvendu.barai@smail.th-koeln.de