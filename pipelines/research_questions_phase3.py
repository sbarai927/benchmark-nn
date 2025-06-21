# app_phase3.py
import os
import shutil
import glob
import papermill as pm
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal

# --- Configuration ---
NOTEBOOK_OUTPUT_DIR = "notebooks_out"
RESULTS_DIR         = "results"
os.makedirs(NOTEBOOK_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Map each RQ to the notebooks it needs and the CSVs it produces
PHASES = {
    "rq1": {
        "notebooks": [
            "rq1_lgbm.ipynb",
            "rq1_xgb.ipynb",
            "rq1_cb.ipynb",
            "rq1_mlp.ipynb",
            "multi_objective_tuning_comparison.ipynb"
        ],
        "results": [
            "rq1_lgbm_final.csv",
            "rq1_xgb_final.csv",
            "rq1_cb_final.csv",
            "rq1_mlp_final.csv",
            "rq1_package_comparison.csv"
        ]
    },
    "rq2": {
        "notebooks": [
            "multi_objective_tuning_comparison.ipynb"
        ],
        "results": [
            "rq2_tradeoff.csv"
        ]
    },
    "rq3": {
        "notebooks": [
            "rq3_robustness.ipynb"
        ],
        "results": [
            "rq3_robustness.csv"
        ]
    }
}

app = FastAPI(
    title="Phase 3 Pipeline",
    description="Run RQ1, RQ2, RQ3 notebooks and fetch their CSV results",
    version="1.0.0"
)

# Model for selecting a single phase
class PhaseIn(BaseModel):
    phase: Literal["rq1", "rq2", "rq3"]

# Helper to run a single notebook via Papermill
def run_notebook(nb_name: str):
    if not os.path.exists(nb_name):
        raise FileNotFoundError(f"Notebook not found: {nb_name}")
    out_path = os.path.join(NOTEBOOK_OUTPUT_DIR, nb_name.replace(".ipynb", "_out.ipynb"))
    pm.execute_notebook(
        input_path=nb_name,
        output_path=out_path,
        log_output=True
    )
    return out_path

# Helper to collect result CSVs
def collect_results(phase: str):
    copied = []
    for fname in PHASES[phase]["results"]:
        # look in cwd for the CSV
        matches = glob.glob(fname)
        if not matches:
            continue
        src = matches[0]
        dst = os.path.join(RESULTS_DIR, os.path.basename(src))
        shutil.copy(src, dst)
        copied.append(os.path.basename(src))
    return copied

# 1) Run RQ1: tuning + merge
@app.post("/run/rq1")
def run_rq1():
    try:
        for nb in PHASES["rq1"]["notebooks"]:
            run_notebook(nb)
        files = collect_results("rq1")
        if not files:
            raise HTTPException(500, detail="RQ1 ran but no CSVs found.")
        return {"status": "rq1 completed", "files": files}
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# 2) Run RQ2: trade-off analysis
@app.post("/run/rq2")
def run_rq2():
    try:
        for nb in PHASES["rq2"]["notebooks"]:
            run_notebook(nb)
        files = collect_results("rq2")
        if not files:
            raise HTTPException(500, detail="RQ2 ran but no CSVs found.")
        return {"status": "rq2 completed", "files": files}
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# 3) Run RQ3: robustness experiments
@app.post("/run/rq3")
def run_rq3():
    try:
        for nb in PHASES["rq3"]["notebooks"]:
            run_notebook(nb)
        files = collect_results("rq3")
        if not files:
            raise HTTPException(500, detail="RQ3 ran but no CSVs found.")
        return {"status": "rq3 completed", "files": files}
    except FileNotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# 4) List all available result CSVs
@app.get("/results")
def list_all_results():
    files = sorted(os.listdir(RESULTS_DIR))
    if not files:
        raise HTTPException(404, detail="No results available. Run /run/rq1, /run/rq2, /run/rq3 first.")
    return {"available_results": files}

# 5) Fetch a specific CSV as JSON records
@app.get("/results/{filename}")
def get_csv(filename: str):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, detail=f"{filename} not found in results/")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

# 6) Health check
@app.get("/health")
def health():
    return {"status": "healthy"}
