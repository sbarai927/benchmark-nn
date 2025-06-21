#!/usr/bin/env python3
"""
Exploration pipeline orchestrating all of our numbered notebooks end-to-end:
 01) Data exploration
 02) Baseline models (LinearRegression & RandomForest)
 03) Optuna-tuned MLP
 04) Hyperopt-tuned MLP
 05) Keras-Tuner MLP
 06) Scikit-Optimize MLP
 07) Optuna-tuned LightGBM
 08) Optuna-tuned XGBoost
 09) Optuna-tuned CatBoost
 10) K-Fold cross-validation wrap-up
"""

import os
import time
import argparse
import logging
import yaml
import warnings

import numpy as np
import pandas as pd
import joblib

# Classical baselines
from sklearn.linear_model import LinearRegression
from sklearn.ensemble   import RandomForestRegressor
from sklearn.metrics    import mean_squared_error, r2_score

# Gradient boosted trees
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

# Neural-net tuners
import optuna
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from skopt import gp_minimize
from skopt.space import Real, Integer
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband, Objective

# Cross-validation
from sklearn.model_selection import KFold

# suppress warnings
warnings.filterwarnings("ignore")

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def load_config(path="config_first.yml"):
    """
    Load pipeline configuration from YAML file.
    Expected keys include:
      data:
        train: path/to/train.pkl
        val:   path/to/val.pkl
        test:  path/to/test.pkl
      seed: 42
      baseline:
        rf_n_estimators: 100
      mlp:
        n_layers:      [1,3]
        units:         [32,256]
        learning_rate: [1e-4,1e-2]
        dropout:       [0.0,0.5]
        batch_size:    256
        max_epochs:    50
        patience:      10
        step_units:    32
        batch_sizes:   [128,512]
        step_batch:    128
      optuna:
        n_trials: 20
      hyperopt:
        max_evals: 20
      skopt:
        n_calls: 20
      lgbm:
        learning_rate: [1e-3,1e-1]
        max_depth:     [3,12]
        subsample:     [0.5,1.0]
        colsample:     [0.5,1.0]
        n_estimators:  1000
        early_stopping_rounds: 50
      xgb:
        learning_rate: [1e-3,1e-1]
        max_depth:     [3,12]
        subsample:     [0.5,1.0]
        colsample:     [0.5,1.0]
        n_estimators:  [100,1000]
        early_stopping_rounds: 50
      cat:
        learning_rate: [1e-2,1e-1]
        depth:         [4,10]
        subsample:     [0.5,1.0]
        colsample:     [0.5,1.0]
        iterations:    2000
        early_stopping_rounds: 50
      cv:
        n_splits: 5
      output: results_summary.csv
    """
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    """Load train/val/test splits via joblib.load(...) as (X, y) pairs."""
    log.info("Loading data splits via joblib.load(...)")
    X_train, y_train = joblib.load(cfg["data"]["train"])
    X_val,   y_val   = joblib.load(cfg["data"]["val"])
    X_test,  y_test  = joblib.load(cfg["data"]["test"])
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_data_exploration(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te):
    log.info("=== 01) Data Exploration ===")
    print(f"Train: {X_tr.shape}   {y_tr.shape}")
    print(f"Val:   {X_val.shape}   {y_val.shape}")
    print(f"Test:  {X_te.shape}   {y_te.shape}\n")
    print("--- y_train distribution ---")
    print(pd.Series(y_tr).describe())
    import matplotlib.pyplot as plt
    pd.Series(y_tr).hist(bins=50)
    plt.title("y_train distribution")
    plt.show()
    print()


def run_baseline_models(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te):
    log.info("=== 02) Baseline Models ===")
    results = []

    # --- Linear Regression ---
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = lr.predict(X_te)
    pred_time = time.time() - t1

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    log.info(f"LinearRegression → RMSE: {rmse:.2f}, R²: {r2:.3f}, train: {train_time:.2f}s, pred: {pred_time:.3f}s")
    results.append({
        "model": "LinearRegression",
        "rmse": rmse, "r2": r2,
        "train_time": train_time, "pred_time": pred_time
    })

    # --- Random Forest ---
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=cfg["baseline"]["rf_n_estimators"],
        random_state=cfg["seed"]
    )
    rf.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred = rf.predict(X_te)
    pred_time = time.time() - t1

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    log.info(f"RandomForest → RMSE: {rmse:.2f}, R²: {r2:.3f}, train: {train_time:.2f}s, pred: {pred_time:.3f}s")
    results.append({
        "model": "RandomForest",
        "rmse": rmse, "r2": r2,
        "train_time": train_time, "pred_time": pred_time
    })

    return results


def run_optuna_mlp(cfg, X_tr, y_tr, X_val, y_val):
    log.info("=== 03) Optuna-Tuned MLP ===")
    def objective(trial):
        # hyperparameter suggestions
        n_layers = trial.suggest_int("n_layers", *cfg["mlp"]["n_layers"])
        units    = trial.suggest_int("units",    *cfg["mlp"]["units"], step=cfg["mlp"]["step_units"])
        lr       = trial.suggest_float("learning_rate", *cfg["mlp"]["learning_rate"], log=True)
        dropout  = trial.suggest_float("dropout", *cfg["mlp"]["dropout"])

        # build model
        model = keras.Sequential()
        for _ in range(n_layers):
            model.add(keras.layers.Dense(units, activation="relu"))
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg["mlp"]["max_epochs"],
            batch_size=cfg["mlp"]["batch_size"],
            callbacks=[keras.callbacks.EarlyStopping(patience=cfg["mlp"]["patience"],
                                                     restore_best_weights=True)],
            verbose=0
        )

        val_rmse = float(model.evaluate(X_val, y_val, verbose=0)[1])
        log.info(f"  trial#{trial.number} → val RMSE {val_rmse:.2f}")
        return val_rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=True)

    best = study.best_trial
    log.info(f"Optuna MLP — Best RMSE: {best.value:.2f}, params: {best.params}")
    return {"model":"Optuna_MLP", "rmse":best.value, **best.params}


def run_hyperopt_mlp(cfg, X_tr, y_tr, X_val, y_val):
    log.info("=== 04) Hyperopt-Tuned MLP ===")
    def fn(space):
        m = keras.Sequential([
            keras.layers.Dense(int(space["units"]), activation="relu"),
            keras.layers.Dropout(space["dropout"]),
            keras.layers.Dense(1)
        ])
        m.compile(
            optimizer=keras.optimizers.Adam(space["lr"]),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        m.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg["mlp"]["max_epochs"],
            batch_size=int(space["batch_size"]),
            callbacks=[keras.callbacks.EarlyStopping(patience=cfg["mlp"]["patience"],
                                                     restore_best_weights=True)],
            verbose=0
        )
        loss, rmse = m.evaluate(X_val, y_val, verbose=0)
        return {"loss": rmse, "status": STATUS_OK}

    space = {
        "units":      hp.quniform("units",      *cfg["mlp"]["units"],  step=cfg["mlp"]["step_units"]),
        "dropout":    hp.uniform("dropout",     *cfg["mlp"]["dropout"]),
        "lr":         hp.loguniform("lr",       *cfg["mlp"]["learning_rate"]),
        "batch_size": hp.quniform("batch_size", *cfg["mlp"]["batch_sizes"], step=cfg["mlp"]["step_batch"])
    }
    trials = Trials()
    best = fmin(fn, space,
                algo=tpe.suggest,
                max_evals=cfg["hyperopt"]["max_evals"],
                trials=trials,
                rstate=np.random.default_rng(cfg["seed"]))
    log.info(f"Hyperopt MLP — Best space: {best}")
    return {"model":"Hyperopt_MLP", **{k:int(v) if k in ['units','batch_size'] else v for k,v in best.items()}}


def run_keras_tuner_mlp(cfg, X_tr, y_tr, X_val, y_val):
    log.info("=== 05) Keras-Tuner Hyperband MLP ===")
    def build_model(hp):
        m = keras.Sequential()
        for _ in range(hp.Int("n_layers", *cfg["mlp"]["n_layers"])):
            m.add(keras.layers.Dense(
                units=hp.Int("units", *cfg["mlp"]["units"], step=cfg["mlp"]["step_units"]),
                activation="relu"))
            m.add(keras.layers.Dropout(hp.Float("dropout", *cfg["mlp"]["dropout"])))
        m.add(keras.layers.Dense(1))
        m.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float("lr", *cfg["mlp"]["learning_rate"], sampling="LOG")),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()]
        )
        return m

    tuner = Hyperband(
        build_model,
        objective=Objective("val_root_mean_squared_error", direction="min"),
        max_epochs=cfg["mlp"]["max_epochs"],
        factor=3,
        seed=cfg["seed"]
    )
    tuner.search(X_tr, y_tr,
                 validation_data=(X_val, y_val),
                 callbacks=[keras.callbacks.EarlyStopping(patience=cfg["mlp"]["patience"])],
                 verbose=0)
    hp_best = tuner.get_best_hyperparameters(1)[0].values
    log.info(f"Keras-Tuner MLP — Best HP: {hp_best}")
    return {"model":"KerasTuner_MLP", **hp_best}


def run_skopt_mlp(cfg, X_tr, y_tr, X_val, y_val):
    log.info("=== 06) Scikit-Optimize MLP ===")
    def sk_objective(params):
        n_layers, units, lr, dropout = params
        m = keras.Sequential()
        for _ in range(int(n_layers)):
            m.add(keras.layers.Dense(int(units), activation="relu"))
            m.add(keras.layers.Dropout(dropout))
        m.add(keras.layers.Dense(1))
        m.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="mse",
                  metrics=[keras.metrics.RootMeanSquaredError()])
        m.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg["mlp"]["max_epochs"],
            batch_size=cfg["mlp"]["batch_size"],
            callbacks=[keras.callbacks.EarlyStopping(patience=cfg["mlp"]["patience"],
                                                     restore_best_weights=True)],
            verbose=0
        )
        return float(m.evaluate(X_val, y_val, verbose=0)[1])

    space = [
        Integer(*cfg["mlp"]["n_layers"], name="n_layers"),
        Integer(*cfg["mlp"]["units"],    name="units"),
        Real(*cfg["mlp"]["learning_rate"], prior="log-uniform", name="lr"),
        Real(*cfg["mlp"]["dropout"],     name="dropout")
    ]
    res = gp_minimize(sk_objective,
                      space,
                      n_calls=cfg["skopt"]["n_calls"],
                      random_state=cfg["seed"])
    best = dict(zip(["n_layers","units","lr","dropout"], res.x))
    log.info(f"Scikit-Optimize MLP — Best: {best}")
    return {"model":"Skopt_MLP", **best}


def run_optuna_lgb(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te):
    log.info("=== 07) Optuna-Tuned LightGBM ===")
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    def objective(trial):
        p = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": trial.suggest_float("learning_rate", *cfg["lgbm"]["learning_rate"], log=True),
            "max_depth":     trial.suggest_int("max_depth", *cfg["lgbm"]["max_depth"]),
            "subsample":     trial.suggest_float("subsample", *cfg["lgbm"]["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg["lgbm"]["colsample"]),
            "seed":          cfg["seed"],
            "verbose":       -1
        }
        bst = lgb.train(
            p,
            dtrain,
            num_boost_round=cfg["lgbm"]["n_estimators"],
            valid_sets=[dval],
            early_stopping_rounds=cfg["lgbm"]["early_stopping_rounds"],
            verbose_eval=False
        )
        preds = bst.predict(X_val, num_iteration=bst.best_iteration)
        return mean_squared_error(y_val, preds, squared=False)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=True)

    best = study.best_trial.params
    log.info(f"Optuna LightGBM — Best: {best}")
    return {"model":"Optuna_LGBM", **best}


def run_optuna_xgb(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te):
    log.info("=== 08) Optuna-Tuned XGBoost ===")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    def objective(trial):
        p = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": trial.suggest_float("learning_rate", *cfg["xgb"]["learning_rate"], log=True),
            "max_depth":     trial.suggest_int("max_depth", *cfg["xgb"]["max_depth"]),
            "subsample":     trial.suggest_float("subsample", *cfg["xgb"]["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg["xgb"]["colsample"]),
            "seed": cfg["seed"],
            "verbosity": 0
        }
        n_rounds = trial.suggest_int("n_estimators", *cfg["xgb"]["n_estimators"])
        bst = xgb.train(
            p,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "valid")],
            early_stopping_rounds=cfg["xgb"]["early_stopping_rounds"],
            verbose_eval=False
        )
        preds = bst.predict(dval, iteration_range=(0, bst.best_iteration))
        return mean_squared_error(y_val, preds, squared=False)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=True)

    best = study.best_trial.params
    log.info(f"Optuna XGBoost — Best: {best}")
    return {"model":"Optuna_XGBoost", **best}


def run_optuna_cat(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te):
    log.info("=== 09) Optuna-Tuned CatBoost ===")
    train_pool = Pool(X_tr, y_tr)
    val_pool   = Pool(X_val, y_val)

    def objective(trial):
        p = {
            "iterations":        cfg["cat"]["iterations"],
            "learning_rate":     trial.suggest_float("learning_rate", *cfg["cat"]["learning_rate"]),
            "depth":             trial.suggest_int("depth", *cfg["cat"]["depth"]),
            "subsample":         trial.suggest_float("subsample", *cfg["cat"]["subsample"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", *cfg["cat"]["colsample"]),
            "loss_function":     "RMSE",
            "random_seed":       cfg["seed"],
            "verbose":           False
        }
        model = CatBoostRegressor(**p)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=cfg["cat"]["early_stopping_rounds"])
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds, squared=False)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
    study.optimize(objective, n_trials=cfg["optuna"]["n_trials"], show_progress_bar=True)

    best = study.best_trial.params
    log.info(f"Optuna CatBoost — Best: {best}")
    return {"model":"Optuna_CatBoost", **best}


def run_kfold_cv(cfg, X, y, model_fn):
    log.info("=== 10) K-Fold Cross-Validation ===")
    kf = KFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])
    records = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        Xt, Xa = X[tr_idx], X[va_idx]
        yt, ya = y[tr_idx], y[va_idx]
        t0 = time.time()
        model = model_fn(Xt, yt, Xa, ya)
        dur = time.time() - t0
        preds = model.predict(Xa)
        if hasattr(preds, "ravel"):
            preds = preds.ravel()
        rmse = mean_squared_error(ya, preds, squared=False)
        r2   = r2_score(ya, preds)
        log.info(f"Fold {fold} → RMSE: {rmse:.2f}, R²: {r2:.3f}, time: {dur:.2f}s")
        records.append({"fold":fold, "rmse":rmse, "r2":r2, "time":dur})
    df = pd.DataFrame(records)
    log.info(f"CV summary → RMSE: {df['rmse'].mean():.2f} ± {df['rmse'].std():.2f}, "
             f"R²: {df['r2'].mean():.3f} ± {df['r2'].std():.3f}")
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Run exploration pipeline steps.")
    p.add_argument("--config", default="config.yml", help="Path to config YAML")
    p.add_argument("--steps", nargs="+", type=int, default=list(range(1,11)),
                   help="Which steps to run (1-10)")
    return p.parse_args()


def main():
    opts = parse_args()
    cfg = load_config(opts.config)

    # load splits
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_data(cfg)
    all_results = []

    # step 1: data exploration
    if 1 in opts.steps:
        run_data_exploration(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te)

    # step 2: classical baselines
    if 2 in opts.steps:
        baselines = run_baseline_models(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te)
        all_results.extend(baselines)

    # steps 3–6: MLP variants
    if 3 in opts.steps:
        all_results.append(run_optuna_mlp(cfg, X_tr, y_tr, X_val, y_val))
    if 4 in opts.steps:
        all_results.append(run_hyperopt_mlp(cfg, X_tr, y_tr, X_val, y_val))
    if 5 in opts.steps:
        all_results.append(run_keras_tuner_mlp(cfg, X_tr, y_tr, X_val, y_val))
    if 6 in opts.steps:
        all_results.append(run_skopt_mlp(cfg, X_tr, y_tr, X_val, y_val))

    # steps 7–9: tuned GBTs
    if 7 in opts.steps:
        all_results.append(run_optuna_lgb(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te))
    if 8 in opts.steps:
        all_results.append(run_optuna_xgb(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te))
    if 9 in opts.steps:
        all_results.append(run_optuna_cat(cfg, X_tr, y_tr, X_val, y_val, X_te, y_te))

    # step 10: final K-Fold CV on best LightGBM (train+val)
    if 10 in opts.steps and cfg.get("use_kfold", True):
        X_all = np.vstack([X_tr, X_val])
        y_all = np.concatenate([y_tr, y_val])
        def final_lgb(Xt, yt, Xv, yv):
            p = cfg["lgbm"]
            dtr = lgb.Dataset(Xt, label=yt)
            dvl = lgb.Dataset(Xv, label=yv, reference=dtr)
            return lgb.train(
                {**p, "metric":"rmse"},
                dtr,
                num_boost_round=p["n_estimators"],
                valid_sets=[dvl],
                early_stopping_rounds=p["early_stopping_rounds"],
                verbose_eval=False
            )
        run_kfold_cv(cfg, X_all, y_all, final_lgb)

    # save summary CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(cfg.get("output", "results_summary.csv"), index=False)
        log.info(f"Saved summary to {cfg.get('output','results_summary.csv')}")


if __name__ == "__main__":
    main()
