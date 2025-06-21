# app.py
import yaml
import pandas as pd
import joblib
import optuna
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.model_selection import KFold

# 1) Load config
with open("config.yml") as f:
    cfg = yaml.safe_load(f)

app = FastAPI(title="Phase 2 Tuning & Serving API")

# 2) Pydantic model for /predict
class PredictIn(BaseModel):
    model: str              # one of "lgbm","xgb","cat","mlp"
    data: list              # list of rows, each a dict of feature→value

# 3) Helper: single Optuna study per package
def tune_package(name, df, target_col="y"):
    space = cfg[name]
    n_trials = cfg["optuna"]["n_trials"]
    cv = KFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])

    def objective(trial):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # suggest hyperparams from config ranges...
        if name == "lgbm":
            import lightgbm as lgb
            params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", *space["learning_rate"]),
                "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                "subsample": trial.suggest_uniform("subsample", *space["subsample"]),
                "colsample_bytree": trial.suggest_uniform("colsample", *space["colsample"]),
                "n_estimators": space["n_estimators"],
                "metric": "rmse",
                "verbosity": -1,
            }
            scores = []
            for train_i, valid_i in cv.split(X):
                dtrain = lgb.Dataset(X.iloc[train_i], label=y.iloc[train_i])
                dvalid = lgb.Dataset(X.iloc[valid_i], label=y.iloc[valid_i])
                bst = lgb.train(
                    params, dtrain,
                    valid_sets=[dvalid],
                    early_stopping_rounds=space["early_stopping_rounds"],
                    verbose_eval=False
                )
                preds = bst.predict(X.iloc[valid_i])
                scores.append(((preds - y.iloc[valid_i]) ** 2).mean() ** 0.5)
            return sum(scores) / len(scores)

        elif name == "xgb":
            import xgboost as xgb
            params = {
                "learning_rate": trial.suggest_loguniform("learning_rate", *space["learning_rate"]),
                "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
                "subsample": trial.suggest_uniform("subsample", *space["subsample"]),
                "colsample_bytree": trial.suggest_uniform("colsample", *space["colsample"]),
                "objective": "reg:squarederror",
                "verbosity": 0,
            }
            scores = []
            for train_i, valid_i in cv.split(X):
                dtrain = xgb.DMatrix(X.iloc[train_i], label=y.iloc[train_i])
                dvalid = xgb.DMatrix(X.iloc[valid_i], label=y.iloc[valid_i])
                bst = xgb.train(
                    params, dtrain,
                    num_boost_round=space["n_estimators"][1],
                    evals=[(dvalid, "valid")],
                    early_stopping_rounds=space["early_stopping_rounds"],
                    verbose_eval=False
                )
                preds = bst.predict(dvalid)
                scores.append(((preds - y.iloc[valid_i]) ** 2).mean() ** 0.5)
            return sum(scores) / len(scores)

        elif name == "cat":
            from catboost import CatBoostRegressor, Pool
            params = {
                "learning_rate": trial.suggest_uniform("learning_rate", *space["learning_rate"]),
                "depth": trial.suggest_int("depth", *space["depth"]),
                "subsample": trial.suggest_uniform("subsample", *space["subsample"]),
                "colsample_bylevel": trial.suggest_uniform("colsample", *space["colsample"]),
                "iterations": space["iterations"],
                "verbose": False,
                "early_stopping_rounds": space["early_stopping_rounds"],
            }
            scores = []
            for train_i, valid_i in cv.split(X):
                model = CatBoostRegressor(**params)
                model.fit(X.iloc[train_i], y.iloc[train_i],
                          eval_set=Pool(X.iloc[valid_i], y.iloc[valid_i]))
                preds = model.predict(X.iloc[valid_i])
                scores.append(((preds - y.iloc[valid_i])**2).mean()**0.5)
            return sum(scores) / len(scores)

        else:  # mlp
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            scores = []
            for train_i, valid_i in cv.split(X):
                # build model per trial
                n_layers = trial.suggest_int("n_layers", *space["n_layers"])
                units    = trial.suggest_int("units", *space["units"])
                rate     = trial.suggest_uniform("dropout", *space["dropout"])
                lr       = trial.suggest_loguniform("learning_rate", *space["learning_rate"])
                model = Sequential()
                model.add(Dense(units, activation="relu", input_shape=(X.shape[1],)))
                for _ in range(n_layers-1):
                    model.add(Dropout(rate))
                    model.add(Dense(units, activation="relu"))
                model.add(Dense(1))
                model.compile(optimizer=Adam(lr), loss="mse")
                model.fit(
                    X.iloc[train_i], y.iloc[train_i],
                    epochs=space["max_epochs"],
                    batch_size=space["batch_size"],
                    validation_data=(X.iloc[valid_i], y.iloc[valid_i]),
                    verbose=0,
                    callbacks=[...],  # early stopping callback
                )
                preds = model.predict(X.iloc[valid_i]).ravel()
                scores.append(((preds - y.iloc[valid_i])**2).mean()**0.5)
            return sum(scores) / len(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # retrain on full data
    best_params = study.best_params
    if name == "mlp":
        joblib.dump(study, f"models/{name}_optuna_study.pkl")
    else:
        # placeholder: retrain & save model
        pass

    return {"package": name,
            "rmse": study.best_value,
            **best_params}

# 4) /train endpoint
@app.post("/train")
def train():
    # load your data; assume a CSV with target column "y"
    df = pd.read_csv(cfg["data_path"])
    results = []
    for pkg in ("lgbm","xgb","cat","mlp"):
        res = tune_package(pkg, df, target_col=cfg["target_col"])
        results.append(res)
    # save summary
    pd.DataFrame(results).to_csv(cfg["output"], index=False)
    return {"status": "trained", "results": results}

# 5) /metrics endpoint
@app.get("/metrics")
def metrics():
    try:
        df = pd.read_csv(cfg["output"])
        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(404, detail="Metrics file not found; run /train first.")

# 6) /predict endpoint
@app.post("/predict")
def predict(req: PredictIn):
    fname = f"models/{req.model}.pkl"
    try:
        model = joblib.load(fname)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Model {req.model} not found.")
    Xnew = pd.DataFrame(req.data)
    preds = model.predict(Xnew)
    return {"predictions": preds.tolist()}

# 7) Health check
@app.get("/health")
def health():
    return {"status": "ok"}
