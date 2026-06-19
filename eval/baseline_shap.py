"""Coefficient / SHAP breakdown of the P1 cheap baseline -> a usable recipe.

"Raw beats SAE" is only actionable if we know WHICH cheap features carry the
signal. P1 is an L1-logistic on 8 prompt stats; for a linear model the exact
SHAP value of feature j on a standardized example is coef_j * (x_j - mean_j),
so mean |coef_j * z_j| over the test set is an exact global attribution -- no
shap dependency needed. We report standardized coefficients and mean-|SHAP|,
ranked, plus the cumulative AUROC recipe (top-1, top-2, top-3 features alone).

Usage:
    python eval/baseline_shap.py \
        --metadata activations/squad_metadata.parquet \
        --out_json eval/results/baseline_shap/squad.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from probing.features import compute_prompt_stats, INPUT_STAT_NAMES

C_GRID = {"C": [1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]}


def fit_logistic(X_tr, y_tr, seed=42):
    base = LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(base, C_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_["C"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="activations/squad_metadata.parquet")
    ap.add_argument("--out_json", default="eval/results/baseline_shap/squad.json")
    args = ap.parse_args()

    df = pd.read_parquet(args.metadata)
    X = compute_prompt_stats(df)
    names = list(INPUT_STAT_NAMES)
    y = df["difficulty"].values
    tr, te = (df["split"] == "train").values, (df["split"] == "test").values

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
    clf, C = fit_logistic(Xtr, y[tr])
    coef = clf.coef_.ravel()
    auroc_full = roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1])

    # Exact linear SHAP: phi_ij = coef_j * z_ij ; global = mean_i |phi_ij| over test.
    shap = np.abs(coef[None, :] * Xte).mean(axis=0)
    order = np.argsort(-np.abs(coef))     # rank by |standardized coef|

    # Cumulative recipe: AUROC using only the top-k features (refit each).
    cum = []
    for k in range(1, min(4, len(names)) + 1):
        idx = order[:k]
        clf_k, _ = fit_logistic(Xtr[:, idx], y[tr])
        a = roc_auc_score(y[te], clf_k.predict_proba(Xte[:, idx])[:, 1])
        cum.append({"k": k, "features": [names[j] for j in idx], "AUROC": float(a)})

    ranked = [{"feature": names[j], "std_coef": float(coef[j]), "mean_abs_shap": float(shap[j])}
              for j in order]
    out = {"chosen_C": C, "P1_full_AUROC": float(auroc_full),
           "ranked_features": ranked, "cumulative_recipe": cum,
           "recipe": [r["feature"] for r in ranked[:3]]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    print(f"P1 full AUROC = {auroc_full:.3f} (C={C})")
    print(f"{'feature':22s} {'std_coef':>10s} {'mean|SHAP|':>11s}")
    for r in ranked:
        print(f"{r['feature']:22s} {r['std_coef']:+10.3f} {r['mean_abs_shap']:11.3f}")
    print("\ncumulative recipe (top-k features alone):")
    for c in cum:
        print(f"  top-{c['k']} {c['AUROC']:.3f}  {c['features']}")
    print(f"\nRecipe (top 3 carry it): {out['recipe']}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
