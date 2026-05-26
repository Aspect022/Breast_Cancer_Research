"""Classical tree/stacking baseline for genomic expression data."""

from __future__ import annotations

from typing import Dict

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _maybe_xgb_classifier(model_cfg: Dict):
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(model_cfg.get("xgb_estimators", 300)),
            max_depth=int(model_cfg.get("xgb_max_depth", 3)),
            learning_rate=float(model_cfg.get("xgb_lr", 0.05)),
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=int(model_cfg.get("n_jobs", 4)),
            random_state=int(model_cfg.get("seed", 42)),
        )
    except Exception:
        return GradientBoostingClassifier(random_state=int(model_cfg.get("seed", 42)))


def build_tree_stack(model_cfg: Dict):
    """Build RF + XGB/GB + SVM stack with logistic meta-learner."""
    seed = int(model_cfg.get("seed", 42))
    n_jobs = int(model_cfg.get("n_jobs", 4))

    rf = RandomForestClassifier(
        n_estimators=int(model_cfg.get("rf_estimators", 500)),
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )
    xgb_or_gb = _maybe_xgb_classifier(model_cfg)
    svm = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("svm", SVC(kernel=model_cfg.get("svm_kernel", "rbf"), probability=True, class_weight="balanced")),
        ]
    )
    meta = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000,
        class_weight="balanced",
        random_state=seed,
    )

    return StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb_or_gb), ("svm", svm)],
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=None,
    )

