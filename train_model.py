

import os
import json
import numpy as np
import joblib
from database import export_training_data
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.multiclass import unique_labels
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
import warnings
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.makedirs("models", exist_ok=True)

EVENTS = ["rain", "heat", "wind", "snow", "haze"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset():
    rows = export_training_data()
    if not rows:
        raise ValueError("No labeled data yet.")

    X = []
    Y = {event: [] for event in EVENTS}

    for r in rows:
        if not r.get("feature_vector"):
            continue

        feat = json.loads(r["feature_vector"])
        X.append(feat)

        for event in EVENTS:
            Y[event].append(r[f"label_{event}"])

    X = np.array(X, dtype=np.float32)
    Y = {k: np.array(v, dtype=int) for k, v in Y.items()}

    print(f"[Train] Dataset: {len(X)} samples, {X.shape[1]} features")
    return X, Y


def build_best_pipeline(X_train, y_train):
    """Try multiple models, return the best one by cross-val F1."""
    from sklearn.model_selection import cross_val_score

    n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])

    candidates = {
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp)),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight="balanced", random_state=42)),
        ]),
        "gradient_boost": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp)),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=4,
                learning_rate=0.1, random_state=42)),
        ]),
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp)),
            ("clf", LogisticRegression(
                class_weight="balanced", max_iter=500,
                C=1.0, random_state=42)),
        ]),
    }

    best_name, best_pipe, best_score = None, None, -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, pipe in candidates.items():
            scores = cross_val_score(pipe, X_train, y_train,
                                     cv=min(3, len(X_train)//2),
                                     scoring="f1", error_score=0)
            mean_score = scores.mean()
            print(f"  [{name}] CV F1 = {mean_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name  = name
                best_pipe  = pipe

    print(f"  -> Best: {best_name} (F1={best_score:.3f})")
    return best_pipe

# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate():
    X, Y = load_dataset()


    results = {}

    for event in EVENTS:
        y = Y[event]

        # Skip if dataset itself has only one class
        if len(set(y)) < 2:
            print(f"[Train] {event}: only one class in dataset, skipping")
            continue

        if min(np.bincount(y)) < 2:
            # too few samples → no stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42
            )
        else:
            # safe to stratify
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

        # Pipeline
        print(f"\n[Train] Selecting best model for {event}...")
        pipe = build_best_pipeline(X_train, y_train)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        print("\n" + "-" * 40)
        print(f"Event: {event.upper()}")

        labels_present = unique_labels(y_test, y_pred)

        if len(labels_present) == 1:
            print(f"[Train] Only one class present: {labels_present}")
            print(classification_report(
                y_test, y_pred,
                labels=labels_present,
                zero_division=0
            ))
        else:
            print(classification_report(
                y_test, y_pred,
                target_names=["No", "Yes"],
                zero_division=0
            ))

        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"AUC-ROC: {auc:.3f}")
        except ValueError:
            auc = None

        results[event] = {
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(auc) if auc else None
        }

        model_path = f"models/model_{event}.pkl"
        joblib.dump(pipe, model_path)
        print(f"[Train] Saved -> {model_path}")
        
    with open("models/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Train] Saved eval results -> models/eval_results.json")
        
    return results    

    


# ── Inference helper (used by main.py) ───────────────────────────────────────

def load_models():
    """Load all saved models. Returns dict {event: pipeline}."""
    models = {}
    for event in EVENTS:
        path = f"models/model_{event}.pkl"
        if os.path.exists(path):
            models[event] = joblib.load(path)
    return models


def predict_with_models(models, X):
    results = {}

    for event, model in models.items():
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0.5

        results[event] = {
            "detected": bool(pred),
            "confidence": float(prob)
        }
    

    return results

if __name__ == "__main__":
    train_and_evaluate()
