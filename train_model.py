import os, io, sys, json, warnings
import numpy as np
import joblib
from database import export_training_data
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score, f1_score)
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.class_weight import compute_sample_weight
from attention_fusion import (train_attention_model, load_attention_models,
                              predict_attention)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                               errors="replace")

os.makedirs("models", exist_ok=True)

EVENTS      = ["rain", "heat", "wind", "snow", "haze"]
FEATURE_DIM = 1437   


#  Data loading 

def load_dataset():
    rows = export_training_data()
    if not rows:
        raise ValueError("No labeled data yet. Run the collector first.")

    X, Y, city_labels = [], {e: [] for e in EVENTS}, []

    for r in rows:
        if not r.get("feature_vector"):
            continue

        feat = json.loads(r["feature_vector"])

        if len(feat) < FEATURE_DIM:
            feat = feat + [0.0] * (FEATURE_DIM - len(feat))
        feat = feat[:FEATURE_DIM]   # truncate if somehow longer

        X.append(feat)
        city_labels.append(r.get("city", "unknown"))
        for event in EVENTS:
            Y[event].append(r[f"label_{event}"])

    X    = np.array(X, dtype=np.float32)
    Y    = {k: np.array(v, dtype=int) for k, v in Y.items()}
    city_labels = np.array(city_labels)

    print(f"[Train] Dataset: {len(X)} samples, {X.shape[1]} features")

    
    city_sample_weights = compute_sample_weight("balanced", city_labels)
    print(f"[Train] City distribution: "
          f"{dict(zip(*np.unique(city_labels, return_counts=True)))}")

    return X, Y, city_sample_weights, city_labels


# Time-based split 

def _time_split(X, y, weights, test_ratio=0.20):
    """Chronological split. Last test_ratio rows = test set."""
    n      = len(X)
    cutoff = int(n * (1 - test_ratio))
    return (X[:cutoff], X[cutoff:],
            y[:cutoff], y[cutoff:],
            weights[:cutoff])   # weights only needed for training


#  sample_weight builder for GradientBoosting + city balance combined

def _combined_weights(y_train, city_weights_train):
    """
    Multiply city-balance weights by class-balance weights.
    This corrects for BOTH imbalanced cities AND imbalanced event labels,
    which GradientBoostingClassifier cannot do via class_weight="balanced".
    """
    class_w  = compute_sample_weight("balanced", y_train)
    combined = city_weights_train * class_w
    # Normalise so weights sum to n_samples (sklearn convention)
    combined = combined / combined.mean()
    return combined


# Model selection

def build_best_pipeline(X_train, y_train, city_weights_train):
    """
    Grid over 4 models using TimeSeriesSplit CV.
    For GBT, passes sample_weight via fit_params.
    Returns (best_pipeline, best_name, best_score).
    """
    n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
    n_cv   = max(2, min(5, len(X_train) // 10))
    tscv   = TimeSeriesSplit(n_splits=n_cv)

    # FIX 2 — sample weights for GBT (and others for consistency)
    sw_combined = _combined_weights(y_train, city_weights_train)

    candidates = {
        "random_forest": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n_comp)),
                ("clf",    RandomForestClassifier(
                               n_estimators=200, max_depth=10,
                               class_weight="balanced", random_state=42)),
            ]),
            None,   # RF handles class_weight natively
        ),
        "gradient_boost": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n_comp)),
                ("clf",    GradientBoostingClassifier(
                               n_estimators=150, max_depth=4,
                               learning_rate=0.08, random_state=42)),
            ]),
            sw_combined,   # FIX 2: pass combined weights to GBT
        ),
        "logistic": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n_comp)),
                ("clf",    LogisticRegression(
                               class_weight="balanced", max_iter=600,
                               C=1.0, random_state=42)),
            ]),
            None,
        ),
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca",    PCA(n_components=n_comp)),
                ("clf",    SVC(kernel="rbf", class_weight="balanced",
                               probability=True, random_state=42)),
            ]),
            None,
        ),
    }

    best_name, best_pipe, best_score = None, None, -1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, (pipe, sw) in candidates.items():
            fit_params = {}
            if sw is not None:
                fit_params["clf__sample_weight"] = sw

            # CV scoring (sample_weight not passed to CV scorer — intentional;
            # the weight is for training fairness, not metric weighting)
            try:
                scores = cross_val_score(
                    pipe, X_train, y_train,
                    cv=tscv, scoring="f1", error_score=0,
                )
            except Exception:
                scores = np.array([0.0])

            mean_score = scores.mean()
            print(f"  [{name}] TimeSeriesCV F1 = {mean_score:.3f}")
            if mean_score > best_score:
                best_score = mean_score
                best_name  = name
                best_pipe  = (pipe, sw)

    print(f"  → Best: {best_name} (F1={best_score:.3f})")
    return best_pipe, best_name


# Late-fusion ensemble (Step 6 — advanced fusion)

def build_late_fusion_ensemble(X_train, y_train, city_weights_train):
    """
    Late fusion: RF + GBT + LR trained independently, soft-voted.
    GBT gets combined sample_weight, others use class_weight="balanced".
    """
    n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
    sw     = _combined_weights(y_train, city_weights_train)

    rf = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_comp)),
                   ("clf", RandomForestClassifier(
                       100, class_weight="balanced", random_state=0))])
    gb = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_comp)),
                   ("clf", GradientBoostingClassifier(100, random_state=0))])
    lr = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_comp)),
                   ("clf", LogisticRegression(
                       class_weight="balanced", max_iter=500, random_state=0))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train,
               clf__sample_weight=sw)   # FIX 2 for GBT in ensemble too
        lr.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr)],
        voting="soft",
    )
    # VotingClassifier with pre-fitted estimators: fit a dummy pass
    ensemble.estimators_ = [rf, gb, lr]
    ensemble.le_         = None
    ensemble.classes_    = np.array([0, 1])
    return ensemble


# Optional LSTM (Step 7 deep learning)

def build_lstm_sequence_model(input_dim, hidden=128, layers=2):
    try:
        import torch, torch.nn as nn
        class WeatherLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, layers,
                                    batch_first=True, dropout=0.3)
                self.fc   = nn.Linear(hidden, 1)
                self.sig  = nn.Sigmoid()
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.sig(self.fc(out[:, -1, :]))
        return WeatherLSTM()
    except ImportError:
        print("[Train] PyTorch not available — LSTM skipped")
        return None


def train_lstm(X, Y, seq_len=5, epochs=20):
    try:
        import torch, torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[LSTM] PyTorch not available — skipping")
        return

    n = len(X)
    if n < seq_len + 10:
        print("[LSTM] Not enough data for sequence training")
        return

    Xs    = np.array([X[i:i+seq_len] for i in range(n - seq_len)])
    split = int(len(Xs) * 0.8)

    for event in EVENTS:
        y = Y[event]
        ys = y[seq_len:]
        if len(set(ys)) < 2:
            continue

        model   = build_lstm_sequence_model(FEATURE_DIM)
        if model is None:
            continue

        Xt = torch.tensor(Xs[:split], dtype=torch.float32)
        yt = torch.tensor(ys[:split], dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(Xs[split:], dtype=torch.float32)
        yv_np = ys[split:]

        loader  = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=False)
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        for ep in range(epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            probs = model(Xv).squeeze().numpy()
        preds = (probs > 0.5).astype(int)
        f1    = f1_score(yv_np, preds, zero_division=0)
        print(f"[LSTM] {event} — Val F1={f1:.3f}")
        torch.save(model.state_dict(), f"models/lstm_{event}.pt")
        print(f"[LSTM] Saved → models/lstm_{event}.pt")


# SHAP explainability

def compute_shap_importances(pipe, X_train, event):
    try:
        import shap
        clf = pipe.named_steps["clf"]
        X_t = pipe[:-1].transform(X_train)
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_t[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0).tolist()
        with open(f"models/shap_{event}.json", "w") as f:
            json.dump({"mean_abs_shap": mean_abs}, f)
        print(f"  [SHAP] Saved → models/shap_{event}.json")
    except Exception as e:
        print(f"  [SHAP] Skipped: {e}")


# Main training routine

def train_and_evaluate():
    X, Y, city_weights, city_labels = load_dataset()
    results = {}

    for event in EVENTS:
        y = Y[event]

        if len(set(y)) < 2:
            print(f"[Train] {event}: only one class — skipping")
            continue

        # ── Time-based split (FIX 4) ───────────────────────────────────────
        X_tr, X_te, y_tr, y_te, cw_tr = _time_split(X, y, city_weights)
        print(f"\n[Train] ── {event.upper()} ──────────────────────────────")
        print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")
        print(f"  Positive rate — train: {y_tr.mean():.2%}, test: {y_te.mean():.2%}")

        (best_pipe, best_sw), best_name = build_best_pipeline(X_tr, y_tr, cw_tr)

        fit_kw = {}
        if best_sw is not None:
            fit_kw["clf__sample_weight"] = _combined_weights(y_tr, cw_tr)
        best_pipe.fit(X_tr, y_tr, **fit_kw)

        y_pred   = best_pipe.predict(X_te)
        y_prob   = best_pipe.predict_proba(X_te)[:, 1]
        single_f1 = float(f1_score(y_te, y_pred, zero_division=0))

        labels_present = unique_labels(y_te, y_pred)
        print(classification_report(
            y_te, y_pred, labels=labels_present,
            target_names=["No","Yes"] if len(labels_present)==2 else None,
            zero_division=0,
        ))

        auc = None
        try:
            auc = roc_auc_score(y_te, y_prob)
            print(f"  AUC-ROC: {auc:.3f}")
        except ValueError:
            pass

        # ── Late-fusion ensemble ───────────────────────────────────────────
        ens_f1_val = None
        ensemble   = None
        if len(X_tr) >= 30:
            try:
                ensemble  = build_late_fusion_ensemble(X_tr, y_tr, cw_tr)
                ens_pred  = ensemble.predict(X_te)
                ens_f1_val = float(f1_score(y_te, ens_pred, zero_division=0))
                print(f"  [Late Fusion] F1={ens_f1_val:.3f}  "
                      f"vs Single F1={single_f1:.3f}")
            except Exception as e:
                print(f"  [Late Fusion] Failed: {e}")

        if ens_f1_val and ens_f1_val > single_f1 and ensemble is not None:
            joblib.dump(ensemble, f"models/model_{event}.pkl")
            chosen_f1   = ens_f1_val
            chosen_type = "late_fusion_ensemble"
            print(f"  → Saved ensemble (F1={ens_f1_val:.3f})")
        else:
            joblib.dump(best_pipe, f"models/model_{event}.pkl")
            chosen_f1   = single_f1
            chosen_type = best_name
            print(f"  → Saved single model: {best_name} (F1={single_f1:.3f})")

        results[event] = {
            "f1":         chosen_f1,
            "auc":        float(auc) if auc else None,
            "model_type": chosen_type,
        }

        # SHAP (only for Pipeline with named clf step)
        if hasattr(best_pipe, "named_steps"):
            compute_shap_importances(best_pipe, X_tr, event)

    with open("models/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Train] Evaluation saved → models/eval_results.json")

    print("\n[Train] Training deep learning models...")

    # (a) LSTM sequence model
    try:
        train_lstm(X, Y)
    except Exception as e:
        print(f"[LSTM] Skipped: {e}")

    # (b) CNN+LSTM+Attention hybrid (spec: "CNN plus LSTM hybrids,
    #     transformer-based architectures")
    print("\n[Train] Training CNN+LSTM+Attention models...")
    attn_results = {}
    for event in EVENTS:
        y = Y[event]
        if len(set(y)) < 2:
            continue
        try:
            val_f1 = train_attention_model(X, y, event, epochs=30)
            if val_f1 is not None:
                attn_results[event] = {"attn_f1": round(val_f1, 3)}
                results[event]["attn_f1"] = round(val_f1, 3)
        except Exception as e:
            print(f"[Attention/{event}] Skipped: {e}")

    with open("models/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Train] All results saved → models/eval_results.json")

    return results


# Inference helpers 

def load_models():
    """Load sklearn models (.pkl) and LSTM state dicts (.pt)."""
    models = {}
    for event in EVENTS:
        path = f"models/model_{event}.pkl"
        if os.path.exists(path):
            models[event] = joblib.load(path)
    return models


def load_lstm_models():
    """Load LSTM sequence models. Returns {event: (model, seq_len)} or {}."""
    try:
        import torch
    except ImportError:
        return {}
    lstm_models = {}
    for event in EVENTS:
        path = f"models/lstm_{event}.pt"
        if not os.path.exists(path):
            continue
        try:
            model = build_lstm_sequence_model(FEATURE_DIM)
            if model is None:
                continue
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            lstm_models[event] = model
            print(f"[LSTM] Loaded model for {event}")
        except Exception as e:
            print(f"[LSTM] Could not load {path}: {e}")
    return lstm_models


def predict_with_lstm(lstm_models: dict, X: "np.ndarray",
                      seq_len: int = 5) -> dict:
    try:
        import torch
    except ImportError:
        return {}
    results = {}
    # Repeat single sample to build a minimal sequence
    X_seq = np.repeat(X, seq_len, axis=0)          # (seq_len, features)
    X_seq = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, features)
    for event, model in lstm_models.items():
        with torch.no_grad():
            prob = model(X_seq).item()
        results[event] = {
            "detected":   prob > 0.5,
            "confidence": round(prob, 4),
        }
    return results


def predict_with_models(models, X):
    """Sklearn model inference. See also predict_with_lstm() and predict_attention()."""
    results = {}
    for event, model in models.items():
        pred = model.predict(X)[0]
        prob = (model.predict_proba(X)[0][1]
                if hasattr(model, "predict_proba") else 0.5)
        results[event] = {"detected": bool(pred), "confidence": float(prob)}
    return results


if __name__ == "__main__":
    train_and_evaluate()
