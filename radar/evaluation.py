"""Shared evaluation helpers: presence threshold and scores, leave-one-session-out runs."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def presence_threshold(energy: np.ndarray, occupied: np.ndarray) -> float:
    """Energy threshold maximising balanced accuracy on the training windows."""
    v = np.unique(energy)
    if len(v) < 2:
        return float(v[0])
    cands = (v[:-1] + v[1:]) / 2
    return float(max(cands, key=lambda t: balanced_accuracy_score(occupied, energy > t)))


def presence_scores(occupied_true: np.ndarray, occupied_pred: np.ndarray) -> dict:
    occ, pred = occupied_true.astype(bool), occupied_pred.astype(bool)
    return {"balanced_accuracy": float(balanced_accuracy_score(occ, pred)),
            "miss_rate": float(np.mean(~pred[occ])), "false_alarm_rate": float(np.mean(pred[~occ]))}


def macro_f1(y, pred, n_classes: int) -> float:
    return float(f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0))


def loso_classical(feats: np.ndarray, y: np.ndarray, sess: np.ndarray, i_energy: int, empty: int,
                   n_classes: int) -> dict:
    """Energy-only LR, features LR, features RF and the energy presence threshold, per test session."""
    out = {m: {} for m in ("energy_lr", "features_lr", "features_rf", "presence_energy")}
    occupied = y != empty
    for s in sorted(set(sess.tolist())):
        tr, te = sess != s, sess == s
        lr1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        p = lr1.fit(feats[tr][:, [i_energy]], y[tr]).predict(feats[te][:, [i_energy]])
        out["energy_lr"][s] = macro_f1(y[te], p, n_classes)
        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        out["features_lr"][s] = macro_f1(y[te], lr.fit(feats[tr], y[tr]).predict(feats[te]), n_classes)
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=0, n_jobs=-1).fit(feats[tr], y[tr])
        out["features_rf"][s] = macro_f1(y[te], rf.predict(feats[te]), n_classes)
        thr = presence_threshold(feats[tr, i_energy], occupied[tr])
        out["presence_energy"][s] = presence_scores(occupied[te], feats[te, i_energy] > thr)["balanced_accuracy"]
    return out


def loso_cnn(maps: np.ndarray, y: np.ndarray, sess: np.ndarray, empty: int, n_classes: int,
             seed: int = 0, epochs: int = 30) -> dict:
    """Small CNN per test session (standardised with training-fold statistics)."""
    from .models import predict, train_cnn
    out = {"cnn": {}, "presence_cnn": {}}
    for s in sorted(set(sess.tolist())):
        tr, te = sess != s, sess == s
        mu = maps[tr].mean(axis=(0, 2, 3), keepdims=True)
        sd = maps[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        model = train_cnn((maps[tr] - mu) / sd, y[tr], n_classes, seed=seed, epochs=epochs)
        p = predict(model, (maps[te] - mu) / sd)
        out["cnn"][s] = macro_f1(y[te], p, n_classes)
        out["presence_cnn"][s] = presence_scores(y[te] != empty, p != empty)["balanced_accuracy"]
    return out
