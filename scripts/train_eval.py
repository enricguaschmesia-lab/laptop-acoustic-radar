"""Leave-one-session-out evaluation of presence detection and activity classification.

Models: majority class, energy threshold (presence only), energy-only logistic regression,
hand-crafted features with logistic regression and a random forest, and the small CNN on
range-Doppler + range-time maps (several seeds). Each fold trains on three sessions and
tests on the fourth; overlapping windows never cross the train/test boundary.

    python scripts/train_eval.py [--seeds 3] [--epochs 30]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radar.evaluation import presence_scores, presence_threshold

ROOT = Path(__file__).resolve().parents[1]

MODELS = ("majority", "energy_lr", "features_lr", "features_rf", "cnn")
LABELS = {"majority": "Majority class", "energy_lr": "Energy only (LR)", "features_lr": "Features + LR",
          "features_rf": "Features + RF", "cnn": "Small CNN (maps)"}


def activity_scores(y: np.ndarray, pred: np.ndarray, sess: np.ndarray, n_classes: int) -> dict:
    folds = {int(s): {"accuracy": float(accuracy_score(y[sess == s], pred[sess == s])),
                      "macro_f1": float(f1_score(y[sess == s], pred[sess == s], average="macro",
                                                 labels=range(n_classes), zero_division=0))}
             for s in np.unique(sess)}
    cm = confusion_matrix(y, pred, labels=range(n_classes))
    return {"pooled_accuracy": float(accuracy_score(y, pred)),
            "pooled_macro_f1": float(f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0)),
            "fold_macro_f1_mean": float(np.mean([f["macro_f1"] for f in folds.values()])),
            "fold_macro_f1_std": float(np.std([f["macro_f1"] for f in folds.values()])),
            "per_class_recall": (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist(),
            "folds": folds, "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "windows.npz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ml")
    args = ap.parse_args()

    import torch

    from radar.models import SmallCNN, n_params, predict, train_cnn
    torch.set_num_threads(max(1, torch.get_num_threads()))

    d = np.load(args.data)
    maps, feats, y, sess = d["maps"].astype(np.float32), d["feats"], d["y"], d["session"]
    classes, names = [str(c) for c in d["classes"]], [str(n) for n in d["feature_names"]]
    n_cls, i_energy, empty = len(classes), names.index("motion_db"), classes.index("empty")
    occupied = y != empty
    print(f"{len(y)} windows, sessions {sorted(set(sess.tolist()))}, maps {maps.shape[1:]}, {len(names)} features")

    preds = {m: np.full(len(y), -1) for m in MODELS if m != "cnn"}
    cnn_preds = [np.full(len(y), -1) for _ in range(args.seeds)]
    energy_presence = np.zeros(len(y), bool)
    thresholds, importances, train_s = {}, [], []
    for s in sorted(set(sess.tolist())):
        tr, te = sess != s, sess == s
        preds["majority"][te] = np.bincount(y[tr], minlength=n_cls).argmax()
        thresholds[s] = presence_threshold(feats[tr, i_energy], occupied[tr])
        energy_presence[te] = feats[te, i_energy] > thresholds[s]
        lr1 = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        preds["energy_lr"][te] = lr1.fit(feats[tr][:, [i_energy]], y[tr]).predict(feats[te][:, [i_energy]])
        lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        preds["features_lr"][te] = lr.fit(feats[tr], y[tr]).predict(feats[te])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=0, n_jobs=-1).fit(feats[tr], y[tr])
        preds["features_rf"][te] = rf.predict(feats[te])
        importances.append(rf.feature_importances_)

        mu = maps[tr].mean(axis=(0, 2, 3), keepdims=True)
        sd = maps[tr].std(axis=(0, 2, 3), keepdims=True) + 1e-6
        x_tr, x_te = (maps[tr] - mu) / sd, (maps[te] - mu) / sd
        for seed in range(args.seeds):
            t0 = time.time()
            model = train_cnn(x_tr, y[tr], n_cls, seed=seed, epochs=args.epochs)
            train_s.append(time.time() - t0)
            cnn_preds[seed][te] = predict(model, x_te)
        f1_fold = [f1_score(y[te], cnn_preds[k][te], average="macro", labels=range(n_cls), zero_division=0)
                   for k in range(args.seeds)]
        print(f"  fold: test session {s} ({te.sum()} windows) | energy thr {thresholds[s]:.1f} dB | macro-F1 "
              f"energy {f1_score(y[te], preds['energy_lr'][te], average='macro'):.2f}, "
              f"LR {f1_score(y[te], preds['features_lr'][te], average='macro'):.2f}, "
              f"RF {f1_score(y[te], preds['features_rf'][te], average='macro'):.2f}, "
              f"CNN {np.mean(f1_fold):.2f}+/-{np.std(f1_fold):.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "predictions.npz", y=y, session=sess, capture=d["capture"],
                        energy_presence=energy_presence, cnn=np.asarray(cnn_preds),
                        **{m: p for m, p in preds.items()})

    # Inference cost of the CNN on one window (CPU, single thread)
    torch.set_num_threads(1)
    probe = SmallCNN(in_ch=maps.shape[1], n_classes=n_cls).eval()
    xb = torch.zeros((1,) + maps.shape[1:])
    with torch.no_grad():
        for _ in range(5):
            probe(xb)
        t0 = time.time()
        for _ in range(50):
            probe(xb)
    infer_ms = (time.time() - t0) / 50 * 1e3

    report = {"n_windows": int(len(y)), "classes": classes, "feature_names": names,
              "presence_energy_thresholds_db": thresholds,
              "presence": {"energy_threshold": presence_scores(occupied, energy_presence)},
              "activity": {}, "rf_importance": dict(zip(names, np.mean(importances, axis=0).tolist(), strict=True)),
              "cnn": {"params": n_params(probe), "seeds": args.seeds, "epochs": args.epochs,
                      "train_seconds_per_fold": float(np.mean(train_s)), "inference_ms_per_window": infer_ms}}
    for m in preds:
        report["activity"][m] = activity_scores(y, preds[m], sess, n_cls)
        report["presence"][m] = presence_scores(occupied, preds[m] != empty)
    per_seed = [activity_scores(y, p, sess, n_cls) for p in cnn_preds]
    cnn = activity_scores(np.tile(y, args.seeds), np.concatenate(cnn_preds), np.tile(sess, args.seeds), n_cls)
    cnn["seed_pooled_macro_f1"] = [r["pooled_macro_f1"] for r in per_seed]
    cnn["seed_pooled_accuracy"] = [r["pooled_accuracy"] for r in per_seed]
    cnn["folds"] = {s: {k: float(np.mean([r["folds"][s][k] for r in per_seed])) for k in ("accuracy", "macro_f1")}
                    for s in per_seed[0]["folds"]}
    report["activity"]["cnn"] = cnn
    report["presence"]["cnn"] = presence_scores(np.tile(occupied, args.seeds), np.concatenate(cnn_preds) != empty)

    # Per-capture accuracy: shows which recordings (if any) carry the errors
    capture = d["capture"]
    all_preds = dict(preds, cnn=cnn_preds[0])
    report["per_capture_accuracy"] = {
        str(c): {m: float(np.mean(p[capture == c] == y[capture == c])) for m, p in all_preds.items()}
        for c in np.unique(capture)}

    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    plot(report, args.out)

    print(f"\n{'model':18s} {'accuracy':>9s} {'macro-F1':>9s} {'fold F1 (mean+/-std)':>21s}  recall "
          + "/".join(classes))
    for m in MODELS:
        a = report["activity"][m]
        extra = (f"  [seeds: {', '.join(f'{v:.2f}' for v in a['seed_pooled_macro_f1'])}]" if m == "cnn" else "")
        print(f"{LABELS[m]:18s} {a['pooled_accuracy']:9.3f} {a['pooled_macro_f1']:9.3f} "
              f"{a['fold_macro_f1_mean']:10.3f} +/- {a['fold_macro_f1_std']:.3f}   "
              + "/".join(f"{r:.2f}" for r in a["per_class_recall"]) + extra)
    print("\nPresence (empty vs occupied):")
    for m, p in report["presence"].items():
        print(f"  {m:17s} balanced acc {p['balanced_accuracy']:.3f}, miss {p['miss_rate']:.3f}, "
              f"false alarm {p['false_alarm_rate']:.3f}")
    print(f"\nCNN: {report['cnn']['params']} parameters, {infer_ms:.2f} ms per window on one CPU thread, "
          f"{report['cnn']['train_seconds_per_fold']:.0f} s training per fold and seed")
    print("\nCaptures with any error (accuracy per model, CNN seed 0):")
    for c, acc in report["per_capture_accuracy"].items():
        if any(v < 1.0 for m, v in acc.items() if m not in ("majority", "energy_lr")):
            print(f"  {c:24s} " + "  ".join(f"{m} {v:.2f}" for m, v in acc.items()))


def plot(report: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = report["classes"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, m in zip(ax, ("energy_lr", "features_rf", "cnn"), strict=True):
        cm = np.asarray(report["activity"][m]["confusion"], float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                       color="white" if cmn[i, j] > 0.6 else "black")
        a.set(xticks=range(len(classes)), xticklabels=classes, yticks=range(len(classes)), yticklabels=classes,
              xlabel="predicted", ylabel="true",
              title=f"{LABELS[m]}\nmacro-F1 {report['activity'][m]['pooled_macro_f1']:.2f}")
    fig.suptitle("Activity, leave-one-session-out (rows normalised)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(MODELS):
        a = report["activity"][m]
        ax.bar(i, a["fold_macro_f1_mean"], yerr=a["fold_macro_f1_std"], color="C0" if m != "cnn" else "C1",
               alpha=0.8, capsize=4)
        ax.plot([i] * len(a["folds"]), [f["macro_f1"] for f in a["folds"].values()], "k.", ms=8)
    ax.set(xticks=range(len(MODELS)), xticklabels=[LABELS[m] for m in MODELS], ylabel="macro-F1 per test session",
           ylim=(0, 1.05), title="Activity classification, leave-one-session-out (dots: test sessions)")
    plt.setp(ax.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(out / "macro_f1.png", dpi=130)
    plt.close(fig)

    imp = report["rf_importance"]
    order = sorted(imp, key=imp.get)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(order, [imp[k] for k in order])
    ax.set(xlabel="Mean decrease in impurity", title="Random-forest feature importance")
    fig.tight_layout()
    fig.savefig(out / "rf_importance.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
