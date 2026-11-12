#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm

CLASS_NAMES = [f"c{i}" for i in range(10)]

# Config
ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "imgs" / "train"
DRIVER_CSV = ROOT / "driver_imgs_list.csv"
CACHE = ROOT / "classical_cache" / "hog_features.joblib"
OUT_DIR = ROOT / "classical_results"

IMG_SIZE = (128, 96)
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 9

N_ESTIMATORS = 200
MAX_DEPTH = None
TEST_SIZE = 0.2
SEED = 42
MAX_PER_CLASS = None 
REBUILD_CACHE = False


def collect_image_records(
    data_root: Path, max_per_class: int | None
) -> list[dict]:
    records: list[dict] = []
    for class_name in CLASS_NAMES:
        class_dir = data_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        paths = sorted(class_dir.glob("*.jpg"))
        if max_per_class is not None:
            paths = paths[:max_per_class]
        label = int(class_name[1:])
        for path in paths:
            records.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "classname": class_name,
                    "label": label,
                }
            )
    if not records:
        raise RuntimeError(f"No images found under {data_root}")
    return records


def extract_hog_vector(
    path: str,
    img_size: tuple[int, int],
    orientations: int,
    pixels_per_cell: tuple[int, int],
    cells_per_block: tuple[int, int],
) -> np.ndarray:
    img = Image.open(path).convert("L").resize(img_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return hog(
        arr,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    )


def build_or_load_features(records: list[dict]) -> dict:
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    if CACHE.exists() and not REBUILD_CACHE:
        cached = load(CACHE)
        if (
            cached.get("filenames") == [r["filename"] for r in records]
            and cached.get("img_size") == IMG_SIZE
            and cached.get("orientations") == ORIENTATIONS
            and cached.get("pixels_per_cell") == PIXELS_PER_CELL
            and cached.get("cells_per_block") == CELLS_PER_BLOCK
        ):
            print(f"Loaded cached features from {CACHE}")
            return cached
        print("Cache mismatch; recomputing HOG features.")

    features = []
    for rec in tqdm(records, desc="Extracting HOG"):
        features.append(
            extract_hog_vector(
                rec["path"],
                IMG_SIZE,
                ORIENTATIONS,
                PIXELS_PER_CELL,
                CELLS_PER_BLOCK,
            )
        )

    payload = {
        "X": np.asarray(features, dtype=np.float32),
        "y": np.asarray([r["label"] for r in records], dtype=np.int64),
        "filenames": [r["filename"] for r in records],
        "classnames": [r["classname"] for r in records],
        "img_size": IMG_SIZE,
        "orientations": ORIENTATIONS,
        "pixels_per_cell": PIXELS_PER_CELL,
        "cells_per_block": CELLS_PER_BLOCK,
    }
    dump(payload, CACHE)
    print(f"Saved HOG cache to {CACHE}  shape={payload['X'].shape}")
    return payload


def attach_drivers(records: list[dict], driver_csv: Path) -> np.ndarray | None:
    if not driver_csv.exists():
        print(
            f"WARNING: {driver_csv} not found. "
            "Driver-aware split will be skipped. "
            "Download driver_imgs_list.csv from the Kaggle competition page."
        )
        return None

    df = pd.read_csv(driver_csv)
    #expected columns: subject, classname, img
    img_col = "img" if "img" in df.columns else df.columns[-1]
    subj_col = "subject" if "subject" in df.columns else df.columns[0]
    mapping = dict(zip(df[img_col].astype(str), df[subj_col].astype(str)))

    drivers = []
    missing = 0
    for rec in records:
        driver = mapping.get(rec["filename"])
        if driver is None:
            missing += 1
            drivers.append("UNKNOWN")
        else:
            drivers.append(driver)

    if missing:
        print(f"WARNING: {missing} images missing from driver CSV.")
    n_drivers = len(set(drivers) - {"UNKNOWN"})
    print(f"Attached drivers for {len(records) - missing} images ({n_drivers} subjects).")
    return np.asarray(drivers)


def evaluate_split(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    out_dir: Path,
) -> dict:
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    print(f"\n=== {name} ===")
    print(f"train={len(train_idx)}  test={len(test_idx)}")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    report = classification_report(
        y_test, pred, target_names=CLASS_NAMES, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_test, pred)

    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")
    print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=4))

    model_path = out_dir / f"rf_{name}.joblib"
    dump(clf, model_path)

    result = {
        "split": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {c: report[c]["f1-score"] for c in CLASS_NAMES},
        "confusion_matrix": cm.tolist(),
        "model_path": str(model_path),
    }
    with open(out_dir / f"metrics_{name}.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / f"confusion_{name}.npy", cm)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning images under {DATA_ROOT}")
    records = collect_image_records(DATA_ROOT, MAX_PER_CLASS)
    print(f"Found {len(records)} images across {len(CLASS_NAMES)} classes")

    feat = build_or_load_features(records)
    X, y = feat["X"], feat["y"]
    drivers = attach_drivers(records, DRIVER_CSV)

    # 1) Naive random split (required comparison)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )
    results = [
        evaluate_split("random_split", X, y, train_idx, test_idx, OUT_DIR)
    ]

    #Split by driver
    if drivers is not None and not np.all(drivers == "UNKNOWN"):
        gss = GroupShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=SEED
        )
        train_idx, test_idx = next(gss.split(X, y, groups=drivers))
        held_out = sorted(set(drivers[test_idx]))
        print(f"Held-out drivers ({len(held_out)}): {', '.join(held_out)}")
        results.append(
            evaluate_split("driver_split", X, y, train_idx, test_idx, OUT_DIR)
        )
        gap = results[0]["accuracy"] - results[1]["accuracy"]
        print(
            f"\nLeakage gap (random - driver accuracy): {gap:.4f} "
            "(large positive => subject leakage)"
        )

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "feature": "HOG",
                "classifier": "RandomForest",
                "img_size": list(IMG_SIZE),
                "orientations": ORIENTATIONS,
                "pixels_per_cell": list(PIXELS_PER_CELL),
                "cells_per_block": list(CELLS_PER_BLOCK),
                "n_estimators": N_ESTIMATORS,
                "seed": SEED,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
