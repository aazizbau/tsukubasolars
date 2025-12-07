"""Grid search MLP for Tsukuba solar detection, with evaluation and optional raster classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import fiona
import joblib
import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search MLP hyperparameters using labeled point samples."
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=Path("E:/data/interim/tsukuba_alphaearth_2024_mosaic_clipped.tif"),
        help="Input embeddings GeoTIFF.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("map/tsukubasolar100labels_with_background.gpkg"),
        help="GeoPackage with labeled point samples.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="tsukubasolar_augmented",
        help="Layer name inside the GeoPackage.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="label",
        help="Attribute containing numeric class labels (e.g., 0 for background, 1 for solar).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples held out for evaluation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Cross-validation folds for grid search.",
    )
    parser.add_argument(
        "--hidden-layer-options",
        type=str,
        nargs="+",
        default=("256,128", "192,96", "128,64", "128,64,32"),
        help='Hidden layer options, comma-separated per option (e.g., "256,128" "128,64").',
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=("relu", "tanh"),
        help="Activation functions to try.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(1e-3, 5e-4, 1e-4),
        help="L2 regularization strengths to try.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(1e-3, 5e-4, 1e-4),
        help="Initial learning rates to try.",
    )
    parser.add_argument(
        "--learning-rate-schedules",
        type=str,
        nargs="+",
        default=("adaptive", "constant"),
        help="Learning rate schedules to try.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=(16, 32, 64),
        help="Batch sizes to try.",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        nargs="+",
        default=("none", "balanced", "pos2", "pos3", "pos4"),
        help=(
            "Class weight options. 'none' means no weighting, 'balanced' uses scikit-learn's "
            "balanced mode, 'posX' sets class_weight={0:1, 1:X}."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        help="Probability thresholds to sweep for evaluation metrics.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Maximum iterations for MLP training.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classification mask.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class_mlp_grid.tif"),
        help="Output classification GeoTIFF path (uint8 mask).",
    )
    parser.add_argument(
        "--prob-output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_prob_mlp_grid.tif"),
        help="Probability GeoTIFF path (float32).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/solar_classifier_model_mlp_grid_improved.joblib"),
        help="Path to save the best estimator.",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Only train/evaluate, skip writing classification rasters.",
    )
    return parser.parse_args()


def parse_hidden_layers(options: tuple[str, ...]) -> list[tuple[int, ...]]:
    parsed: list[tuple[int, ...]] = []
    for opt in options:
        if not opt:
            continue
        parts = [p.strip() for p in opt.split(",") if p.strip()]
        if not parts:
            continue
        parsed.append(tuple(int(p) for p in parts))
    if not parsed:
        raise ValueError("No valid hidden layer options parsed.")
    return parsed


def parse_class_weights(options: tuple[str, ...]) -> list[dict | str | None]:
    parsed: list[dict | str | None] = []
    for opt in options:
        if opt == "none":
            parsed.append(None)
        elif opt == "balanced":
            parsed.append("balanced")
        elif opt.startswith("pos"):
            try:
                val = float(opt.replace("pos", ""))
                parsed.append({0: 1.0, 1: val})
            except ValueError:
                continue
        else:
            continue
    if not parsed:
        parsed = [None]
    return parsed


def apply_class_weight_oversample(X: np.ndarray, y: np.ndarray, weight_spec) -> tuple[np.ndarray, np.ndarray]:
    """Approximate class weighting by oversampling positives."""
    if weight_spec is None:
        return X, y
    pos_mask = y == 1
    neg_mask = y == 0
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    if pos_count == 0 or neg_count == 0:
        return X, y

    if weight_spec == "balanced":
        pos_weight = neg_count / pos_count
    elif isinstance(weight_spec, dict):
        pos_weight = float(weight_spec.get(1, 1.0))
    else:
        pos_weight = 1.0

    if pos_weight <= 1.0:
        return X, y

    # Repeat positives to approximate weighting.
    repeat_times = max(1, int(np.floor(pos_weight))) - 1
    frac = pos_weight - np.floor(pos_weight)

    X_pos = X[pos_mask]
    y_pos = y[pos_mask]
    reps_list = [X_pos] * repeat_times if repeat_times > 0 else []
    y_reps_list = [y_pos] * repeat_times if repeat_times > 0 else []

    if frac > 1e-6:
        extra = max(1, int(round(frac * pos_count)))
        reps_list.append(X_pos[:extra])
        y_reps_list.append(y_pos[:extra])

    if reps_list:
        X_bal = np.concatenate([X, *reps_list], axis=0)
        y_bal = np.concatenate([y, *y_reps_list], axis=0)
    else:
        X_bal, y_bal = X, y
    return X_bal, y_bal


def reproject_point(
    x: float, y: float, src_crs: str | dict | None, dst_crs: str | dict
) -> Tuple[float, float]:
    if not src_crs or src_crs == dst_crs:
        return x, y
    xs, ys = transform(src_crs, dst_crs, [x], [y])
    return xs[0], ys[0]


def load_training_samples(
    dataset: rasterio.io.DatasetReader,
    vector_path: Path,
    layer: str,
    label_field: str,
) -> Tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    with fiona.open(vector_path, layer=layer) as src:
        src_crs = src.crs_wkt or src.crs
        for feat in src:
            props = feat.get("properties") or {}
            if label_field not in props:
                raise KeyError(f"Feature missing label field '{label_field}'.")
            label_value = props[label_field]
            try:
                label_int = int(label_value)
            except Exception as exc:
                raise ValueError(f"Label '{label_value}' is not numeric.") from exc

            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates")
            if coords is None:
                continue
            if isinstance(coords, (list, tuple)):
                if len(coords) >= 2 and not isinstance(coords[0], (list, tuple)):
                    x, y = coords[0], coords[1]
                elif len(coords) and isinstance(coords[0], (list, tuple)):
                    first = coords[0]
                    if len(first) >= 2:
                        x, y = first[0], first[1]
                    else:
                        continue
                else:
                    continue
            else:
                continue

            x, y = reproject_point(x, y, src_crs, dataset.crs)
            pixel_values = next(dataset.sample([(x, y)])).astype(np.float32)

            if not np.all(np.isfinite(pixel_values)):
                continue
            nodata = dataset.nodata
            if nodata is not None:
                if np.isnan(nodata) and np.all(np.isnan(pixel_values)):
                    continue
                if not np.isnan(nodata) and np.all(pixel_values == nodata):
                    continue

            features.append(pixel_values)
            labels.append(label_int)

    if not features:
        raise RuntimeError("No valid training samples extracted from the provided points.")

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise RuntimeError(
            f"Need at least two classes for classification. Found {unique_labels}."
        )

    return np.vstack(features), np.asarray(labels, dtype=np.int32)


def classify_raster(
    dataset: rasterio.io.DatasetReader,
    model,
    output_path: Path,
    prob_output_path: Path,
    prob_threshold: float,
) -> None:
    """Apply model to each pixel and write class/probability rasters."""
    class_meta = dataset.meta.copy()
    class_meta.update(count=1, dtype="uint8", nodata=0)
    prob_meta = dataset.meta.copy()
    prob_meta.update(count=1, dtype="float32", nodata=None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prob_output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **class_meta) as dst_class, rasterio.open(
        prob_output_path, "w", **prob_meta
    ) as dst_prob:
        for idx, (_, window) in enumerate(dataset.block_windows(1), start=1):
            block_data = dataset.read(window=window).astype(np.float32)
            bands, height, width = block_data.shape
            flat = block_data.reshape(bands, -1).T

            valid_mask = np.all(np.isfinite(flat), axis=1)
            nodata = dataset.nodata
            if nodata is not None:
                if np.isnan(nodata):
                    valid_mask &= ~np.all(np.isnan(flat), axis=1)
                else:
                    valid_mask &= ~np.all(flat == nodata, axis=1)

            predictions = np.zeros(flat.shape[0], dtype=np.uint8)
            probabilities = np.zeros(flat.shape[0], dtype=np.float32)

            if np.any(valid_mask):
                prob_vals = model.predict_proba(flat[valid_mask])[:, 1]
                probabilities[valid_mask] = prob_vals.astype(np.float32)
                predictions[valid_mask] = (prob_vals >= prob_threshold).astype(np.uint8)

            dst_class.write(predictions.reshape(height, width), 1, window=window)
            dst_prob.write(probabilities.reshape(height, width), 1, window=window)

            if idx % 50 == 0:
                print(f"Processed {idx} windows ...")


def main() -> None:
    args = parse_args()
    hidden_layer_grid = parse_hidden_layers(tuple(args.hidden_layer_options))
    class_weight_grid = parse_class_weights(tuple(args.class_weights))
    print("Opening raster ...")
    with rasterio.open(args.raster) as dataset:
        print("Sampling training data ...")
        X, y = load_training_samples(dataset, args.samples, args.layer, args.label_field)
        print(f"Training samples: {X.shape[0]} pixels with {X.shape[1]} features.")

        print("Splitting train/test ...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )

        best_overall = None
        best_overall_f1 = -1.0
        best_overall_thresh = None
        best_overall_params = None
        best_overall_cv = None
        best_overall_model = None
        best_overall_weight = None

        for cw in class_weight_grid:
            X_train_bal, y_train_bal = apply_class_weight_oversample(X_train, y_train, cw)

            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            max_iter=args.max_iter,
                            solver="adam",
                            random_state=42,
                            early_stopping=True,
                            n_iter_no_change=10,
                            validation_fraction=0.2,
                        ),
                    ),
                ]
            )

            param_grid = {
                "model__hidden_layer_sizes": hidden_layer_grid,
                "model__activation": list(args.activations),
                "model__alpha": list(args.alphas),
                "model__learning_rate_init": list(args.learning_rates),
                "model__learning_rate": list(args.learning_rate_schedules),
                "model__batch_size": list(args.batch_sizes),
            }
            print(f"Grid search parameters (class weight {cw}):", param_grid)

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=args.cv_folds,
                n_jobs=-1,
                verbose=1,
            )
            print(f"Running grid search (class weight {cw}) ...")
            grid.fit(X_train_bal, y_train_bal)

            best_model = grid.best_estimator_
            print("Best params:", grid.best_params_)
            print(f"CV best score (accuracy): {grid.best_score_:.4f}")

            print("Evaluating on hold-out set ...")
            prob_test = best_model.predict_proba(X_test)[:, 1]

            def metrics_for_threshold(th: float):
                preds = (prob_test >= th).astype(np.uint8)
                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_test, preds, average="binary", zero_division=0
                )
                return acc, cm, prec, rec, f1

            for th in args.thresholds:
                acc, cm, prec, rec, f1 = metrics_for_threshold(th)
                print(
                    f"[cw={cw}] Threshold {th:.2f}: acc={acc*100:.2f}% prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}"
                )
                print(cm)
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_overall_thresh = th
                    best_overall_params = grid.best_params_
                    best_overall_cv = grid.best_score_
                    best_overall_model = best_model
                    best_overall_weight = cw

        if best_overall_model is None:
            raise RuntimeError("No model was trained.")

        print(f"Selected best model with class weight {best_overall_weight}")
        print(f"Best params: {best_overall_params}")
        print(f"CV best score (accuracy): {best_overall_cv:.4f}")
        print(f"Best threshold by F1: {best_overall_thresh:.2f} (F1={best_overall_f1:.3f})")

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_overall_model, args.model_output)
        print(f"Saved best model to {args.model_output}")

        if not args.skip_classify:
            print("Classifying full raster with best model ...")
            classify_raster(
                dataset,
                best_overall_model,
                args.output,
                prob_output_path=args.prob_output,
                prob_threshold=best_overall_thresh if best_overall_thresh is not None else args.prob_threshold,
            )
            print(f"Saved classification raster to {args.output}")
            print(f"Saved probability raster to {args.prob_output}")


if __name__ == "__main__":
    main()
