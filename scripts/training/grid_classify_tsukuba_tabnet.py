"""Grid search TabNet for Tsukuba solar detection (includes hard negatives)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import fiona
import joblib
import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search TabNet hyperparameters using labeled point samples (with hard negatives)."
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
        help="GeoPackage with labeled point samples (positives, background, hard negatives).",
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
        help="Attribute containing numeric class labels (0/1).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples held out for evaluation.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        help="Probability thresholds to sweep for evaluation metrics.",
    )
    parser.add_argument(
        "--n-d",
        type=int,
        nargs="+",
        default=(8, 16),
        help="Dimensionality of the decision prediction layer (n_d) to try.",
    )
    parser.add_argument(
        "--n-a",
        type=int,
        nargs="+",
        default=(8, 16),
        help="Dimensionality of the attention embedding (n_a) to try.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        nargs="+",
        default=(3, 4),
        help="Number of sequential steps to try.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        nargs="+",
        default=(1.0, 1.5),
        help="Relaxation factor for feature reuse.",
    )
    parser.add_argument(
        "--lambda-sparse",
        type=float,
        nargs="+",
        default=(1e-4, 1e-3),
        help="Sparsity regularization for TabNet.",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=(1e-3, 5e-4),
        help="Learning rates to try.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Maximum epochs for TabNet training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--virtual-batch-size",
        type=int,
        default=128,
        help="Virtual batch size for Ghost Batch Normalization.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        nargs="+",
        default=("none", "balanced", "pos2", "pos3"),
        help="Class weight options: none | balanced | posX for class1 weight X (class0 weight 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class_tabnet_grid.tif"),
        help="Output classification GeoTIFF path (uint8 mask).",
    )
    parser.add_argument(
        "--prob-output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_prob_tabnet_grid.tif"),
        help="Probability GeoTIFF path (float32).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/solar_classifier_model_tabnet_grid"),
        help="Path prefix to save the best TabNet model (.zip will be appended).",
    )
    parser.add_argument(
        "--skip-classify",
        action="store_true",
        help="Only train/evaluate, skip writing classification rasters.",
    )
    return parser.parse_args()


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
    """Apply model to every pixel and write class/probability rasters."""
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


def evaluate_thresholds(y_true: np.ndarray, prob: np.ndarray, thresholds: Iterable[float]):
    best_f1 = -1.0
    best_thresh = None
    best_metrics = None
    for th in thresholds:
        preds = (prob >= th).astype(np.uint8)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        print(
            f"Threshold {th:.2f}: acc={acc*100:.2f}% prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}"
        )
        print(cm)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_metrics = (acc, prec, rec, f1, cm)
    return best_thresh, best_metrics


def parse_class_weights(options: tuple[str, ...], y: np.ndarray) -> list[list[float] | None]:
    counts = np.bincount(y, minlength=2)
    weights: list[list[float] | None] = []
    for opt in options:
        if opt == "none":
            weights.append(None)
        elif opt == "balanced":
            # Simple balanced weighting: w1 scaled to balance class counts.
            if counts[1] == 0:
                weights.append(None)
            else:
                w1 = counts[0] / counts[1]
                weights.append([1.0, w1])
        elif opt.startswith("pos"):
            try:
                val = float(opt.replace("pos", ""))
                weights.append([1.0, val])
            except ValueError:
                continue
    if not weights:
        weights.append(None)
    return weights


def make_sample_weights(y: np.ndarray, class_weights: list[float] | None) -> np.ndarray | None:
    if class_weights is None:
        return None
    w0, w1 = class_weights
    weights = np.ones_like(y, dtype=float)
    weights[y == 0] = w0
    weights[y == 1] = w1
    return weights


def main() -> None:
    args = parse_args()
    print("Opening raster ...")
    with rasterio.open(args.raster) as dataset:
        print("Sampling training data ...")
        X, y = load_training_samples(dataset, args.samples, args.layer, args.label_field)
        print(f"Training samples: {X.shape[0]} pixels with {X.shape[1]} features.")

        print("Splitting train/test ...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
        class_weights_grid = parse_class_weights(tuple(args.class_weights), y_train)

        best_f1 = -1.0
        best_model = None
        best_thresh = None
        best_params = None
        best_metrics = None

        for cw in class_weights_grid:
            for n_d in args.n_d:
                for n_a in args.n_a:
                    for n_steps in args.n_steps:
                        for gamma in args.gamma:
                            for lambda_sparse in args.lambda_sparse:
                                for lr in args.learning_rates:
                                    params = dict(
                                    n_d=n_d,
                                    n_a=n_a,
                                    n_steps=n_steps,
                                    gamma=gamma,
                                    lambda_sparse=lambda_sparse,
                                    learning_rate=lr,
                                    class_weights=cw,
                                )
                                    print(f"Training TabNet with params: {params}")
                                    X_tr, X_val, y_tr, y_val = train_test_split(
                                        X_train,
                                        y_train,
                                        test_size=0.2,
                                        stratify=y_train,
                                        random_state=42,
                                    )
                                    train_weights = make_sample_weights(y_tr, cw)
                                    val_weights = make_sample_weights(y_val, cw)
                                    clf = TabNetClassifier(
                                        n_d=n_d,
                                        n_a=n_a,
                                        n_steps=n_steps,
                                        gamma=gamma,
                                        lambda_sparse=lambda_sparse,
                                        optimizer_params={"lr": lr},
                                        n_independent=1,
                                        n_shared=1,
                                        seed=42,
                                        verbose=0,
                                    )
                                    fit_kwargs = dict(
                                        X_train=X_tr,
                                        y_train=y_tr,
                                        eval_set=[(X_val, y_val)],
                                        eval_name=["val"],
                                        eval_metric=["accuracy"],
                                        patience=args.patience,
                                        max_epochs=args.max_epochs,
                                        batch_size=args.batch_size,
                                        virtual_batch_size=args.virtual_batch_size,
                                        drop_last=False,
                                    )
                                    if train_weights is not None:
                                        fit_kwargs["weights"] = train_weights
                                    clf.fit(**fit_kwargs)
                                    prob_test = clf.predict_proba(X_test)[:, 1]
                                    th, metrics = evaluate_thresholds(
                                        y_test, prob_test, args.thresholds
                                    )
                                    if metrics:
                                        acc, prec, rec, f1, cm = metrics
                                        if f1 > best_f1:
                                            best_f1 = f1
                                            best_thresh = th
                                            best_model = clf
                                            best_params = params
                                            best_metrics = dict(
                                                accuracy=acc, precision=prec, recall=rec, f1=f1, cm=cm
                                            )
                                    print(f"Finished params: {params}")

        if best_model is None:
            raise RuntimeError("No model trained successfully.")

        print("Selected best model:")
        print(best_params)
        if best_metrics:
            print(
                f"Best metrics at threshold {best_thresh:.2f}: "
                f"acc={best_metrics['accuracy']*100:.2f}% "
                f"prec={best_metrics['precision']:.3f} "
                f"rec={best_metrics['recall']:.3f} "
                f"f1={best_metrics['f1']:.3f}"
            )

        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        save_prefix = str(args.model_output)
        best_model.save_model(save_prefix)
        print(f"Saved best TabNet model to {save_prefix}.zip")
        joblib.dump(
            {"params": best_params, "threshold": best_thresh, "metrics": best_metrics},
            Path(save_prefix).with_suffix(".meta.joblib"),
        )

        if not args.skip_classify:
            print("Classifying full raster with best model ...")
            classify_raster(
                dataset,
                best_model,
                args.output,
                prob_output_path=args.prob_output,
                prob_threshold=best_thresh if best_thresh is not None else 0.5,
            )
            print(f"Saved classification raster to {args.output}")
            print(f"Saved probability raster to {args.prob_output}")


if __name__ == "__main__":
    main()
