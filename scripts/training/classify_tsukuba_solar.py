"""Train a pixel-wise solar panel classifier from labeled points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import fiona
import numpy as np
import rasterio
import joblib
from rasterio.warp import transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample the 64-band Tsukuba embeddings at labeled points and "
            "train a pixel classifier (KNN or Random Forest) to detect solar panels."
        )
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
        "--model",
        choices=("knn", "random-forest"),
        default="knn",
        help="Classifier to train.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=5,
        help="Neighbors for KNN (ignored for random forest).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees for random forest (ignored for KNN).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class.tif"),
        help="Output classification GeoTIFF path (uint8 mask).",
    )
    parser.add_argument(
        "--prob-output",
        type=Path,
        default=Path("map/tsukuba_solarpanel_prob.tif"),
        help="Optional probability GeoTIFF path (float32).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/solar_classifier_model.joblib"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.5,
        help="Probability threshold to convert to class mask.",
    )
    return parser.parse_args()


def reproject_point(
    x: float, y: float, src_crs: str | dict | None, dst_crs: str | dict
) -> Tuple[float, float]:
    """Reproject a point coordinate if necessary."""
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
    """Sample raster pixels at labeled vector points."""
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
            # Handle 2D or 3D point coordinates; discard Z if present.
            if isinstance(coords, (list, tuple)):
                if len(coords) >= 2 and not isinstance(coords[0], (list, tuple)):
                    x, y = coords[0], coords[1]
                elif len(coords) and isinstance(coords[0], (list, tuple)):
                    # Fallback for unexpected nesting: grab first pair.
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


def build_model(args: argparse.Namespace):
    if args.model == "knn":
        if args.num_neighbors < 1:
            raise ValueError("Number of neighbors must be positive.")
        return KNeighborsClassifier(n_neighbors=args.num_neighbors, n_jobs=-1)
    if args.n_estimators < 1:
        raise ValueError("Number of trees must be positive.")
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        n_jobs=-1,
        random_state=42,
    )


def classify_raster(
    dataset: rasterio.io.DatasetReader,
    model,
    output_path: Path,
    prob_output_path: Path | None = None,
    prob_threshold: float = 0.5,
) -> None:
    """Apply the trained model to every pixel and write the classification raster."""
    meta = dataset.meta.copy()
    meta.update(count=1, dtype="uint8", nodata=0)

    prob_meta = None
    if prob_output_path:
        prob_meta = dataset.meta.copy()
        prob_meta.update(count=1, dtype="float32", nodata=None)
        prob_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst_class, (
        rasterio.open(prob_output_path, "w", **prob_meta)
        if prob_meta
        else rasterio.Env()
    ) as dst_prob:
        for idx, (block_index, window) in enumerate(dataset.block_windows(1), start=1):
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
            probabilities = (
                np.zeros(flat.shape[0], dtype=np.float32) if prob_output_path else None
            )

            if np.any(valid_mask):
                if prob_output_path and hasattr(model, "predict_proba"):
                    prob_vals = model.predict_proba(flat[valid_mask])[:, 1]
                elif prob_output_path:
                    prob_vals = model.predict(flat[valid_mask]).astype(np.float32)
                else:
                    prob_vals = None

                if prob_vals is not None and probabilities is not None:
                    probabilities[valid_mask] = prob_vals.astype(np.float32)
                    preds = (probabilities[valid_mask] >= prob_threshold).astype(
                        np.uint8
                    )
                else:
                    preds = np.asarray(model.predict(flat[valid_mask]), dtype=np.uint8)
                predictions[valid_mask] = preds

            predictions = predictions.reshape(height, width)
            dst_class.write(predictions, 1, window=window)

            if prob_output_path and probabilities is not None and dst_prob:
                dst_prob.write(
                    probabilities.reshape(height, width), 1, window=window
                )

            if idx % 50 == 0:
                print(f"Processed {idx} windows ...")


def main() -> None:
    args = parse_args()
    print("Opening raster ...")
    with rasterio.open(args.raster) as dataset:
        print("Sampling training data ...")
        X, y = load_training_samples(dataset, args.samples, args.layer, args.label_field)
        print(f"Training samples: {X.shape[0]} pixels with {X.shape[1]} features.")

        print(f"Training {args.model} classifier ...")
        model = build_model(args)
        model.fit(X, y)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_output)
        print(f"Model saved: {args.model_output}")

        print("Model trained. Classifying raster ...")
        classify_raster(
            dataset,
            model,
            args.output,
            prob_output_path=args.prob_output,
            prob_threshold=args.prob_threshold,
        )
    print(f"Saved classification raster to {args.output}")
    if args.prob_output:
        print(f"Saved probability raster to {args.prob_output}")


if __name__ == "__main__":
    main()
