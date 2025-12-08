"""Clean MLP prediction raster by removing tiny positive blobs and evaluate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import fiona
import numpy as np
import rasterio
from rasterio.warp import transform
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove small connected positive components from the MLP prediction raster "
            "and compare metrics before/after cleaning."
        )
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class_mlp_grid.tif"),
        help="Input classification raster (MLP).",
    )
    parser.add_argument(
        "--cleaned-raster",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class_mlp_grid_cleaned.tif"),
        help="Output cleaned raster path.",
    )
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("map/tsukubasolar100labels_with_background.gpkg"),
        help="GeoPackage with labeled samples.",
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
        help="Label field name (0/1).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=9,
        help="Remove connected positive components with pixel count <= this value.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("map/mlp_cleaning_metrics.csv"),
        help="CSV file to store metrics before/after cleaning.",
    )
    return parser.parse_args()


def reproject_point(
    x: float, y: float, src_crs: str | dict | None, dst_crs: str | dict
) -> Tuple[float, float]:
    if not src_crs or src_crs == dst_crs:
        return x, y
    xs, ys = transform(src_crs, dst_crs, [x], [y])
    return xs[0], ys[0]


def remove_small_components(arr: np.ndarray, min_size: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)  # 8-connected
    labeled, num = ndimage.label(arr == 1, structure=structure)
    if num == 0:
        return arr
    counts = np.bincount(labeled.ravel())
    remove = np.isin(labeled, np.where(counts <= min_size)[0])
    cleaned = arr.copy()
    cleaned[remove] = 0
    return cleaned


def evaluate_raster(
    dataset: rasterio.io.DatasetReader,
    raster_array: np.ndarray,
    vector_path: Path,
    layer: str,
    label_field: str,
) -> Tuple[float, float, float, float, np.ndarray]:
    preds: list[int] = []
    labels: list[int] = []
    with fiona.open(vector_path, layer=layer) as src:
        src_crs = src.crs_wkt or src.crs
        for feat in src:
            props = feat.get("properties") or {}
            if label_field not in props:
                continue
            try:
                label_val = int(props[label_field])
            except Exception:
                continue
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates")
            if coords is None:
                continue
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                x, y = coords[0], coords[1]
            else:
                continue
            x, y = reproject_point(x, y, src_crs, dataset.crs)
            row, col = dataset.index(x, y)
            if row < 0 or col < 0 or row >= raster_array.shape[0] or col >= raster_array.shape[1]:
                continue
            pred_val = int(raster_array[row, col])
            preds.append(pred_val)
            labels.append(label_val)
    preds_arr = np.asarray(preds, dtype=int)
    labels_arr = np.asarray(labels, dtype=int)
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1])
    acc = accuracy_score(labels_arr, preds_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, average="binary", zero_division=0
    )
    return acc, prec, rec, f1, cm


def main() -> None:
    args = parse_args()
    with rasterio.open(args.raster) as src:
        data = src.read(1)
        meta = src.meta.copy()
        acc_before, prec_before, rec_before, f1_before, cm_before = evaluate_raster(
            src, data, args.samples, args.layer, args.label_field
        )

        cleaned_data = remove_small_components(data, args.min_size)
        acc_after, prec_after, rec_after, f1_after, cm_after = evaluate_raster(
            src, cleaned_data, args.samples, args.layer, args.label_field
        )

        meta.update(dtype="uint8", count=1)
        args.cleaned_raster.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.cleaned_raster, "w", **meta) as dst:
            dst.write(cleaned_data.astype(np.uint8), 1)

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"])
        writer.writerow(
            [
                "before",
                f"{acc_before:.4f}",
                f"{prec_before:.4f}",
                f"{rec_before:.4f}",
                f"{f1_before:.4f}",
                cm_before[0, 0],
                cm_before[0, 1],
                cm_before[1, 0],
                cm_before[1, 1],
            ]
        )
        writer.writerow(
            [
                "after",
                f"{acc_after:.4f}",
                f"{prec_after:.4f}",
                f"{rec_after:.4f}",
                f"{f1_after:.4f}",
                cm_after[0, 0],
                cm_after[0, 1],
                cm_after[1, 0],
                cm_after[1, 1],
            ]
        )
    print("Metrics saved to", args.csv_output)
    print("Cleaned raster saved to", args.cleaned_raster)


if __name__ == "__main__":
    main()
