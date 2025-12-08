"""Classify Tsukuba solar panels on 2017 mosaic using trained RF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Tsukuba solar panels on a mosaic using a trained RF model."
    )
    parser.add_argument(
        "--input-raster",
        type=Path,
        default=Path("E:/data/interim/tsukuba_alphaearth_2017_mosaic_clipped.tif"),
        help="Input embeddings GeoTIFF to classify.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/solar_classifier_model_rf_grid.joblib"),
        help="Trained RF model path.",
    )
    parser.add_argument(
        "--output-raster",
        type=Path,
        default=Path("map/tsukuba_solarpanel_class_rf_grid_2017.tif"),
        help="Output classification raster path (uint8 mask).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for positive class.",
    )
    return parser.parse_args()


def classify_raster(
    raster_path: Path, model, output_path: Path, prob_threshold: float
) -> None:
    with rasterio.open(raster_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype="uint8", nodata=0)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **meta) as dst:
            for idx, (_, window) in enumerate(src.block_windows(1), start=1):
                block = src.read(window=window).astype(np.float32)
                bands, height, width = block.shape
                flat = block.reshape(bands, -1).T

                valid = np.all(np.isfinite(flat), axis=1)
                nodata = src.nodata
                if nodata is not None:
                    if np.isnan(nodata):
                        valid &= ~np.all(np.isnan(flat), axis=1)
                    else:
                        valid &= ~np.all(flat == nodata, axis=1)

                preds = np.zeros(flat.shape[0], dtype=np.uint8)
                if np.any(valid):
                    prob = model.predict_proba(flat[valid])[:, 1]
                    preds[valid] = (prob >= prob_threshold).astype(np.uint8)

                dst.write(preds.reshape(height, width), 1, window=window)

                if idx % 50 == 0:
                    print(f"Processed {idx} windows ...")


def main() -> None:
    args = parse_args()
    print(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)
    print(f"Classifying raster {args.input_raster} ...")
    classify_raster(args.input_raster, model, args.output_raster, prob_threshold=args.threshold)
    print(f"Saved classification raster to {args.output_raster}")


if __name__ == "__main__":
    main()
