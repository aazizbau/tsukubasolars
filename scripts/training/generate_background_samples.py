"""Generate background training points for Tsukuba solar classification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import fiona
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform
from rasterio.windows import Window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random background points from the Tsukuba embeddings mosaic "
            "to augment solar-panel training data."
        )
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=Path("E:/data/interim/tsukuba_alphaearth_2024_mosaic_clipped.tif"),
        help="Input embeddings GeoTIFF used for sampling.",
    )
    parser.add_argument(
        "--positives",
        type=Path,
        default=Path("map/tsukubasolar100label_gp.gpkg"),
        help="GeoPackage containing positive solar panel points.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="tsukubasolar100",
        help="Layer name for the positive samples.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="label",
        help="Name of the label column to write in the output file.",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="Label value assigned to existing solar samples.",
    )
    parser.add_argument(
        "--background-label",
        type=int,
        default=0,
        help="Label value for generated background samples.",
    )
    parser.add_argument(
        "--num-background",
        type=int,
        default=200,
        help="Number of background points to sample.",
    )
    parser.add_argument(
        "--min-distance-pixels",
        type=float,
        default=5.0,
        help="Minimum pixel distance from positive samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("map/tsukuba_solarlabel_with_background.gpkg"),
        help="Output GeoPackage path containing positives + background samples.",
    )
    parser.add_argument(
        "--output-layer",
        type=str,
        default="tsukuba_solarlabel_augmented",
        help="Layer name for the output GeoPackage.",
    )
    return parser.parse_args()


def reproject_point(
    x: float,
    y: float,
    src_crs: str | dict | None,
    dst_crs: str | dict | None,
) -> Tuple[float, float]:
    if not src_crs or src_crs == dst_crs:
        return x, y
    xs, ys = transform(src_crs, dst_crs, [x], [y])
    return xs[0], ys[0]


def load_positive_pixels(
    dataset: rasterio.io.DatasetReader,
    vector_path: Path,
    layer: str,
    label_field: str,
    positive_label: int,
) -> Tuple[list[dict], list[Tuple[int, int]], str | dict | None]:
    """Load positive features and track their pixel coordinates."""
    features: list[dict] = []
    pixels: list[Tuple[int, int]] = []

    with fiona.open(vector_path, layer=layer) as src:
        output_crs = src.crs_wkt or src.crs or dataset.crs
        src_crs = src.crs_wkt or src.crs

        for feat in src:
            geom = feat.get("geometry")
            if not geom or geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates")
            if not coords or len(coords) < 2:
                continue
            x, y = coords[:2]

            rx, ry = reproject_point(x, y, src_crs, dataset.crs)
            col, row = dataset.index(rx, ry)
            if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
                continue
            pixels.append((row, col))
            properties = {label_field: positive_label, "source": "positive"}
            features.append({"type": "Feature", "geometry": geom, "properties": properties})

    if not features:
        raise RuntimeError("No positive point geometries found.")
    return features, pixels, output_crs


def is_valid_pixel(
    dataset: rasterio.io.DatasetReader,
    row: int,
    col: int,
    nodata,
) -> bool:
    if row < 0 or row >= dataset.height or col < 0 or col >= dataset.width:
        return False
    sample = dataset.read(1, window=Window(col, row, 1, 1))
    value = sample[0, 0]
    if not np.isfinite(value):
        return False
    if nodata is None:
        return True
    if np.isnan(nodata):
        return not np.isnan(value)
    return value != nodata


def too_close(
    row: int,
    col: int,
    positive_pixels: Iterable[Tuple[int, int]],
    min_distance: float,
) -> bool:
    if min_distance <= 0:
        return False
    min_sq = min_distance * min_distance
    for prow, pcol in positive_pixels:
        dr = row - prow
        dc = col - pcol
        if dr * dr + dc * dc < min_sq:
            return True
    return False


def sample_background_points(
    dataset: rasterio.io.DatasetReader,
    positive_pixels: list[Tuple[int, int]],
    num_samples: int,
    min_distance: float,
    rng: np.random.Generator,
) -> list[Tuple[int, int]]:
    samples: list[Tuple[int, int]] = []
    attempts = 0
    nodata = dataset.nodata
    while len(samples) < num_samples:
        attempts += 1
        if attempts > num_samples * 100:
            raise RuntimeError(
                "Unable to sample sufficient background points. "
                "Consider lowering min_distance or num_background."
            )
        row = int(rng.integers(0, dataset.height))
        col = int(rng.integers(0, dataset.width))
        if too_close(row, col, positive_pixels, min_distance):
            continue
        window = rasterio.windows.Window(col, row, 1, 1)
        sample = dataset.read(1, window=window)[0, 0]
        if not np.isfinite(sample):
            continue
        if nodata is not None:
            if np.isnan(nodata) and np.isnan(sample):
                continue
            if not np.isnan(nodata) and sample == nodata:
                continue
        samples.append((row, col))
    return samples


def write_augmented_samples(
    output_path: Path,
    layer: str,
    crs,
    label_field: str,
    features: list[dict],
) -> None:
    schema = {
        "geometry": "Point",
        "properties": {
            label_field: "int",
            "source": "str",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(
        output_path,
        mode="w",
        driver="GPKG",
        schema=schema,
        crs=crs,
        layer=layer,
    ) as dst:
        for feature in features:
            dst.write(feature)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print("Opening raster ...")
    with rasterio.open(args.raster) as dataset:
        print("Loading positive samples ...")
        positive_features, positive_pixels, output_crs = load_positive_pixels(
            dataset, args.positives, args.layer, args.label_field, args.positive_label
        )
        print(f"Loaded {len(positive_features)} positive samples.")

        print("Sampling background points ...")
        background_pixels = sample_background_points(
            dataset,
            positive_pixels,
            args.num_background,
            args.min_distance_pixels,
            rng,
        )
        print(f"Generated {len(background_pixels)} background points.")

        features = positive_features.copy()
        for row, col in background_pixels:
            x_raster, y_raster = xy(dataset.transform, row, col, offset="center")
            x_out, y_out = reproject_point(x_raster, y_raster, dataset.crs, output_crs)
            geom = {"type": "Point", "coordinates": (x_out, y_out)}
            props = {args.label_field: args.background_label, "source": "background"}
            features.append({"type": "Feature", "geometry": geom, "properties": props})

    print(f"Writing augmented dataset to {args.output} (layer: {args.output_layer}) ...")
    write_augmented_samples(args.output, args.output_layer, output_crs, args.label_field, features)
    print("Background sampling complete.")


if __name__ == "__main__":
    main()
