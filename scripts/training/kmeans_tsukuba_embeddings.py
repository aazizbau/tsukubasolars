"""Run K-means clustering on the Tsukuba AlphaEarth embeddings mosaic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster the 64-dimensional AlphaEarth embeddings using K-means "
            "and write a mask GeoTIFF with cluster IDs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("E:/data/interim/tsukuba_alphaearth_2024_mosaic_clipped.tif"),
        help="Path to the clipped Tsukuba embeddings mosaic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("E:/data/tsukuba_alphaearth_2024_clusters.tif"),
        help="Destination for the cluster raster.",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of K-means clusters to generate.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for the K-means solver.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of K-means initializations.",
    )
    return parser.parse_args()


def flatten_raster(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened raster data and valid pixel mask."""
    data = dataset.read().astype(np.float32)  # (bands, rows, cols)
    bands, height, width = data.shape

    flat = data.reshape(bands, -1).T  # (pixels, bands)
    valid_mask = np.all(np.isfinite(flat), axis=1)

    nodata = dataset.nodata
    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.all(np.isnan(flat), axis=1)
        else:
            nodata_mask = np.all(flat == nodata, axis=1)
        valid_mask &= ~nodata_mask

    if not np.any(valid_mask):
        raise RuntimeError("No valid pixels found in the input raster.")

    return flat, valid_mask.reshape(height, width)


def write_cluster_raster(
    output_path: Path,
    template: rasterio.io.DatasetReader,
    cluster_map: np.ndarray,
) -> None:
    """Persist the cluster map as a GeoTIFF."""
    meta = template.meta.copy()
    meta.update(count=1, dtype="uint16", nodata=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(cluster_map.astype(np.uint16), 1)


def run_clustering(args: argparse.Namespace) -> None:
    if args.num_clusters < 2:
        raise ValueError("Number of clusters must be at least 2.")
    if args.num_clusters > 65535:
        raise ValueError("Number of clusters must be less than 65536.")

    print(f"Loading mosaic from {args.input} ...")
    with rasterio.open(args.input) as src:
        flat_data, valid_mask = flatten_raster(src)

        valid_pixels = flat_data[valid_mask.ravel()]
        print(
            f"Found {valid_pixels.shape[0]:,} valid pixels with "
            f"{flat_data.shape[1]} features each."
        )

        print(
            f"Running K-means with k={args.num_clusters}, "
            f"max_iter={args.max_iter}, n_init={args.n_init} ..."
        )
        kmeans = KMeans(
            n_clusters=args.num_clusters,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=42,
        )
        labels = kmeans.fit_predict(valid_pixels) + 1  # ensure IDs start at 1

        cluster_map = np.zeros(valid_mask.size, dtype=np.uint16)
        cluster_map[valid_mask.ravel()] = labels
        cluster_map = cluster_map.reshape(valid_mask.shape)

        print(f"Writing cluster raster to {args.output} ...")
        write_cluster_raster(args.output, src, cluster_map)
    print("Clustering complete.")


def main() -> None:
    args = parse_args()
    run_clustering(args)


if __name__ == "__main__":
    main()
