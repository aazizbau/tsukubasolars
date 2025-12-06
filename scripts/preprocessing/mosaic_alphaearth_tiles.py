"""Mosaic Tsukuba AlphaEarth tiles into a single GeoTIFF using streaming writes."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import rasterio
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download import download_alphaearth_embeddings as downloader


def list_tile_paths(base: Path) -> list[Path]:
    parent, stem, suffix = downloader.resolve_output_template(base)
    pattern = f"{stem}_r*_c*{suffix}"
    paths = sorted(parent.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No tiles found matching pattern {pattern} in directory {parent}"
        )
    return paths


def compute_mosaic_metadata(tile_paths: list[Path]) -> dict:
    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf

    ref_meta = None
    for path in tile_paths:
        with rasterio.open(path) as ds:
            bounds = ds.bounds
            minx = min(minx, bounds.left)
            miny = min(miny, bounds.bottom)
            maxx = max(maxx, bounds.right)
            maxy = max(maxy, bounds.top)

            if ref_meta is None:
                ref_meta = ds.meta.copy()
            else:
                if ds.crs != ref_meta["crs"]:
                    raise ValueError(f"CRS mismatch for {path}")
                if (ds.transform.a != ref_meta["transform"].a) or (
                    ds.transform.e != ref_meta["transform"].e
                ):
                    raise ValueError(f"Resolution/transform mismatch for {path}")

    if ref_meta is None:
        raise RuntimeError("No tiles found.")

    pixel_width = ref_meta["transform"].a
    pixel_height = -ref_meta["transform"].e

    width = int(math.ceil((maxx - minx) / pixel_width))
    height = int(math.ceil((maxy - miny) / pixel_height))

    transform = from_origin(minx, maxy, pixel_width, pixel_height)

    meta = ref_meta.copy()
    meta.update({"width": width, "height": height, "transform": transform})

    if meta.get("nodata") is None:
        meta["nodata"] = 0

    return meta


def mosaic_tiles(tile_paths: list[Path], output: Path) -> None:
    meta = compute_mosaic_metadata(tile_paths)
    output.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output, "w", **meta) as dst:
        for idx, tile_path in enumerate(tile_paths, start=1):
            with rasterio.open(tile_path) as src:
                data = src.read()
                window = from_bounds(
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                    meta["transform"],
                    precision=6,
                ).round_offsets().round_lengths()

                if window.width != src.width or window.height != src.height:
                    raise ValueError(f"Window mismatch for {tile_path}")

                dst.write(data, window=window)

            if idx % 250 == 0 or idx == len(tile_paths):
                print(f"Wrote {idx}/{len(tile_paths)} tiles ...")

    print(f"Saved mosaic to {output}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic Tsukuba AlphaEarth tiles.")
    parser.add_argument(
        "--input-base",
        type=Path,
        default=Path("data/raw/embeddings/tsukuba_alphaearth_2024.tif"),
        help="Base path used for Tsukuba per-tile downloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/tsukuba_alphaearth_2024_mosaic.tif"),
        help="Output Tsukuba GeoTIFF path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tile_paths = list_tile_paths(args.input_base)
    print(f"Found {len(tile_paths)} tiles. Building mosaic ...")
    mosaic_tiles(tile_paths, args.output)


if __name__ == "__main__":
    main()
