"""Clip the Tsukuba AlphaEarth mosaic to the city boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import fiona
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import transform_geom


def read_clip_geometries(
    vector_path: Path, layer: str, target_crs: CRS
) -> list[dict]:
    """Return geometries from the vector layer, reprojected to target_crs."""
    with fiona.open(vector_path, layer=layer) as src:
        source_crs = CRS(src.crs) if src.crs else None
        geoms: list[dict] = []
        for feature in src:
            geom = feature["geometry"]
            if source_crs and source_crs != target_crs:
                geom = transform_geom(
                    source_crs.to_string(),
                    target_crs.to_string(),
                    geom,
                )
            geoms.append(geom)

    if not geoms:
        raise RuntimeError(
            f"No geometries found in {vector_path} (layer {layer})."
        )
    return geoms


def clip_mosaic(
    mosaic_path: Path,
    vector_path: Path,
    layer: str,
    output_path: Path,
) -> None:
    """Clip the mosaic raster to the supplied vector boundary."""
    with rasterio.open(mosaic_path) as src:
        geoms = read_clip_geometries(vector_path, layer, CRS(src.crs))
        data, transform = mask(src, geoms, crop=True)
        meta = src.meta.copy()
        meta.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data)

    print(f"Saved clipped mosaic to {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clip the Tsukuba AlphaEarth mosaic by the city boundary."
    )
    parser.add_argument(
        "--mosaic",
        type=Path,
        default=Path("data/interim/tsukuba_alphaearth_2024_mosaic.tif"),
        help="Input mosaic GeoTIFF path.",
    )
    parser.add_argument(
        "--vector",
        type=Path,
        default=Path("map/tsukuba_gp.gpkg"),
        help="Vector boundary data (GeoPackage).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="tsukuba",
        help="Layer name inside the GeoPackage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/tsukuba_alphaearth_2024_mosaic_clipped.tif"),
        help="Destination path for the clipped mosaic.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    clip_mosaic(
        mosaic_path=args.mosaic,
        vector_path=args.vector,
        layer=args.layer,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
