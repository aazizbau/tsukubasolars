"""Verify AlphaEarth embedding tile downloads for the Tsukuba region.

Usage:
    python scripts/download/check_alphaearth_tiles.py \
        --output data/raw/embeddings/tsukuba_alphaearth_2024.tif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import ee

TSUKUBA_BBOX = [
    [139.93097, 36.25236],
    [140.01016, 35.91349],
    [140.23176, 35.91425],
    [140.20968, 36.26074],
    [139.93097, 36.25236],
]


def initialize_earth_engine(project: str | None = None) -> None:
    try:
        ee.Initialize()
    except ee.EEException:
        print("Authenticating to Google Earth Engine ...")
        ee.Authenticate()
        ee.Initialize(project=project)
    else:
        print("Google Earth Engine initialized.")


def create_tsukuba_geometry() -> ee.Geometry:
    return ee.Geometry.Polygon([TSUKUBA_BBOX])


def km_to_deg_lat(kilometers: float) -> float:
    return kilometers / 111.32


def km_to_deg_lon(kilometers: float, reference_lat: float) -> float:
    cos_lat = math.cos(math.radians(reference_lat))
    if cos_lat == 0:
        raise ValueError("Cannot compute longitude degrees at the poles.")
    return kilometers / (111.32 * cos_lat)


def get_bounds_from_polygon(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    lons = [pt[0] for pt in polygon]
    lats = [pt[1] for pt in polygon]
    return min(lons), min(lats), max(lons), max(lats)


def iterate_tiles(
    geometry: ee.Geometry,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
) -> Iterable[Tuple[int, int]]:
    min_lon, min_lat, max_lon, max_lat = get_bounds_from_polygon(TSUKUBA_BBOX)
    ref_lat = 0.5 * (min_lat + max_lat)

    tile_width_deg = km_to_deg_lon(tile_width_km, ref_lat)
    tile_height_deg = km_to_deg_lat(tile_height_km)
    overlap_lon_deg = km_to_deg_lon(overlap_km, ref_lat)
    overlap_lat_deg = km_to_deg_lat(overlap_km)

    if tile_width_deg <= 0 or tile_height_deg <= 0:
        raise ValueError("Tile width/height must be greater than zero.")

    step_lon = tile_width_deg - overlap_lon_deg
    step_lat = tile_height_deg - overlap_lat_deg
    if step_lon <= 0 or step_lat <= 0:
        raise ValueError("Tile overlap must be smaller than the tile dimensions.")

    row = 0
    lat_start = min_lat
    while lat_start < max_lat:
        lat_end = min(lat_start + tile_height_deg, max_lat)
        col = 0
        lon_start = min_lon
        while lon_start < max_lon:
            lon_end = min(lon_start + tile_width_deg, max_lon)
            rect = ee.Geometry.Rectangle(
                [lon_start, lat_start, lon_end, lat_end],
                geodesic=False,
            )
            tile_region = rect.intersection(geometry, ee.ErrorMargin(1))
            area = tile_region.area(maxError=1).getInfo()
            if area and area > 0:
                yield row, col
            lon_start += step_lon
            col += 1
        lat_start += step_lat
        row += 1


def resolve_output_template(base: Path) -> Tuple[Path, str, str]:
    if base.suffix:
        return base.parent, base.stem, base.suffix
    return base, "tile", ".tif"


def check_tiles(
    output: Path,
    tile_width_km: float,
    tile_height_km: float,
    overlap_km: float,
    project: str | None,
) -> None:
    initialize_earth_engine(project=project)
    geometry = create_tsukuba_geometry()

    parent, stem, suffix = resolve_output_template(output)
    parent.mkdir(parents=True, exist_ok=True)

    missing: list[Path] = []
    total = 0
    for row, col in iterate_tiles(geometry, tile_width_km, tile_height_km, overlap_km):
        total += 1
        tile_path = parent / f"{stem}_r{row:02d}_c{col:02d}{suffix}"
        if not tile_path.exists():
            missing.append(tile_path)

    print(f"Expected tiles: {total}")
    print(f"Found tiles: {total - len(missing)}")
    if missing:
        print(f"Missing tiles ({len(missing)}):")
        for path in missing:
            print(f"  {path}")
    else:
        print("All tiles present.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check AlphaEarth tile downloads.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/embeddings/tsukuba_alphaearth_2024.tif"),
        help="Base output path used for downloads.",
    )
    parser.add_argument(
        "--tile-width-km",
        type=float,
        default=2.5,
        help="Tile width in kilometers (default: 2.5 km).",
    )
    parser.add_argument(
        "--tile-height-km",
        type=float,
        default=2.5,
        help="Tile height in kilometers (default: 2.5 km).",
    )
    parser.add_argument(
        "--tile-overlap-km",
        type=float,
        default=0.5,
        help="Overlap between tiles in kilometers (default: 0.5 km).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional Earth Engine project ID.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    check_tiles(
        output=args.output,
        tile_width_km=args.tile_width_km,
        tile_height_km=args.tile_height_km,
        overlap_km=args.tile_overlap_km,
        project=args.project,
    )


if __name__ == "__main__":
    main()
