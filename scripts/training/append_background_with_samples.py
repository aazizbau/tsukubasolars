"""Append positive samples into an augmented GeoPackage with background points."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import fiona


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append positive point samples into an existing augmented GeoPackage "
            "containing background points."
        )
    )
    parser.add_argument(
        "--positives",
        type=Path,
        default=Path("map/tsukubasolar100label_gp.gpkg"),
        help="GeoPackage containing positive points.",
    )
    parser.add_argument(
        "--positive-layer",
        type=str,
        default="tsukubasolar100",
        help="Layer name for positive samples.",
    )
    parser.add_argument(
        "--augmented",
        type=Path,
        default=Path("map/tsukubasolar100labels_with_background.gpkg"),
        help="Target GeoPackage that already has background points.",
    )
    parser.add_argument(
        "--augmented-layer",
        type=str,
        default="tsukubasolar_augmented",
        help="Layer name inside the target GeoPackage.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="label",
        help="Name of the label attribute.",
    )
    parser.add_argument(
        "--label-value",
        type=int,
        default=1,
        help="Label value to assign to all appended positive samples.",
    )
    parser.add_argument(
        "--source-field",
        type=str,
        default="source",
        help="Name of the source attribute.",
    )
    parser.add_argument(
        "--source-value",
        type=str,
        default="positive",
        help="Source value to assign to all appended positive samples.",
    )
    parser.add_argument(
        "--overwrite-source",
        action="store_true",
        help="If set, overwrite the target layer instead of appending.",
    )
    return parser.parse_args()


def layer_exists(path: Path, layer: str) -> bool:
    if not path.exists():
        return False
    try:
        return layer in fiona.listlayers(path)
    except Exception:
        return False


def main() -> None:
    args = parse_args()

    # Load positive samples and set attributes.
    positives = gpd.read_file(args.positives, layer=args.positive_layer)
    positives[args.label_field] = args.label_value
    positives[args.source_field] = args.source_value

    # Ensure target directory exists.
    args.augmented.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.overwrite_source or not layer_exists(args.augmented, args.augmented_layer) else "a"

    # When appending to an existing layer, geopandas requires schema alignment.
    if mode == "a":
        existing = gpd.read_file(args.augmented, layer=args.augmented_layer)
        # Align columns: keep existing columns order, add missing ones to positives.
        for col in existing.columns:
            if col not in positives.columns:
                positives[col] = None
        # Also keep any new columns from positives by ordering to match existing first.
        positives = positives[existing.columns]

    positives.to_file(
        args.augmented,
        layer=args.augmented_layer,
        driver="GPKG",
        mode=mode,
    )
    print(
        f"Appended {len(positives)} positives to {args.augmented} (layer: {args.augmented_layer})"
    )


if __name__ == "__main__":
    main()
