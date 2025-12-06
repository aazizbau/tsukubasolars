# Tsukuba AlphaEarth Utilities

Helper scripts for downloading, validating, and mosaicking Google AlphaEarth embeddings over the Tsukuba area of interest.

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Each script exposes self-documented CLI options; pass `--help` to see defaults and flags.

## Download Tiles

```bash
python scripts/download/download_alphaearth_embeddings.py \
    --help
```

Run without `--help` (or after adjusting flags) to authenticate with Earth Engine, build the Tsukuba AOI mosaic, and export tiles to `data/raw/embeddings/tsukuba_alphaearth_2024.tif` by default.

## Resume Missing Tile Downloads

```bash
python scripts/download/download_missing_alphaearth_tiles.py \
    --help
```

This script checks for tiles that are absent on disk (based on the same naming convention) and re-downloads only the missing ones.

## Verify Tile Coverage

```bash
python scripts/download/check_alphaearth_tiles.py \
    --help
```

It reports how many Tsukuba tiles are expected vs. present, and lists any missing paths.

## Mosaic Tiles

```bash
python scripts/preprocessing/mosaic_alphaearth_tiles.py \
    --help
```

After tiles finish downloading, use this script to stream-write a single GeoTIFF mosaic (default output `data/interim/tsukuba_alphaearth_2024_mosaic.tif`).

## Clip Mosaic to Tsukuba Boundary

```bash
python scripts/preprocessing/clip_tsukuba_mosaic.py \
    --help
```

This crops the mosaic using the Tsukuba GeoPackage boundary (`map/tsukuba_gp.gpkg`, layer `tsukuba`) and writes the clipped raster (default `data/interim/tsukuba_alphaearth_2024_mosaic_clipped.tif`).

## K-means Clustering

```bash
python scripts/training/kmeans_tsukuba_embeddings.py \
    --help
```

Runs K-means across all 64 embedding bands and saves a cluster ID mask (values 1..k).

## Solar Panel Classification

```bash
python scripts/training/classify_tsukuba_solar.py \
    --help
```

Samples the embeddings at labeled solar panel points (`map/tsukuba_solarlabel_gp.gpkg`), trains a KNN or random-forest classifier, and writes a solar/non-solar mask GeoTIFF.

## Generate Background Samples

```bash
python scripts/training/generate_background_samples.py \
    --help
```

Creates a new GeoPackage combining your positive labels with automatically sampled background points (label 0) to balance the training set; handy before adding hard negatives such as parking lots.
