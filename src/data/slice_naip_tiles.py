import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def slice_tile(tif_path, output_rgb_dir, output_4band_dir, patch_size=640, overlap=0.1, min_valid=0.8):
    stride = int(patch_size * (1 - overlap))
    site_name = tif_path.stem
    written = 0
    skipped = 0
    with rasterio.open(tif_path) as src:
        width, height, n_bands = src.width, src.height, src.count
        log.info(f"{site_name}: {width}x{height}px, {n_bands} bands, GSD={src.res[0]:.3f}m")
        x_starts = list(range(0, width - patch_size + 1, stride))
        y_starts = list(range(0, height - patch_size + 1, stride))
        if x_starts and x_starts[-1] + patch_size < width:
            x_starts.append(width - patch_size)
        if y_starts and y_starts[-1] + patch_size < height:
            y_starts.append(height - patch_size)
        with tqdm(total=len(x_starts)*len(y_starts), desc=f"Slicing {site_name}") as pbar:
            for row_i, y in enumerate(y_starts):
                for col_i, x in enumerate(x_starts):
                    pbar.update(1)
                    window = Window(x, y, patch_size, patch_size)
                    data = src.read(window=window)
                    if np.sum(data[0] > 0) / (patch_size * patch_size) < min_valid:
                        skipped += 1
                        continue
                    patch_name = f"{site_name}_r{row_i:04d}_c{col_i:04d}"
                    rgb = np.stack([data[0], data[1], data[2]], axis=-1)
                    if rgb.dtype != np.uint8:
                        rgb = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6) * 255).astype(np.uint8)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_rgb_dir / f"{patch_name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    p4_path = output_4band_dir / f"{patch_name}.tif"
                    transform = rasterio.windows.transform(window, src.transform)
                    profile = src.profile.copy()
                    profile.update({"width": patch_size, "height": patch_size, "transform": transform, "count": min(n_bands, 4)})
                    with rasterio.open(p4_path, "w", **profile) as dst:
                        dst.write(data[:4])
                    written += 1
    log.info(f"{site_name}: {written} patches written, {skipped} skipped")
    return written

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=Path, default=Path("data/naip/raw"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/naip/patches"))
    parser.add_argument("--patch_size", type=int,  default=640)
    parser.add_argument("--overlap",    type=float, default=0.1)
    parser.add_argument("--min_valid",  type=float, default=0.8)
    args = parser.parse_args()

    rgb_dir   = args.output_dir / "images"
    band4_dir = args.output_dir / "images_4band"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    band4_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(args.input_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif files in {args.input_dir}")
    log.info(f"Found {len(tifs)} GeoTIFFs")
    total = sum(slice_tile(t, rgb_dir, band4_dir, args.patch_size, args.overlap, args.min_valid) for t in tifs)
    log.info(f"Total patches: {total}")
    log.info(f"RGB patches for CVAT: {rgb_dir}")

if __name__ == "__main__":
    main()
