#!/usr/bin/env python3
"""
Pseudo-Color RGB Generation from Hyperspectral Data.

Pipeline:
  1. Load spectral (5ms.tif), dark, sky, and grayscale (view2.tif) images
  2. Compute reflectance: (spec - dark) / (sky - dark)
  3. Extract RGB bands at coords_dict positions
     R ← Band 41 (650nm), G ← Band 21 (550nm), B ← Band 15 (520nm)
  4. Apply registration offset: spectral points shifted by dx=195, dy=-30 (from user calibration)
  5. Render as sparse scatter → nearest-neighbor interpolation → full image
  6. Histogram stretch + gray-world white balance
  7. Save summary panel, pure RGB, overlay, blended images

Usage:
  python scripts/pseudo_color.py
  python scripts/pseudo_color.py --dx 0       # override registration offset
  python scripts/pseudo_color.py --auto       # ORB feature-based auto-registration
"""

import argparse
import json
import os
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import tifffile

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import subtract_dark_current, compute_reflectance

# ── Constants ──
WAVELENGTHS = np.arange(445, 906, 5, dtype=int)  # 93 bands
R_BAND = 41   # 650nm
G_BAND = 21   # 550nm
B_BAND = 15   # 520nm (sensor has near-zero response for < 520nm)
REG_DX = 195  # registration offset (horizontal): spectral → grayscale
REG_DY = -30  # registration offset (vertical)
R_PATCH = 4   # scatter dot radius


def load_data(data_dir: str, hi_dir: str) -> tuple:
    """Load TIF images, coords_dict, compute reflectance."""
    paths = {
        "spec_base": os.path.join(data_dir, "5ms.tif"),
        "dark": os.path.join(data_dir, "P11070000.tif"),
        "illuminance": os.path.join(data_dir, "5ms_sky.tif"),
        "gray": os.path.join(data_dir, "view2.tif"),
    }
    images = {}
    for name, p in paths.items():
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)
        images[name] = tifffile.imread(p)

    with open(os.path.join(hi_dir, "coords_dict.json")) as f:
        coords_dict = json.load(f)

    img_spec = subtract_dark_current(images["spec_base"], images["dark"])
    img_sky = subtract_dark_current(images["illuminance"], images["dark"])
    reflect = compute_reflectance(img_spec, img_sky)
    reflect = np.clip(reflect, 0, None)

    print(f"  Reflectance: {reflect.shape}, range=[{reflect.min():.4f}, {reflect.max():.4f}]")
    return images, coords_dict, reflect


def extract_rgb(reflect: np.ndarray, coords_dict: dict) -> tuple:
    """
    Extract R/G/B reflectance at each calibration point.

    Uses band 0 (first band) position as the spatial reference,
    following the old code convention.
    """
    max_b = max(R_BAND, G_BAND, B_BAND)
    items = [(k, v) for k, v in coords_dict.items() if len(v) > max_b]
    n = len(items)
    rgb = np.zeros((n, 3), dtype=np.float64)
    pos = np.zeros((n, 2), dtype=int)

    for i, (_, spec) in enumerate(items):
        for ch, bi in [(0, B_BAND), (1, G_BAND), (2, R_BAND)]:
            y = np.clip(spec[bi][1], 0, reflect.shape[0] - 1)
            x = np.clip(spec[bi][2], 0, reflect.shape[1] - 1)
            rgb[i, ch] = reflect[y, x]

        # Spatial reference: band 0 (first band) position
        y = np.clip(spec[0][1], 0, 2047)
        x = np.clip(spec[0][2], 0, 2047)
        pos[i] = [y, x]

    # Filter extreme values (likely noise/saturated pixels)
    mask = (rgb.min(axis=1) >= 0) & (rgb.max(axis=1) <= 2.0)
    rgb = rgb[mask]
    pos = pos[mask]
    print(f"  RGB: {rgb.shape} (filtered: {mask.sum()}/{len(mask)})")
    return rgb, pos


def render_image(rgb_pts: np.ndarray, pos: np.ndarray,
                 shape: tuple, dx: int, dy: int = 0) -> np.ndarray:
    """
    Render sparse RGB points to a full image.

    Steps:
    1. Apply registration offset (dx, dy)
    2. Scatter plot small patches
    3. Nearest-neighbor interpolation to fill gaps
    """
    H, W = shape
    y_pos = pos[:, 0] + dy  # spectral → grayscale alignment
    x_pos = pos[:, 1] + dx

    ok = (y_pos >= 0) & (y_pos < H) & (x_pos >= 0) & (x_pos < W)
    y_pos, x_pos, rgb_pts = y_pos[ok], x_pos[ok], rgb_pts[ok]

    img = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    half = R_PATCH // 2

    for i in range(len(y_pos)):
        y1 = max(0, y_pos[i] - half)
        y2 = min(H, y_pos[i] + half + 1)
        x1 = max(0, x_pos[i] - half)
        x2 = min(W, x_pos[i] + half + 1)
        for c in range(3):
            img[y1:y2, x1:x2, c] += rgb_pts[i, c]
        weight[y1:y2, x1:x2] += 1.0

    mask = weight > 0
    for c in range(3):
        img[:, :, c] = np.divide(img[:, :, c], weight,
                                 out=np.zeros_like(img[:, :, c]), where=mask)

    # Fill gaps
    fy, fx = np.where(~mask)
    if len(fy) > 0:
        print(f"  Interpolating {len(fy)}/{H*W} pixels...")
        for c in range(3):
            grid = griddata((np.where(mask)[0], np.where(mask)[1]),
                            img[mask, c], (fy, fx), method="nearest")
            img[fy, fx, c] = np.clip(grid, 0, None)

    return img


def normalize(img: np.ndarray) -> np.ndarray:
    """Per-channel histogram stretch + gray-world white balance."""
    out = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        lo = np.percentile(ch[ch > 0], 1.0) if (ch > 0).any() else 0
        hi = np.percentile(ch, 99.0)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-10), 0, 1) if hi > lo else ch
    # Gray-world
    means = np.array([out[:, :, c].mean() for c in range(3)])
    target = means.mean()
    scales = np.clip(np.where(means > 1e-8, target / means, 1.0), 0.3, 3.0)
    for c in range(3):
        out[:, :, c] = np.clip(out[:, :, c] * scales[c], 0, 1)
    return out


def auto_register(rgb_img: np.ndarray, gray_img: np.ndarray) -> tuple:
    """ORB feature matching to estimate registration offset."""
    try:
        import cv2
    except ImportError:
        return None

    spec_8u = (np.clip(rgb_img * 255, 0, 255)).astype(np.uint8)
    spec_gray = cv2.cvtColor(spec_8u, cv2.COLOR_RGB2GRAY)
    g = gray_img.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    gray_8u = np.clip((g - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)

    orb = cv2.ORB_create(nfeatures=5000)
    kp1, d1 = orb.detectAndCompute(spec_gray, None)
    kp2, d2 = orb.detectAndCompute(gray_8u, None)
    if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)[:100]

    if len(matches) < 4:
        return None

    shifts = np.array([[kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1],
                        kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]]
                       for m in matches])
    # Trim outliers
    y_s = np.sort(shifts[:, 0])
    x_s = np.sort(shifts[:, 1])
    n = len(y_s)
    trim = n // 5
    dy = int(np.mean(y_s[trim:-trim])) if n > 10 else int(np.median(y_s))
    dx = int(np.mean(x_s[trim:-trim])) if n > 10 else int(np.median(x_s))
    return (dy, dx)


def save_results(rgb, gray, output_dir, dx_val, dy_val=0):
    """Save all output images."""
    H, W = gray.shape
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    gd = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    g3 = np.stack([gd] * 3, axis=-1)
    has_data = rgb.sum(axis=2) > 0

    # ── Summary panel ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    ax.imshow(gd, cmap="gray")
    ax.set_title("(a) Grayscale (view2.tif)")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(rgb)
    ax.set_title("(b) Pseudo-Color RGB")
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(gd, cmap="gray")
    ov = np.zeros((H, W, 4))
    ov[has_data, :3] = rgb[has_data]
    ov[has_data, 3] = 0.85
    ax.imshow(ov)
    ax.set_title("(c) Scatter + Gray")
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(g3 * 0.4 + rgb * 0.6)
    ax.set_title("(d) Blend: 40% Gray + 60% Color")
    ax.axis("off")

    ax = axes[1, 1]
    colors = ["blue", "green", "red"]
    labels = [f"B ({WAVELENGTHS[B_BAND]}nm)", f"G ({WAVELENGTHS[G_BAND]}nm)",
              f"R ({WAVELENGTHS[R_BAND]}nm)"]
    for c in range(3):
        vals = rgb[has_data, c]
        if len(vals) > 0:
            ax.hist(vals, bins=80, alpha=0.5, color=colors[c], label=labels[c])
    ax.set_xlabel("Reflectance (normalized)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.set_title("(e) Channel Histograms")

    ax = axes[1, 2]
    ax.axis("off")
    info = (
        f"Mapping:\n"
        f"  R = {WAVELENGTHS[R_BAND]}nm (b{R_BAND})\n"
        f"  G = {WAVELENGTHS[G_BAND]}nm (b{G_BAND})\n"
        f"  B = {WAVELENGTHS[B_BAND]}nm (b{B_BAND})\n\n"
        f"Scene pts: {has_data.sum()}\n"
        f"Image: {H}x{W}\n"
        f"Registration: dx={dx_val}, dy={dy_val}\n"
        f"Histogram clip: 1%-99%\n"
        f"White balance: gray-world"
    )
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_color_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # ── Pure RGB ──
    fig2, ax2 = plt.subplots(figsize=(12, 12))
    ax2.imshow(rgb)
    ax2.set_title(f"Pseudo-Color ({WAVELENGTHS[B_BAND]}/{WAVELENGTHS[G_BAND]}/{WAVELENGTHS[R_BAND]}nm)",
                  fontsize=13)
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_color_rgb.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # ── Overlay ──
    fig3, ax3 = plt.subplots(figsize=(12, 12))
    ax3.imshow(gd, cmap="gray")
    ov2 = np.zeros((H, W, 4))
    ov2[:, :, :3] = rgb
    ov2[:, :, 3] = 0.7
    ax3.imshow(ov2)
    ax3.set_title("Pseudo-Color Overlay on Grayscale", fontsize=13)
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_color_overlay.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # ── Blended ──
    fig4, ax4 = plt.subplots(figsize=(12, 12))
    ax4.imshow(g3 * 0.35 + rgb * 0.65)
    ax4.set_title("Blended: 35% Gray + 65% Color", fontsize=13)
    ax4.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pseudo_color_blended.png"), dpi=200,
                bbox_inches="tight")
    plt.close()

    # ── NPZ data ──
    np.savez_compressed(
        os.path.join(output_dir, "pseudo_color.npz"),
        rgb=rgb, gray=gd,
        channels_nm=np.array([WAVELENGTHS[B_BAND], WAVELENGTHS[G_BAND], WAVELENGTHS[R_BAND]]),
    )

    print(f"  Output files in {output_dir}/:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    {f:35s} {size//1024:>6} KB")


def main():
    parser = argparse.ArgumentParser(description="Hyperspectral Pseudo-Color")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--hi-dir", default=None)
    parser.add_argument("--output", default="output/pseudo_color")
    parser.add_argument("--dx", type=int, default=REG_DX,
                        help=f"Registration x-offset (default: {REG_DX})")
    parser.add_argument("--dy", type=int, default=REG_DY,
                        help=f"Registration y-offset (default: {REG_DY})")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-register with ORB feature matching")
    parser.add_argument("--r-band", type=int, default=R_BAND)
    parser.add_argument("--g-band", type=int, default=G_BAND)
    parser.add_argument("--b-band", type=int, default=B_BAND)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(base, "..", "data", "1")
    hi_dir = args.hi_dir or os.path.join(base, "..", "HI")
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    print("=" * 55)
    print("  Pseudo-Color RGB Generation")
    print("=" * 55)
    print(f"  R = {WAVELENGTHS[args.r_band]}nm  G = {WAVELENGTHS[args.g_band]}nm  "
          f"B = {WAVELENGTHS[args.b_band]}nm")
    print(f"  Registration offset: dx={args.dx}, dy={args.dy}")

    # 1. Load
    print("\n[1/5] Loading data...")
    images, coords_dict, reflect = load_data(data_dir, hi_dir)

    # 2. Extract
    print("\n[2/5] Extracting RGB vectors...")
    rgb_pts, positions = extract_rgb(reflect, coords_dict)

    # 3. Auto-register
    dx_use = args.dx
    dy_use = args.dy
    if args.auto:
        print("\n[*] Auto-registration...")
        prelim = render_image(rgb_pts, positions, images["gray"].shape, 0)
        prelim = normalize(prelim)
        result = auto_register(prelim, images["gray"])
        if result:
            dy_auto, dx_auto = result
            print(f"  Auto: dy={dy_auto}, dx={dx_auto} (overriding --dx --dy)")
            dx_use = dx_auto
            dy_use = dy_auto
        else:
            print("  Auto-registration failed, using --dx --dy")

    # 4. Render
    print(f"\n[3/5] Rendering (dx={dx_use}, dy={dy_use})...")
    rgb = render_image(rgb_pts, positions, images["gray"].shape, dx_use, dy_use)

    # 5. Normalize
    print("\n[4/5] Normalizing + white balance...")
    rgb = normalize(rgb)

    # 6. Save
    print("\n[5/5] Saving...")
    save_results(rgb, images["gray"], output_dir, dx_use, dy_use)

    elapsed = time.time() - t0
    print(f"\n{'=' * 55}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
