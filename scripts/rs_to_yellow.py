import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def convert_to_yellow(input_path: Path, output_path: Path) -> None:
    img = Image.open(input_path).convert("RGB")
    data = np.array(img, dtype=np.float32) / 255.0

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.zeros_like(maxc)
    np.divide(maxc - minc, maxc, out=s, where=maxc != 0)

    delta = maxc - minc
    h = np.zeros_like(maxc)

    mask = delta != 0
    rm = mask & (maxc == r)
    gm = mask & (maxc == g)
    bm = mask & (maxc == b)

    h[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / delta[gm] + 2
    h[bm] = (r[bm] - g[bm]) / delta[bm] + 4
    h = h / 6.0

    non_bg = (s > 0.05) & (v > 0.05)
    h[non_bg] = 0.167

    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    result = np.zeros_like(data)
    variants = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for idx, (ri, gi, bi) in enumerate(variants):
        mask_i = i == idx
        result[:, :, 0][mask_i] = ri[mask_i]
        result[:, :, 1][mask_i] = gi[mask_i]
        result[:, :, 2][mask_i] = bi[mask_i]

    result[~non_bg] = data[~non_bg]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img = Image.fromarray((result * 255).astype(np.uint8), "RGB")
    result_img.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shift non-background hues in an image toward yellow using Pillow and numpy."
    )
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to save the transformed image")
    args = parser.parse_args()

    convert_to_yellow(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
