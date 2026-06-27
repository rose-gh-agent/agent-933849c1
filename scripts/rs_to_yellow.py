import argparse
from pathlib import Path

import cv2
import numpy as np


def convert_teal_metal_to_gold(input_path: Path, output_path: Path) -> None:
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    active_mask = (s > 20) & (v > 20)
    teal_mask = active_mask & (h >= 75) & (h <= 125)

    if not np.any(teal_mask):
        raise RuntimeError("No teal/cyan metallic regions detected in the image.")

    new_h = h.copy()
    source_low = 75.0
    source_high = 125.0
    target_low = 18.0
    target_high = 38.0

    # Remap the detected teal/cyan band into a tighter gold range while
    # preserving tonal detail through the untouched saturation/value channels.
    remapped = np.interp(h[teal_mask], [source_low, source_high], [target_low, target_high])
    new_h[teal_mask] = remapped % 180.0
    hsv[:, :, 0] = new_h

    # A small saturation lift helps the remapped metal read as gold without
    # disturbing brightness or global contrast.
    new_s = s.copy()
    new_s[teal_mask] = np.clip(s[teal_mask] * 1.15, 0, 255)
    hsv[:, :, 1] = new_s

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shift teal/cyan metallic image elements toward golden yellow."
    )
    parser.add_argument("input", type=Path, help="Path to the source image")
    parser.add_argument("output", type=Path, help="Path to save the transformed image")
    args = parser.parse_args()

    convert_teal_metal_to_gold(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
