import argparse
import os

import numpy as np
from PIL import Image


DEFAULT_SIZE = 512
DEFAULT_OUTPUT = os.path.join("experiments", "stress_test_density.png")


def build_stress_density(size):
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(lin, lin, indexing="xy")
    density = 0.2 * np.exp(-20.0 * (x * x + y * y))
    density += 0.2 * (np.sin(np.pi * x) ** 2) * (np.sin(np.pi * y) ** 2)
    density /= max(float(density.max()), 1e-8)
    return density.astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic stress-test density image.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    density = build_stress_density(args.size)
    image_u8 = np.clip(255.0 - density * 255.0, 0.0, 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(image_u8, mode="L").save(args.output)
    print(f"Saved stress map to: {args.output}")


if __name__ == "__main__":
    main()
