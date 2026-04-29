import argparse
from pathlib import Path

import cv2
import numpy as np


def build_quadratic_density_gradient(width, height, exponent):
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    density_1d = x ** np.float32(exponent)
    density_2d = np.tile(density_1d, (height, 1))
    return (1.0 - density_2d).astype(np.float32, copy=False)


def build_stress_density(size):
    lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(lin, lin, indexing="xy")
    density = 0.2 * np.exp(-20.0 * (x * x + y * y))
    density += 0.2 * (np.sin(np.pi * x) ** 2) * (np.sin(np.pi * y) ** 2)
    density /= max(float(density.max()), 1e-8)
    return density.astype(np.float32, copy=False)


def quarter_capacities(exponent, quarters=4):
    edges = np.linspace(0.0, 1.0, quarters + 1, dtype=np.float64)
    return [((end ** (exponent + 1.0)) - (start ** (exponent + 1.0))) * 100.0 for start, end in zip(edges[:-1], edges[1:])]


def run_gradient(args):
    image_pixels = build_quadratic_density_gradient(args.width, args.height, args.k)
    image_u8 = np.clip(image_pixels * 255.0, 0.0, 255.0).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_u8):
        raise RuntimeError(f"Failed to write PNG image to {output_path}")

    capacities = quarter_capacities(args.k)
    print("Verification of Quarter Capacities:")
    print(f"Q1 (0-25%):   {capacities[0]:.2f}%")
    print(f"Q2 (25-50%):  {capacities[1]:.2f}%")
    print(f"Q3 (50-75%):  {capacities[2]:.2f}%")
    print(f"Q4 (75-100%): {capacities[3]:.2f}%")
    print(f"Saved gradient to: {output_path}")


def run_stress(args):
    density = build_stress_density(args.size)
    image_u8 = np.clip(255.0 - density * 255.0, 0.0, 255.0).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_u8):
        raise RuntimeError(f"Failed to write stress image to {output_path}")

    print(f"Saved stress map to: {output_path}")


###########
# Parsers #
###########

def build_gradient_parser():
    DEFAULT_WIDTH = 1000
    DEFAULT_HEIGHT = 250
    DEFAULT_EXPONENT = 2.0
    DEFAULT_OUTPUT = Path("experiments") / "quadratic_density_gradient.png"

    parser = argparse.ArgumentParser(description="Generate a quadratic density gradient image.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--k", type=float, default=DEFAULT_EXPONENT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def build_stress_parser():
    DEFAULT_SIZE = 512
    DEFAULT_OUTPUT = Path("experiments") / "stress_test_density.png"

    parser = argparse.ArgumentParser(description="Generate the synthetic stress-test density image.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate experiment density images.")
    parser.add_argument("option", nargs="?", default="gradient", help="Choose which generator to run.")
    args, remaining = parser.parse_known_args(argv)

    if args.option == "gradient":
        gradient_parser = build_gradient_parser()
        gradient_args = gradient_parser.parse_args(remaining)
        run_gradient(gradient_args)
    elif args.option == "stress":
        stress_parser = build_stress_parser()
        stress_args = stress_parser.parse_args(remaining)
        run_stress(stress_args)
    else:
        raise ValueError(f"Unknown option: {args.option}")


if __name__ == "__main__":
    main()