from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.encoders import CLIPEncoder, DINOv2Encoder, SigLIPEncoder


ENCODERS = {
    "clip": CLIPEncoder,
    "siglip": SigLIPEncoder,
    "dinov2": DINOv2Encoder,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a normalized image embedding.")
    parser.add_argument("--encoder", choices=ENCODERS, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, for example: cpu, cuda, cuda:0, or mps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    encoder = ENCODERS[args.encoder](device=args.device)
    embedding = encoder.encode_image(args.image)

    print(f"encoder: {args.encoder}")
    print(f"device: {encoder.device}")
    print(f"embedding shape: {tuple(embedding.shape)}")
    print(f"embedding norm: {embedding.norm().item():.4f}")


if __name__ == "__main__":
    main()
