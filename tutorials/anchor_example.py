#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal semantic-anchor example using Rex Omni.

Note:
The released Rex-Omni checkpoint is not anchor-tuned by default.
This script validates the new task interface and parser path first.
"""

import argparse
import json

from PIL import Image

from rex_omni import RexOmniWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Rex-Omni anchor task example")
    parser.add_argument(
        "--model_path",
        default="IDEA-Research/Rex-Omni",
        help="Model path or HuggingFace repo id",
    )
    parser.add_argument(
        "--image_path",
        default="tutorials/detection_example/test_images/cafe.jpg",
        help="Input image path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rex_model = RexOmniWrapper(
        model_path=args.model_path,
        backend="transformers",
        max_tokens=512,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.05,
    )

    image = Image.open(args.image_path).convert("RGB")
    categories = ["man", "woman", "cup", "laptop"]

    results = rex_model.inference(
        images=image,
        task="anchor",
        categories=categories,
    )

    result = results[0]
    print("=== Prompt ===")
    print(result["prompt"])
    print("\n=== Raw Output ===")
    print(result["raw_output"])
    print("\n=== Parsed Anchors ===")
    print(json.dumps(result["extracted_predictions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
