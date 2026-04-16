#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build anchor labels from COCO-style bbox annotations.

Output JSONL format (one line per image):
{
  "image_name": ".../xxx.jpg",
  "image_id": 123,
  "width": 1280,
  "height": 720,
  "task": "anchor",
  "categories": ["person", "car"],
  "prompt": "Detect person, car. Output semantic anchors.",
  "annotation": {
    "anchors": [
      {
        "phrase": "person",
        "coord_id": 2,
        "grid_size": 500,
        "x_grid": 311,
        "y_grid": 194,
        "x_bin_999": 623,
        "y_bin_999": 388,
        "scale_id": 4,
        "ratio_id": 3,
        "bbox_xyxy": [x0, y0, x1, y1],
        "bbox_xywh": [x, y, w, h]
      }
    ],
    "target_text": "<|object_ref_start|>person<|object_ref_end|><|anchor_start|><2><311><194><4><3><|anchor_end|>"
  }
}
"""

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _parse_csv_numbers(value: str, as_int: bool = False) -> List[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if as_int:
        return [int(x) for x in items]
    return [float(x) for x in items]


def _load_yaml(path: str) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from exc

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data


def _resolve_args_with_config(args: argparse.Namespace) -> Dict:
    cfg = {}
    if args.config:
        cfg = _load_yaml(args.config)

    merged = {
        "coco_json": cfg.get("coco_json"),
        "output_jsonl": cfg.get("output_jsonl"),
        "image_root": cfg.get("image_root", ""),
        "grids": cfg.get("grids", [100, 250, 500]),
        "selection_eta": cfg.get("selection_eta", 0.25),
        "scale_bins": cfg.get("scale_bins", [0.03, 0.06, 0.10, 0.16, 0.25, 0.40, 0.64, 1.0]),
        "ratio_bins": cfg.get("ratio_bins", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]),
        "include_crowd": cfg.get("include_crowd", False),
        "min_box_size": cfg.get("min_box_size", 2.0),
        "max_images": cfg.get("max_images", 0),
        "token_style": cfg.get("token_style", "plain"),
        "sort_instances": cfg.get("sort_instances", "xy"),
        "save_stats_json": cfg.get("save_stats_json", ""),
    }

    # CLI overrides config
    if args.coco_json is not None:
        merged["coco_json"] = args.coco_json
    if args.output_jsonl is not None:
        merged["output_jsonl"] = args.output_jsonl
    if args.image_root is not None:
        merged["image_root"] = args.image_root
    if args.grids is not None:
        merged["grids"] = _parse_csv_numbers(args.grids, as_int=True)
    if args.selection_eta is not None:
        merged["selection_eta"] = args.selection_eta
    if args.scale_bins is not None:
        merged["scale_bins"] = _parse_csv_numbers(args.scale_bins, as_int=False)
    if args.ratio_bins is not None:
        merged["ratio_bins"] = _parse_csv_numbers(args.ratio_bins, as_int=False)
    if args.include_crowd:
        merged["include_crowd"] = True
    if args.min_box_size is not None:
        merged["min_box_size"] = args.min_box_size
    if args.max_images is not None:
        merged["max_images"] = args.max_images
    if args.token_style is not None:
        merged["token_style"] = args.token_style
    if args.sort_instances is not None:
        merged["sort_instances"] = args.sort_instances
    if args.save_stats_json is not None:
        merged["save_stats_json"] = args.save_stats_json

    if not merged["coco_json"]:
        raise ValueError("coco_json is required (via --coco_json or --config)")
    if not merged["output_jsonl"]:
        raise ValueError("output_jsonl is required (via --output_jsonl or --config)")

    return merged


def _clip_xywh_to_image(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x0 = max(0.0, min(float(img_w), x))
    y0 = max(0.0, min(float(img_h), y))
    x1 = max(0.0, min(float(img_w), x + w))
    y1 = max(0.0, min(float(img_h), y + h))
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)


def _quantize_by_edges(value: float, edges: List[float]) -> int:
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return max(0, len(edges) - 1)


def _select_grid(
    w_ratio: float,
    h_ratio: float,
    grid_sizes: List[int],
    eta: float,
) -> Tuple[int, int]:
    """
    Select the coarsest grid that satisfies quantization error constraint:
      1 / (2G) <= eta * min(w_ratio, h_ratio)
    """
    short_ratio = max(min(w_ratio, h_ratio), 1e-6)
    required = 1.0 / (2.0 * max(eta, 1e-6) * short_ratio)
    for gid, g in enumerate(grid_sizes):
        if g >= required:
            return gid, g
    return len(grid_sizes) - 1, grid_sizes[-1]


def _make_anchor_token(anchor: Dict, token_style: str) -> str:
    if token_style == "prefixed":
        return (
            f"<g{anchor['coord_id']}>"
            f"<x_{anchor['x_grid']}>"
            f"<y_{anchor['y_grid']}>"
            f"<s{anchor['scale_id']}>"
            f"<r{anchor['ratio_id']}>"
        )
    return (
        f"<{anchor['coord_id']}>"
        f"<{anchor['x_grid']}>"
        f"<{anchor['y_grid']}>"
        f"<{anchor['scale_id']}>"
        f"<{anchor['ratio_id']}>"
    )


def _sort_anchors(anchors: List[Dict], mode: str) -> List[Dict]:
    if mode == "yx":
        return sorted(
            anchors,
            key=lambda a: (a["phrase"], a["y_bin_999"], a["x_bin_999"]),
        )
    if mode == "area_desc":
        return sorted(
            anchors,
            key=lambda a: (a["phrase"], -a["bbox_xywh"][2] * a["bbox_xywh"][3]),
        )
    # default: "xy"
    return sorted(
        anchors,
        key=lambda a: (a["phrase"], a["x_bin_999"], a["y_bin_999"]),
    )


def _build_target_text(anchors: List[Dict], token_style: str) -> str:
    grouped = defaultdict(list)
    for a in anchors:
        grouped[a["phrase"]].append(a)

    chunks = []
    for phrase in sorted(grouped.keys()):
        tokens = [_make_anchor_token(x, token_style=token_style) for x in grouped[phrase]]
        chunk = (
            f"<|object_ref_start|>{phrase}<|object_ref_end|>"
            f"<|anc_s|>{','.join(tokens)}<|anc_e|>"
        )
        chunks.append(chunk)
    return ", ".join(chunks)


def _iter_image_ids(images: Dict[int, Dict], max_images: int) -> Iterable[int]:
    image_ids = sorted(images.keys())
    if max_images and max_images > 0:
        image_ids = image_ids[:max_images]
    return image_ids


def build_anchor_labels(cfg: Dict) -> Dict:
    with open(cfg["coco_json"], "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    cat_id_to_name = {int(x["id"]): str(x["name"]) for x in categories}
    image_id_to_info = {int(x["id"]): x for x in images}

    ann_by_image = defaultdict(list)
    for ann in annotations:
        if not cfg["include_crowd"] and int(ann.get("iscrowd", 0)) == 1:
            continue
        image_id = int(ann["image_id"])
        if image_id not in image_id_to_info:
            continue
        ann_by_image[image_id].append(ann)

    os.makedirs(os.path.dirname(cfg["output_jsonl"]) or ".", exist_ok=True)
    fout = open(cfg["output_jsonl"], "w", encoding="utf-8")

    total_images = 0
    total_instances = 0
    grid_hist = defaultdict(int)
    scale_hist = defaultdict(int)
    ratio_hist = defaultdict(int)

    for image_id in _iter_image_ids(image_id_to_info, cfg["max_images"]):
        info = image_id_to_info[image_id]
        img_w = int(info["width"])
        img_h = int(info["height"])
        file_name = info["file_name"]

        image_anns = ann_by_image.get(image_id, [])
        anchors: List[Dict] = []

        for ann in image_anns:
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            x, y, bw, bh = [float(v) for v in bbox]
            x, y, bw, bh = _clip_xywh_to_image(x, y, bw, bh, img_w, img_h)
            if bw < cfg["min_box_size"] or bh < cfg["min_box_size"]:
                continue

            cat_name = cat_id_to_name.get(int(ann["category_id"]), str(ann["category_id"]))

            w_ratio = bw / float(img_w)
            h_ratio = bh / float(img_h)
            coord_id, grid_size = _select_grid(
                w_ratio=w_ratio,
                h_ratio=h_ratio,
                grid_sizes=[int(x) for x in cfg["grids"]],
                eta=float(cfg["selection_eta"]),
            )

            cx = (x + bw / 2.0) / float(img_w)
            cy = (y + bh / 2.0) / float(img_h)
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))

            x_grid = int(round(cx * (grid_size - 1)))
            y_grid = int(round(cy * (grid_size - 1)))
            x_bin_999 = int(round(cx * 999.0))
            y_bin_999 = int(round(cy * 999.0))

            # Scale token from sqrt(area_ratio): robust for very long/flat boxes.
            area_ratio = max((bw * bh) / float(img_w * img_h), 1e-12)
            scale_value = math.sqrt(area_ratio)
            scale_id = _quantize_by_edges(scale_value, [float(x) for x in cfg["scale_bins"]])

            # Ratio token from w/h
            ratio_value = max(bw / max(bh, 1e-6), 1e-6)
            ratio_id = _quantize_by_edges(ratio_value, [float(x) for x in cfg["ratio_bins"]])

            x0 = x
            y0 = y
            x1 = x + bw
            y1 = y + bh

            anchor = {
                "phrase": cat_name,
                "coord_id": int(coord_id),
                "grid_size": int(grid_size),
                "x_grid": int(x_grid),
                "y_grid": int(y_grid),
                "x_bin_999": int(x_bin_999),
                "y_bin_999": int(y_bin_999),
                "scale_id": int(scale_id),
                "ratio_id": int(ratio_id),
                "bbox_xyxy": [x0, y0, x1, y1],
                "bbox_xywh": [x, y, bw, bh],
                "ann_id": int(ann.get("id", -1)),
            }
            anchors.append(anchor)

            grid_hist[int(grid_size)] += 1
            scale_hist[int(scale_id)] += 1
            ratio_hist[int(ratio_id)] += 1

        if not anchors:
            continue

        anchors = _sort_anchors(anchors, mode=cfg["sort_instances"])
        categories_in_image = sorted(list({x["phrase"] for x in anchors}))
        target_text = _build_target_text(anchors, token_style=cfg["token_style"])

        image_name = (
            os.path.join(cfg["image_root"], file_name)
            if cfg["image_root"]
            else file_name
        )

        sample = {
            "image_name": image_name,
            "image_id": image_id,
            "width": img_w,
            "height": img_h,
            "task": "anchor",
            "categories": categories_in_image,
            "prompt": f"Detect {', '.join(categories_in_image)}. Output semantic anchors.",
            "annotation": {
                "anchors": anchors,
                "target_text": target_text,
            },
        }
        fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

        total_images += 1
        total_instances += len(anchors)

    fout.close()

    stats = {
        "total_images": total_images,
        "total_instances": total_instances,
        "avg_instances_per_image": (
            float(total_instances) / max(total_images, 1)
        ),
        "grid_hist": dict(sorted(grid_hist.items(), key=lambda x: x[0])),
        "scale_hist": dict(sorted(scale_hist.items(), key=lambda x: x[0])),
        "ratio_hist": dict(sorted(ratio_hist.items(), key=lambda x: x[0])),
        "config": cfg,
    }

    if cfg["save_stats_json"]:
        os.makedirs(os.path.dirname(cfg["save_stats_json"]) or ".", exist_ok=True)
        with open(cfg["save_stats_json"], "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build anchor labels (g,x,y,s,r) from COCO bbox annotations."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")

    parser.add_argument("--coco_json", type=str, default=None, help="Path to COCO annotation json")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Output jsonl path")
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional image root to prepend to file_name",
    )
    parser.add_argument(
        "--grids",
        type=str,
        default=None,
        help='Grid sizes, e.g. "100,250,500"',
    )
    parser.add_argument(
        "--selection_eta",
        type=float,
        default=None,
        help="Quantization tolerance coefficient for grid selection",
    )
    parser.add_argument(
        "--scale_bins",
        type=str,
        default=None,
        help='Scale-bin edges for sqrt(area_ratio), e.g. "0.03,0.06,0.1,0.16,0.25,0.4,0.64,1.0"',
    )
    parser.add_argument(
        "--ratio_bins",
        type=str,
        default=None,
        help='Ratio-bin edges for w/h, e.g. "0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0"',
    )
    parser.add_argument(
        "--include_crowd",
        action="store_true",
        help="Include iscrowd=1 annotations",
    )
    parser.add_argument(
        "--min_box_size",
        type=float,
        default=None,
        help="Filter tiny boxes where width or height < min_box_size (pixels)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Use only first N images by image_id order; 0 means all",
    )
    parser.add_argument(
        "--token_style",
        type=str,
        default=None,
        choices=["plain", "prefixed"],
        help="Target token style in target_text",
    )
    parser.add_argument(
        "--sort_instances",
        type=str,
        default=None,
        choices=["xy", "yx", "area_desc"],
        help="Sort anchors within each phrase",
    )
    parser.add_argument(
        "--save_stats_json",
        type=str,
        default=None,
        help="Optional path to save build stats",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = _resolve_args_with_config(args)
    stats = build_anchor_labels(cfg)

    print("Anchor label build done.")
    print(f"  total_images: {stats['total_images']}")
    print(f"  total_instances: {stats['total_instances']}")
    print(f"  avg_instances_per_image: {stats['avg_instances_per_image']:.3f}")
    print(f"  output_jsonl: {cfg['output_jsonl']}")
    if cfg["save_stats_json"]:
        print(f"  stats_json: {cfg['save_stats_json']}")


if __name__ == "__main__":
    main()
