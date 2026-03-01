"""Utility for running person detection + simple risk scoring on an image.

This is intended for quick experimentation outside of the main pipeline.  It
leverages the same ``EdgeDetector`` interface that the core pipeline uses so
that you can easily switch the model configuration (ultralytics vs custom
YOLOv26, enable augmentation, etc.) through the YAML config file.

Usage example::

    python scripts/detect_and_risk.py \
        --image path/to/frame.jpg \
        --model weights.pt \
        --output out.jpg

If ``--annotations`` and ``--save-predictions`` are provided the script will
also run COCO-style evaluation (requires ``pycocotools``) and print a simple
person AP score which can help track accuracy when you are tuning parameters
or retraining models.  The risk scoring logic is toy‑level: any detected
person whose bounding box overlaps >30% with another person is marked
"high" risk.
"""

import argparse
import sys
from pathlib import Path

# ensure project root is on path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from typing import Optional

from urbanai.edge_ai.edge_inference_runner.detector import EdgeDetector
from urbanai.logging import logger


def risk_from_detections(detections: list) -> str:
    """Very simple spatial-overlap risk scoring.

    If any two person boxes overlap by more than 30% of the smaller area we
    call the frame "high risk".  Otherwise it is "low risk".  This is just a
    placeholder; the pipeline's ``RiskEngine`` contains more sophisticated
    logic.
    """
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            box1 = np.array(detections[i]["bbox"])
            box2 = np.array(detections[j]["bbox"])
            # compute intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                continue
            inter = (x2 - x1) * (y2 - y1)
            area_smaller = min(
                (box1[2] - box1[0]) * (box1[3] - box1[1]),
                (box2[2] - box2[0]) * (box2[3] - box2[1]),
            )
            if inter / (area_smaller + 1e-9) > 0.3:
                return "high"
    return "low"


def main():
    parser = argparse.ArgumentParser(description="Run person detector + risk")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", required=False, help="Model weights path")
    parser.add_argument("--output", required=False, help="Output image with boxes")
    parser.add_argument("--annotations", required=False, help="COCO json for evaluation")
    parser.add_argument("--save-predictions", required=False, help="Path to dump predictions (COCO format)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Could not load image {args.image}")
        return

    detector = EdgeDetector(model_path=args.model) if args.model else EdgeDetector()
    res = detector.detect(img)
    detections = [d.to_dict() for d in res.get("detections", [])]
    risk = risk_from_detections(detections)
    logger.info(f"Detected {len(detections)} objects, risk level: {risk}")

    # draw boxes
    vis = img.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{d['class_name']}:{d['confidence']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    if args.output:
        cv2.imwrite(args.output, vis)
        logger.info(f"Wrote output to {args.output}")

    # optional evaluation
    if args.annotations and args.save_predictions:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            logger.error("pycocotools not installed, cannot evaluate")
            return
        # build coco-style results
        coco_gt = COCO(args.annotations)
        preds = []
        for d in detections:
            if d["class_name"] != "person":
                continue
            x1, y1, x2, y2 = d["bbox"]
            w = x2 - x1
            h = y2 - y1
            preds.append({
                "image_id": int(Path(args.image).stem),
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "score": d["confidence"]
            })
        import json
        with open(args.save_predictions, "w") as f:
            json.dump(preds, f)
        coco_dt = coco_gt.loadRes(args.save_predictions)
        evaler = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaler.params.imgIds = [int(Path(args.image).stem)]
        evaler.evaluate()
        evaler.accumulate()
        evaler.summarize()


if __name__ == "__main__":
    main()
