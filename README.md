Yes — that’s actually a cleaner approach for YOLOv7, because LoadImages() is designed for paths, not NumPy arrays.
So the new flow becomes:
✅ Detect views from drawing → crop view images → save them as files
✅ Run dimension detector using view image paths
✅ OCR each DIM crop → JSON output
This avoids the ndarray issue completely.
✅ Updated Pipeline Strategy (Path-based)
Copy code
Text
drawing.jpg
   ↓ (View model)
detect View boxes
   ↓
crop each view
   ↓
save as:
runs/<job_id>/views/view_00.jpg
runs/<job_id>/views/view_01.jpg
   ↓
(DIM model runs on those saved view image paths)
   ↓
DIM crops + OCR → output.json
✅ Updated runner.py (Full working version)
This version:
Saves views to disk
Passes view file paths into YOLO DIM detector
Produces structured JSON
pipeline/runner.py
Copy code
Python
import os
import json
import cv2
from pathlib import Path

from detectors.yolo_detector import YOLODetector
from processing.crop import crop
from ocr.tess_ocr import recognize


class DrawingPipeline:
    def __init__(
        self,
        view_weights="weights/0519_view_best.pt",
        dim_weights="weights/0324_dim_and_tol_best.pt",
        device="0",
        img_size=640,
    ):
        self.view_detector = YOLODetector(view_weights, img_size=img_size, device=device)
        self.dim_detector = YOLODetector(dim_weights, img_size=img_size, device=device)

    def run(self, drawing_path, out_dir="runs/pipeline_run"):
        drawing_path = str(drawing_path)
        out_dir = Path(out_dir)

        # Output folders
        views_dir = out_dir / "views"
        dim_dir = out_dir / "dims"
        views_dir.mkdir(parents=True, exist_ok=True)
        dim_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Stage 1: Detect Views (on original drawing)
        # -----------------------------
        view_dets = self.view_detector.detect(drawing_path)

        # Read original drawing once
        original = cv2.imread(drawing_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {drawing_path}")

        # Keep only View detections
        view_dets = [d for d in view_dets if d["label"] == "View"]
        view_dets = sorted(view_dets, key=lambda d: d["xyxy"][0])  # left-to-right

        results = {
            "drawing_path": drawing_path,
            "views": []
        }

        # -----------------------------
        # Stage 2: Save each view image to disk
        # -----------------------------
        for i, v in enumerate(view_dets):
            view_crop = crop(original, v["xyxy"])
            view_path = views_dir / f"view_{i:02d}.jpg"
            cv2.imwrite(str(view_path), view_crop)

            view_record = {
                "view_id": i,
                "view_path": str(view_path),
                "bbox": v["xyxy"],
                "dimensions": []
            }

            # -----------------------------
            # Stage 3: Detect DIM on saved view path
            # -----------------------------
            dim_dets = self.dim_detector.detect(str(view_path))
            dim_dets = [d for d in dim_dets if d["label"] == "DIM"]
            dim_dets = sorted(dim_dets, key=lambda d: d["xyxy"][0])

            # Load the view image once for cropping DIM
            view_img = cv2.imread(str(view_path))

            for j, d in enumerate(dim_dets):
                dim_crop = crop(view_img, d["xyxy"])
                dim_crop_path = dim_dir / f"view_{i:02d}_dim_{j:02d}.jpg"
                cv2.imwrite(str(dim_crop_path), dim_crop)

                # OCR
                text = recognize(dim_crop)

                view_record["dimensions"].append({
                    "dim_id": j,
                    "bbox": d["xyxy"],
                    "confidence": d["confidence"],
                    "crop_path": str(dim_crop_path),
                    "text": text
                })

            results["views"].append(view_record)

        # Save JSON
        json_path = out_dir / "output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Done. Output saved at: {json_path}")
        return results
✅ Minimal main.py
Copy code
Python
from runner import DrawingPipeline

pipeline = DrawingPipeline(
    view_weights="weights/0519_view_best.pt",
    dim_weights="weights/0324_dim_and_tol_best.pt",
    device="0"
)

pipeline.run("input/example_dwg_page-0001.jpg", out_dir="runs/test1")
✅ What gets saved to disk
Example output folder:
Copy code
Text
runs/test1/
│
├── views/
│   ├── view_00.jpg
│   ├── view_01.jpg
│   └── view_02.jpg
│
├── dims/
│   ├── view_00_dim_00.jpg
│   ├── view_00_dim_01.jpg
│   └── view_01_dim_00.jpg
│
└── output.json
✅ Why this is the best solution for you
Advantages
✅ No ndarray → LoadImages() crash
✅ Keeps YOLOv7 pipeline unchanged
✅ Debuggable (you can visually inspect saved crops)
✅ Easy to add later:
FCF stage
tolerance parsing
CAD export
Tradeoff
Slightly slower due to disk write/read, but for engineering drawings this is totally fine.
If you want, I can also add:
saving annotated images (view boxes + dim boxes)
confidence filtering
auto job_id timestamp folder
support for PDF → image pages