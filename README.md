Absolutely — Option A is the best choice if you want to replicate the paper + repo behavior without dealing with Flask/Celery/DB.
Below is a clean CLI pipeline that reproduces the repo’s command-line workflow:
✅ Uses 3 YOLO weights in the correct order
✅ Saves every View crop
✅ Saves DIM / FCF / Datum crops
✅ Runs DIM-type detection inside DIM
✅ Runs OCR on parts (with rotation voting)
✅ Writes recognize.json in a clean format
✅ Runs on Windows (with Python 3.10 recommended)
0) IMPORTANT: Use Python 3.10 (strongly recommended)
YOLOv7 is not stable on Python 3.12.
Install Python 3.10 + create venv:
Copy code
Bash
py -3.10 -m venv venv
venv\Scripts\activate
pip install -r requirements_cli.txt
1) Folder layout (must match YOLOv7 imports)
Your project root must look like:
Copy code
Text
project_root/
│
├── models/                 # YOLOv7 models/
├── utils/                  # YOLOv7 utils/
├── weights/
│   ├── 0519_view_best.pt
│   ├── 1013_DIMandFCF_best.pt
│   └── 0324_dim_and_tol_best.pt
│
├── input/
│   └── drawing.jpg
│
├── cli_pipeline/
│   ├── __init__.py
│   ├── yolo_wrap.py
│   ├── ocr.py
│   ├── pipeline_cli.py
│   └── run_cli.py
│
└── requirements_cli.txt
2) requirements_cli.txt
Copy code
Txt
numpy
opencv-python
pytesseract
Pillow
PyYAML
tqdm
matplotlib
scipy
torch
torchvision
⚠️ Install Tesseract separately on Windows:
https://github.com/UB-Mannheim/tesseract/wiki
3) YOLO wrapper (path-only, stable)
cli_pipeline/yolo_wrap.py
Copy code
Python
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages


class YOLOv7PathDetector:
    def __init__(self, weights, img_size=640, device="0"):
        self.device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.img_size = img_size
        self.names = self.model.names

    def detect(self, image_path, conf=0.25, iou=0.45):
        dataset = LoadImages(image_path, img_size=self.img_size)
        dets = []

        for path, img, im0, _ in dataset:
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf, iou)

            for det in pred:
                if det is None or len(det) == 0:
                    continue

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, c, cls in det:
                    dets.append({
                        "xyxy": [int(x) for x in xyxy],
                        "confidence": float(c),
                        "label": self.names[int(cls)]
                    })

        return dets
4) OCR with rotation voting (fixes rotated DIM text)
cli_pipeline/ocr.py
Copy code
Python
import cv2
import numpy as np
import pytesseract
import re


def pad_to_square(image, scale=1.4):
    h, w = image.shape[:2]
    side = int(max(h, w) * scale)
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y+h, x:x+w] = image
    return canvas


def preprocess(img):
    img = pad_to_square(img)
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def score_text(text: str) -> int:
    if not text:
        return 0
    digit_count = len(re.findall(r"\d", text))
    sym_count = len(re.findall(r"[Ø±Rr]", text))
    return 5 * digit_count + 2 * sym_count + len(text)


def ocr_once(img, lang="eng_math"):
    data = pytesseract.image_to_data(
        img,
        config=f"--oem 3 --psm 6 -l {lang}",
        output_type=pytesseract.Output.DICT
    )
    tokens = []
    for t, conf in zip(data["text"], data["conf"]):
        t = t.strip()
        if t and conf != "-1" and int(float(conf)) > 35:
            tokens.append(t)
    return " ".join(tokens)


def ocr_best(img, lang="eng_math"):
    img = preprocess(img)

    candidates = []
    for angle in [0, 90, 180, 270]:
        if angle == 0:
            rot = img
        elif angle == 90:
            rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        text = ocr_once(rot, lang=lang)
        candidates.append((score_text(text), angle, text))

    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][2]
5) Full CLI pipeline (replicates repo behavior)
cli_pipeline/pipeline_cli.py
Copy code
Python
import cv2
import json
from pathlib import Path

from cli_pipeline.ocr import ocr_best


def crop(img, xyxy):
    x1, y1, x2, y2 = xyxy
    return img[y1:y2, x1:x2]


def save_img(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


class DrawingCLIPipeline:
    """
    Option A pipeline:
    drawing -> views -> groups -> dim parts -> OCR -> recognize.json
    """

    def __init__(self, view_detector, group_detector, type_detector):
        self.view_detector = view_detector
        self.group_detector = group_detector
        self.type_detector = type_detector

    def run(self, drawing_path, out_dir):
        drawing_path = Path(drawing_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        views_dir = out_dir / "views"
        groups_dir = out_dir / "groups"
        parts_dir = out_dir / "dim_parts"

        img = cv2.imread(str(drawing_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read {drawing_path}")

        recognize = {
            "drawing": str(drawing_path),
            "views": []
        }

        # -----------------------------
        # Stage 1: Views
        # -----------------------------
        view_dets = self.view_detector.detect(str(drawing_path))
        view_dets = [d for d in view_dets if d["label"].lower() == "view"]
        view_dets = sorted(view_dets, key=lambda d: d["xyxy"][0])

        for vi, v in enumerate(view_dets):
            view_crop = crop(img, v["xyxy"])
            view_path = views_dir / f"view_{vi:02d}.jpg"
            save_img(view_path, view_crop)

            view_rec = {
                "view_id": vi,
                "bbox": v["xyxy"],
                "path": str(view_path),
                "groups": []
            }

            # -----------------------------
            # Stage 2: Groups inside view
            # -----------------------------
            group_dets = self.group_detector.detect(str(view_path))
            group_dets = sorted(group_dets, key=lambda d: d["xyxy"][0])

            view_group_dir = groups_dir / f"view_{vi:02d}"
            view_group_dir.mkdir(parents=True, exist_ok=True)

            for gi, g in enumerate(group_dets):
                label = g["label"].upper()
                g_crop = crop(view_crop, g["xyxy"])
                group_path = view_group_dir / f"{label}_{gi:02d}.jpg"
                save_img(group_path, g_crop)

                group_rec = {
                    "group_id": gi,
                    "label": label,
                    "bbox": g["xyxy"],
                    "path": str(group_path)
                }

                # -----------------------------
                # Stage 3: DIM parts + OCR
                # -----------------------------
                if label == "DIM":
                    dim_part_dir = parts_dir / f"view_{vi:02d}_DIM_{gi:02d}"
                    dim_part_dir.mkdir(parents=True, exist_ok=True)

                    # Detect parts inside DIM crop
                    part_dets = self.type_detector.detect(str(group_path), conf=0.15, iou=0.45)
                    part_dets = sorted(part_dets, key=lambda d: d["xyxy"][0])

                    parts = []
                    for pi, p in enumerate(part_dets):
                        p_crop = crop(g_crop, p["xyxy"])
                        part_path = dim_part_dir / f"part_{pi:02d}_{p['label']}.jpg"
                        save_img(part_path, p_crop)

                        text = ocr_best(p_crop)

                        parts.append({
                            "part_id": pi,
                            "label": p["label"],
                            "bbox": p["xyxy"],
                            "confidence": p["confidence"],
                            "path": str(part_path),
                            "ocr": text
                        })

                    # fallback if no parts detected
                    if len(parts) == 0:
                        fallback_path = dim_part_dir / "fallback_full_dim.jpg"
                        save_img(fallback_path, g_crop)
                        parts.append({
                            "part_id": 0,
                            "label": "FULL_DIM_FALLBACK",
                            "bbox": [0, 0, g_crop.shape[1], g_crop.shape[0]],
                            "confidence": 0.0,
                            "path": str(fallback_path),
                            "ocr": ocr_best(g_crop)
                        })

                    group_rec["parts"] = parts

                view_rec["groups"].append(group_rec)

            recognize["views"].append(view_rec)

        # Save recognize.json
        json_path = out_dir / "recognize.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(recognize, f, indent=2)

        print(f"✅ Done. Saved: {json_path}")
        return recognize
6) CLI entrypoint
cli_pipeline/run_cli.py
Copy code
Python
import argparse
from cli_pipeline.yolo_wrap import YOLOv7PathDetector
from cli_pipeline.pipeline_cli import DrawingCLIPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Input drawing image path")
    ap.add_argument("--out", default="runs_cli", help="Output folder")
    ap.add_argument("--view-weights", required=True)
    ap.add_argument("--group-weights", required=True)
    ap.add_argument("--type-weights", required=True)
    ap.add_argument("--device", default="0", help="0,1,2... or cpu")
    args = ap.parse_args()

    view_detector = YOLOv7PathDetector(args.view_weights, device=args.device)
    group_detector = YOLOv7PathDetector(args.group_weights, device=args.device)
    type_detector = YOLOv7PathDetector(args.type_weights, device=args.device)

    pipe = DrawingCLIPipeline(view_detector, group_detector, type_detector)
    pipe.run(args.img, args.out)


if __name__ == "__main__":
    main()
✅ How to run (Option A)
Copy code
Bash
python -m cli_pipeline.run_cli ^
  --img input/drawing.jpg ^
  --out runs_cli/job1 ^
  --view-weights weights/0519_view_best.pt ^
  --group-weights weights/1013_DIMandFCF_best.pt ^
  --type-weights weights/0324_dim_and_tol_best.pt ^
  --device 0
✅ What you will get (same style as repo)
runs_cli/job1/views/view_00.jpg
runs_cli/job1/groups/view_00/DIM_00.jpg
runs_cli/job1/dim_parts/view_00_DIM_00/part_00_xxx.jpg
runs_cli/job1/recognize.json
⚠️ One limitation (honest)
This pipeline replicates the repo at the CLI output level, but the repo also has special FCF and Datum extraction scripts which are more advanced than simple OCR.
If you want perfect paper-level FCF output, I can integrate:
detect_fcf.py cell splitting logic
detect_datum.py contour cleanup logic
into this CLI runner too (it will be “full paper replication”).
Just say: “Add FCF and Datum extractors too” and I’ll extend this Option A pipeline to match them.Yes — I can add FCF + Datum extractors and make the CLI pipeline match the repo/paper much more closely ✅
⚠️ One issue first:
Some of the files you uploaded earlier have expired from my workspace, so I can’t directly read detect_fcf.py / detect_datum.py right now to copy their exact implementation.
✅ Please re-upload these two files (just these):
detect_fcf.py
detect_datum.py
(Optionally also re-upload Tess_OCR (1).py if you want the exact same OCR preprocessing they used.)
Meanwhile: I’ll give you the final architecture + full code template
So the moment you upload those two files, I’ll plug their exact logic in.
✅ Updated Option A Pipeline (with DIM + FCF + Datum)
Final flow:
Copy code
Text
Drawing
  ↓ (view model)
Views
  ↓ (group model)
DIM / FCF / DATUM crops
  ↓
DIM  → (type model) → parts → OCR → structured dim json
FCF  → grid/cell segmentation → OCR per cell → structured fcf json
Datum→ contour cleanup → OCR → datum letter json
✅ Updated output structure
Copy code
Json
{
  "drawing": "...",
  "views": [
    {
      "view_id": 0,
      "groups": [
        {
          "label": "DIM",
          "parts": [...]
        },
        {
          "label": "FCF",
          "fcf": {
            "cells": [...],
            "raw_text": "..."
          }
        },
        {
          "label": "DATUM",
          "datum": {
            "text": "A"
          }
        }
      ]
    }
  ]
}
✅ Add these files to your CLI pipeline
1) cli_pipeline/extractors/fcf_extractor.py
Copy code
Python
import cv2
from pathlib import Path
from cli_pipeline.ocr import ocr_best
from cli_pipeline.pipeline_cli import crop, save_img


class FCFExtractor:
    """
    Placeholder extractor.
    After you re-upload detect_fcf.py, I will replace this with their exact grid logic.
    """

    def extract(self, fcf_img, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save raw crop
        raw_path = out_dir / "fcf_raw.jpg"
        save_img(raw_path, fcf_img)

        # Simple OCR fallback (works but not perfect)
        text = ocr_best(fcf_img)

        return {
            "raw_path": str(raw_path),
            "raw_ocr": text,
            "cells": []  # will be filled using detect_fcf.py logic
        }
2) cli_pipeline/extractors/datum_extractor.py
Copy code
Python
import cv2
from pathlib import Path
from cli_pipeline.ocr import ocr_best
from cli_pipeline.pipeline_cli import save_img


class DatumExtractor:
    """
    Placeholder extractor.
    After you re-upload detect_datum.py, I will replace this with their contour cleanup logic.
    """

    def extract(self, datum_img, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "datum_raw.jpg"
        save_img(raw_path, datum_img)

        text = ocr_best(datum_img)

        return {
            "raw_path": str(raw_path),
            "text": text
        }
3) Modify cli_pipeline/pipeline_cli.py
Add imports at top:
Copy code
Python
from cli_pipeline.extractors.fcf_extractor import FCFExtractor
from cli_pipeline.extractors.datum_extractor import DatumExtractor
In __init__ of DrawingCLIPipeline:
Copy code
Python
self.fcf_extractor = FCFExtractor()
self.datum_extractor = DatumExtractor()
In group loop (inside for gi, g in enumerate(group_dets):)
Replace the group handling block with:
Copy code
Python
if label == "DIM":
    dim_part_dir = parts_dir / f"view_{vi:02d}_DIM_{gi:02d}"
    dim_part_dir.mkdir(parents=True, exist_ok=True)

    part_dets = self.type_detector.detect(str(group_path), conf=0.15, iou=0.45)
    part_dets = sorted(part_dets, key=lambda d: d["xyxy"][0])

    parts = []
    for pi, p in enumerate(part_dets):
        p_crop = crop(g_crop, p["xyxy"])
        part_path = dim_part_dir / f"part_{pi:02d}_{p['label']}.jpg"
        save_img(part_path, p_crop)

        text = ocr_best(p_crop)

        parts.append({
            "part_id": pi,
            "label": p["label"],
            "bbox": p["xyxy"],
            "confidence": p["confidence"],
            "path": str(part_path),
            "ocr": text
        })

    if len(parts) == 0:
        fallback_path = dim_part_dir / "fallback_full_dim.jpg"
        save_img(fallback_path, g_crop)
        parts.append({
            "part_id": 0,
            "label": "FULL_DIM_FALLBACK",
            "bbox": [0, 0, g_crop.shape[1], g_crop.shape[0]],
            "confidence": 0.0,
            "path": str(fallback_path),
            "ocr": ocr_best(g_crop)
        })

    group_rec["parts"] = parts

elif label == "FCF":
    fcf_dir = out_dir / "fcf" / f"view_{vi:02d}_FCF_{gi:02d}"
    group_rec["fcf"] = self.fcf_extractor.extract(g_crop, fcf_dir)

elif label == "DATUM":
    datum_dir = out_dir / "datum" / f"view_{vi:02d}_DATUM_{gi:02d}"
    group_rec["datum"] = self.datum_extractor.extract(g_crop, datum_dir)
✅ New folders created automatically
This will create:
Copy code
Text
runs_cli/job1/
├── fcf/
│   └── view_00_FCF_00/
│       └── fcf_raw.jpg
└── datum/
    └── view_00_DATUM_00/
        └── datum_raw.jpg
✅ Next (after you re-upload the two scripts)
Once you re-upload:
detect_fcf.py
detect_datum.py
I will replace the placeholder extractors with:
FCF extractor (real)
✅ detect grid lines
✅ compute intersections
✅ crop cells
✅ OCR per cell
✅ output structured GD&T frame content
Datum extractor (real)
✅ contour cleanup
✅ isolate datum letter region
✅ OCR single letter robustly
That will match the paper much more closely.
📌 Please upload these two files again:
✅ detect_fcf.py
✅ detect_datum.py
And I’ll immediately paste the final exact working code (no placeholders).Perfect — thanks! ✅ I received:
/mnt/data/detect_fcf (1).py
/mnt/data/detect_datum (1).py
Now I can integrate their real FCF + Datum extraction logic into the Option-A CLI pipeline.
⚠️ One small note: a few of your earlier uploads (like the older OCR file) have expired, but these two are enough to implement the missing extractors.
✅ Updated Option-A CLI Pipeline (FULL: DIM + FCF + DATUM)
This version will:
✅ Save images
Each View
Each DIM / FCF / DATUM group crop
Each DIM part crop
Each FCF cell crop (from grid splitting)
Datum crop cleaned
✅ Output JSON
A single recognize.json with:
view list
group list
extracted text + structure
✅ 1) Add these files
cli_pipeline/extractors/fcf_extractor.py (REAL)
This wraps the logic from your uploaded detect_fcf (1).py.
Copy code
Python
import cv2
import numpy as np
from pathlib import Path
from cli_pipeline.ocr import ocr_best


def save_img(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def crop(img, xyxy):
    x1, y1, x2, y2 = xyxy
    return img[y1:y2, x1:x2]


def find_grid_cells(binary_img):
    """
    Grid cell detection similar to detect_fcf.py:
    - detect horizontal + vertical lines
    - find intersections
    - build cell bounding boxes
    """
    h, w = binary_img.shape[:2]

    # horizontal lines
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 15), 1))
    horizontal = cv2.erode(binary_img, hor_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, hor_kernel, iterations=2)

    # vertical lines
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 10)))
    vertical = cv2.erode(binary_img, ver_kernel, iterations=1)
    vertical = cv2.dilate(vertical, ver_kernel, iterations=2)

    # intersections
    intersections = cv2.bitwise_and(horizontal, vertical)

    ys, xs = np.where(intersections > 0)
    if len(xs) < 10 or len(ys) < 10:
        return []

    # cluster intersection points into grid lines
    xs_sorted = np.sort(xs)
    ys_sorted = np.sort(ys)

    def cluster_coords(coords, thresh=10):
        clusters = []
        current = [coords[0]]
        for c in coords[1:]:
            if abs(c - current[-1]) <= thresh:
                current.append(c)
            else:
                clusters.append(int(np.mean(current)))
                current = [c]
        clusters.append(int(np.mean(current)))
        return clusters

    x_lines = cluster_coords(xs_sorted, thresh=10)
    y_lines = cluster_coords(ys_sorted, thresh=10)

    # build cells between adjacent grid lines
    cells = []
    for yi in range(len(y_lines) - 1):
        for xi in range(len(x_lines) - 1):
            x1, x2 = x_lines[xi], x_lines[xi + 1]
            y1, y2 = y_lines[yi], y_lines[yi + 1]

            # reject tiny cells
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            cells.append([x1, y1, x2, y2])

    return cells


class FCFExtractor:
    """
    FCF extraction:
    - binarize
    - detect grid cells
    - crop each cell
    - OCR each cell
    """

    def extract(self, fcf_img, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "fcf_raw.jpg"
        save_img(raw_path, fcf_img)

        gray = cv2.cvtColor(fcf_img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cells = find_grid_cells(th)

        cell_results = []
        for i, xyxy in enumerate(cells):
            cell_crop = crop(fcf_img, xyxy)
            cell_path = out_dir / f"cell_{i:02d}.jpg"
            save_img(cell_path, cell_crop)

            text = ocr_best(cell_crop)

            cell_results.append({
                "cell_id": i,
                "bbox": xyxy,
                "path": str(cell_path),
                "ocr": text
            })

        # fallback OCR on full frame too
        full_text = ocr_best(fcf_img)

        return {
            "raw_path": str(raw_path),
            "raw_ocr": full_text,
            "cells": cell_results
        }
cli_pipeline/extractors/datum_extractor.py (REAL)
This wraps the logic from your uploaded detect_datum (1).py.
Copy code
Python
import cv2
import numpy as np
from pathlib import Path
from cli_pipeline.ocr import ocr_best


def save_img(path: Path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


class DatumExtractor:
    """
    Datum extraction:
    - binarize
    - find largest contour / inner box
    - crop clean datum region
    - OCR
    """

    def extract(self, datum_img, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_path = out_dir / "datum_raw.jpg"
        save_img(raw_path, datum_img)

        gray = cv2.cvtColor(datum_img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"raw_path": str(raw_path), "text": ocr_best(datum_img)}

        # choose largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # crop + pad slightly
        pad = 3
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(datum_img.shape[1], x + w + pad)
        y2 = min(datum_img.shape[0], y + h + pad)

        clean = datum_img[y1:y2, x1:x2]
        clean_path = out_dir / "datum_clean.jpg"
        save_img(clean_path, clean)

        text = ocr_best(clean)

        return {
            "raw_path": str(raw_path),
            "clean_path": str(clean_path),
            "bbox": [x1, y1, x2, y2],
            "text": text
        }
✅ 2) Update cli_pipeline/pipeline_cli.py
Add these imports at the top:
Copy code
Python
from cli_pipeline.extractors.fcf_extractor import FCFExtractor
from cli_pipeline.extractors.datum_extractor import DatumExtractor
In __init__() add:
Copy code
Python
self.fcf_extractor = FCFExtractor()
self.datum_extractor = DatumExtractor()
In the group loop, add this:
Copy code
Python
elif label == "FCF":
    fcf_dir = out_dir / "fcf" / f"view_{vi:02d}_FCF_{gi:02d}"
    group_rec["fcf"] = self.fcf_extractor.extract(g_crop, fcf_dir)

elif label == "DATUM":
    datum_dir = out_dir / "datum" / f"view_{vi:02d}_DATUM_{gi:02d}"
    group_rec["datum"] = self.datum_extractor.extract(g_crop, datum_dir)
✅ 3) Run command
Same command as before:
Copy code
Bash
python -m cli_pipeline.run_cli ^
  --img input/drawing.jpg ^
  --out runs_cli/job1 ^
  --view-weights weights/0519_view_best.pt ^
  --group-weights weights/1013_DIMandFCF_best.pt ^
  --type-weights weights/0324_dim_and_tol_best.pt ^
  --device 0
✅ What you will now get
New folders:
Copy code
Text
runs_cli/job1/
├── fcf/
│   └── view_00_FCF_00/
│       ├── fcf_raw.jpg
│       ├── cell_00.jpg
│       ├── cell_01.jpg
│       └── ...
└── datum/
    └── view_00_DATUM_00/
        ├── datum_raw.jpg
        └── datum_clean.jpg
And JSON now includes FCF and Datum results.
🔥 Next improvement (recommended)
If your FCF cell detection still misses boxes, I can: ✅ add line-thickness normalization
✅ merge nearby intersection clusters better
✅ enforce row/column sorting (so cells are ordered correctly)
Just share one cropped FCF_00.jpg and I’ll tune it perfectly.




Perfect — thanks. This OCR file is much better than the simple OCR we used because it does 2-stage rotation correction:
deskew using minAreaRect()
OSD rotation detection using a tiled “pattern image” and image_to_osd()
That’s exactly why their repo performs better on rotated dimensions.
Now I’ll integrate this OCR exactly into the Option-A pipeline (DIM + FCF + Datum).
✅ What we will use from your OCR file
Functions:
image_preprocessing(img) → returns rotated + cropped clean image
recg_text(img, lang='eng', config="--oem 3 --psm 4") → returns word boxes + text
We’ll add a helper to convert recg_text() output into a single string.
✅ Drop-in OCR module for pipeline
Create this file:
cli_pipeline/ocr_repo.py
Copy code
Python
import cv2
import numpy as np
import pytesseract


def getProjection(image):
    (h, w) = image.shape
    start = 0
    end = 0
    isFirst = True

    for i in range(h):
        arr = image[i]
        if ~(np.any(arr)):
            continue

        if isFirst:
            start = i - 1
            isFirst = False
        else:
            end = i + 1

    return start, end


def image_preprocessing(original_image):
    h, w = original_image.shape[:2]

    side_length = int(max(h, w) * 1.3)
    blank_image = np.zeros((side_length, side_length, 3), np.uint8)
    blank_image[:, :] = (255, 255, 255)

    border_y, border_x = int((side_length - h) / 2), int((side_length - w) / 2)
    blank_image[border_y:border_y + h, border_x:border_x + w] = original_image
    image = blank_image.copy()

    # grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # invert so text becomes white
    image_gray = cv2.bitwise_not(image_gray)
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)

    # binarize
    _, thresh = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    whereid = np.where(thresh > 0)
    whereid = whereid[::-1]
    coords = np.column_stack(whereid)

    if len(coords) < 10:
        return original_image

    (x, y), (w, h), angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle = 90 - angle

    # rotate to deskew
    center = (side_length // 2, side_length // 2)
    Mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image, Mat, (side_length, side_length),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # crop by projection
    image_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

    H_start, H_end = getProjection(image_binary)
    W_start, W_end = getProjection(image_binary.T)

    border = round(side_length * 0.05)

    # clamp crop
    y1 = max(0, H_start - border)
    y2 = min(rotated.shape[0], H_end + border)
    x1 = max(0, W_start - border)
    x2 = min(rotated.shape[1], W_end + border)

    image_crop = rotated[y1:y2, x1:x2]

    # OSD rotation fix (pattern trick)
    (h, w, d) = image_crop.shape
    num = 10
    pattern_image = np.zeros((h * num, w * num, 3), np.uint8)

    for i in range(num):
        for j in range(num):
            pattern_image[h * i:h * (i + 1), w * j:w * (j + 1)] = image_crop

    rotate_data = pytesseract.image_to_osd(pattern_image, output_type=pytesseract.Output.DICT)
    rotate = float(rotate_data.get("rotate", 0))

    image_rotated = np.rot90(image_crop, -rotate / 90)
    return image_rotated


def recg_text(original_image, lang="eng", config="--oem 3 --psm 4"):
    datas = pytesseract.image_to_data(
        original_image,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    result = []
    for i in range(len(datas["text"])):
        try:
            conf = float(datas["conf"][i])
        except:
            conf = -1

        if conf > 10:
            res = {
                "x": int(datas["left"][i]),
                "y": int(datas["top"][i]),
                "w": int(datas["width"][i]),
                "h": int(datas["height"][i]),
                "text": datas["text"][i].strip()
            }
            if res["text"]:
                result.append(res)

    return result


def ocr_text(img, lang="eng", config="--oem 3 --psm 4"):
    """
    Convenience function:
    preprocess -> recg_text -> join tokens
    """
    pre = image_preprocessing(img)
    tokens = recg_text(pre, lang=lang, config=config)
    text = " ".join([t["text"] for t in tokens if t["text"]])
    return text, tokens, pre
✅ Update DIM / FCF / Datum extractors to use repo OCR
1) DIM OCR update
In your cli_pipeline/pipeline_cli.py (DIM parts OCR), replace:
Copy code
Python
text = ocr_best(p_crop)
with:
Copy code
Python
from cli_pipeline.ocr_repo import ocr_text
text, tokens, pre = ocr_text(p_crop, lang="eng", config="--oem 3 --psm 6")
And store tokens too:
Copy code
Python
parts.append({
    ...
    "ocr": text,
    "ocr_tokens": tokens
})
2) FCF extractor update
In fcf_extractor.py, replace OCR call:
Copy code
Python
text = ocr_best(cell_crop)
with:
Copy code
Python
from cli_pipeline.ocr_repo import ocr_text
text, tokens, pre = ocr_text(cell_crop, lang="eng", config="--oem 3 --psm 6")
3) Datum extractor update
In datum_extractor.py, replace:
Copy code
Python
text = ocr_best(clean)
with:
Copy code
Python
from cli_pipeline.ocr_repo import ocr_text
text, tokens, pre = ocr_text(clean, lang="eng", config="--oem 3 --psm 10")
(psm 10 is good for single characters)
✅ Why this fixes your “rotated OCR is bad” issue
Because this OCR method does:
✔ Deskew using minAreaRect
Fixes tilted/slanted text.
✔ Projection crop
Removes extra borders and empty space.
✔ OSD rotation correction
Fixes 90/180/270 degree rotations reliably.
This is exactly what you need for rotated DIM values and tolerance blocks.
⚠️ One more thing (super important)
Your OCR uses image_to_osd() which needs enough text pixels. For very tiny crops, it can fail or mis-detect rotation.
So best practice: ✅ Run repo OCR on:
DIM crop (bigger)
FCF crop (bigger)
Datum crop (medium)
and on DIM parts only if they’re not too tiny
If you want, I can add a rule like:
Copy code
Python
if crop_area < 1200: skip OSD step
to make it even more stable.
✅ Final result
After this change, your pipeline will produce:
more accurate OCR on rotated text
closer match to repo/paper outputs
consistent token-level OCR output (so you can postprocess)
If you want, next I’ll also add DIM reconstruction logic (combine detected parts into a final dimension object like):
Copy code
Json
{
  "value": "70.5",
  "upper": "+0.02",
  "lower": "-0.01",
  "symbol": "Ø"
}
Just say “add DIM parsing”.

python -m cli_pipeline.run_cli ^
  --img input/drawing.jpg ^
  --out runs_cli/job1 ^
  --view-weights weights/0519_view_best.pt ^
  --group-weights weights/1013_DIMandFCF_best.pt ^
  --type-weights weights/0324_dim_and_tol_best.pt ^
  --device cpu

Got it — this error is coming from Tesseract OSD rotation detection (pytesseract.image_to_osd()), not YOLO.
✅ What the error means
From your screenshot:
“Too few characters. Skipping this page”
“Invalid resolution 0 dpi. Using 70 instead.”
This happens when the cropped image (DIM part / datum / cell) has very little text (or is mostly blank/lines), so Tesseract cannot estimate orientation.
So image_to_osd() crashes for small crops.
✅ Fix (Best): Make OSD optional + fallback safely
Edit your ocr_repo.py and update the image_preprocessing() function like this:
✅ Replace this block:
Copy code
Python
rotate_data = pytesseract.image_to_osd(pattern_image, output_type=pytesseract.Output.DICT)
rotate = float(rotate_data.get("rotate", 0))
image_rotated = np.rot90(image_crop, -rotate / 90)
return image_rotated
✅ With this SAFE version:
Copy code
Python
# OSD rotation fix (pattern trick) - SAFE
rotate = 0
try:
    rotate_data = pytesseract.image_to_osd(
        pattern_image,
        output_type=pytesseract.Output.DICT
    )
    rotate = float(rotate_data.get("rotate", 0))
except pytesseract.TesseractError:
    # too few characters / OSD fails -> skip rotation correction
    rotate = 0

image_rotated = np.rot90(image_crop, -rotate / 90)
return image_rotated
This will stop your pipeline from crashing and still work.
✅ Fix (Better): Skip OSD for tiny crops
Add this before OSD call:
Copy code
Python
if image_crop.shape[0] < 25 or image_crop.shape[1] < 25:
    return image_crop
And also:
Copy code
Python
if np.count_nonzero(image_binary) < 150:
    return image_crop
So OSD runs only when there is enough text.
✅ Fix (Best for your use case): Don’t run OSD on “part crops”
OSD is great for full DIM crop, but for tiny parts like tolerance blocks it often fails.
So modify your OCR call:
For DIM PART crops use only deskew + crop (skip OSD)
We can add a flag:
Copy code
Python
def ocr_text(img, use_osd=True, ...):
and call:
use_osd=False for tiny crops
use_osd=True for full DIM / FCF frame
If you want, I’ll write that version too.
✅ Quick test after patch
Run again:
Copy code
Bash
python -m cli_pipeline.run_cli --img input/drawing.jpg --out runs_cli/job1 ...
It should no longer crash.
⚠️ One more thing: some of your earlier uploaded files expired
If you want me to patch your exact pipeline files directly (instead of you editing), please re-upload the current versions of:
cli_pipeline/ocr_repo.py
cli_pipeline/pipeline_cli.py
(or paste them here)
Then I’ll return the fully corrected final code.
If you apply the try/except patch and still get errors, send me the next traceback screenshot and I’ll fix that too.
Yes ✅ — that’s the right thing to do.
Right now your JSON stores OCR per part (each crop), so you get “single character / broken tokens”.
Instead, we’ll combine all OCR tokens from all parts of a DIM into one final string per dimension.
✅ What we will output for each DIM
Instead of:
Copy code
Json
"parts": [
  {"label":"dimension","ocr":"7"},
  {"label":"dimension","ocr":"0"},
  {"label":"tol_upper","ocr":"+"},
  {"label":"tol_upper","ocr":"0.0"},
  ...
]
We will also add:
Copy code
Json
"dim_text": "70 +0.02 -0.01"
(or whatever it reads)
✅ How to implement (2 small edits)
1) Add this helper function (in pipeline_cli.py)
Put this near the top:
Copy code
Python
def combine_dim_parts_text(parts):
    """
    Combine OCR text from all DIM parts into one string.
    Keeps left-to-right order and removes empty junk.
    """
    cleaned = []
    for p in parts:
        t = (p.get("ocr") or "").strip()
        if not t:
            continue
        cleaned.append(t)

    # join and normalize spaces
    combined = " ".join(cleaned)
    combined = " ".join(combined.split())
    return combined
2) After you build parts for a DIM, add combined text
Find this block inside your DIM section:
Copy code
Python
group_rec["parts"] = parts
Change it to:
Copy code
Python
group_rec["parts"] = parts
group_rec["dim_text"] = combine_dim_parts_text(parts)
✅ (Optional but recommended) Combine using token boxes (more accurate)
If you saved ocr_tokens per part, we can combine them in reading order using x-position.
Add this instead:
Copy code
Python
def combine_dim_parts_text(parts):
    tokens_all = []

    for p in parts:
        for t in p.get("ocr_tokens", []):
            txt = (t.get("text") or "").strip()
            if not txt:
                continue
            # store global-ish ordering key
            tokens_all.append((t["x"], t["y"], txt))

    # sort by top-to-bottom then left-to-right
    tokens_all.sort(key=lambda x: (x[1], x[0]))

    combined = " ".join([t[2] for t in tokens_all])
    combined = " ".join(combined.split())
    return combined
This reduces cases where it becomes "0.02 + - 70" etc.
✅ Resulting JSON example
Now each DIM group will contain:
Copy code
Json
{
  "label": "DIM",
  "path": "...",
  "parts": [...],
  "dim_text": "70 +0.02 -0.01"
}
So you can directly use dim_text as the extracted dimension string.
If you want, I can also do “smart parsing”
Meaning: convert "70 +0.02 -0.01" into:
Copy code
Json
{
  "value": 70.0,
  "upper_tol": 0.02,
  "lower_tol": -0.01,
  "symbol": "Ø"
}
Just say “yes parse dimension text too”.