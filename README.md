Perfect ✅
I’ll generate a complete working codebase that matches the repo + paper pipeline, and improves the “DIM type OCR” using the better logic from your uploaded files.
⚠️ Important note before code:
Your uploaded detect_fcf.py, detect_datum.py, and Tess_OCR (1).py are written as standalone scripts (with their own CLI / paths).
So the cleanest way is:
✅ Wrap them as “modules” and call their functions directly
(so you don’t need subprocess, and you still save every view + dim image)
✅ FINAL PROJECT STRUCTURE
Create this folder structure:
Copy code
Text
project_root/
│
├── models/                     # from YOLOv7 repo
├── utils/                      # from YOLOv7 repo
├── weights/
│   ├── 0519_view_best.pt
│   ├── 1013_DIMandFCF_best.pt
│   └── 0324_dim_and_tol_best.pt
│
├── input/
│   └── example.jpg
│
├── pipeline/
│   ├── __init__.py
│
│   ├── detectors/
│   │   ├── __init__.py
│   │   └── yolo_detector.py
│
│   ├── processing/
│   │   ├── __init__.py
│   │   └── image_ops.py
│
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── tess_ocr_best.py
│
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── dim_extractor.py
│   │   ├── fcf_extractor.py
│   │   └── datum_extractor.py
│
│   ├── runner.py
│   └── main.py
│
└── requirements.txt
✅ requirements.txt
Copy code
Txt
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.5
opencv-python>=4.5.3
Pillow>=8.2.0
PyYAML>=5.4.1
scipy>=1.5.4
tqdm>=4.64.0
matplotlib>=3.3.4
pytesseract>=0.3.10
✅ pipeline/detectors/yolo_detector.py
Copy code
Python
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages


class YOLODetector:
    def __init__(self, weights, img_size=640, device="0"):
        self.device = torch.device(f"cuda:{device}" if device != "cpu" else "cpu")
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.img_size = img_size
        self.names = self.model.names

    def detect(self, image_path, conf=0.25, iou=0.45):
        dataset = LoadImages(image_path, img_size=self.img_size)
        detections = []

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
                    detections.append({
                        "xyxy": [int(x) for x in xyxy],
                        "confidence": float(c),
                        "label": self.names[int(cls)]
                    })

        return detections
✅ pipeline/processing/image_ops.py
Copy code
Python
import cv2
from pathlib import Path


def crop(image, xyxy):
    x1, y1, x2, y2 = xyxy
    return image[y1:y2, x1:x2]


def save_image(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
✅ pipeline/ocr/tess_ocr_best.py  (IMPROVED OCR)
This is adapted from your uploaded Tess_OCR (1).py and simplified.
Copy code
Python
import cv2
import numpy as np
import pytesseract


def pad_to_square(image, scale=1.3):
    h, w = image.shape[:2]
    side = int(max(h, w) * scale)
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y+h, x:x+w] = image
    return canvas


def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(th > 0)[::-1])
    if len(coords) < 10:
        return image

    (_, _), (_, _), angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle = 90 - angle

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def recognize_text(image, lang="eng_math"):
    img = pad_to_square(image)
    img = deskew(img)

    data = pytesseract.image_to_data(
        img,
        config=f"--oem 3 --psm 6 -l {lang}",
        output_type=pytesseract.Output.DICT
    )

    tokens = []
    for t, conf in zip(data["text"], data["conf"]):
        if t.strip() and int(conf) > 40:
            tokens.append(t.strip())

    return " ".join(tokens)
✅ pipeline/extractors/dim_extractor.py
Uses type model (0324_dim_and_tol_best.pt) and OCR on each part crop.
Copy code
Python
from pathlib import Path
from pipeline.processing.image_ops import crop, save_image
from pipeline.ocr.tess_ocr_best import recognize_text


class DIMExtractor:
    def __init__(self, type_detector):
        self.type_detector = type_detector

    def extract(self, dim_image_path: Path, out_parts_dir: Path):
        """
        Input: DIM crop path
        Output: parts list with OCR
        """
        dim_img = __import__("cv2").imread(str(dim_image_path))

        part_dets = self.type_detector.detect(str(dim_image_path))
        part_dets = sorted(part_dets, key=lambda d: d["xyxy"][0])

        results = []

        for pi, p in enumerate(part_dets):
            part_label = p["label"]
            part_crop = crop(dim_img, p["xyxy"])

            part_path = out_parts_dir / f"part_{pi:02d}_{part_label}.jpg"
            save_image(part_path, part_crop)

            text = recognize_text(part_crop)

            results.append({
                "part_id": pi,
                "label": part_label,
                "bbox": p["xyxy"],
                "confidence": p["confidence"],
                "path": str(part_path),
                "ocr": text
            })

        return results
✅ pipeline/extractors/fcf_extractor.py
This is a safe wrapper around your detect_fcf.py style logic.
For now I’m giving you a strong baseline (grid-based segmentation is complex).
This version will:
Save FCF crop
Run OCR directly (still improved)
Return structured result
You can later plug the exact detect_fcf.py cell-splitting logic inside this file.
Copy code
Python
from pipeline.ocr.tess_ocr_best import recognize_text


class FCFExtractor:
    def extract(self, fcf_crop_img):
        text = recognize_text(fcf_crop_img)
        return {
            "raw_ocr": text
        }
✅ pipeline/extractors/datum_extractor.py
Also a wrapper.
Copy code
Python
from pipeline.ocr.tess_ocr_best import recognize_text


class DatumExtractor:
    def extract(self, datum_crop_img):
        text = recognize_text(datum_crop_img)
        return {
            "datum": text
        }
✅ pipeline/runner.py  (FULL 3-STAGE PIPELINE)
This saves every view + every DIM crop + every part crop.
Copy code
Python
import json
from pathlib import Path
from datetime import datetime

from pipeline.detectors.yolo_detector import YOLODetector
from pipeline.processing.image_ops import read_image, crop, save_image
from pipeline.extractors.dim_extractor import DIMExtractor
from pipeline.extractors.fcf_extractor import FCFExtractor
from pipeline.extractors.datum_extractor import DatumExtractor


class DrawingPipeline:
    def __init__(self, view_w, group_w, type_w, device="0"):
        self.view_detector = YOLODetector(view_w, device=device)
        self.group_detector = YOLODetector(group_w, device=device)
        self.type_detector = YOLODetector(type_w, device=device)

        self.dim_extractor = DIMExtractor(self.type_detector)
        self.fcf_extractor = FCFExtractor()
        self.datum_extractor = DatumExtractor()

    def run(self, drawing_path, out_root="runs"):
        drawing_path = Path(drawing_path)
        img = read_image(drawing_path)

        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
        out_dir = Path(out_root) / job_id

        views_dir = out_dir / "views"
        groups_dir = out_dir / "groups"
        dim_parts_dir = out_dir / "dim_parts"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------
        # Stage 1: Detect Views
        # ------------------------
        view_dets = self.view_detector.detect(str(drawing_path))
        view_dets = [d for d in view_dets if d["label"].lower() == "view"]
        view_dets = sorted(view_dets, key=lambda d: d["xyxy"][0])

        output = {"drawing": str(drawing_path), "views": []}

        for vi, v in enumerate(view_dets):
            view_crop = crop(img, v["xyxy"])
            view_path = views_dir / f"view_{vi:02d}.jpg"
            save_image(view_path, view_crop)

            view_record = {
                "view_id": vi,
                "bbox": v["xyxy"],
                "path": str(view_path),
                "groups": []
            }

            # ------------------------
            # Stage 2: Detect Groups
            # ------------------------
            group_dets = self.group_detector.detect(str(view_path))
            group_dets = sorted(group_dets, key=lambda d: d["xyxy"][0])

            view_group_dir = groups_dir / f"view_{vi:02d}"
            view_group_dir.mkdir(parents=True, exist_ok=True)

            for gi, g in enumerate(group_dets):
                label = g["label"].upper()
                g_crop = crop(view_crop, g["xyxy"])
                group_path = view_group_dir / f"{label}_{gi:02d}.jpg"
                save_image(group_path, g_crop)

                group_record = {
                    "group_id": gi,
                    "label": label,
                    "bbox": g["xyxy"],
                    "path": str(group_path)
                }

                # ------------------------
                # Stage 3: Extract Info
                # ------------------------
                if label == "DIM":
                    part_folder = dim_parts_dir / f"view_{vi:02d}_DIM_{gi:02d}"
                    part_folder.mkdir(parents=True, exist_ok=True)

                    parts = self.dim_extractor.extract(group_path, part_folder)
                    group_record["parts"] = parts

                elif label == "FCF":
                    group_record["fcf"] = self.fcf_extractor.extract(g_crop)

                elif label == "DATUM":
                    group_record["datum"] = self.datum_extractor.extract(g_crop)

                view_record["groups"].append(group_record)

            output["views"].append(view_record)

        # Save JSON
        json_path = out_dir / "output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"✅ Done. Output saved at: {json_path}")
        return output
✅ pipeline/main.py
Copy code
Python
from pipeline.runner import DrawingPipeline

pipeline = DrawingPipeline(
    view_w="weights/0519_view_best.pt",
    group_w="weights/1013_DIMandFCF_best.pt",
    type_w="weights/0324_dim_and_tol_best.pt",
    device="0"   # use "cpu" if needed
)

pipeline.run("input/example.jpg", out_root="runs")
✅ What you’ll see after running
You will visually verify everything:
runs/job_xxx/views/view_00.jpg
runs/job_xxx/groups/view_00/DIM_00.jpg
runs/job_xxx/dim_parts/view_00_DIM_00/part_00_value.jpg
So debugging becomes easy.
🚀 Next Improvement (THIS will fix “bad dim OCR” even more)
Right now DIM parts are detected, but OCR can still struggle because:
Dimension line fragments appear inside crop
Tolerance boxes overlap
Symbols are weird
So next I can add:
✅ line removal using morphology before OCR
✅ class-wise OCR config (different psm for each part type)
✅ merge OCR tokens into final DIM_VALUE, UPPER, LOWER
Just tell me and I’ll upgrade dim_extractor.py to produce final structured output like:
Copy code
Json
{
  "dim_value": 70.5,
  "upper_tol": 0.02,
  "lower_tol": 0.01,
  "raw": "70.5 +0.02 -0.01"
}
Want that next?