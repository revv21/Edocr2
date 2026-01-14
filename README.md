Awesome — below is a complete, clean 3-stage pipeline (exactly like the paper) that:
✅ Uses your 3 YOLO weight files in the correct order
✅ Saves each View image
✅ Saves each DIM group image
✅ Saves each DIM type crop (value/tolerance parts)
✅ Runs OCR on the smallest crops
✅ Produces a final JSON output linking everything
This is the most “engineering-grade” version while still being simple.
✅ Final Folder Output (what you will get)
After running on drawing.jpg, you’ll get:
Copy code
Text
runs/job_YYYYMMDD_HHMMSS/
│
├── views/
│   ├── view_00.jpg
│   ├── view_01.jpg
│   └── ...
│
├── groups/
│   ├── view_00/
│   │   ├── DIM_00.jpg
│   │   ├── DIM_01.jpg
│   │   ├── FCF_00.jpg
│   │   └── Datum_00.jpg
│   └── view_01/
│       ├── DIM_00.jpg
│       └── ...
│
├── dim_parts/
│   ├── view_00_DIM_00/
│   │   ├── part_00_<class>.jpg
│   │   ├── part_01_<class>.jpg
│   │   └── ...
│   └── view_00_DIM_01/
│       └── ...
│
├── annotated/
│   ├── drawing_views.jpg
│   ├── view_00_groups.jpg
│   └── ...
│
└── output.json
✅ 1) pipeline/detectors/yolo_detector.py
This uses YOLOv7 internals (models/, utils/) from the same repo.
Copy code
Python
import torch
import numpy as np
import cv2

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

    def detect_path(self, image_path, conf=0.25, iou=0.45):
        """
        Detect objects in an image path (recommended).
        Returns list of dicts: {xyxy, confidence, label}
        """
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
✅ 2) pipeline/processing/image_ops.py
Copy code
Python
import cv2
import numpy as np


def crop(image, xyxy):
    x1, y1, x2, y2 = xyxy
    return image[y1:y2, x1:x2]


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["xyxy"]
        label = d["label"]
        conf = d["confidence"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            img, f"{label} {conf:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return img
✅ 3) pipeline/ocr/tess_ocr.py  (your same logic)
Copy code
Python
import cv2
import numpy as np
import pytesseract


def preprocess(image):
    h, w = image.shape[:2]
    side = int(max(h, w) * 1.3)

    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y+h, x:x+w] = image

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0)[::-1])
    if len(coords) < 10:
        return canvas

    (_, _), (_, _), angle = cv2.minAreaRect(coords)
    if angle < -45:
        angle = 90 - angle

    M = cv2.getRotationMatrix2D((side // 2, side // 2), angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (side, side),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def recognize(image):
    img = preprocess(image)
    data = pytesseract.image_to_data(
        img,
        config="--oem 3 --psm 6 -l eng_math",
        output_type=pytesseract.Output.DICT
    )
    tokens = [t.strip() for t in data["text"] if t.strip()]
    return " ".join(tokens)
✅ 4) pipeline/runner.py (FULL pipeline)
Copy code
Python
import json
import cv2
from pathlib import Path
from datetime import datetime

from pipeline.detectors.yolo_detector import YOLODetector
from pipeline.processing.image_ops import crop, save_image, draw_boxes
from pipeline.ocr.tess_ocr import recognize


class DrawingPipeline:
    def __init__(self, view_w, group_w, type_w, device="0"):
        self.view_detector = YOLODetector(view_w, device=device)
        self.group_detector = YOLODetector(group_w, device=device)
        self.type_detector = YOLODetector(type_w, device=device)

    def run(self, drawing_path, out_root="runs"):
        drawing_path = Path(drawing_path)
        img = cv2.imread(str(drawing_path))
        if img is None:
            raise FileNotFoundError(f"Could not read: {drawing_path}")

        job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
        out_dir = Path(out_root) / job_id

        views_dir = out_dir / "views"
        groups_dir = out_dir / "groups"
        parts_dir = out_dir / "dim_parts"
        annotated_dir = out_dir / "annotated"

        # -------------------------
        # Stage 1: View Detection
        # -------------------------
        view_dets = self.view_detector.detect_path(str(drawing_path))
        view_dets = [d for d in view_dets if d["label"].lower() == "view"]
        view_dets = sorted(view_dets, key=lambda d: d["xyxy"][0])

        annotated_views = draw_boxes(img, view_dets)
        save_image(annotated_dir / "drawing_views.jpg", annotated_views)

        output = {
            "drawing": str(drawing_path),
            "views": []
        }

        # Save each view crop
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

            # -------------------------
            # Stage 2: Group Detection inside View
            # -------------------------
            group_dets = self.group_detector.detect_path(str(view_path))
            group_dets = sorted(group_dets, key=lambda d: d["xyxy"][0])

            annotated_groups = draw_boxes(view_crop, group_dets)
            save_image(annotated_dir / f"view_{vi:02d}_groups.jpg", annotated_groups)

            view_group_folder = groups_dir / f"view_{vi:02d}"
            view_group_folder.mkdir(parents=True, exist_ok=True)

            # Save each group crop (DIM / FCF / Datum)
            for gi, g in enumerate(group_dets):
                label = g["label"]
                g_crop = crop(view_crop, g["xyxy"])

                group_path = view_group_folder / f"{label}_{gi:02d}.jpg"
                save_image(group_path, g_crop)

                group_record = {
                    "group_id": gi,
                    "label": label,
                    "bbox": g["xyxy"],
                    "path": str(group_path),
                    "parts": []
                }

                # -------------------------
                # Stage 3: Type Detection inside DIM only
                # -------------------------
                if label.upper() == "DIM":
                    part_dets = self.type_detector.detect_path(str(group_path))
                    part_dets = sorted(part_dets, key=lambda d: d["xyxy"][0])

                    dim_part_folder = parts_dir / f"view_{vi:02d}_DIM_{gi:02d}"
                    dim_part_folder.mkdir(parents=True, exist_ok=True)

                    # Save each detected part + OCR
                    for pi, p in enumerate(part_dets):
                        part_label = p["label"]
                        p_crop = crop(g_crop, p["xyxy"])

                        part_path = dim_part_folder / f"part_{pi:02d}_{part_label}.jpg"
                        save_image(part_path, p_crop)

                        text = recognize(p_crop)

                        group_record["parts"].append({
                            "part_id": pi,
                            "label": part_label,
                            "bbox": p["xyxy"],
                            "path": str(part_path),
                            "ocr": text,
                            "confidence": p["confidence"]
                        })

                view_record["groups"].append(group_record)

            output["views"].append(view_record)

        # Save JSON
        json_path = out_dir / "output.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        print(f"✅ Done! Output saved to: {json_path}")
        return output
✅ 5) main.py
Copy code
Python
from pipeline.runner import DrawingPipeline

pipeline = DrawingPipeline(
    view_w="weights/0519_view_best.pt",
    group_w="weights/1013_DIMandFCF_best.pt",
    type_w="weights/0324_dim_and_tol_best.pt",
    device="0"   # use "cpu" if no GPU
)

pipeline.run("input/example_dwg_page-0001.jpg", out_root="runs")
🔥 Important Notes (so it works immediately)
✅ Your YOLOv7 folders must be at project root:
Copy code
Text
project/
├── models/
├── utils/
├── weights/
├── pipeline/
└── main.py
✅ Tesseract must be installed on Windows
And if needed, add:
Copy code
Python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
If you want, I can also add:
✅ “Only keep DIM groups”
✅ Confidence thresholds per stage
✅ Save annotated images for DIM parts too
✅ Batch processing for a folder of drawings
Just say the word.