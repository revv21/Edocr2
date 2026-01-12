Perfect — this is doable cleanly without changing the trained YOLO models at all 👍
Below I’ll give you a new, modular pipeline that:
Uses the exact same YOLOv7 weights
Reuses the logic from the 4 files you uploaded
Removes script duplication
Produces clean structured outputs
Is easy to extend later (arrowheads, CAD, agents)
I’ll stay very concrete and show real runnable Python code, not pseudo-talk.
0️⃣ What we are preserving from the original repo
Original file
What we keep
detect_view.py
YOLO inference + cropping logic
detect_dim.py
Detection + OCR → JSON
detect_custom.py
Aspect-ratio safe cropping
Tess_OCR.py
Deskew + OCR preprocessing
We do not retrain or touch YOLO weights.
1️⃣ New folder structure (minimal, practical)
Copy code
Text
pipeline/
│
├── detectors/
│   └── yolo.py
│
├── ocr/
│   └── tess.py
│
├── processing/
│   ├── crop.py
│   └── pad.py
│
├── pipeline/
│   ├── view_stage.py
│   ├── dim_stage.py
│   └── runner.py
│
├── schemas/
│   └── output.py
│
└── main.py
No over-engineering — just separation of concerns.
2️⃣ YOLO wrapper (shared by all stages)
detectors/yolo.py
Copy code
Python
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages
import numpy as np

class YOLODetector:
    def __init__(self, weights, device='0', img_size=640):
        self.device = torch.device(f'cuda:{device}' if device != 'cpu' else 'cpu')
        self.model = attempt_load(weights, map_location=self.device)
        self.model.eval()
        self.img_size = img_size
        self.names = self.model.names

    def detect(self, source, conf=0.25, iou=0.45):
        dataset = LoadImages(source, img_size=self.img_size)
        results = []

        for path, img, im0, _ in dataset:
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = self.model(img)[0]
            pred = non_max_suppression(pred, conf, iou)

            for det in pred:
                if det is None:
                    continue

                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in det:
                    results.append({
                        "xyxy": [int(x) for x in xyxy],
                        "conf": float(conf),
                        "label": self.names[int(cls)],
                        "image": im0,
                        "path": path
                    })
        return results
👉 This replaces YOLO code duplication across all original scripts.
3️⃣ OCR logic (directly reused)
ocr/tess.py
This is almost exactly your Tess_OCR.py, just wrapped cleanly.
Copy code
Python
import cv2
import numpy as np
import pytesseract

def image_preprocessing(image):
    h, w = image.shape[:2]
    side = int(max(h, w) * 1.3)

    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side-h)//2, (side-w)//2
    canvas[y:y+h, x:x+w] = image

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0)[::-1])
    (_, _), (_, _), angle = cv2.minAreaRect(coords)

    if angle < -45:
        angle = 90 - angle

    M = cv2.getRotationMatrix2D((side//2, side//2), angle, 1)
    rotated = cv2.warpAffine(canvas, M, (side, side),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def recognize_text(image):
    processed = image_preprocessing(image)
    data = pytesseract.image_to_data(
        processed,
        config="--oem 3 --psm 6 -l eng_math",
        output_type=pytesseract.Output.DICT
    )
    text = ''.join([t for t in data['text'] if t.strip()])
    return text
4️⃣ Cropping utilities
processing/crop.py
Copy code
Python
def crop(image, xyxy):
    x1, y1, x2, y2 = xyxy
    return image[y1:y2, x1:x2]
processing/pad.py
Copy code
Python
import numpy as np

def pad_if_needed(img):
    h, w = img.shape[:2]
    if 0.6 < w/h < 1.6:
        return img

    side = int(max(h, w) * 1.3)
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side-h)//2, (side-w)//2
    canvas[y:y+h, x:x+w] = img
    return canvas
This logic comes from detect_custom.py.
5️⃣ View detection stage
pipeline/view_stage.py
Copy code
Python
from processing.crop import crop

class ViewStage:
    def __init__(self, detector):
        self.detector = detector

    def run(self, image_path):
        detections = self.detector.detect(image_path)
        views = []

        for d in detections:
            if d['label'] == 'View':
                view_img = crop(d['image'], d['xyxy'])
                views.append({
                    "image": view_img,
                    "bbox": d['xyxy']
                })
        return views
This replaces detect_view.py.
6️⃣ Dimension detection + OCR stage
pipeline/dim_stage.py
Copy code
Python
from processing.crop import crop
from processing.pad import pad_if_needed
from ocr.tess import recognize_text

class DimensionStage:
    def __init__(self, detector):
        self.detector = detector

    def run(self, image_path):
        detections = self.detector.detect(image_path)
        dims = []

        for d in detections:
            if d['label'] == 'DIM':
                img = crop(d['image'], d['xyxy'])
                img = pad_if_needed(img)
                text = recognize_text(img)

                dims.append({
                    "bbox": d['xyxy'],
                    "confidence": d['conf'],
                    "text": text
                })
        return dims
This replaces detect_dim.py + Tess_OCR coupling.
7️⃣ Output schema
schemas/output.py
Copy code
Python
def make_output(image_name, dimensions):
    return {
        "image": image_name,
        "dimensions": dimensions
    }
8️⃣ Pipeline runner
pipeline/runner.py
Copy code
Python
from detectors.yolo import YOLODetector
from pipeline.view_stage import ViewStage
from pipeline.dim_stage import DimensionStage

class DrawingPipeline:
    def __init__(self, view_weights, dim_weights):
        self.view_detector = YOLODetector(view_weights)
        self.dim_detector = YOLODetector(dim_weights)

        self.view_stage = ViewStage(self.view_detector)
        self.dim_stage = DimensionStage(self.dim_detector)

    def run(self, image_path):
        views = self.view_stage.run(image_path)
        results = []

        for i, view in enumerate(views):
            dims = self.dim_stage.run(image_path)
            results.append({
                "view_id": i,
                "dimensions": dims
            })
        return results
9️⃣ Entry point
main.py
Copy code
Python
import json
from pipeline.runner import DrawingPipeline

pipeline = DrawingPipeline(
    view_weights="weights/view.pt",
    dim_weights="weights/dim.pt"
)

result = pipeline.run("input/drawing.png")

with open("output.json", "w") as f:
    json.dump(result, f, indent=2)
🔑 What you gain immediately
✅ Same YOLO models
✅ Same OCR behavior
✅ Same cropping logic
❌ No duplicated YOLO scripts
❌ No OCR inside detection code
🚀 What becomes easy next
Now you can add cleanly:
Arrowhead detector → geometry/arrowheads.py
Dimension–line association
Manufacturing-critical dimension classifier
CAD API adapter (CATIA/NX)
LLM reasoning on output.json
If you want, next I can:
Add arrowhead + leader detection
Parse tolerances (±, limits, Ø, R)
Map dimensions → geometry
Generate CATIA API calls
Just tell me the next step.