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

Yes — below is a complete runnable Python implementation of Approach 1:
✅ Detect the inner border rectangle (drawing frame) using edges + Hough lines
✅ Deskew (fix rotation)
✅ Crop only the drawing area, removing the template/title block
✅ Works even if template dimensions change (as long as there is a clear frame)
✅ Install requirements
Copy code
Bash
pip install opencv-python numpy
✅ Script: extract_drawing_area.py
Copy code
Python
import os
import cv2
import numpy as np


def order_points(pts):
    """Return points ordered as: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_largest_rect_contour(binary_img, min_area_ratio=0.15):
    """
    Find the largest contour that looks like a rectangle.
    binary_img should be white=foreground, black=background.
    """
    h, w = binary_img.shape[:2]
    min_area = min_area_ratio * (h * w)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Prefer quadrilaterals
        if len(approx) == 4:
            if area > best_area:
                best_area = area
                best = approx

    return best  # shape (4,1,2) or None


def deskew_using_hough(gray):
    """
    Estimate skew angle using Hough lines and rotate to correct it.
    Returns deskewed image.
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                            minLineLength=int(0.3 * gray.shape[1]),
                            maxLineGap=20)

    if lines is None:
        return gray, 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Keep near-horizontal lines only
        if -30 < angle < 30:
            angles.append(angle)

    if len(angles) < 5:
        return gray, 0.0

    median_angle = float(np.median(angles))

    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated, median_angle


def extract_drawing_area(image_bgr, padding=10, debug=False):
    """
    Main extraction:
    1) grayscale
    2) deskew
    3) detect inner frame rectangle
    4) crop inside it
    """
    original = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Deskew (important for scanned/photographed drawings)
    gray_deskewed, angle = deskew_using_hough(gray)
    bgr_deskewed = cv2.cvtColor(gray_deskewed, cv2.COLOR_GRAY2BGR)

    # Preprocess for border detection
    blur = cv2.GaussianBlur(gray_deskewed, (5, 5), 0)

    # Adaptive threshold makes it work on different lighting/scan conditions
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # Connect border lines using morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    rect = find_largest_rect_contour(closed, min_area_ratio=0.20)

    if rect is None:
        # fallback: try slightly different settings
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed2 = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2, iterations=2)
        rect = find_largest_rect_contour(closed2, min_area_ratio=0.15)

        if rect is None:
            raise RuntimeError("Could not detect inner drawing frame rectangle.")

    pts = rect.reshape(4, 2)
    pts = order_points(pts)

    # Crop bounding rectangle (simpler and stable)
    x_min = int(max(0, np.min(pts[:, 0]) + padding))
    y_min = int(max(0, np.min(pts[:, 1]) + padding))
    x_max = int(min(bgr_deskewed.shape[1], np.max(pts[:, 0]) - padding))
    y_max = int(min(bgr_deskewed.shape[0], np.max(pts[:, 1]) - padding))

    cropped = bgr_deskewed[y_min:y_max, x_min:x_max].copy()

    debug_images = {}
    if debug:
        dbg = bgr_deskewed.copy()
        cv2.polylines(dbg, [pts.astype(np.int32)], True, (0, 0, 255), 3)
        debug_images = {
            "gray_deskewed": gray_deskewed,
            "threshold": th,
            "closed": closed,
            "detected_rect": dbg,
            "cropped": cropped
        }

    return cropped, {
        "deskew_angle_deg": angle,
        "crop_box": (x_min, y_min, x_max, y_max),
        "debug": debug_images
    }


def process_folder(input_dir, output_dir, debug_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(exts):
            continue

        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)

        if img is None:
            print(f"[WARN] Could not read: {in_path}")
            continue

        try:
            cropped, info = extract_drawing_area(img, padding=15, debug=bool(debug_dir))
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, cropped)
            print(f"[OK] {fname} -> cropped (angle={info['deskew_angle_deg']:.2f}°)")

            if debug_dir:
                base = os.path.splitext(fname)[0]
                for k, v in info["debug"].items():
                    if v is None:
                        continue
                    dbg_path = os.path.join(debug_dir, f"{base}__{k}.png")
                    if len(v.shape) == 2:
                        cv2.imwrite(dbg_path, v)
                    else:
                        cv2.imwrite(dbg_path, v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    # Example usage:
    # Put your input images in ./input_images/
    # Output will go to ./output_cropped/
    input_dir = "input_images"
    output_dir = "output_cropped"
    debug_dir = "debug_outputs"  # set to None to disable

    process_folder(input_dir, output_dir, debug_dir=debug_dir)
✅ How to run
Put your drawings (images) here:
Copy code

input_images/
  drawing1.png
  drawing2.jpg
Run:
Copy code
Bash
python extract_drawing_area.py
Cropped output will appear in:
Copy code

output_cropped/
Debug visualizations will appear in:
Copy code

debug_outputs/
Notes / tuning knobs (important)
If your drawings vary a lot, these are the first parameters to adjust:
1) min_area_ratio
In find_largest_rect_contour(...)
Increase to 0.3 if it mistakenly detects title blocks instead of full border
Decrease to 0.1 if it fails to find a rectangle
2) padding
In extract_drawing_area(..., padding=15)
Increase if border lines remain in crop.
3) Morphology kernel sizes
These two lines:
Copy code
Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
If borders are broken/faint → increase sizes.
What this gives you (output)
You will get an image that is basically: ✅ the main drawing region
❌ without the outer template margins + title block tables (in most cases)
If you want, I can also give you an improved version that:
removes title block area only (bottom/right strip)
handles multiple frames
works directly on PDF → image → crop (when you run internally)