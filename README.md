Sure — here is a single clean final script that:
✅ Reads images from a folder
✅ Detects outer + inner rectangle borders
✅ Selects the inner rectangle first (not the outer)
✅ Crops only the inner drawing area
✅ Saves cropped outputs + optional debug images
✅ Final Code: extract_inner_frame.py
Copy code
Python
import os
import cv2
import numpy as np


# ----------------------------
# Utility functions
# ----------------------------
def order_points(pts):
    """Order points as: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_to_bbox(quad):
    """Convert 4-point quad -> (xmin, ymin, xmax, ymax)"""
    pts = quad.reshape(4, 2)
    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


# ----------------------------
# Core detection logic
# ----------------------------
def find_rectangle_candidates(binary_img, min_area_ratio=0.03):
    """
    Find quadrilateral contours (rectangles) from a binary image.
    Returns list of quads sorted by area descending.
    """
    h, w = binary_img.shape[:2]
    min_area = min_area_ratio * (h * w)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    quads = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx)

    quads.sort(key=lambda q: cv2.contourArea(q), reverse=True)
    return quads


def pick_inner_rectangle(quads, img_shape):
    """
    Pick the inner frame rectangle.
    Assumption:
      - Largest rectangle = outer border
      - Next best rectangle = inner border
    But we apply a scoring rule to avoid picking title block.
    """
    if len(quads) == 0:
        return None

    h, w = img_shape[:2]
    img_area = h * w

    # If only one rectangle exists, return it
    if len(quads) == 1:
        return quads[0]

    outer = quads[0]
    outer_bbox = quad_to_bbox(outer)
    outer_area = bbox_area(outer_bbox)

    best = None
    best_score = -1

    for q in quads[1:]:
        bbox = quad_to_bbox(q)
        area = bbox_area(bbox)

        # Must be smaller than outer border
        if area > 0.95 * outer_area:
            continue

        # Must still be reasonably large to be the drawing frame
        # (title block rectangles are usually much smaller)
        if area < 0.20 * img_area:
            continue

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Prefer centered rectangles (inner frame is usually centered)
        center_dist = ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) / (w * w + h * h)

        # Score: large area + near center
        score = (area / img_area) - 0.35 * center_dist

        if score > best_score:
            best_score = score
            best = q

    # Fallback: 2nd largest rectangle
    if best is None:
        best = quads[1]

    return best


def extract_inner_frame(image_bgr, padding=15, debug=False):
    """
    Extract/crop inner drawing frame from engineering drawing image.
    Returns cropped_image, info dict
    """
    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize: invert so lines become white
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # Morphological close to connect broken border lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detect rectangle candidates
    quads = find_rectangle_candidates(closed, min_area_ratio=0.02)
    if len(quads) == 0:
        raise RuntimeError("No rectangle candidates found. Try increasing DPI or morphology kernel size.")

    inner_quad = pick_inner_rectangle(quads, image_bgr.shape)
    if inner_quad is None:
        raise RuntimeError("Could not select inner frame rectangle.")

    pts = order_points(inner_quad.reshape(4, 2))
    x_min = int(max(0, np.min(pts[:, 0]) + padding))
    y_min = int(max(0, np.min(pts[:, 1]) + padding))
    x_max = int(min(w, np.max(pts[:, 0]) - padding))
    y_max = int(min(h, np.max(pts[:, 1]) - padding))

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Invalid crop box detected. Try reducing padding.")

    cropped = image_bgr[y_min:y_max, x_min:x_max].copy()

    info = {
        "crop_box": (x_min, y_min, x_max, y_max),
        "num_rectangles_found": len(quads),
    }

    if debug:
        dbg_candidates = image_bgr.copy()
        for i, q in enumerate(quads[:6]):  # show top 6
            p = order_points(q.reshape(4, 2)).astype(np.int32)
            cv2.polylines(dbg_candidates, [p], True, (0, 255, 0), 2)
            cv2.putText(dbg_candidates, f"R{i}", tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        dbg_inner = image_bgr.copy()
        cv2.polylines(dbg_inner, [pts.astype(np.int32)], True, (0, 0, 255), 3)

        info["debug_images"] = {
            "threshold": th,
            "closed": closed,
            "rect_candidates": dbg_candidates,
            "chosen_inner_rect": dbg_inner,
            "cropped": cropped,
        }

    return cropped, info


# ----------------------------
# Batch processing
# ----------------------------
def process_folder(input_dir, output_dir, debug_dir=None):
    os.makedirs(output_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(valid_exts):
            continue

        in_path = os.path.join(input_dir, fname)
        img = cv2.imread(in_path)

        if img is None:
            print(f"[WARN] Could not read: {in_path}")
            continue

        try:
            cropped, info = extract_inner_frame(img, padding=15, debug=bool(debug_dir))
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, cropped)

            print(f"[OK] {fname} -> cropped | rectangles={info['num_rectangles_found']}")

            if debug_dir and "debug_images" in info:
                base = os.path.splitext(fname)[0]
                for k, v in info["debug_images"].items():
                    dbg_path = os.path.join(debug_dir, f"{base}__{k}.png")
                    cv2.imwrite(dbg_path, v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    # Put your images inside input_images/
    # Cropped output will be saved into output_cropped/
    # Debug images will be saved into debug_outputs/
    input_dir = "input_images"
    output_dir = "output_cropped"
    debug_dir = "debug_outputs"  # set to None to disable debug outputs

    process_folder(input_dir, output_dir, debug_dir=debug_dir)
✅ How to use
Folder structure
Copy code

project/
  extract_inner_frame.py
  input_images/
    sample1.png
    sample2.jpg
Run
Copy code
Bash
python extract_inner_frame.py
Outputs:
output_cropped/ → final cropped inner drawing region
debug_outputs/ → threshold + candidate rectangles + chosen rectangle
🔧 If it still picks wrong rectangle (rare cases)
If it mistakenly picks title block:
Inside pick_inner_rectangle() change:
Copy code
Python
if area < 0.20 * img_area:
to
Copy code
Python
if area < 0.30 * img_area:
If it fails to find inner border:
Increase morphology:
Copy code
Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
If you want next: I can give you a version that guarantees outer+inner border detection using Hough line intersections (even better for engineering drawings).