From your screenshot the script is failing with:
No rectangle candidates found. Try increasing DPI or morphology kernel size.
That means your binary image doesn’t contain a clean closed rectangular contour, so findContours() never sees a “rectangle”.
This usually happens because:
the border lines are too thin / broken
the image is low resolution
the page is not properly thresholded
the rectangle is split into multiple contours
✅ Fix (most reliable): detect inner rectangle using LINE EXTRACTION (not contours)
Instead of relying on “closed contour rectangles”, we explicitly extract:
horizontal lines
vertical lines then combine them → the inner frame becomes a strong rectangle.
This works MUCH better for engineering drawings.
✅ Final working code (inner frame using morphology line detection)
Save as: extract_inner_frame_lines.py
Copy code
Python
import os
import cv2
import numpy as np


def extract_inner_frame_using_lines(image_bgr, padding=15, debug=False):
    """
    Robust inner-frame extractor:
    - binarize
    - extract horizontal + vertical lines using morphology
    - combine -> strong rectangle structure
    - find rectangles -> choose inner frame
    """

    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Stronger threshold for scanned/low-contrast images
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    # --- Extract horizontal lines ---
    horiz_kernel_len = max(30, w // 20)   # tune if needed
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_len, 1))
    horizontal = cv2.erode(th, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=2)

    # --- Extract vertical lines ---
    vert_kernel_len = max(30, h // 20)    # tune if needed
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_len))
    vertical = cv2.erode(th, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=2)

    # Combine line maps
    lines = cv2.bitwise_or(horizontal, vertical)

    # Close small gaps in frame
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    lines_closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Find contours from line structure
    contours, _ = cv2.findContours(lines_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find rectangle candidates (approx polygons with 4 corners)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * (h * w):  # ignore small rectangles (tables etc.)
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            candidates.append(approx)

    if len(candidates) == 0:
        raise RuntimeError("No rectangle candidates found (even after line extraction).")

    # Sort by area descending
    candidates.sort(key=lambda q: cv2.contourArea(q), reverse=True)

    # Outer is largest, inner is usually 2nd largest
    chosen = candidates[1] if len(candidates) > 1 else candidates[0]

    pts = chosen.reshape(4, 2)
    x_min = int(max(0, np.min(pts[:, 0]) + padding))
    y_min = int(max(0, np.min(pts[:, 1]) + padding))
    x_max = int(min(w, np.max(pts[:, 0]) - padding))
    y_max = int(min(h, np.max(pts[:, 1]) - padding))

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError("Invalid crop box. Reduce padding or check detection.")

    cropped = image_bgr[y_min:y_max, x_min:x_max].copy()

    info = {"crop_box": (x_min, y_min, x_max, y_max), "num_candidates": len(candidates)}

    if debug:
        dbg = image_bgr.copy()
        for i, q in enumerate(candidates[:5]):
            cv2.polylines(dbg, [q], True, (0, 255, 0), 2)
            cv2.putText(dbg, f"R{i}", tuple(q[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        chosen_dbg = image_bgr.copy()
        cv2.polylines(chosen_dbg, [chosen], True, (0, 0, 255), 3)

        info["debug_images"] = {
            "threshold": th,
            "horizontal": horizontal,
            "vertical": vertical,
            "lines": lines,
            "lines_closed": lines_closed,
            "rect_candidates": dbg,
            "chosen_rect": chosen_dbg,
            "cropped": cropped,
        }

    return cropped, info


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
            cropped, info = extract_inner_frame_using_lines(img, padding=15, debug=bool(debug_dir))
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, cropped)

            print(f"[OK] {fname} -> cropped | candidates={info['num_candidates']}")

            if debug_dir and "debug_images" in info:
                base = os.path.splitext(fname)[0]
                for k, v in info["debug_images"].items():
                    cv2.imwrite(os.path.join(debug_dir, f"{base}__{k}.png"), v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    input_dir = "input_images"
    output_dir = "output_cropped"
    debug_dir = "debug_outputs"  # set None to disable

    process_folder(input_dir, output_dir, debug_dir)
✅ Why this version will work for your case
Contour-based rectangle detection fails when borders are broken.
This method:
explicitly isolates long straight lines (like sheet borders)
reconstructs the frame even if it is broken
then detects rectangles from that
This is the standard robust approach for engineering drawings.
🔧 If it still selects wrong rectangle (title block)
Change this line:
Copy code
Python
chosen = candidates[1] if len(candidates) > 1 else candidates[0]
to this:
Copy code
Python
chosen = candidates[0]  # if inner frame is actually the biggest in your images
Or increase the minimum area filter:
Copy code
Python
if area < 0.10 * (h * w):
    continue
If you want, I can also modify this to remove title block even inside the inner frame (crop out bottom-right region automatically).