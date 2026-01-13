Ahh got it — that changes everything 👍
If the actual drawing region is a closed polygon (not a rectangle), then any “inner rectangle border” method will fail / be unstable, because there simply isn’t a consistent inner rectangle to detect.
In that case, the best classical approach is:
✅ Detect the biggest closed polygon contour (the drawing boundary)
→ then crop to its bounding box OR mask everything outside it
This works even when:
template size changes
drawing boundary is irregular
title block exists
border lines are not rectangular
✅ Best Approach for Closed Polygon Drawing Boundary
Pipeline
PDF → high DPI image
Convert to grayscale + threshold
Morphological close (connect broken boundary lines)
Find contours
Pick the largest closed contour that is NOT the page border
Crop / mask using that contour
✅ Final End-to-End Code (PDF → detect polygon → extract only drawing)
Install
Copy code
Bash
pip install opencv-python numpy pymupdf
✅ Script: extract_polygon_drawing.py
Copy code
Python
import os
import cv2
import numpy as np
import fitz  # PyMuPDF


# ----------------------------
# PDF -> Image (High DPI)
# ----------------------------
def pdf_page_to_image(pdf_path, page_number=0, dpi=500):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


# ----------------------------
# Polygon extraction
# ----------------------------
def extract_largest_polygon_region(image_bgr, padding=10, debug=False):
    """
    Extract the largest closed polygon region (drawing area) excluding page border.
    Returns:
      - cropped image (bounding box crop)
      - masked image (everything outside polygon removed)
    """

    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Improve contrast (important for light template lines)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive threshold -> lines become white (foreground)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    # Close gaps so polygon becomes a single closed contour
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found. Try increasing DPI or morphology kernel.")

    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_area = H * W

    chosen = None
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Skip tiny things
        if area < 0.05 * page_area:
            continue

        # Skip page outer border (usually almost full page)
        if area > 0.95 * page_area:
            continue

        chosen = cnt
        break

    if chosen is None:
        # fallback: just take 2nd largest contour
        chosen = contours[1] if len(contours) > 1 else contours[0]

    # Approx polygon (smooth contour)
    peri = cv2.arcLength(chosen, True)
    approx = cv2.approxPolyDP(chosen, 0.01 * peri, True)

    # Bounding box crop
    x, y, w, h = cv2.boundingRect(approx)
    x1 = max(0, x + padding)
    y1 = max(0, y + padding)
    x2 = min(W, x + w - padding)
    y2 = min(H, y + h - padding)

    cropped = image_bgr[y1:y2, x1:x2].copy()

    # Mask everything outside polygon
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)

    masked_full = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    masked_crop = masked_full[y1:y2, x1:x2].copy()

    info = {
        "bbox": (x1, y1, x2, y2),
        "polygon_points": len(approx),
        "chosen_area_ratio": float(cv2.contourArea(chosen) / page_area)
    }

    if debug:
        dbg = image_bgr.copy()
        cv2.drawContours(dbg, [approx], -1, (0, 0, 255), 3)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)

        info["debug_images"] = {
            "threshold": th,
            "closed": closed,
            "chosen_polygon": dbg,
            "mask": mask,
            "masked_full": masked_full,
            "cropped_bbox": cropped,
            "masked_crop": masked_crop,
        }

    return cropped, masked_crop, info


# ----------------------------
# Batch PDF folder
# ----------------------------
def process_pdf_folder(pdf_dir, out_dir, dpi=500, debug_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, fname)
        base = os.path.splitext(fname)[0]

        try:
            img = pdf_page_to_image(pdf_path, page_number=0, dpi=dpi)

            bbox_crop, masked_crop, info = extract_largest_polygon_region(
                img, padding=10, debug=bool(debug_dir)
            )

            # Save both versions
            cv2.imwrite(os.path.join(out_dir, f"{base}_bbox_crop.png"), bbox_crop)
            cv2.imwrite(os.path.join(out_dir, f"{base}_masked_crop.png"), masked_crop)

            print(f"[OK] {fname} -> polygon_pts={info['polygon_points']} area_ratio={info['chosen_area_ratio']:.3f}")

            if debug_dir and "debug_images" in info:
                for k, v in info["debug_images"].items():
                    cv2.imwrite(os.path.join(debug_dir, f"{base}__{k}.png"), v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    pdf_dir = "input_pdfs"
    out_dir = "output_extracted"
    debug_dir = "debug_outputs"  # set None to disable

    process_pdf_folder(pdf_dir, out_dir, dpi=500, debug_dir=debug_dir)
✅ Output Files
For each PDF you’ll get:
*_bbox_crop.png
➡️ Cropped bounding box around polygon (fast + easy)
*_masked_crop.png
➡️ Cleanest output: template removed because everything outside polygon is black/transparent-ish
🔧 Tuning knobs (important)
If it picks the wrong polygon:
Increase minimum contour size
Change:
Copy code
Python
if area < 0.05 * page_area:
to:
Copy code
Python
if area < 0.15 * page_area:
If polygon boundary is broken
Increase closing kernel:
Copy code
Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
⭐ Best practice recommendation
For production use: ✅ use masked_crop output
because even if the bounding box includes some template area, the mask removes it.
If you want, I can also modify this to output a transparent PNG (template fully removed with alpha channel).