Yep — in your example the outer sheet border is a perfect strong rectangle, so any “largest contour / strongest rectangle” logic will always lock onto it first.
To avoid the outer border, you need to do 2-stage detection:
✅ Stage 1: detect the outer border rectangle
✅ Stage 2: remove/ignore a margin band near the outer border
✅ Stage 3: search for the next largest closed polygon/contour inside (your drawing region)
This guarantees you don’t pick the sheet border.
✅ Best Fix for Your Case (Outer border ignored)
What this code does
Convert PDF → high DPI image
Threshold
Detect outer border rectangle
Create an “inner search ROI” by shrinking inside that border
Find the largest closed contour inside ROI
Crop to that contour’s bounding box
✅ Final Working Code (PDF → ignore outer border → extract inner drawing polygon)
Install
Copy code
Bash
pip install opencv-python numpy pymupdf
Script: extract_drawing_ignore_outer_border.py
Copy code
Python
import os
import cv2
import numpy as np
import fitz  # PyMuPDF


# ----------------------------
# PDF -> Image
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
# Helpers
# ----------------------------
def find_outer_border_rect(binary_img, min_area_ratio=0.80):
    """
    Find the outer sheet border rectangle.
    binary_img should have lines as white (255).
    Returns bounding box (x1,y1,x2,y2) of outer border.
    """
    H, W = binary_img.shape[:2]
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    page_area = H * W
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * page_area:
            continue

        # Approx polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

        # Outer border usually is a 4-corner-ish contour
        x, y, w, h = cv2.boundingRect(approx)
        return (x, y, x + w, y + h)

    # fallback: take largest contour bbox
    x, y, w, h = cv2.boundingRect(contours[0])
    return (x, y, x + w, y + h)


def extract_largest_inner_contour(binary_img, roi_box, min_area_ratio=0.02):
    """
    Search for largest contour inside ROI.
    """
    x1, y1, x2, y2 = roi_box
    roi = binary_img[y1:y2, x1:x2]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    roi_area = (y2 - y1) * (x2 - x1)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * roi_area:
            continue
        return cnt, (x1, y1)  # contour + offset

    return contours[0], (x1, y1)


# ----------------------------
# Main extraction
# ----------------------------
def extract_drawing_region(image_bgr, outer_shrink_ratio=0.06, padding=10, debug=False):
    """
    1) Threshold
    2) Find outer border bbox
    3) Shrink bbox to ignore template border region
    4) Find largest contour inside -> drawing region polygon
    5) Crop
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to make lines white
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )

    # Connect broken border lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Outer border bbox
    outer = find_outer_border_rect(closed, min_area_ratio=0.70)
    if outer is None:
        raise RuntimeError("Outer border not found.")

    ox1, oy1, ox2, oy2 = outer

    # Shrink inside outer border so we never select it again
    shrink_x = int((ox2 - ox1) * outer_shrink_ratio)
    shrink_y = int((oy2 - oy1) * outer_shrink_ratio)

    ix1 = min(W - 1, ox1 + shrink_x)
    iy1 = min(H - 1, oy1 + shrink_y)
    ix2 = max(0, ox2 - shrink_x)
    iy2 = max(0, oy2 - shrink_y)

    if ix2 <= ix1 or iy2 <= iy1:
        raise RuntimeError("Invalid inner ROI after shrinking. Reduce outer_shrink_ratio.")

    # Find largest contour inside inner ROI
    result = extract_largest_inner_contour(closed, (ix1, iy1, ix2, iy2), min_area_ratio=0.02)
    if result is None:
        raise RuntimeError("No inner drawing contour found inside ROI.")

    cnt, (offx, offy) = result

    # Move contour back to full-image coordinates
    cnt_full = cnt + np.array([[[offx, offy]]], dtype=np.int32)

    # Approx polygon (smooth)
    peri = cv2.arcLength(cnt_full, True)
    approx = cv2.approxPolyDP(cnt_full, 0.01 * peri, True)

    # Crop bbox of polygon
    x, y, w, h = cv2.boundingRect(approx)
    x1 = max(0, x + padding)
    y1 = max(0, y + padding)
    x2 = min(W, x + w - padding)
    y2 = min(H, y + h - padding)

    cropped = image_bgr[y1:y2, x1:x2].copy()

    info = {
        "outer_border_box": outer,
        "inner_roi_box": (ix1, iy1, ix2, iy2),
        "final_crop_box": (x1, y1, x2, y2),
        "polygon_points": len(approx),
    }

    if debug:
        dbg = image_bgr.copy()

        # Draw outer border (blue)
        cv2.rectangle(dbg, (ox1, oy1), (ox2, oy2), (255, 0, 0), 4)

        # Draw inner ROI (green)
        cv2.rectangle(dbg, (ix1, iy1), (ix2, iy2), (0, 255, 0), 3)

        # Draw chosen polygon (red)
        cv2.drawContours(dbg, [approx], -1, (0, 0, 255), 3)

        info["debug_images"] = {
            "threshold": th,
            "closed": closed,
            "debug_overlay": dbg,
            "cropped": cropped,
        }

    return cropped, info


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
            cropped, info = extract_drawing_region(
                img,
                outer_shrink_ratio=0.06,  # increase if it still picks outer border
                padding=10,
                debug=bool(debug_dir),
            )

            cv2.imwrite(os.path.join(out_dir, f"{base}_drawing.png"), cropped)
            print(f"[OK] {fname} -> extracted | polygon_pts={info['polygon_points']}")

            if debug_dir and "debug_images" in info:
                for k, v in info["debug_images"].items():
                    cv2.imwrite(os.path.join(debug_dir, f"{base}__{k}.png"), v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    pdf_dir = "input_pdfs"
    out_dir = "output_extracted"
    debug_dir = "debug_outputs"  # set None to disable debug outputs

    process_pdf_folder(pdf_dir, out_dir, dpi=500, debug_dir=debug_dir)
✅ What you should tweak for your template
If it STILL picks the outer border:
Increase shrink:
Copy code
Python
outer_shrink_ratio=0.10
If it crops too much inside:
Reduce shrink:
Copy code
Python
outer_shrink_ratio=0.04
If it fails to find inner contour:
Increase morphology:
Copy code
Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
⭐ Why this works on your sample
Your sheet has:
a very strong outer border
a strong template on left/bottom
drawing in the middle
By shrinking inside the outer border first, we force the algorithm to look for the next biggest closed region, which will be your drawing polygon.
If you want, I can also modify this to output a transparent PNG mask (template fully removed, only drawing visible).