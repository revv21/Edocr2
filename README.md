Yes — that’s exactly the right way to do it ✅
Your “red boundary” is basically:
Cut LEFT (remove template column)
Cut RIGHT (remove right template strip)
Cut BOTTOM (remove title block)
Then cut TOP (only slightly, and only for the remaining region)
So we should not use contours/polygons at all.
We should do a 4-sided crop detection using line-density / projection profiles, but in a staged way:
✅ Best Approach (matches your requirement)
Stage 1: Detect left + right crop from vertical line density
Stage 2: Detect bottom crop from horizontal line density
Stage 3: After cropping L/R/B, detect top crop inside remaining ROI
This avoids the top header strip confusing the left/right detection.
✅ Final Working Code (4-side staged crop)
Install
Copy code
Bash
pip install opencv-python numpy pymupdf
Script: extract_red_boundary_crop.py
Copy code
Python
import os
import cv2
import numpy as np
import fitz


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
# Projection helpers
# ----------------------------
def compute_binary_lines(gray):
    """
    Make a binary image where lines/text are white.
    """
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )
    return th


def extract_vertical_lines(bin_img):
    H, W = bin_img.shape[:2]
    v_len = max(50, H // 15)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v = cv2.erode(bin_img, v_kernel, iterations=1)
    v = cv2.dilate(v, v_kernel, iterations=2)
    return v


def extract_horizontal_lines(bin_img):
    H, W = bin_img.shape[:2]
    h_len = max(50, W // 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h = cv2.erode(bin_img, h_kernel, iterations=1)
    h = cv2.dilate(h, h_kernel, iterations=2)
    return h


def find_left_cut(vertical_lines, threshold=0.10, max_search_ratio=0.40):
    """
    Find where the left template ends.
    threshold is a ratio of active pixels per column.
    """
    H, W = vertical_lines.shape[:2]
    col_sum = np.sum(vertical_lines > 0, axis=0)
    col_ratio = col_sum / float(H)

    max_x = int(max_search_ratio * W)
    left_end = 0

    for x in range(max_x):
        if col_ratio[x] > threshold:
            left_end = x

    return left_end


def find_right_cut(vertical_lines, threshold=0.10, max_search_ratio=0.40):
    """
    Find where the right template starts.
    """
    H, W = vertical_lines.shape[:2]
    col_sum = np.sum(vertical_lines > 0, axis=0)
    col_ratio = col_sum / float(H)

    min_x = int((1.0 - max_search_ratio) * W)
    right_start = W - 1

    for x in range(W - 1, min_x, -1):
        if col_ratio[x] > threshold:
            right_start = x

    return right_start


def find_bottom_cut(horizontal_lines, threshold=0.08, max_search_ratio=0.50):
    """
    Find where the bottom title block starts.
    """
    H, W = horizontal_lines.shape[:2]
    row_sum = np.sum(horizontal_lines > 0, axis=1)
    row_ratio = row_sum / float(W)

    min_y = int((1.0 - max_search_ratio) * H)
    bottom_start = H - 1

    for y in range(H - 1, min_y, -1):
        if row_ratio[y] > threshold:
            bottom_start = y

    return bottom_start


def find_top_cut(horizontal_lines_roi, threshold=0.05, max_search_ratio=0.20):
    """
    Find where top header ends INSIDE the already cropped ROI.
    """
    H, W = horizontal_lines_roi.shape[:2]
    row_sum = np.sum(horizontal_lines_roi > 0, axis=1)
    row_ratio = row_sum / float(W)

    max_y = int(max_search_ratio * H)
    top_end = 0

    for y in range(max_y):
        if row_ratio[y] > threshold:
            top_end = y

    return top_end


# ----------------------------
# Main staged crop
# ----------------------------
def extract_red_boundary_crop(image_bgr, padding=20, debug=False):
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    bin_img = compute_binary_lines(gray)

    # Extract strong vertical/horizontal template lines
    vlines = extract_vertical_lines(bin_img)
    hlines = extract_horizontal_lines(bin_img)

    # ---- Stage 1: LEFT + RIGHT crop ----
    left_end = find_left_cut(vlines, threshold=0.10, max_search_ratio=0.45)
    right_start = find_right_cut(vlines, threshold=0.10, max_search_ratio=0.45)

    x1 = max(0, left_end + padding)
    x2 = min(W, right_start - padding)

    if x2 <= x1:
        raise RuntimeError("Invalid left/right crop. Tune thresholds or padding.")

    # ---- Stage 2: BOTTOM crop (using full width, but we can also use cropped width) ----
    bottom_start = find_bottom_cut(hlines, threshold=0.08, max_search_ratio=0.55)
    y2 = min(H, bottom_start - padding)

    if y2 <= 0:
        raise RuntimeError("Invalid bottom crop. Tune threshold/padding.")

    # ---- Stage 3: TOP crop (only inside the ROI after L/R/B crop) ----
    roi_hlines = hlines[0:y2, x1:x2]
    top_end = find_top_cut(roi_hlines, threshold=0.05, max_search_ratio=0.25)

    y1 = max(0, top_end + padding)

    if y2 <= y1:
        raise RuntimeError("Invalid top crop. Tune top threshold/padding.")

    cropped = image_bgr[y1:y2, x1:x2].copy()

    info = {
        "crop_box": (x1, y1, x2, y2),
        "left_end": left_end,
        "right_start": right_start,
        "bottom_start": bottom_start,
        "top_end": top_end,
    }

    if debug:
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 4)

        info["debug_images"] = {
            "binary": bin_img,
            "vlines": vlines,
            "hlines": hlines,
            "overlay": overlay,
            "cropped": cropped,
        }

    return cropped, info


# ----------------------------
# Batch processing PDFs
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
            cropped, info = extract_red_boundary_crop(img, padding=20, debug=bool(debug_dir))

            out_path = os.path.join(out_dir, f"{base}_extracted.png")
            cv2.imwrite(out_path, cropped)

            print(f"[OK] {fname} -> crop={info['crop_box']}")

            if debug_dir and "debug_images" in info:
                for k, v in info["debug_images"].items():
                    cv2.imwrite(os.path.join(debug_dir, f"{base}__{k}.png"), v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    pdf_dir = "input_pdfs"
    out_dir = "output_extracted"
    debug_dir = "debug_outputs"  # set None to disable debug

    process_pdf_folder(pdf_dir, out_dir, dpi=500, debug_dir=debug_dir)
🔧 Tuning knobs (you will likely adjust these once)
Inside extract_red_boundary_crop():
If left crop is not enough:
Copy code
Python
left_end = find_left_cut(vlines, threshold=0.08, max_search_ratio=0.50)
If right crop is not enough:
Copy code
Python
right_start = find_right_cut(vlines, threshold=0.08, max_search_ratio=0.50)
If bottom crop is not enough:
Copy code
Python
bottom_start = find_bottom_cut(hlines, threshold=0.06, max_search_ratio=0.60)
If top crop removes too much:
Copy code
Python
top_end = find_top_cut(roi_hlines, threshold=0.07, max_search_ratio=0.20)
✅ Why this will match your red crop
Because it does exactly what you described:
Left/Right/Bottom cropping using global template lines
Then Top crop only after the other 3 sides are fixed
So the top strip doesn’t “pollute” the left/right decision.
If you want, I can also make this return the crop in PDF coordinates so you can crop the PDF directly (without converting to image).