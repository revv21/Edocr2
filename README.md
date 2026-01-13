Got it — that means your line-density thresholds are too aggressive, so it’s thinking the “template area” extends deeper into the page than it actually does.
✅ Fix = make the left/right/bottom detection less sensitive and more stable by:
Key changes
Use only the strongest “template separator lines” (not every small line)
Detect one dominant vertical separator on left and one on right
Detect one dominant horizontal separator at bottom
Add bounds so it can’t crop beyond reasonable limits
This works really well for standardized drawings like yours.
✅ Final Stable Version (Bounded + less aggressive)
What it guarantees
Left crop will never exceed 25% of width
Right crop will never exceed 25% of width
Bottom crop will never exceed 35% of height
Top crop is still detected last (as you wanted)
✅ Script: extract_red_crop_bounded.py
Copy code
Python
import os
import cv2
import numpy as np
import fitz


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


def compute_binary(gray):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35, 5
    )


def extract_vertical_lines(bin_img):
    H, W = bin_img.shape[:2]
    v_len = max(80, H // 10)  # longer -> only strong long borders survive
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v = cv2.erode(bin_img, v_kernel, iterations=1)
    v = cv2.dilate(v, v_kernel, iterations=2)
    return v


def extract_horizontal_lines(bin_img):
    H, W = bin_img.shape[:2]
    h_len = max(80, W // 10)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h = cv2.erode(bin_img, h_kernel, iterations=1)
    h = cv2.dilate(h, h_kernel, iterations=2)
    return h


def find_left_separator(vlines, W, max_left_ratio=0.25):
    """
    Find the strongest vertical line near the left side.
    """
    H = vlines.shape[0]
    col_sum = np.sum(vlines > 0, axis=0)
    col_ratio = col_sum / float(H)

    max_x = int(max_left_ratio * W)
    search = col_ratio[:max_x]

    # pick peak column (strongest vertical separator)
    x = int(np.argmax(search))
    return x


def find_right_separator(vlines, W, max_right_ratio=0.25):
    """
    Find the strongest vertical line near the right side.
    """
    H = vlines.shape[0]
    col_sum = np.sum(vlines > 0, axis=0)
    col_ratio = col_sum / float(H)

    min_x = int((1.0 - max_right_ratio) * W)
    search = col_ratio[min_x:]

    x = int(np.argmax(search)) + min_x
    return x


def find_bottom_separator(hlines, H, max_bottom_ratio=0.35):
    """
    Find strongest horizontal line near bottom.
    """
    W = hlines.shape[1]
    row_sum = np.sum(hlines > 0, axis=1)
    row_ratio = row_sum / float(W)

    min_y = int((1.0 - max_bottom_ratio) * H)
    search = row_ratio[min_y:]

    y = int(np.argmax(search)) + min_y
    return y


def find_top_separator(hlines_roi, max_top_ratio=0.20):
    """
    Find strongest horizontal line near top inside ROI.
    """
    H, W = hlines_roi.shape[:2]
    row_sum = np.sum(hlines_roi > 0, axis=1)
    row_ratio = row_sum / float(W)

    max_y = int(max_top_ratio * H)
    search = row_ratio[:max_y]

    y = int(np.argmax(search))
    return y


def extract_red_boundary_crop(image_bgr, pad_lr=15, pad_bottom=15, pad_top=10, debug=False):
    """
    Less aggressive cropping:
    - Find ONE separator line left, right, bottom
    - Crop inside them with small padding
    - Then crop top inside ROI
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    bin_img = compute_binary(gray)
    vlines = extract_vertical_lines(bin_img)
    hlines = extract_horizontal_lines(bin_img)

    # Stage 1: left + right separators
    left_sep = find_left_separator(vlines, W, max_left_ratio=0.25)
    right_sep = find_right_separator(vlines, W, max_right_ratio=0.25)

    # ensure ordering
    if right_sep <= left_sep:
        raise RuntimeError("Bad left/right separators. Check line extraction.")

    x1 = max(0, left_sep + pad_lr)
    x2 = min(W, right_sep - pad_lr)

    # Stage 2: bottom separator
    bottom_sep = find_bottom_separator(hlines, H, max_bottom_ratio=0.35)
    y2 = min(H, bottom_sep - pad_bottom)

    if x2 <= x1 or y2 <= 0:
        raise RuntimeError("Invalid crop after L/R/B stage. Reduce padding.")

    # Stage 3: top separator inside ROI
    roi_hlines = hlines[:y2, x1:x2]
    top_sep = find_top_separator(roi_hlines, max_top_ratio=0.20)

    y1 = max(0, top_sep + pad_top)

    if y2 <= y1:
        raise RuntimeError("Invalid crop after top stage. Reduce pad_top.")

    cropped = image_bgr[y1:y2, x1:x2].copy()

    info = {
        "crop_box": (x1, y1, x2, y2),
        "left_sep": left_sep,
        "right_sep": right_sep,
        "bottom_sep": bottom_sep,
        "top_sep": top_sep,
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
            cropped, info = extract_red_boundary_crop(img, debug=bool(debug_dir))

            cv2.imwrite(os.path.join(out_dir, f"{base}_extracted.png"), cropped)
            print(f"[OK] {fname} -> crop={info['crop_box']}")

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
✅ If left/right/bottom are STILL too much
Reduce padding first:
Copy code
Python
cropped, info = extract_red_boundary_crop(
    img,
    pad_lr=5,
    pad_bottom=5,
    pad_top=10
)
And if it’s still too much, reduce the “max crop bounds”:
Left bound from 0.25 → 0.20
Right bound from 0.25 → 0.20
Bottom bound from 0.35 → 0.30
Why this version won’t overcrop
Instead of “everything above threshold”, we pick the single strongest separator line in a limited region.
So it won’t accidentally interpret internal drawing lines as template boundary.
If you want, I can add a debug image that draws the detected separator lines (left/right/bottom/top) so you can see exactly what it’s locking onto.