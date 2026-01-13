Perfect — if they’re proper PDFs converted to images, then the #1 reason all these methods fail is the conversion DPI is too low (inner border becomes 1–2 px thin and breaks).
So the best fix is:
✅ Convert PDF → image at 400–600 DPI
✅ Then run Hough-line inner frame detection
Below is a final end-to-end code that does BOTH:
✅ Final Solution (PDF → High DPI image → inner frame crop)
1) Install
Copy code
Bash
pip install opencv-python numpy pymupdf
✅ Script: pdf_crop_inner_frame.py
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
    """
    Convert a PDF page to a high-resolution BGR image using PyMuPDF.
    dpi=400-600 is recommended for engineering drawings.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Convert to BGR for OpenCV
    if pix.n == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img_bgr


# ----------------------------
# Inner frame using Hough lines
# ----------------------------
def extract_inner_frame_hough(image_bgr, padding=15, debug=False):
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Light blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Strong edges
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Hough line detection
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=int(0.30 * min(H, W)),
        maxLineGap=40
    )

    if lines is None:
        raise RuntimeError("No Hough lines detected. Try increasing DPI or lowering Hough threshold.")

    vertical = []
    horizontal = []

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # horizontal
        if dy < 12 and dx > 0.35 * W:
            horizontal.append((x1, y1, x2, y2))

        # vertical
        if dx < 12 and dy > 0.35 * H:
            vertical.append((x1, y1, x2, y2))

    if len(vertical) < 2 or len(horizontal) < 2:
        raise RuntimeError(f"Not enough border lines found (v={len(vertical)}, h={len(horizontal)}).")

    vx = sorted([int((l[0] + l[2]) / 2) for l in vertical])
    hy = sorted([int((l[1] + l[3]) / 2) for l in horizontal])

    # Pick inner-most extremes but avoid absolute page edge
    def pick_inner_extremes(vals, size):
        margin = int(0.02 * size)
        filtered = [v for v in vals if margin < v < (size - margin)]
        if len(filtered) < 2:
            filtered = vals
        return filtered[0], filtered[-1]

    left_x, right_x = pick_inner_extremes(vx, W)
    top_y, bottom_y = pick_inner_extremes(hy, H)

    left_x, right_x = min(left_x, right_x), max(left_x, right_x)
    top_y, bottom_y = min(top_y, bottom_y), max(top_y, bottom_y)

    x1 = max(0, left_x + padding)
    x2 = min(W, right_x - padding)
    y1 = max(0, top_y + padding)
    y2 = min(H, bottom_y - padding)

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid crop box. Reduce padding or check detected lines.")

    cropped = image_bgr[y1:y2, x1:x2].copy()

    info = {
        "crop_box": (x1, y1, x2, y2),
        "num_lines": len(lines),
        "num_vertical": len(vertical),
        "num_horizontal": len(horizontal),
    }

    if debug:
        dbg = image_bgr.copy()
        for (x1l, y1l, x2l, y2l) in vertical:
            cv2.line(dbg, (x1l, y1l), (x2l, y2l), (0, 255, 0), 2)
        for (x1l, y1l, x2l, y2l) in horizontal:
            cv2.line(dbg, (x1l, y1l), (x2l, y2l), (255, 0, 0), 2)

        chosen_dbg = image_bgr.copy()
        cv2.rectangle(chosen_dbg, (x1, y1), (x2, y2), (0, 0, 255), 3)

        info["debug_images"] = {
            "edges": edges,
            "all_lines": dbg,
            "chosen_rect": chosen_dbg,
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

        try:
            img = pdf_page_to_image(pdf_path, page_number=0, dpi=dpi)
            cropped, info = extract_inner_frame_hough(img, padding=20, debug=bool(debug_dir))

            base = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{base}_cropped.png")
            cv2.imwrite(out_path, cropped)

            print(f"[OK] {fname} -> cropped | v={info['num_vertical']} h={info['num_horizontal']}")

            if debug_dir and "debug_images" in info:
                for k, v in info["debug_images"].items():
                    cv2.imwrite(os.path.join(debug_dir, f"{base}__{k}.png"), v)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")


if __name__ == "__main__":
    pdf_dir = "input_pdfs"
    out_dir = "output_cropped"
    debug_dir = "debug_outputs"  # set None to disable

    process_pdf_folder(pdf_dir, out_dir, dpi=500, debug_dir=debug_dir)
✅ How to run
Folder layout
Copy code

project/
  pdf_crop_inner_frame.py
  input_pdfs/
    drawing1.pdf
    drawing2.pdf
Run
Copy code
Bash
python pdf_crop_inner_frame.py
Outputs:
output_cropped/*.png → cropped inner drawing region
debug_outputs/* → debug overlays + edges
🔧 Tuning (very important)
If it fails on some PDFs:
If no lines detected
Increase DPI:
Copy code
Python
dpi=600
If it detects only short lines
Reduce minLineLength:
Copy code
Python
minLineLength=int(0.20 * min(H, W))
If inner border is faint
Increase dilation:
Copy code
Python
edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
If you want the BEST possible reliability
The next upgrade is: ✅ detect outer border first, then search inside that ROI for the inner border
(this avoids confusing title blocks / tables)
If you want, I’ll give that “two-stage border detection” version too.