import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import json
import re
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point

# -----------------------------
# CONFIG
# -----------------------------
DPI = 500
MIN_OCR_CONF = 60
ARROW_MAX_AREA = 400
GEOM_SEARCH_RADIUS = 20  # pixels

DIM_REGEX = re.compile(
    r"(Ø?\d+(\.\d+)?(\s*[±+−]\s*\d+(\.\d+)?)?)"
)

# -----------------------------
# STEP 1: PDF → IMAGE
# -----------------------------
def rasterize_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# -----------------------------
# STEP 2: PREPROCESSING
# -----------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img, gray

# -----------------------------
# STEP 3: DETECT LINES
# -----------------------------
def detect_lines(bin_img):
    lines = cv2.HoughLinesP(
        bin_img, 1, np.pi / 180,
        threshold=120,
        minLineLength=40,
        maxLineGap=8
    )
    result = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            result.append(((x1, y1), (x2, y2)))
    return result

# -----------------------------
# STEP 4: DETECT CIRCLES (HOLES)
# -----------------------------
def detect_circles(gray):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=5
    )
    result = []
    if circles is not None:
        for c in circles[0]:
            result.append((int(c[0]), int(c[1]), int(c[2])))
    return result

# -----------------------------
# STEP 5: DETECT ARROWHEADS
# -----------------------------
def detect_arrowheads(bin_img):
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    arrows = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > ARROW_MAX_AREA:
            continue
        approx = cv2.approxPolyDP(
            c, 0.02 * cv2.arcLength(c, True), True
        )
        if len(approx) == 3:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                arrows.append((cx, cy))
    return arrows

# -----------------------------
# STEP 6: OCR TEXT
# -----------------------------
def ocr_dimensions(img):
    config = "--psm 6 -c tessedit_char_whitelist=0123456789.+-±Ø⌀"
    data = pytesseract.image_to_data(
        img, config=config, output_type=pytesseract.Output.DICT
    )
    dims = []
    for i, txt in enumerate(data["text"]):
        if int(data["conf"][i]) < MIN_OCR_CONF:
            continue
        txt = txt.strip()
        if DIM_REGEX.match(txt):
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            dims.append({
                "text": txt,
                "bbox": (x, y, w, h),
                "center": (x + w / 2, y + h / 2),
                "confidence": int(data["conf"][i]) / 100.0
            })
    return dims

# -----------------------------
# STEP 7: GEOMETRY ASSOCIATION
# -----------------------------
def associate_dimension(dim, arrowheads, circles):
    if not arrowheads:
        return None, 0.0

    arrow_tree = KDTree(arrowheads)
    dist, idx = arrow_tree.query(dim["center"])
    arrow = arrowheads[idx]

    if not circles:
        return None, 0.0

    circle_centers = [(c[0], c[1]) for c in circles]
    circle_tree = KDTree(circle_centers)
    cdist, cidx = circle_tree.query(arrow)

    if cdist < GEOM_SEARCH_RADIUS:
        return f"circle_{cidx}", max(0.0, 1.0 - cdist / GEOM_SEARCH_RADIUS)

    return None, 0.0

# -----------------------------
# MAIN INGESTION
# -----------------------------
def ingest(pdf_path):
    img = rasterize_pdf(pdf_path)
    bin_img, gray = preprocess(img)

    lines = detect_lines(bin_img)
    circles = detect_circles(gray)
    arrowheads = detect_arrowheads(bin_img)
    dimensions = ocr_dimensions(img)

    results = []
    for dim in dimensions:
        ref, geom_conf = associate_dimension(dim, arrowheads, circles)
        results.append({
            "raw_text": dim["text"],
            "nominal": dim["text"],
            "referenced_geometry": ref,
            "confidence": round(
                0.6 * dim["confidence"] + 0.4 * geom_conf, 2
            )
        })

    return {
        "dimensions": results,
        "num_lines": len(lines),
        "num_circles": len(circles),
        "num_arrowheads": len(arrowheads)
    }

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    output = ingest("input.pdf")
    with open("output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
