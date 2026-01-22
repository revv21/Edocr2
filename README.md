Yes ✅ — here’s a reliable arrowhead detector for engineering drawings (raster images).
It uses contours + triangle detection + filtering, which works well without training.
You’ll get:
arrowhead bounding boxes
arrowhead tip point (important for dimension matching)
a debug image showing detected arrowheads
✅ Install
Copy code
Bash
pip install opencv-python numpy
✅ detect_arrowheads.py (Contour + Triangle based)
Copy code
Python
import cv2
import numpy as np
import math


# -----------------------------
# Helpers
# -----------------------------
def angle(p1, p2, p3):
    """
    angle at p2 formed by p1-p2-p3 in degrees
    """
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cosv = np.clip(np.dot(a, b) / (na * nb), -1, 1)
    return math.degrees(math.acos(cosv))


def contour_centroid(cnt):
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def triangle_tip_point(tri_pts):
    """
    tri_pts: 3 points of triangle
    Tip is usually the sharpest vertex (smallest internal angle).
    """
    pts = [tuple(p) for p in tri_pts]
    angs = [
        angle(pts[1], pts[0], pts[2]),
        angle(pts[0], pts[1], pts[2]),
        angle(pts[0], pts[2], pts[1]),
    ]
    tip_idx = int(np.argmin(angs))
    return pts[tip_idx], angs[tip_idx]


# -----------------------------
# Main detector
# -----------------------------
def detect_arrowheads(
    img_bgr,
    min_area=30,
    max_area=2000,
    min_sharp_angle=15,
    max_sharp_angle=75,
    aspect_ratio_range=(0.3, 3.5),
    debug=False
):
    """
    Returns list of arrowheads:
    [
      {
        "bbox": (x,y,w,h),
        "tip": (tx,ty),
        "confidence": float (heuristic),
        "contour": cnt
      }, ...
    ]
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold works well for drawings
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Close small gaps in arrowheads
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    arrowheads = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0 or h == 0:
            continue

        aspect = w / h
        if not (aspect_ratio_range[0] <= aspect <= aspect_ratio_range[1]):
            continue

        # Approximate polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # Arrowhead is usually triangle-ish
        if len(approx) != 3:
            continue

        tri = approx.reshape(-1, 2)
        tip, sharp_angle = triangle_tip_point(tri)

        if not (min_sharp_angle <= sharp_angle <= max_sharp_angle):
            continue

        # Heuristic confidence: sharper tip + reasonable size
        conf = max(0.0, 1.0 - (sharp_angle / 90.0))

        arrowheads.append({
            "bbox": (x, y, w, h),
            "tip": tip,
            "confidence": float(conf),
            "contour": cnt
        })

    # Sort by confidence descending
    arrowheads = sorted(arrowheads, key=lambda a: a["confidence"], reverse=True)

    if debug:
        dbg = img_bgr.copy()
        for a in arrowheads:
            x, y, w, h = a["bbox"]
            tx, ty = a["tip"]
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(dbg, (tx, ty), 4, (0, 0, 255), -1)
            cv2.putText(
                dbg,
                f"{a['confidence']:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
        return arrowheads, bin_img, dbg

    return arrowheads, bin_img, None


# -----------------------------
# Run on image
# -----------------------------
if __name__ == "__main__":
    img_path = "input/example.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    arrowheads, bin_img, dbg = detect_arrowheads(img, debug=True)

    print("Arrowheads detected:", len(arrowheads))

    cv2.imwrite("arrowheads_binary.png", bin_img)
    if dbg is not None:
        cv2.imwrite("arrowheads_detected.png", dbg)

    print("✅ Saved: arrowheads_binary.png, arrowheads_detected.png")
✅ Output files
You’ll get:
arrowheads_binary.png → thresholded image used for contour detection
arrowheads_detected.png → arrowheads drawn with:
green bbox
red dot = tip point
🔧 Tuning (important)
If it detects too many triangles (noise), tighten:
Copy code
Python
min_area = 80
max_area = 1200
max_sharp_angle = 60
If it detects too few, relax:
Copy code
Python
min_area = 15
max_sharp_angle = 85
⭐ Best upgrade (recommended)
Once this works, we should run arrowhead detection only inside:
the detected DIM group crop
or near extension lines
That reduces false positives massively.
If you want, I can also provide: ✅ arrowhead detection directly from SVG segments (even cleaner)
because arrowheads become small triangular polylines in vector form.