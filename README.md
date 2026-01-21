Sure ✅ — here is clean, runnable Python code to detect:
Straight lines (HoughLinesP)
Circles (HoughCircles)
Curves / arcs (contour extraction + polyline approximation)
This works on raster engineering drawings (PNG/JPG from PDF).
✅ Install requirements
Copy code
Bash
pip install opencv-python numpy
✅ detect_geometry.py (Lines + Circles + Curves)
Copy code
Python
import cv2
import numpy as np


# -----------------------------
# Preprocessing (important)
# -----------------------------
def preprocess(img_bgr):
    """
    Returns a clean binary image suitable for line/circle detection.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Remove small noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold works well for scanned drawings
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Remove tiny speckles
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)

    return gray, bin_img


# -----------------------------
# Line Detection (HoughLinesP)
# -----------------------------
def detect_lines(bin_img):
    """
    Detect straight line segments using probabilistic Hough transform.
    Returns list of segments [(x1,y1,x2,y2), ...]
    """
    edges = cv2.Canny(bin_img, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=40,
        maxLineGap=8
    )

    segments = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            segments.append((x1, y1, x2, y2))

    return segments


# -----------------------------
# Circle Detection (HoughCircles)
# -----------------------------
def detect_circles(gray_img):
    """
    Detect circles (full circles). Good for holes, bolt circles, etc.
    Returns list of circles [(x,y,r), ...]
    """
    # HoughCircles expects blurred grayscale
    blur = cv2.medianBlur(gray_img, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=120,
        param2=35,
        minRadius=3,
        maxRadius=300
    )

    results = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        for x, y, r in circles:
            results.append((x, y, r))

    return results


# -----------------------------
# Curve / Arc Detection (Contours)
# -----------------------------
def detect_curves(bin_img):
    """
    Extract curved contours (arcs/splines) by finding contours and approximating them.
    Returns list of polylines (each polyline is Nx2 array of points).
    """
    contours, _ = cv2.findContours(
        bin_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    polylines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:  # ignore tiny noise
            continue

        # Approximate contour to reduce points
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # approx is a polygon; curves will have many vertices
        if len(approx) >= 8:
            poly = approx.reshape(-1, 2)
            polylines.append(poly)

    return polylines


# -----------------------------
# Visualization
# -----------------------------
def draw_results(img_bgr, lines, circles, curves):
    out = img_bgr.copy()

    # Draw lines
    for (x1, y1, x2, y2) in lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw circles
    for (x, y, r) in circles:
        cv2.circle(out, (x, y), r, (255, 0, 0), 2)
        cv2.circle(out, (x, y), 2, (255, 0, 0), 3)

    # Draw curves (polylines)
    for poly in curves:
        cv2.polylines(out, [poly.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    return out


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    img_path = "input/example.jpg"
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Could not read: {img_path}")

    gray, bin_img = preprocess(img)

    lines = detect_lines(bin_img)
    circles = detect_circles(gray)
    curves = detect_curves(bin_img)

    print(f"Lines detected: {len(lines)}")
    print(f"Circles detected: {len(circles)}")
    print(f"Curves detected: {len(curves)}")

    vis = draw_results(img, lines, circles, curves)
    cv2.imwrite("geometry_detected.png", vis)

    print("✅ Saved output: geometry_detected.png")
✅ Output
You will get:
geometry_detected.png
with:
🟩 green = straight lines
🟦 blue = circles
🟥 red = curve contours
⚠️ Notes (important for engineering drawings)
Why curves detection is “approximate”
In raster drawings, arcs are usually part of contours.
So we detect them as curved polylines, not perfect analytic arcs.
If you want true arc parameters (center, radius, start angle, end angle), we can fit circles to contour segments using least squares.
Next (if you want)
I can upgrade this into a robust geometry extractor that returns:
merged collinear line segments
deduplicated circles
fitted arcs with (cx, cy, r, θ1, θ2)
Just say “yes fit arcs too”.