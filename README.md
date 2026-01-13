Got it — that’s happening because the outer sheet border is the largest rectangle, so the “largest-rectangle contour” logic will always pick it.
To get the inner drawing frame, you need to explicitly detect rectangles and choose the “second-largest” (or best inner) rectangle, not the largest.
Below is a drop-in improved version that:
✅ Detects multiple rectangular candidates
✅ Sorts by area
✅ Picks the best inner frame (usually the 2nd largest)
✅ Still works even if sizes change
✅ Fix: detect multiple rectangles and pick inner one
Replace your rectangle-finding logic with this
Copy code
Python
import cv2
import numpy as np


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def rect_area(quad):
    quad = quad.reshape(4, 2)
    x = quad[:, 0]
    y = quad[:, 1]
    return float(cv2.contourArea(np.array([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]], dtype=np.float32)))


def find_rect_candidates(binary_img, min_area_ratio=0.05):
    """
    Returns a list of quadrilateral contours sorted by area descending.
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


def pick_inner_frame(quads, img_shape):
    """
    Strategy:
    - quads[0] is usually outer border
    - quads[1] is usually inner drawing frame
    But sometimes extra rectangles exist, so we pick a quad that:
      - is smaller than outer
      - still reasonably large
      - centered-ish
    """
    if len(quads) == 0:
        return None
    if len(quads) == 1:
        return quads[0]

    h, w = img_shape[:2]
    outer = quads[0]
    outer_area = cv2.contourArea(outer)

    best = None
    best_score = -1

    for q in quads[1:]:
        area = cv2.contourArea(q)

        # must be significantly smaller than outer border
        if area > 0.95 * outer_area:
            continue

        # must still be big enough to be the drawing frame
        if area < 0.20 * (h * w):
            continue

        pts = order_points(q.reshape(4, 2))
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        # score = area - penalty for being far from center
        center_penalty = ((cx - w/2)**2 + (cy - h/2)**2) / (w*w + h*h)
        score = (area / (h * w)) - 0.3 * center_penalty

        if score > best_score:
            best_score = score
            best = q

    # fallback: 2nd largest quad
    if best is None:
        best = quads[1]

    return best
✅ Updated extraction function (full working)
Replace your extract_drawing_area() with this version:
Copy code
Python
def extract_inner_drawing_frame(image_bgr, padding=10, debug=False):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    th = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

    # IMPORTANT: we want to connect border lines strongly
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find rectangle candidates
    quads = find_rect_candidates(closed, min_area_ratio=0.03)

    if len(quads) == 0:
        raise RuntimeError("No rectangle candidates found.")

    inner = pick_inner_frame(quads, image_bgr.shape)
    if inner is None:
        raise RuntimeError("Could not select inner frame rectangle.")

    pts = order_points(inner.reshape(4, 2))

    x_min = int(max(0, np.min(pts[:, 0]) + padding))
    y_min = int(max(0, np.min(pts[:, 1]) + padding))
    x_max = int(min(image_bgr.shape[1], np.max(pts[:, 0]) - padding))
    y_max = int(min(image_bgr.shape[0], np.max(pts[:, 1]) - padding))

    cropped = image_bgr[y_min:y_max, x_min:x_max].copy()

    debug_images = {}
    if debug:
        dbg = image_bgr.copy()
        for i, q in enumerate(quads[:5]):  # show top 5 rectangles
            p = order_points(q.reshape(4, 2)).astype(np.int32)
            cv2.polylines(dbg, [p], True, (0, 255, 0), 2)
            cv2.putText(dbg, f"#{i}", tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        inner_dbg = image_bgr.copy()
        cv2.polylines(inner_dbg, [pts.astype(np.int32)], True, (0, 0, 255), 3)

        debug_images = {
            "threshold": th,
            "closed": closed,
            "rect_candidates": dbg,
            "chosen_inner_rect": inner_dbg,
            "cropped": cropped
        }

    return cropped, {"crop_box": (x_min, y_min, x_max, y_max), "debug": debug_images}
✅ How to use it in your folder pipeline
Inside your loop, call:
Copy code
Python
cropped, info = extract_inner_drawing_frame(img, padding=15, debug=True)
cv2.imwrite(out_path, cropped)
Why this works
Instead of “largest rectangle = frame”, we do:
detect all rectangles
assume largest = outer sheet border
choose the best inner rectangle based on:
smaller than outer
still large enough
roughly centered
fallback = 2nd largest
This matches typical engineering drawings: Outer border (sheet) > Inner border (drawing area) > Title blocks (smaller rectangles)
If it still fails on some files
These are the 2 most common reasons:
1) Inner border is faint/broken
Increase morphology strength:
Copy code
Python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
2) Title block is being selected as “inner”
Increase minimum area threshold in pick_inner_frame:
Copy code
Python
if area < 0.30 * (h * w):
    continue
If you want, I can also give you an even more stable method that explicitly detects two rectangles by finding long horizontal + vertical lines and computing intersections (that works extremely well for templates).