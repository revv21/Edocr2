Yep — both issues are expected with HoughLinesP on engineering drawings:
✅ Why lines are missed
broken edges (scan noise / gaps)
thin faint lines → Canny misses them
dashed/hidden lines break into pieces
text overlaps and interrupts lines
✅ Why “double lines” happen
edges produce two contours (both sides of a thick line)
Hough detects each side as a separate line → you see “parallel duplicates”
✅ Fix Strategy (works well in drawings)
A) Detect lines on a “skeletonized” single-pixel image
This collapses thick lines into one centerline → removes double edges.
B) Merge collinear segments afterward
Even with good detection, Hough returns fragments → we merge them.
✅ Best Practical Code: line detection without double lines
This version does:
✅ binarize
✅ remove text-ish small blobs
✅ skeletonize lines (single pixel)
✅ HoughLinesP
✅ merge collinear duplicates
✅ line_detection_fixed.py
Copy code
Python
import cv2
import numpy as np
import math


# -----------------------------
# Skeletonization (Zhang-Suen style)
# -----------------------------
def skeletonize(binary_img):
    """
    binary_img: 0 background, 255 foreground
    returns skeletonized binary image
    """
    img = binary_img.copy()
    img[img > 0] = 255

    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open_img)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel


# -----------------------------
# Preprocess for line extraction
# -----------------------------
def preprocess_for_lines(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Better for scanned drawings
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Remove tiny blobs (text specks / noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    cleaned = np.zeros_like(bin_img)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 40:  # keep larger components (lines)
            cleaned[labels == i] = 255

    # Bridge small gaps in dashed lines
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Skeletonize → removes double edges
    skel = skeletonize(cleaned)

    return skel


# -----------------------------
# Hough line detection
# -----------------------------
def detect_lines_skeleton(skel_img):
    edges = skel_img  # already 1px lines

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=15
    )

    segs = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            segs.append((x1, y1, x2, y2))
    return segs


# -----------------------------
# Merge collinear segments
# -----------------------------
def angle_of_line(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def line_distance(l1, l2):
    # distance between midpoints
    x1, y1, x2, y2 = l1
    a1, b1, a2, b2 = l2
    m1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    m2 = ((a1 + a2) / 2, (b1 + b2) / 2)
    return math.hypot(m1[0] - m2[0], m1[1] - m2[1])


def merge_lines(lines, angle_thresh=6, dist_thresh=15):
    """
    Merge lines that are nearly collinear and close.
    Simple but works well for drawings.
    """
    merged = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue
        x1, y1, x2, y2 = lines[i]
        ang1 = angle_of_line(x1, y1, x2, y2)

        group = [lines[i]]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            a1, b1, a2, b2 = lines[j]
            ang2 = angle_of_line(a1, b1, a2, b2)

            if abs(ang1 - ang2) < angle_thresh or abs(abs(ang1 - ang2) - 180) < angle_thresh:
                if line_distance(lines[i], lines[j]) < dist_thresh:
                    group.append(lines[j])
                    used[j] = True

        # Merge group into one long segment by extreme endpoints
        pts = []
        for g in group:
            pts.append((g[0], g[1]))
            pts.append((g[2], g[3]))

        # choose farthest pair of points
        best = None
        best_d = -1
        for p in pts:
            for q in pts:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d > best_d:
                    best_d = d
                    best = (p, q)

        (mx1, my1), (mx2, my2) = best
        merged.append((int(mx1), int(my1), int(mx2), int(my2)))

    return merged


# -----------------------------
# Visualization
# -----------------------------
def draw_lines(img_bgr, lines):
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in lines:
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


if __name__ == "__main__":
    img_path = "input/example.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    skel = preprocess_for_lines(img)
    raw_lines = detect_lines_skeleton(skel)
    merged_lines = merge_lines(raw_lines)

    print("Raw lines:", len(raw_lines))
    print("Merged lines:", len(merged_lines))

    cv2.imwrite("debug_skeleton.png", skel)
    cv2.imwrite("lines_raw.png", draw_lines(img, raw_lines))
    cv2.imwrite("lines_merged.png", draw_lines(img, merged_lines))

    print("✅ Saved: debug_skeleton.png, lines_raw.png, lines_merged.png")
✅ What this solves
✅ “Double lines”
Skeletonization converts thick edges into one-pixel centerline, so Hough sees only one.
✅ “Missing lines”
We add:
morphological close → bridges gaps
connected-component cleanup → reduces noise
better Hough parameters for broken segments
🔧 If you still miss dashed/hidden lines
Increase:
Copy code
Python
maxLineGap=25
threshold=60
minLineLength=30
🔥 Best debug outputs to inspect
This script saves:
debug_skeleton.png → shows if lines are continuous
lines_raw.png → raw Hough output
lines_merged.png → cleaned final lines
If debug_skeleton.png looks broken, Hough will always miss lines.
If you upload one sample drawing image, I can tune the exact thresholds (blockSize, Canny, minLineLength) so it detects almost all edges correctly.