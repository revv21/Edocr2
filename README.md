Yep — I can see exactly why from your screenshot.
The missing lines are mostly the actual part edges (black outlines), while the algorithm is strongly picking up dimension lines / leader lines (thin + long), and some part edges are being broken by:
hatching
thick outlines → double edges
low contrast / scanning noise
gaps where the outline meets other geometry
So we need a more engineering-drawing-specific line extractor.
✅ Best fix: detect “STRUCTURE lines” separately from “ANNOTATION lines”
In drawings, there are 2 line categories:
1) Geometry edges (part outline)
thicker
continuous
often axis-aligned
should be detected
2) Dimension/leader lines (annotation)
thin
extend into white space
should be ignored OR separated
Your current method mixes both, and misses some structure edges.
✅ Improved approach (works much better)
Step A: Extract edges with LSD (Line Segment Detector)
OpenCV’s LSD is much better than Hough for engineering drawings because:
it finds line segments directly from gradients
less sensitive to broken edges
reduces duplicates
Step B: Merge collinear segments
Step C: Filter out annotation lines (optional)
✅ Code: Robust Line Detection using LSD + merging
Create: detect_lines_lsd.py
Copy code
Python
import cv2
import numpy as np
import math


def preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Slight blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def detect_lines_lsd(gray):
    """
    Returns line segments from LSD:
    [(x1,y1,x2,y2), ...]
    """
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(gray)

    segs = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            segs.append((int(x1), int(y1), int(x2), int(y2)))
    return segs


def line_length(l):
    x1, y1, x2, y2 = l
    return math.hypot(x2 - x1, y2 - y1)


def angle_deg(l):
    x1, y1, x2, y2 = l
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def midpoint(l):
    x1, y1, x2, y2 = l
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def merge_lines(lines, angle_thresh=6, dist_thresh=12):
    """
    Merge nearly collinear and nearby segments into longer segments.
    """
    lines = [l for l in lines if line_length(l) > 30]  # remove tiny clutter
    used = [False] * len(lines)
    merged = []

    for i in range(len(lines)):
        if used[i]:
            continue

        base = lines[i]
        base_ang = angle_deg(base)

        group = [base]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            ang = angle_deg(lines[j])
            # handle wrap-around angle
            da = abs(base_ang - ang)
            da = min(da, abs(da - 180))

            if da < angle_thresh:
                m1 = midpoint(base)
                m2 = midpoint(lines[j])
                if math.hypot(m1[0] - m2[0], m1[1] - m2[1]) < dist_thresh:
                    group.append(lines[j])
                    used[j] = True

        # merge group by farthest endpoints
        pts = []
        for g in group:
            pts.append((g[0], g[1]))
            pts.append((g[2], g[3]))

        best_pair = None
        best_d = -1
        for p in pts:
            for q in pts:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d > best_d:
                    best_d = d
                    best_pair = (p, q)

        (x1, y1), (x2, y2) = best_pair
        merged.append((int(x1), int(y1), int(x2), int(y2)))

    return merged


def draw_lines(img_bgr, lines, color=(0, 255, 0)):
    out = img_bgr.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (x1, y1), (x2, y2), color, 2)
    return out


if __name__ == "__main__":
    img_path = "input/example.jpg"
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)

    gray = preprocess(img)
    raw_lines = detect_lines_lsd(gray)
    merged_lines = merge_lines(raw_lines)

    print("Raw LSD lines:", len(raw_lines))
    print("Merged lines:", len(merged_lines))

    cv2.imwrite("lsd_raw.png", draw_lines(img, raw_lines))
    cv2.imwrite("lsd_merged.png", draw_lines(img, merged_lines))

    print("✅ Saved: lsd_raw.png and lsd_merged.png")
✅ Why this will detect the missing lines in your screenshot
Your missing lines are thick, continuous structural edges.
LSD detects them much better than Hough because it follows the gradient of the edge even when:
broken by hatch lines
not perfectly sharp
partially occluded
🔥 Optional: Remove annotation lines (dimension lines)
If you want only part outline lines, you can filter:
remove long thin horizontal/vertical lines that extend into white space
We can add:
“line must lie near black pixels on both sides”
or “line must be part of large connected component”
But first get all lines properly.
⚙️ If still missing some edges
Change thresholds:
In merge_lines():
Copy code
Python
lines = [l for l in lines if line_length(l) > 15]
In preprocess: replace histogram equalization with CLAHE:
Copy code
Python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
Next step (once lines are correct)
After line detection works, we can do: ✅ line classification: geometry vs dimension
✅ dimension-to-line matching using arrowheads/extension lines
If you want, upload the actual input/example.jpg (not screenshot) and I’ll tune the LSD + filters so it detects all part edges cleanly.