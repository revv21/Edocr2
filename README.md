Perfect ✅ — below is a single final pipeline script that does everything end-to-end:
✅ Input: drawing.svg
✅ Output:
geometry.json (merged lines + detected arcs + detected circles)
geometry_preview.png (visual verification)
It solves your exact problem:
✅ slanted straight edges broken into tiny segments → merged into long lines
✅ arcs/circles made of tiny segments → reconstructed by chaining + circle fitting
✅ keeps clutter visible but highlights extracted primitives
🔥 Brief logic / pipeline (what we do)
Step 1: Extract raw SVG line segments
Most converters store everything as <path> segments → we extract all Line segments.
Step 2: Merge “broken straight lines”
Many slanted edges become many tiny collinear segments. So we:
chain connected segments by endpoint proximity
check if the chain is “almost straight” using PCA + max deviation
merge the chain into one long line
➡️ Output: merged_lines
Step 3: Build curve chains from remaining tiny segments
From the leftover segments (not merged into straight lines), we:
chain connected segments again
these chains often represent arcs/circles/hatching
Step 4: Fit circles to chains
For each chain:
fit a circle using least squares
compute RMSE error
compute angle coverage (how much of circle it covers)
If:
coverage ~ full → circle
coverage partial → arc
➡️ Output: detected_circles, detected_arcs
Step 5: Save JSON + Preview Image
draw merged lines (black)
draw arcs (red)
draw circles (blue)
draw leftover clutter (light gray)
✅ Install
Copy code
Bash
pip install svgpathtools pillow numpy
✅ Final Script: svg_geometry_pipeline.py
Copy code
Python
import math
import json
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line


# -----------------------------
# Basic helpers
# -----------------------------
def complex_to_xy(z):
    return np.array([float(z.real), float(z.imag)], dtype=np.float32)


def seg_length(p1, p2):
    return float(np.linalg.norm(p2 - p1))


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    return math.degrees(math.acos(cosv))


def point_line_distance(pt, a, b):
    v = b - a
    if np.linalg.norm(v) < 1e-6:
        return float(np.linalg.norm(pt - a))
    return float(np.abs(np.cross(v, pt - a)) / np.linalg.norm(v))


# -----------------------------
# Extract raw SVG line segments
# -----------------------------
def extract_svg_lines(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)
    segs = []

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                p1 = complex_to_xy(seg.start)
                p2 = complex_to_xy(seg.end)
                if seg_length(p1, p2) > 1e-6:
                    segs.append((p1, p2))

    return segs


# -----------------------------
# Chain segments by endpoint proximity
# -----------------------------
def chain_segments(segments, join_dist=2.5, max_turn_deg=70):
    """
    Connect segments into chains:
    chain = [p0, p1, p2, ...]
    """
    unused = [True] * len(segments)

    endpoints = []
    for i, (p1, p2) in enumerate(segments):
        endpoints.append((i, 0, p1[0], p1[1]))
        endpoints.append((i, 1, p2[0], p2[1]))
    endpoints = np.array(endpoints, dtype=np.float32)

    def candidates(pt):
        dx = endpoints[:, 2] - pt[0]
        dy = endpoints[:, 3] - pt[1]
        dist = np.sqrt(dx * dx + dy * dy)
        return np.where(dist <= join_dist)[0]

    chains = []

    for i in range(len(segments)):
        if not unused[i]:
            continue

        p1, p2 = segments[i]
        unused[i] = False
        chain = [p1, p2]

        # forward extend
        last_dir = p2 - p1
        while True:
            tip = chain[-1]
            cand_idxs = candidates(tip)

            best = None
            best_turn = 1e9

            for ci in cand_idxs:
                seg_id = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])

                if seg_id < 0 or seg_id >= len(segments):
                    continue
                if not unused[seg_id]:
                    continue

                a, b = segments[seg_id]
                if end_id == 0:
                    cur, nxt = a, b
                else:
                    cur, nxt = b, a

                if np.linalg.norm(cur - tip) > join_dist:
                    continue

                new_dir = nxt - cur
                turn = angle_between(last_dir, new_dir)

                if turn <= max_turn_deg and turn < best_turn:
                    best_turn = turn
                    best = (seg_id, nxt, new_dir)

            if best is None:
                break

            seg_id, nxt, new_dir = best
            unused[seg_id] = False
            chain.append(nxt)
            last_dir = new_dir

        # backward extend
        last_dir = chain[0] - chain[1]
        while True:
            tip = chain[0]
            cand_idxs = candidates(tip)

            best = None
            best_turn = 1e9

            for ci in cand_idxs:
                seg_id = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])

                if seg_id < 0 or seg_id >= len(segments):
                    continue
                if not unused[seg_id]:
                    continue

                a, b = segments[seg_id]
                if end_id == 0:
                    cur, nxt = a, b
                else:
                    cur, nxt = b, a

                if np.linalg.norm(cur - tip) > join_dist:
                    continue

                new_dir = nxt - cur
                turn = angle_between(last_dir, new_dir)

                if turn <= max_turn_deg and turn < best_turn:
                    best_turn = turn
                    best = (seg_id, nxt, new_dir)

            if best is None:
                break

            seg_id, nxt, new_dir = best
            unused[seg_id] = False
            chain.insert(0, nxt)
            last_dir = new_dir

        if len(chain) >= 2:
            chains.append(chain)

    return chains


# -----------------------------
# Straightness test + merge chain
# -----------------------------
def chain_is_straight(chain_pts, max_dev=1.0):
    pts = np.asarray(chain_pts, dtype=np.float32)

    # PCA direction
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]

    a = mean - 1000 * direction
    b = mean + 1000 * direction

    devs = [point_line_distance(p, a, b) for p in pts]
    return max(devs) <= max_dev


def merge_chain_to_line(chain_pts):
    pts = np.asarray(chain_pts, dtype=np.float32)
    best_d = -1
    best_pair = None

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.linalg.norm(pts[j] - pts[i])
            if d > best_d:
                best_d = d
                best_pair = (pts[i], pts[j])

    return best_pair  # (p1,p2)


def merge_broken_straight_lines(raw_segments, join_dist=2.5, straight_dev=1.0):
    """
    Chains segments and merges straight chains into long lines.
    Returns (merged_lines, leftover_segments)
    """
    chains = chain_segments(raw_segments, join_dist=join_dist, max_turn_deg=60)

    merged_lines = []
    leftover_segments = []

    for ch in chains:
        if len(ch) >= 3 and chain_is_straight(ch, max_dev=straight_dev):
            merged_lines.append(merge_chain_to_line(ch))
        else:
            for i in range(len(ch) - 1):
                leftover_segments.append((ch[i], ch[i + 1]))

    return merged_lines, leftover_segments


# -----------------------------
# Circle fitting from chains
# -----------------------------
def fit_circle_least_squares(points):
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol

    cx = -a / 2.0
    cy = -b_ / 2.0
    r = math.sqrt(max(0.0, cx * cx + cy * cy - c))

    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((d - r) ** 2)))

    return cx, cy, r, rmse


def arc_coverage_degrees(points, cx, cy):
    pts = np.asarray(points, dtype=np.float32)
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    ang = (ang + 360.0) % 360.0
    ang_sorted = np.sort(ang)

    diffs = np.diff(ang_sorted)
    wrap_gap = (ang_sorted[0] + 360.0) - ang_sorted[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    max_gap = float(np.max(gaps))
    return 360.0 - max_gap


def detect_arcs_circles_from_leftover(leftover_segments,
                                      join_dist=2.5,
                                      rmse_thresh=1.2,
                                      min_radius=8,
                                      min_coverage_deg=60):
    """
    Chain leftover segments -> fit circles.
    """
    chains = chain_segments(leftover_segments, join_dist=join_dist, max_turn_deg=80)

    arcs = []
    circles = []
    other = []

    for ch in chains:
        if len(ch) < 10:
            other.append(ch)
            continue

        cx, cy, r, rmse = fit_circle_least_squares(ch)
        if r < min_radius or rmse > rmse_thresh:
            other.append(ch)
            continue

        coverage = arc_coverage_degrees(ch, cx, cy)
        if coverage < min_coverage_deg:
            other.append(ch)
            continue

        item = {
            "center": [float(cx), float(cy)],
            "radius": float(r),
            "rmse": float(rmse),
            "coverage_deg": float(coverage),
            "polyline": [[float(p[0]), float(p[1])] for p in ch]
        }

        if coverage > 300:
            circles.append(item)
        else:
            arcs.append(item)

    return arcs, circles, other


# -----------------------------
# Preview rendering
# -----------------------------
def render_preview(raw_lines, merged_lines, leftover_segments, arcs, circles,
                   out_path="geometry_preview.png", size=2000, pad=30):
    pts = []

    for p1, p2 in raw_lines:
        pts.append(p1); pts.append(p2)
    for p1, p2 in merged_lines:
        pts.append(p1); pts.append(p2)
    for p1, p2 in leftover_segments:
        pts.append(p1); pts.append(p2)
    for a in arcs:
        for p in a["polyline"]:
            pts.append(np.array(p, dtype=np.float32))
    for c in circles:
        for p in c["polyline"]:
            pts.append(np.array(p, dtype=np.float32))

    pts = np.asarray(pts, dtype=np.float32)
    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
    w = maxx - minx
    h = maxy - miny
    scale = min((size - 2 * pad) / (w + 1e-6), (size - 2 * pad) / (h + 1e-6))

    def map_pt(p):
        X = pad + (p[0] - minx) * scale
        Y = pad + (p[1] - miny) * scale
        return (float(X), float(Y))

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # raw lines (very light gray)
    for p1, p2 in raw_lines:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(230, 230, 230), width=1)

    # merged long lines (black)
    for p1, p2 in merged_lines:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=2)

    # arcs (red)
    for a in arcs:
        pts2 = [map_pt(np.array(p, dtype=np.float32)) for p in a["polyline"]]
        draw.line(pts2, fill=(255, 0, 0), width=3)

    # circles (blue outline)
    for c in circles:
        cx, cy = c["center"]
        r = c["radius"]
        C = np.array([cx, cy], dtype=np.float32)
        CX, CY = map_pt(C)
        R = r * scale
        draw.ellipse((CX - R, CY - R, CX + R, CY + R), outline=(0, 0, 255), width=3)

    img.save(out_path)
    print(f"✅ Saved preview: {out_path}")


# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    # 1) Extract all raw line segments from SVG
    raw_lines = extract_svg_lines(svg_file)
    print("Raw SVG line segments:", len(raw_lines))

    # 2) Merge broken straight lines (including slanted)
    merged_lines, leftover_segments = merge_broken_straight_lines(
        raw_segments=raw_lines,
        join_dist=2.5,
        straight_dev=1.0
    )
    print("Merged straight lines:", len(merged_lines))
    print("Leftover segments:", len(leftover_segments))

    # 3) Detect arcs/circles from leftover segments
    arcs, circles, other_chains = detect_arcs_circles_from_leftover(
        leftover_segments,
        join_dist=3.0,
        rmse_thresh=1.2,
        min_radius=8,
        min_coverage_deg=60
    )
    print("Detected arcs:", len(arcs))
    print("Detected circles:", len(circles))

    # 4) Save JSON
    geometry = {
        "merged_lines": [
            {
                "x1": float(p1[0]), "y1": float(p1[1]),
                "x2": float(p2[0]), "y2": float(p2[1]),
            }
            for (p1, p2) in merged_lines
        ],
        "arcs": arcs,
        "circles": circles
    }

    with open("geometry.json", "w", encoding="utf-8") as f:
        json.dump(geometry, f, indent=2)

    print("✅ Saved geometry.json")

    # 5) Render preview
    render_preview(
        raw_lines=raw_lines,
        merged_lines=merged_lines,
        leftover_segments=leftover_segments,
        arcs=arcs,
        circles=circles,
        out_path="geometry_preview.png",
        size=2000
    )
✅ What you should expect
geometry_preview.png
Black = merged straight edges (including slanted edges)
Red = arcs detected from leftover tiny segments
Blue = circles detected from leftover tiny segments
Light gray = raw clutter (just for context)
geometry.json
Contains clean primitives you can now use for:
dimension matching
snapping arrowheads to nearest edge/circle/arc
tolerance analysis later
🔧 Tuning tips (if output isn’t perfect)
If slanted lines still don’t merge:
Increase:
Copy code
Python
join_dist=4.0
straight_dev=1.5
If wrong merges happen (hatching becomes a “line”):
Decrease:
Copy code
Python
straight_dev=0.6
If arcs are missed:
Relax:
Copy code
Python
rmse_thresh=2.0
min_coverage_deg=40
If too many false circles:
Tighten:
Copy code
Python
rmse_thresh=0.8
min_radius=15
If you want, next I can integrate this directly into your view → DIM detection pipeline, so for each view crop you generate:
view_00_geometry.json
view_00_geometry_preview.png
and later we can do dimension-to-geometry matching on that view only (much easier + faster).