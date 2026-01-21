Yes — that’s exactly the problem when your SVG stores arcs/circles as lots of tiny straight segments.
So when you increase min_line_length, you’re accidentally deleting the segments that form the arcs/circles.
✅ The right solution is:
⭐ Extract arcs/circles by grouping small line segments into curve clusters, then fitting circles.
Instead of filtering small lines away, we use them.
✅ Method that works (robust on cluttered SVG)
Step 1: Extract all Line segments from SVG paths
Step 2: Build “chains” of connected segments
Segments belong to the same chain if:
endpoints are close (<= join_dist)
direction changes smoothly
Step 3: For each chain:
sample points along it
fit a circle using least squares
compute fit error
if error small and arc-length is large enough → accept as circle/arc
Step 4: Classify:
Circle if chain covers most of 360°
Arc if it covers a partial angle
This extracts circles/arcs even when represented as 200 tiny lines.
✅ One Final Code: Extract Lines + Detect Circles/Arcs from Tiny Segments + Preview
Install
Copy code
Bash
pip install svgpathtools pillow numpy
svg_extract_arcs_circles_from_lines.py
Copy code
Python
import math
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line


# -----------------------------
# Geometry helpers
# -----------------------------
def complex_to_xy(z):
    return np.array([float(z.real), float(z.imag)], dtype=np.float32)


def seg_len(p1, p2):
    return float(np.linalg.norm(p2 - p1))


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))
    return math.degrees(math.acos(cosv))


# -----------------------------
# Circle fitting (least squares)
# -----------------------------
def fit_circle_least_squares(points):
    """
    Fit circle x^2 + y^2 + Ax + By + C = 0
    Returns (cx, cy, r, rmse)
    """
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    # Solve least squares
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol

    cx = -a / 2.0
    cy = -b_ / 2.0
    r = math.sqrt(max(0.0, cx * cx + cy * cy - c))

    # RMSE of radius error
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmse = float(np.sqrt(np.mean((d - r) ** 2)))

    return cx, cy, r, rmse


def arc_coverage_degrees(points, cx, cy):
    pts = np.asarray(points, dtype=np.float32)
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    ang = (ang + 360.0) % 360.0
    ang_sorted = np.sort(ang)

    # coverage = max gap complement
    diffs = np.diff(ang_sorted)
    wrap_gap = (ang_sorted[0] + 360.0) - ang_sorted[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    max_gap = float(np.max(gaps))
    coverage = 360.0 - max_gap
    return coverage


# -----------------------------
# Extract line segments from SVG
# -----------------------------
def extract_lines(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)
    segs = []

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                p1 = complex_to_xy(seg.start)
                p2 = complex_to_xy(seg.end)
                if seg_len(p1, p2) > 1e-3:
                    segs.append((p1, p2))

    return segs, svg_attr


# -----------------------------
# Build chains of connected small segments
# -----------------------------
def build_chains(segments, join_dist=2.5, max_turn_deg=45):
    """
    Connect segments into chains based on endpoint proximity and smooth direction.
    Returns list of chains, each chain is list of points [p0,p1,p2,...]
    """
    unused = [True] * len(segments)

    # Build endpoint index for quick lookup
    endpoints = []
    for i, (p1, p2) in enumerate(segments):
        endpoints.append((i, 0, p1))  # segment i, endpoint 0
        endpoints.append((i, 1, p2))  # segment i, endpoint 1
    endpoints = np.array([(i, e, p[0], p[1]) for (i, e, p) in endpoints], dtype=np.float32)

    chains = []

    def find_candidates(pt):
        # brute force nearest endpoints (OK for moderate SVG size)
        dx = endpoints[:, 2] - pt[0]
        dy = endpoints[:, 3] - pt[1]
        dist = np.sqrt(dx * dx + dy * dy)
        idxs = np.where(dist <= join_dist)[0]
        return idxs

    for i in range(len(segments)):
        if not unused[i]:
            continue

        p1, p2 = segments[i]
        unused[i] = False

        chain = [p1, p2]
        last_dir = p2 - p1

        # extend forward
        while True:
            tip = chain[-1]
            cand_idxs = find_candidates(tip)
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
                    nxt = b
                    cur = a
                else:
                    nxt = a
                    cur = b

                # ensure the endpoint matches chain tip
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

        # extend backward
        last_dir = chain[0] - chain[1]
        while True:
            tip = chain[0]
            cand_idxs = find_candidates(tip)
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
                    nxt = b
                    cur = a
                else:
                    nxt = a
                    cur = b

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

        # Keep meaningful chains
        if len(chain) >= 8:
            chains.append(chain)

    return chains


# -----------------------------
# Detect arcs/circles from chains
# -----------------------------
def detect_arcs_circles(chains, rmse_thresh=1.5, min_radius=5, min_coverage_deg=40):
    arcs = []
    circles = []

    for pts in chains:
        pts_np = np.asarray(pts, dtype=np.float32)

        # skip too small chains
        if len(pts_np) < 10:
            continue

        cx, cy, r, rmse = fit_circle_least_squares(pts_np)

        if r < min_radius:
            continue
        if rmse > rmse_thresh:
            continue

        coverage = arc_coverage_degrees(pts_np, cx, cy)
        if coverage < min_coverage_deg:
            continue

        item = {
            "center": (float(cx), float(cy)),
            "radius": float(r),
            "rmse": float(rmse),
            "coverage_deg": float(coverage),
            "points": pts_np
        }

        # Circle if almost full coverage
        if coverage > 300:
            circles.append(item)
        else:
            arcs.append(item)

    return arcs, circles


# -----------------------------
# Render preview
# -----------------------------
def render_preview(segments, arcs, circles, out_path="svg_arc_circle_preview.png", size=2000, pad=30):
    # Collect all points to set bounds
    all_pts = []
    for p1, p2 in segments:
        all_pts.append(p1)
        all_pts.append(p2)
    for a in arcs:
        all_pts.extend(a["points"])
    for c in circles:
        all_pts.extend(c["points"])

    all_pts = np.asarray(all_pts, dtype=np.float32)
    minx, miny = float(all_pts[:, 0].min()), float(all_pts[:, 1].min())
    maxx, maxy = float(all_pts[:, 0].max()), float(all_pts[:, 1].max())
    w = maxx - minx
    h = maxy - miny
    scale = min((size - 2 * pad) / (w + 1e-6), (size - 2 * pad) / (h + 1e-6))

    def map_pt(p):
        X = pad + (p[0] - minx) * scale
        Y = pad + (p[1] - miny) * scale
        return (float(X), float(Y))

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw all raw segments lightly (gray)
    for p1, p2 in segments:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(180, 180, 180), width=1)

    # Draw arcs (red polyline)
    for a in arcs:
        mapped = [map_pt(p) for p in a["points"]]
        draw.line(mapped, fill=(255, 0, 0), width=3)

    # Draw circles (blue outline)
    for c in circles:
        cx, cy = c["center"]
        r = c["radius"]

        C = np.array([cx, cy], dtype=np.float32)
        CX, CY = map_pt(C)
        R = r * scale
        draw.ellipse((CX - R, CY - R, CX + R, CY + R), outline=(0, 0, 255), width=3)

    img.save(out_path)
    print(f"✅ Saved: {out_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    segments, svg_attr = extract_lines(svg_file)
    print("Raw line segments:", len(segments))

    # Build chains from tiny segments (this is the key)
    chains = build_chains(segments, join_dist=3.0, max_turn_deg=55)
    print("Chains built:", len(chains))

    # Detect arcs/circles from chains
    arcs, circles = detect_arcs_circles(
        chains,
        rmse_thresh=1.2,       # tighter = fewer false arcs
        min_radius=8,
        min_coverage_deg=60
    )

    print("Detected arcs:", len(arcs))
    print("Detected circles:", len(circles))

    render_preview(segments, arcs, circles, out_path="svg_arc_circle_preview.png", size=2000)
✅ What you’ll see in svg_arc_circle_preview.png
Gray = all raw line segments (clutter)
Red = detected arc chains (from tiny segments)
Blue = detected circles (from tiny segments)
This is exactly what you want:
don’t delete tiny segments → use them to reconstruct arcs/circles.
🔧 Tuning knobs (very important)
If you get too many false arcs:
Increase strictness:
Copy code
Python
rmse_thresh = 0.8
min_coverage_deg = 90
If arcs are missed:
Relax:
Copy code
Python
join_dist = 5.0
rmse_thresh = 2.0
min_coverage_deg = 40
If circles are missed:
Relax circle threshold:
Copy code
Python
coverage > 260   # instead of 300
⭐ Why this works better than filtering
Filtering removes the arc itself.
This method reconstructs arcs from the clutter.
If you upload one SVG, I can tune these 4 parameters to your converter output and make it very clean:
join_dist
max_turn_deg
rmse_thresh
`min_coverage