import math
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line


# -----------------------------
# Helpers
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


# -----------------------------
# Circle fit (least squares)
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
    coverage = 360.0 - max_gap
    return coverage


# -----------------------------
# Extract line segments from SVG
# -----------------------------
def extract_line_segments(svg_path):
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
# Build chains ONLY from tiny segments
# -----------------------------
def build_chains_from_small_segments(segments, join_dist=2.5, max_turn_deg=55):
    """
    segments: list of (p1,p2)
    Returns chains, each chain is list of points [p0,p1,p2,...]
    """
    unused = [True] * len(segments)

    # Endpoint index: [seg_id, endpoint_id, x, y]
    endpoints = []
    for i, (p1, p2) in enumerate(segments):
        endpoints.append((i, 0, p1[0], p1[1]))
        endpoints.append((i, 1, p2[0], p2[1]))
    endpoints = np.array(endpoints, dtype=np.float32)

    def find_endpoint_candidates(pt):
        dx = endpoints[:, 2] - pt[0]
        dy = endpoints[:, 3] - pt[1]
        dist = np.sqrt(dx * dx + dy * dy)
        idxs = np.where(dist <= join_dist)[0]
        return idxs

    chains = []

    for i in range(len(segments)):
        if not unused[i]:
            continue

        p1, p2 = segments[i]
        unused[i] = False

        chain = [p1, p2]
        last_dir = p2 - p1

        # Extend forward
        while True:
            tip = chain[-1]
            cand_idxs = find_endpoint_candidates(tip)

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
                # choose the endpoint that connects to current tip
                if end_id == 0:
                    cur = a
                    nxt = b
                else:
                    cur = b
                    nxt = a

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

        # Extend backward
        last_dir = chain[0] - chain[1]
        while True:
            tip = chain[0]
            cand_idxs = find_endpoint_candidates(tip)

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
                    cur = a
                    nxt = b
                else:
                    cur = b
                    nxt = a

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

        # Keep meaningful chains only
        if len(chain) >= 10:
            chains.append(chain)

    return chains


# -----------------------------
# Detect arcs/circles from chains
# -----------------------------
def detect_arcs_and_circles(chains, rmse_thresh=1.2, min_radius=8, min_coverage_deg=60):
    arcs = []
    circles = []

    for pts in chains:
        pts_np = np.asarray(pts, dtype=np.float32)

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

        if coverage > 300:
            circles.append(item)
        else:
            arcs.append(item)

    return arcs, circles


# -----------------------------
# Render preview
# -----------------------------
def render_preview(all_lines, arcs, circles, out_path="svg_geometry_preview.png", size=2000, pad=30):
    # Collect all points for bounds
    pts = []
    for p1, p2 in all_lines:
        pts.append(p1)
        pts.append(p2)
    for a in arcs:
        pts.extend(a["points"])
    for c in circles:
        pts.extend(c["points"])

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

    # Draw ALL lines in light gray (for context)
    for p1, p2 in all_lines:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(200, 200, 200), width=1)

    # Draw arcs in red
    for a in arcs:
        mapped = [map_pt(p) for p in a["points"]]
        draw.line(mapped, fill=(255, 0, 0), width=3)

    # Draw circles in blue
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
# Main
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    all_lines = extract_line_segments(svg_file)

    # Split: small segments used for arcs/circles, large segments are true edges
    SMALL_LEN = 1.0

    small_lines = []
    large_lines = []
    for p1, p2 in all_lines:
        if seg_length(p1, p2) < SMALL_LEN:
            small_lines.append((p1, p2))
        else:
            large_lines.append((p1, p2))

    print("Total line segments:", len(all_lines))
    print("Small segments (<1):", len(small_lines))
    print("Large segments (>=1):", len(large_lines))

    # Build chains only from small segments
    chains = build_chains_from_small_segments(
        small_lines,
        join_dist=3.0,
        max_turn_deg=55
    )
    print("Chains from small segments:", len(chains))

    # Fit arcs/circles from those chains
    arcs, circles = detect_arcs_and_circles(
        chains,
        rmse_thresh=1.2,
        min_radius=8,
        min_coverage_deg=60
    )

    print("Detected arcs:", len(arcs))
    print("Detected circles:", len(circles))

    # Render everything for verification:
    # - all lines in gray
    # - arcs in red
    # - circles in blue
    render_preview(all_lines, arcs, circles, out_path="svg_geometry_preview.png")