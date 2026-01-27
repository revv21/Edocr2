import math
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line

# ============================================================
# Helpers
# ============================================================
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

def round_pt(p, prec=3):
    return (round(float(p[0]), prec), round(float(p[1]), prec))

# ============================================================
# 1) Extract SVG line segments
# ============================================================
def extract_line_segments(svg_path):
    paths, _, _ = svg2paths2(svg_path)
    segs = []
    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                p1 = complex_to_xy(seg.start)
                p2 = complex_to_xy(seg.end)
                if seg_length(p1, p2) > 1e-6:
                    segs.append((p1, p2))
    return segs

# ============================================================
# 2) ARC / CIRCLE DETECTION (UNCHANGED WORKING LOGIC)
# ============================================================
def fit_circle_least_squares(points):
    pts = np.asarray(points, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
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
    ang = np.sort(ang)
    diffs = np.diff(ang)
    wrap_gap = (ang[0] + 360.0) - ang[-1]
    return 360.0 - max(np.max(diffs), wrap_gap)

def build_chains_from_small_segments(segments, join_dist=3.0, max_turn_deg=55):
    unused = [True] * len(segments)
    endpoints = []

    for i, (p1, p2) in enumerate(segments):
        endpoints.append((i, 0, p1))
        endpoints.append((i, 1, p2))

    chains = []

    def candidates(pt):
        out = []
        for sid, eid, p in endpoints:
            if unused[sid] and np.linalg.norm(p - pt) <= join_dist:
                out.append((sid, eid))
        return out

    for i in range(len(segments)):
        if not unused[i]:
            continue

        p1, p2 = segments[i]
        unused[i] = False
        chain = [p1, p2]
        last_dir = p2 - p1

        # forward
        while True:
            tip = chain[-1]
            best = None
            best_turn = 1e9
            for sid, eid in candidates(tip):
                if not unused[sid]:
                    continue
                a, b = segments[sid]
                cur, nxt = (a, b) if eid == 0 else (b, a)
                if np.linalg.norm(cur - tip) > 3.0:
                    continue
                turn = angle_between(last_dir, nxt - cur)
                if turn <= max_turn_deg and turn < best_turn:
                    best = (sid, nxt, nxt - cur)
                    best_turn = turn
            if best is None:
                break
            sid, nxt, new_dir = best
            unused[sid] = False
            chain.append(nxt)
            last_dir = new_dir

        # backward
        last_dir = chain[0] - chain[1]
        while True:
            tip = chain[0]
            best = None
            best_turn = 1e9
            for sid, eid in candidates(tip):
                if not unused[sid]:
                    continue
                a, b = segments[sid]
                cur, nxt = (a, b) if eid == 0 else (b, a)
                if np.linalg.norm(cur - tip) > 3.0:
                    continue
                turn = angle_between(last_dir, nxt - cur)
                if turn <= max_turn_deg and turn < best_turn:
                    best = (sid, nxt, nxt - cur)
                    best_turn = turn
            if best is None:
                break
            sid, nxt, new_dir = best
            unused[sid] = False
            chain.insert(0, nxt)
            last_dir = new_dir

        if len(chain) >= 10:
            chains.append(chain)

    return chains

def detect_arcs_and_circles(all_segments):
    SMALL_LEN = 1.0
    small = [(p1, p2) for p1, p2 in all_segments if seg_length(p1, p2) < SMALL_LEN]
    chains = build_chains_from_small_segments(small)

    arcs, circles = [], []
    used_small_segments = set()

    for pts in chains:
        pts_np = np.asarray(pts, dtype=np.float32)
        cx, cy, r, rmse = fit_circle_least_squares(pts_np)
        if r < 8 or rmse > 1.2:
            continue
        cov = arc_coverage_degrees(pts_np, cx, cy)
        if cov < 60:
            continue

        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            used_small_segments.add((round_pt(a), round_pt(b)))
            used_small_segments.add((round_pt(b), round_pt(a)))

        item = {"center": (cx, cy), "radius": r, "points": pts_np}
        if cov > 300:
            circles.append(item)
        else:
            arcs.append(item)

    return arcs, circles, used_small_segments

# ============================================================
# 3) ARROWHEAD DETECTION (UNCHANGED WORKING LOGIC)
# ============================================================
def angle(p1, p2, p3):
    a = p1 - p2
    b = p3 - p2
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return math.degrees(math.acos(cosv))

def triangle_tip(pts):
    angs = [
        angle(pts[1], pts[0], pts[2]),
        angle(pts[0], pts[1], pts[2]),
        angle(pts[0], pts[2], pts[1]),
    ]
    return pts[np.argmin(angs)], min(angs)

def detect_arrowheads(segments):
    small = [(p1, p2) for p1, p2 in segments if 0.2 <= seg_length(p1, p2) <= 12]
    graph, edges = {}, set()

    def add_edge(a, b):
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
        edges.add(tuple(sorted([a, b])))

    for p1, p2 in small:
        a, b = round_pt(p1), round_pt(p2)
        if a != b:
            add_edge(a, b)

    used_small_segments = set()

    for u in graph:
        for v in graph[u]:
            for w in graph[u]:
                if v != w and tuple(sorted([v, w])) in edges:
                    pts = [np.array(u), np.array(v), np.array(w)]
                    _, sharp = triangle_tip(pts)
                    if 10 <= sharp <= 75:
                        used_small_segments.add((u, v))
                        used_small_segments.add((v, w))
                        used_small_segments.add((u, w))

    return used_small_segments

# ============================================================
# 4) REMOVE ONLY SMALL SEGMENTS USED BY ARCS / ARROWHEADS
# ============================================================
def remove_used_small_segments(all_segments, used_small_segments, small_len=1.0):
    remaining = []
    for p1, p2 in all_segments:
        if seg_length(p1, p2) >= small_len:
            remaining.append((p1, p2))
            continue
        key = (round_pt(p1), round_pt(p2))
        if key not in used_small_segments:
            remaining.append((p1, p2))
    return remaining

# ============================================================
# 5) KEEP ONLY LONG LINES
# ============================================================
def keep_only_long_lines(segments, min_len=30.0):
    return [(p1, p2) for p1, p2 in segments if seg_length(p1, p2) >= min_len]

# ============================================================
# 6) PREVIEW
# ============================================================
def render_preview(all_segments, long_lines, out="final_long_lines.png", size=2000):
    pts = np.array([p for s in all_segments for p in s])
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    scale = (size - 40) / max(maxx - minx, maxy - miny)

    def map_pt(p):
        return (20 + (p[0] - minx) * scale,
                20 + (p[1] - miny) * scale)

    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)

    for p1, p2 in all_segments:
        d.line((*map_pt(p1), *map_pt(p2)), fill=(220, 220, 220), width=1)

    for p1, p2 in long_lines:
        d.line((*map_pt(p1), *map_pt(p2)), fill=(0, 0, 0), width=3)

    img.save(out)
    print("✅ Saved:", out)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    segments = extract_line_segments(svg_file)
    print("Total segments:", len(segments))

    arcs, circles, arc_used = detect_arcs_and_circles(segments)
    print("Arcs:", len(arcs), "Circles:", len(circles))

    arrow_used = detect_arrowheads(segments)
    print("Arrowhead small segments:", len(arrow_used))

    used_small = arc_used.union(arrow_used)

    remaining = remove_used_small_segments(segments, used_small)
    long_lines = keep_only_long_lines(remaining)

    print("Long lines:", len(long_lines))
    render_preview(segments, long_lines)