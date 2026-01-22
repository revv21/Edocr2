Awesome — since both arc/circle detection and arrowhead detection are working for you, the correct next step is:
✅ Run those first → remove the segments they consume
✅ Then merge the remaining segments into clean straight lines
Below I’m giving you a single clean modular pipeline that does exactly this in 3 stages:
✅ Final Pipeline Order
Stage 1 — Detect arcs & circles
Uses your “small segments chaining + circle fitting”
Marks those segments as USED
Stage 2 — Detect arrowheads
Uses your triangle graph method
Marks those segments as USED
Stage 3 — Merge remaining segments into straight lines
Chains connected segments
Checks if chain is straight (PCA deviation)
Merges into 1 long line
✅ One Final Code: svg_pipeline_arcs_arrowheads_merge_lines.py
Put your SVG at: input/drawing.svg
Copy code
Python
import math
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line


# ============================================================
# 0) Common helpers
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


def point_line_distance(pt, a, b):
    v = b - a
    if np.linalg.norm(v) < 1e-6:
        return float(np.linalg.norm(pt - a))
    return float(np.abs(np.cross(v, pt - a)) / np.linalg.norm(v))


def round_point(p, grid=1.0):
    return (round(float(p[0]) / grid) * grid, round(float(p[1]) / grid) * grid)


# ============================================================
# 1) Extract SVG line segments
# ============================================================
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

    return segs, svg_attr


# ============================================================
# 2) Arc/Circle detection (your logic + segment-id tracking)
# ============================================================
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
    coverage = 360.0 - max_gap
    return coverage


def build_chains_from_small_segments(seg_records, join_dist=2.5, max_turn_deg=55):
    """
    seg_records: [{"id":i, "p1":..., "p2":...}, ...]
    Returns chains:
      [{"seg_ids":[...], "points":[p0,p1,p2,...]}]
    """
    n = len(seg_records)
    unused = [True] * n

    endpoints = []
    for idx, rec in enumerate(seg_records):
        p1, p2 = rec["p1"], rec["p2"]
        endpoints.append((idx, 0, p1[0], p1[1]))
        endpoints.append((idx, 1, p2[0], p2[1]))
    endpoints = np.array(endpoints, dtype=np.float32)

    def find_endpoint_candidates(pt):
        dx = endpoints[:, 2] - pt[0]
        dy = endpoints[:, 3] - pt[1]
        dist = np.sqrt(dx * dx + dy * dy)
        return np.where(dist <= join_dist)[0]

    chains = []

    for i in range(n):
        if not unused[i]:
            continue

        rec = seg_records[i]
        p1, p2 = rec["p1"], rec["p2"]
        unused[i] = False

        chain_pts = [p1, p2]
        chain_seg_idxs = [i]

        last_dir = p2 - p1

        # forward extend
        while True:
            tip = chain_pts[-1]
            cand_idxs = find_endpoint_candidates(tip)

            best = None
            best_turn = 1e9

            for ci in cand_idxs:
                seg_idx = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])

                if seg_idx < 0 or seg_idx >= n:
                    continue
                if not unused[seg_idx]:
                    continue

                rec2 = seg_records[seg_idx]
                a, b = rec2["p1"], rec2["p2"]

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
                    best = (seg_idx, nxt, new_dir)

            if best is None:
                break

            seg_idx, nxt, new_dir = best
            unused[seg_idx] = False
            chain_seg_idxs.append(seg_idx)
            chain_pts.append(nxt)
            last_dir = new_dir

        # backward extend
        last_dir = chain_pts[0] - chain_pts[1]
        while True:
            tip = chain_pts[0]
            cand_idxs = find_endpoint_candidates(tip)

            best = None
            best_turn = 1e9

            for ci in cand_idxs:
                seg_idx = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])

                if seg_idx < 0 or seg_idx >= n:
                    continue
                if not unused[seg_idx]:
                    continue

                rec2 = seg_records[seg_idx]
                a, b = rec2["p1"], rec2["p2"]

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
                    best = (seg_idx, nxt, new_dir)

            if best is None:
                break

            seg_idx, nxt, new_dir = best
            unused[seg_idx] = False
            chain_seg_idxs.append(seg_idx)
            chain_pts.insert(0, nxt)
            last_dir = new_dir

        if len(chain_pts) >= 10:
            chains.append({
                "seg_ids": [seg_records[k]["id"] for k in chain_seg_idxs],
                "points": chain_pts
            })

    return chains


def detect_arcs_and_circles_from_segments(all_segments, small_len=1.0,
                                         join_dist=3.0, max_turn_deg=55,
                                         rmse_thresh=1.2, min_radius=8, min_coverage_deg=60):
    """
    Returns:
      arcs, circles, used_segment_ids
    """
    seg_records = []
    for sid, (p1, p2) in enumerate(all_segments):
        if seg_length(p1, p2) < small_len:
            seg_records.append({"id": sid, "p1": p1, "p2": p2})

    chains = build_chains_from_small_segments(seg_records, join_dist=join_dist, max_turn_deg=max_turn_deg)

    arcs = []
    circles = []
    used_ids = set()

    for ch in chains:
        pts_np = np.asarray(ch["points"], dtype=np.float32)
        cx, cy, r, rmse = fit_circle_least_squares(pts_np)

        if r < min_radius or rmse > rmse_thresh:
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

        for sid in ch["seg_ids"]:
            used_ids.add(sid)

        if coverage > 300:
            circles.append(item)
        else:
            arcs.append(item)

    return arcs, circles, used_ids


# ============================================================
# 3) Arrowhead detection (your triangle logic + used segments)
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


def triangle_tip(tri_pts):
    p0, p1, p2 = tri_pts
    ang0 = angle(p1, p0, p2)
    ang1 = angle(p0, p1, p2)
    ang2 = angle(p0, p2, p1)
    angs = [ang0, ang1, ang2]
    idx = int(np.argmin(angs))
    return tri_pts[idx], float(angs[idx])


def detect_arrowhead_triangles_with_used_ids(
    segments,
    max_edge_len=12.0,
    min_edge_len=0.2,
    quant_grid=1.0,
    min_area=0.8,
    max_area=300.0,
    min_sharp_angle=10,
    max_sharp_angle=75
):
    """
    Returns arrowheads, used_segment_ids
    """
    # keep only small segments
    small = []
    small_ids = []
    for sid, (p1, p2) in enumerate(segments):
        L = seg_length(p1, p2)
        if min_edge_len <= L <= max_edge_len:
            small.append((p1, p2))
            small_ids.append(sid)

    graph = {}
    edges = set()
    edge_to_segids = {}

    def add_edge(a, b, seg_id):
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)

        key = tuple(sorted([a, b]))
        edges.add(key)
        if key not in edge_to_segids:
            edge_to_segids[key] = []
        edge_to_segids[key].append(seg_id)

    # build graph
    for idx, (p1, p2) in enumerate(small):
        seg_id = small_ids[idx]
        a = round_point(p1, grid=quant_grid)
        b = round_point(p2, grid=quant_grid)
        if a == b:
            continue
        add_edge(a, b, seg_id)

    triangles = set()
    nodes = list(graph.keys())

    for u in nodes:
        nbrs = graph[u]
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v = nbrs[i]
                w = nbrs[j]
                if tuple(sorted([v, w])) in edges:
                    tri = tuple(sorted([u, v, w]))
                    triangles.add(tri)

    arrowheads = []
    used_ids = set()

    for tri in triangles:
        pts = [
            np.array([tri[0][0], tri[0][1]], dtype=np.float32),
            np.array([tri[1][0], tri[1][1]], dtype=np.float32),
            np.array([tri[2][0], tri[2][1]], dtype=np.float32),
        ]

        # area
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        area = abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))
        if not (min_area <= area <= max_area):
            continue

        tip, sharp = triangle_tip(pts)
        if not (min_sharp_angle <= sharp <= max_sharp_angle):
            continue

        # mark segment ids used by this triangle
        u, v, w = tri
        e1 = tuple(sorted([u, v]))
        e2 = tuple(sorted([v, w]))
        e3 = tuple(sorted([u, w]))

        for e in [e1, e2, e3]:
            for sid in edge_to_segids.get(e, []):
                used_ids.add(sid)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

        arrowheads.append({
            "vertices": [(float(p[0]), float(p[1])) for p in pts],
            "tip": (float(tip[0]), float(tip[1])),
            "sharp_angle": sharp,
            "bbox": bbox,
            "area": float(area),
        })

    # dedupe by bbox
    uniq = []
    seen = set()
    for a in arrowheads:
        key = tuple(round(v, 1) for v in a["bbox"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    uniq.sort(key=lambda x: x["sharp_angle"])
    return uniq, used_ids


# ============================================================
# 4) Merge remaining lines (straight edges)
# ============================================================
def chain_segments_simple(segments, join_dist=2.5):
    """
    segments: list of (p1,p2)
    returns list of chains: each chain is [p0,p1,p2,...]
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

        # forward
        while True:
            tip = chain[-1]
            cand = candidates(tip)

            best = None
            best_dist = 1e9

            for ci in cand:
                seg_id = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])
                if not unused[seg_id]:
                    continue

                a, b = segments[seg_id]
                cur, nxt = (a, b) if end_id == 0 else (b, a)

                d = np.linalg.norm(cur - tip)
                if d < best_dist:
                    best_dist = d
                    best = (seg_id, nxt)

            if best is None:
                break

            seg_id, nxt = best
            unused[seg_id] = False
            chain.append(nxt)

        # backward
        while True:
            tip = chain[0]
            cand = candidates(tip)

            best = None
            best_dist = 1e9

            for ci in cand:
                seg_id = int(endpoints[ci, 0])
                end_id = int(endpoints[ci, 1])
                if not unused[seg_id]:
                    continue

                a, b = segments[seg_id]
                cur, nxt = (a, b) if end_id == 0 else (b, a)

                d = np.linalg.norm(cur - tip)
                if d < best_dist:
                    best_dist = d
                    best = (seg_id, nxt)

            if best is None:
                break

            seg_id, nxt = best
            unused[seg_id] = False
            chain.insert(0, nxt)

        chains.append(chain)

    return chains


def chain_is_straight(chain_pts, max_dev=1.0):
    pts = np.asarray(chain_pts, dtype=np.float32)
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
    return best_pair


def merge_remaining_lines(segments, join_dist=2.5, straight_dev=1.0):
    chains = chain_segments_simple(segments, join_dist=join_dist)
    merged = []
    leftovers = []

    for ch in chains:
        if len(ch) >= 3 and chain_is_straight(ch, max_dev=straight_dev):
            merged.append(merge_chain_to_line(ch))
        else:
            for i in range(len(ch) - 1):
                leftovers.append((ch[i], ch[i + 1]))

    return merged, leftovers


# ============================================================
# 5) Preview rendering
# ============================================================
def render_final_preview(all_segments, arcs, circles, arrowheads, merged_lines,
                         out_path="final_pipeline_preview.png", size=2000, pad=30):
    pts = []
    for p1, p2 in all_segments:
        pts.append(p1); pts.append(p2)
    for a in arcs:
        pts.extend(a["points"])
    for c in circles:
        pts.extend(c["points"])
    for m1, m2 in merged_lines:
        pts.append(m1); pts.append(m2)

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

    # all segments light gray
    for p1, p2 in all_segments:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(230, 230, 230), width=1)

    # arcs red
    for a in arcs:
        mapped = [map_pt(p) for p in a["points"]]
        draw.line(mapped, fill=(255, 0, 0), width=3)

    # circles blue
    for c in circles:
        cx, cy = c["center"]
        r = c["radius"]
        C = np.array([cx, cy], dtype=np.float32)
        CX, CY = map_pt(C)
        R = r * scale
        draw.ellipse((CX - R, CY - R, CX + R, CY + R), outline=(0, 0, 255), width=3)

    # arrowheads green triangles + red tip
    for a in arrowheads:
        verts = [np.array(v, dtype=np.float32) for v in a["vertices"]]
        tip = np.array(a["tip"], dtype=np.float32)
        mapped_verts = [map_pt(v) for v in verts]
        draw.polygon(mapped_verts, outline=(0, 200, 0), fill=None)
        tx, ty = map_pt(tip)
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(255, 0, 0))

    # merged lines black
    for p1, p2 in merged_lines:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=2)

    img.save(out_path)
    print(f"✅ Saved: {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    segments, _ = extract_line_segments(svg_file)
    print("Total segments:", len(segments))

    # 1) Detect arcs/circles first
    arcs, circles, used_arc_circle = detect_arcs_and_circles_from_segments(
        segments,
        small_len=1.0,
        join_dist=3.0,
        max_turn_deg=55,
        rmse_thresh=1.2,
        min_radius=8,
        min_coverage_deg=60
    )
    print("Arcs:", len(arcs), "Circles:", len(circles))
    print("Segments used by arcs/circles:", len(used_arc_circle))

    # remove those segments
    remaining1 = [(p1, p2) for sid, (p1, p2) in enumerate(segments) if sid not in used_arc_circle]
    print("Remaining after arc/circle removal:", len(remaining1))

    # 2) Detect arrowheads next
    arrowheads, used_arrow = detect_arrowhead_triangles_with_used_ids(
        remaining1,
        max_edge_len=12.0,
        min_edge_len=0.2,
        quant_grid=1.0,
        min_area=0.8,
        max_area=300.0,
        min_sharp_angle=10,
        max_sharp_angle=75
    )
    print("Arrowheads:", len(arrowheads))
    print("Segments used by arrowheads:", len(used_arrow))

    # remove arrowhead segments
    remaining2 = [(p1, p2) for sid, (p1, p2) in enumerate(remaining1) if sid not in used_arrow]
    print("Remaining after arrowhead removal:", len(remaining2))

    # 3) Merge remaining into straight lines
    merged_lines, leftover = merge_remaining_lines(
        remaining2,
        join_dist=3.0,
        straight_dev=1.0
    )
    print("Merged straight lines:", len(merged_lines))
    print("Leftover (unmerged):", len(leftover))

    # 4) Preview
    render_final_preview(
        all_segments=segments,
        arcs=arcs,
        circles=circles,
        arrowheads=arrowheads,
        merged_lines=merged_lines,
        out_path="final_pipeline_preview.png"
    )
✅ What you get
Output image:
📌 final_pipeline_preview.png
Shows:
Light gray = all SVG segments
Red = detected arcs
Blue = detected circles
Green triangles = arrowheads
Black = merged straight lines
🔧 Important tuning knobs
If arcs are wrongly removed (slanted edges being taken as arcs)
Decrease arc detection sensitivity:
reduce join_dist
reduce max_turn_deg
decrease rmse_thresh to 0.8
If arrowheads are too many
reduce max_edge_len
reduce max_area
If straight lines don’t merge well
increase join_dist to 4.0
increase straight_dev to 1.5
⚠️ Small note
Some of the repo files you uploaded earlier have expired on my side, so if you want me to integrate this into your full YOLO view/dim pipeline later, you may need to upload them again.
If you want, next I can: ✅ export a clean geometry.json with IDs
✅ allow you to query “which dimension matched which line/circle/arc” using arrowhead tips