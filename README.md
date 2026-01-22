Yes ✅ — SVG-based arrowhead detection is actually cleaner and more reliable than raster, because arrowheads become small triangular shapes made of short line segments.
Below is a direct SVG arrowhead detector that:
✅ loads SVG
✅ extracts all line segments
✅ finds small closed triangles (3 edges)
✅ returns:
triangle vertices
tip point (sharpest vertex)
bounding box
saves a preview image svg_arrowheads_preview.png
⚠️ Note
Some earlier uploaded repo files expired on my side, but this code is standalone and doesn’t need them.
✅ Install
Copy code
Bash
pip install svgpathtools pillow numpy
✅ Final Code: svg_detect_arrowheads.py
Copy code
Python
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


def angle(p1, p2, p3):
    """
    Angle at p2 for triangle p1-p2-p3 (degrees)
    """
    a = p1 - p2
    b = p3 - p2
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return math.degrees(math.acos(cosv))


def triangle_tip(tri_pts):
    """
    tri_pts: list of 3 points (np arrays)
    tip = sharpest corner (min internal angle)
    """
    p0, p1, p2 = tri_pts
    ang0 = angle(p1, p0, p2)
    ang1 = angle(p0, p1, p2)
    ang2 = angle(p0, p2, p1)

    angs = [ang0, ang1, ang2]
    idx = int(np.argmin(angs))
    return tri_pts[idx], float(angs[idx])


def round_point(p, grid=1.0):
    """
    Quantize point so near-equal endpoints can match.
    grid=1.0 means round to integer coords.
    """
    return (round(float(p[0]) / grid) * grid, round(float(p[1]) / grid) * grid)


# -----------------------------
# Extract SVG line segments
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

    return segs, svg_attr


# -----------------------------
# Detect triangles (arrowheads)
# -----------------------------
def detect_arrowhead_triangles(
    segments,
    max_edge_len=10.0,
    min_edge_len=0.3,
    join_tol=1.5,
    quant_grid=1.0,
    min_area=0.5,
    max_area=200.0,
    min_sharp_angle=10,
    max_sharp_angle=75
):
    """
    Detect arrowheads as small closed triangles made of 3 short line segments.
    Returns list of arrowheads:
    [
      {
        "vertices": [p0,p1,p2],
        "tip": tip_point,
        "sharp_angle": angle,
        "bbox": (minx,miny,maxx,maxy)
      }
    ]
    """

    # Keep only small segments (arrowheads are small)
    small = []
    for p1, p2 in segments:
        L = seg_length(p1, p2)
        if min_edge_len <= L <= max_edge_len:
            small.append((p1, p2))

    # Build adjacency by quantized endpoints
    # node -> list of connected nodes
    graph = {}
    edges = set()

    def add_edge(a, b):
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []
        graph[a].append(b)
        graph[b].append(a)
        edges.add(tuple(sorted([a, b])))

    # Insert edges
    for p1, p2 in small:
        a = round_point(p1, grid=quant_grid)
        b = round_point(p2, grid=quant_grid)

        # ignore tiny degenerate edges
        if a == b:
            continue

        add_edge(a, b)

    # Find triangles in the graph: (u,v,w) where all 3 edges exist
    nodes = list(graph.keys())
    triangles = set()

    for u in nodes:
        nbrs = graph[u]
        if len(nbrs) < 2:
            continue
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                v = nbrs[i]
                w = nbrs[j]
                # check if v-w is an edge
                if tuple(sorted([v, w])) in edges:
                    tri = tuple(sorted([u, v, w]))
                    triangles.add(tri)

    arrowheads = []

    for tri in triangles:
        pts = [np.array([tri[0][0], tri[0][1]], dtype=np.float32),
               np.array([tri[1][0], tri[1][1]], dtype=np.float32),
               np.array([tri[2][0], tri[2][1]], dtype=np.float32)]

        # Area filter
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        area = abs(0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)))
        if not (min_area <= area <= max_area):
            continue

        tip, sharp = triangle_tip(pts)
        if not (min_sharp_angle <= sharp <= max_sharp_angle):
            continue

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

    # Remove duplicates (same bbox)
    uniq = []
    seen = set()
    for a in arrowheads:
        key = tuple(round(v, 1) for v in a["bbox"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    # sort by sharpness (more arrow-like)
    uniq.sort(key=lambda x: x["sharp_angle"])
    return uniq


# -----------------------------
# Render preview
# -----------------------------
def render_preview(segments, arrowheads, out_path="svg_arrowheads_preview.png", size=2000, pad=30):
    # bounds from all segments
    pts = []
    for p1, p2 in segments:
        pts.append(p1)
        pts.append(p2)
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

    # draw all segments light gray
    for p1, p2 in segments:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(210, 210, 210), width=1)

    # draw arrowheads
    for a in arrowheads:
        verts = [np.array(v, dtype=np.float32) for v in a["vertices"]]
        tip = np.array(a["tip"], dtype=np.float32)

        mapped_verts = [map_pt(v) for v in verts]
        draw.polygon(mapped_verts, outline=(0, 200, 0), fill=None)

        tx, ty = map_pt(tip)
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), fill=(255, 0, 0))

    img.save(out_path)
    print(f"✅ Saved preview: {out_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    segments, svg_attr = extract_svg_lines(svg_file)
    print("Total SVG line segments:", len(segments))

    arrowheads = detect_arrowhead_triangles(
        segments,
        max_edge_len=12.0,      # tune depending on SVG scale
        min_edge_len=0.2,
        quant_grid=1.0,
        min_area=0.8,
        max_area=300.0,
        min_sharp_angle=10,
        max_sharp_angle=75
    )

    print("Arrowheads detected:", len(arrowheads))

    render_preview(segments, arrowheads, out_path="svg_arrowheads_preview.png")
✅ Output
You’ll get:
📌 svg_arrowheads_preview.png
light gray = all SVG segments
green triangle outlines = detected arrowheads
red dot = arrowhead tip
🔧 If it detects too many / too few arrowheads
SVG units vary a LOT depending on converter.
If too many false triangles:
decrease max_edge_len (like 6.0)
increase min_area (like 3.0)
If too few:
increase max_edge_len (like 20.0)
increase quant_grid slightly (2.0)
If you upload your SVG, I can tune these thresholds exactly so it detects only real arrowheads from dimension lines.