That’s excellent news ✅ — if your SVG contains the drawing geometry as real vector paths, then you can stop fighting CV noise and extract:
lines
polylines
arcs/curves
circles (sometimes)
directly from the SVG.
This will be much more reliable than Hough/LSD on raster.
✅ What this means practically
If the SVG has vectorized elements, then:
You can do geometry extraction like CAD:
Parse SVG <path>, <line>, <polyline>, <circle>, <ellipse>
Convert to geometric primitives
Use those primitives for dimension matching
This avoids: ❌ double-line detection
❌ missing edges
❌ random circles
❌ hatch noise messing Hough
⚠️ One warning
Most PDF→SVG converters convert everything into paths.
So circles may appear as:
<circle ...> (rare, best case)
<ellipse ...> (sometimes)
<path d="..."> using Bezier curves (most common)
Still fine — we can parse them.
✅ Next step: Extract all geometry from SVG (Python)
Here is a clean script that:
✅ reads SVG
✅ extracts:
straight lines (including from paths)
Bezier curve segments
circles/ellipses if present
✅ saves JSON output
✅ Install
Copy code
Bash
pip install svgpathtools numpy
✅ svg_extract_geometry.py
Copy code
Python
from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc
import numpy as np
import json


def complex_to_xy(z):
    return float(z.real), float(z.imag)


def sample_curve(seg, n=30):
    """Sample points along a curve segment."""
    pts = []
    for t in np.linspace(0, 1, n):
        z = seg.point(t)
        pts.append(complex_to_xy(z))
    return pts


def extract_geometry(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)

    out = {
        "lines": [],
        "curves": [],
        "arcs": [],
        "meta": {
            "svg_width": svg_attr.get("width"),
            "svg_height": svg_attr.get("height"),
            "viewBox": svg_attr.get("viewBox"),
        }
    }

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                x1, y1 = complex_to_xy(seg.start)
                x2, y2 = complex_to_xy(seg.end)
                out["lines"].append({
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                })

            elif isinstance(seg, Arc):
                # Arc has center/radius info
                out["arcs"].append({
                    "start": complex_to_xy(seg.start),
                    "end": complex_to_xy(seg.end),
                    "radius": (float(seg.radius.real), float(seg.radius.imag)),
                    "rotation": float(seg.rotation),
                    "large_arc": bool(seg.large_arc),
                    "sweep": bool(seg.sweep),
                    "samples": sample_curve(seg, n=25)
                })

            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                out["curves"].append({
                    "type": seg.__class__.__name__,
                    "start": complex_to_xy(seg.start),
                    "end": complex_to_xy(seg.end),
                    "samples": sample_curve(seg, n=30)
                })

            else:
                # unknown segment type
                out["curves"].append({
                    "type": str(type(seg)),
                    "samples": sample_curve(seg, n=20)
                })

    return out


if __name__ == "__main__":
    svg_file = "input/drawing.svg"
    geom = extract_geometry(svg_file)

    with open("svg_geometry.json", "w") as f:
        json.dump(geom, f, indent=2)

    print("✅ Saved svg_geometry.json")
    print("Lines:", len(geom["lines"]))
    print("Arcs:", len(geom["arcs"]))
    print("Curves:", len(geom["curves"]))
✅ What you get
A file:
📄 svg_geometry.json
containing:
all extracted straight segments
sampled points for curves/arcs
This is already enough to do matching.
⭐ Bonus: Detect circles from SVG curves (simple trick)
Even if SVG has no <circle>, you can detect circles by:
take curve samples
fit circle using least squares
if fit error is small → it’s a circle
I can give that next.
🔥 Why this will help your “dimension matching”
Now your geometry primitives are clean and stable:
no “double edges”
no missing lines
no random circle hallucinations
So matching becomes a pure geometry problem: arrow tip → nearest SVG line/arc/circle.
If you want, upload your SVG (or a small sample)
I’ll tell you: ✅ whether circles are true circles or Bezier paths
✅ how to extract only “part outline” and ignore hatching/text
✅ how to map SVG coordinates back to your raster image coordinates
Yes ✅ — the easiest way to verify is to render the extracted SVG lines onto an image (so you visually confirm what got extracted).
Below is a complete script that:
✅ reads your SVG
✅ extracts only straight line segments
✅ draws them into a PNG overlay
✅ saves svg_lines_preview.png
✅ Install
Copy code
Bash
pip install svgpathtools pillow numpy
✅ preview_svg_lines.py
Copy code
Python
from svgpathtools import svg2paths2, Line
from PIL import Image, ImageDraw
import numpy as np


def parse_viewbox(svg_attr):
    """
    Returns (minx, miny, width, height)
    """
    vb = svg_attr.get("viewBox", None)
    if vb is None:
        return None
    vals = [float(x) for x in vb.replace(",", " ").split()]
    if len(vals) != 4:
        return None
    return vals[0], vals[1], vals[2], vals[3]


def extract_lines(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)

    lines = []
    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                x1, y1 = float(seg.start.real), float(seg.start.imag)
                x2, y2 = float(seg.end.real), float(seg.end.imag)
                lines.append((x1, y1, x2, y2))

    return lines, svg_attr


def normalize_to_canvas(lines, viewbox, out_w=2000, out_h=2000, pad=20):
    """
    Map SVG coords -> image coords.
    """
    if viewbox is None:
        # fallback: infer bounds from lines
        xs = [p for l in lines for p in (l[0], l[2])]
        ys = [p for l in lines for p in (l[1], l[3])]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        vb_w, vb_h = maxx - minx, maxy - miny
    else:
        minx, miny, vb_w, vb_h = viewbox

    scale_x = (out_w - 2 * pad) / (vb_w if vb_w > 1e-6 else 1.0)
    scale_y = (out_h - 2 * pad) / (vb_h if vb_h > 1e-6 else 1.0)
    scale = min(scale_x, scale_y)

    mapped = []
    for x1, y1, x2, y2 in lines:
        X1 = pad + (x1 - minx) * scale
        Y1 = pad + (y1 - miny) * scale
        X2 = pad + (x2 - minx) * scale
        Y2 = pad + (y2 - miny) * scale
        mapped.append((X1, Y1, X2, Y2))

    return mapped


def draw_lines(lines_xy, out_path="svg_lines_preview.png", out_w=2000, out_h=2000):
    img = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw all lines in black
    for x1, y1, x2, y2 in lines_xy:
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=1)

    img.save(out_path)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    lines, svg_attr = extract_lines(svg_file)
    print("Extracted straight lines:", len(lines))

    viewbox = parse_viewbox(svg_attr)
    mapped = normalize_to_canvas(lines, viewbox, out_w=2000, out_h=2000)

    draw_lines(mapped, out_path="svg_lines_preview.png")
✅ Output
You will get:
📌 svg_lines_preview.png
This will show only the straight segments extracted from the SVG.
🔥 If your SVG contains everything as paths (curves too)
Then straight lines might be few.
In that case, we can extract “line-like segments” from paths by checking if a segment is almost straight (even if it’s stored as a cubic curve).
If you want, I’ll extend the script to: ✅ extract near-straight Bezier segments too.
If you upload your drawing.svg, I can also generate:
a colored overlay (part edges vs dimension lines)
filtering to remove hatching/text path

from svgpathtools import svg2paths2, Line, Arc, CubicBezier, QuadraticBezier

paths, attrs, svg_attr = svg2paths2("input/drawing.svg")

cnt = {"Line":0, "Arc":0, "CubicBezier":0, "QuadraticBezier":0}
for p in paths:
    for seg in p:
        name = seg.__class__.__name__
        if name in cnt:
            cnt[name] += 1

print(cnt)
Sure ✅ Here is one final single Python script that:
✅ Loads an SVG
✅ Extracts:
Lines (Line)
Arcs (Arc)
Curves (CubicBezier / QuadraticBezier) → shown as “arcs/curves”
Circles/Ellipses (if present as SVG shapes)
✅ Draws everything into a verification image
✅ Saves svg_geometry_preview.png
✅ Install
Copy code
Bash
pip install svgpathtools pillow numpy
✅ Final Script: svg_detect_geometry_preview.py
Copy code
Python
import math
import numpy as np
from PIL import Image, ImageDraw

from svgpathtools import svg2paths2, Line, Arc, CubicBezier, QuadraticBezier
import xml.etree.ElementTree as ET


# -----------------------------
# Helpers
# -----------------------------
def complex_to_xy(z):
    return float(z.real), float(z.imag)


def parse_viewbox(svg_attr):
    """
    Returns (minx, miny, width, height) or None
    """
    vb = svg_attr.get("viewBox", None)
    if vb is None:
        return None
    vals = [float(x) for x in vb.replace(",", " ").split()]
    if len(vals) != 4:
        return None
    return vals[0], vals[1], vals[2], vals[3]


def infer_bounds_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def map_xy(x, y, minx, miny, scale, pad):
    X = pad + (x - minx) * scale
    Y = pad + (y - miny) * scale
    return X, Y


def sample_segment(seg, n=30):
    pts = []
    for t in np.linspace(0, 1, n):
        z = seg.point(t)
        pts.append(complex_to_xy(z))
    return pts


def length_line(l):
    x1, y1, x2, y2 = l
    return math.hypot(x2 - x1, y2 - y1)


# -----------------------------
# Extract paths (lines/arcs/curves)
# -----------------------------
def extract_from_paths(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)

    lines = []
    arcs = []
    curves = []

    all_points = []

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                x1, y1 = complex_to_xy(seg.start)
                x2, y2 = complex_to_xy(seg.end)
                lines.append((x1, y1, x2, y2))
                all_points.extend([(x1, y1), (x2, y2)])

            elif isinstance(seg, Arc):
                pts = sample_segment(seg, n=40)
                arcs.append(pts)
                all_points.extend(pts)

            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                pts = sample_segment(seg, n=40)
                curves.append(pts)
                all_points.extend(pts)

            else:
                pts = sample_segment(seg, n=25)
                curves.append(pts)
                all_points.extend(pts)

    return lines, arcs, curves, svg_attr, all_points


# -----------------------------
# Extract circles/ellipses from SVG tags
# -----------------------------
def extract_circles_ellipses(svg_path):
    """
    Extract <circle> and <ellipse> elements if present.
    Returns list of dicts:
    circles: {"cx","cy","r"}
    ellipses: {"cx","cy","rx","ry"}
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    # SVG namespaces handling
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0] + "}"

    circles = []
    ellipses = []

    for elem in root.iter():
        tag = elem.tag
        if tag == f"{ns}circle":
            cx = float(elem.attrib.get("cx", 0))
            cy = float(elem.attrib.get("cy", 0))
            r = float(elem.attrib.get("r", 0))
            if r > 0:
                circles.append({"cx": cx, "cy": cy, "r": r})

        if tag == f"{ns}ellipse":
            cx = float(elem.attrib.get("cx", 0))
            cy = float(elem.attrib.get("cy", 0))
            rx = float(elem.attrib.get("rx", 0))
            ry = float(elem.attrib.get("ry", 0))
            if rx > 0 and ry > 0:
                ellipses.append({"cx": cx, "cy": cy, "rx": rx, "ry": ry})

    return circles, ellipses


# -----------------------------
# Render preview
# -----------------------------
def render_preview(
    lines,
    arcs,
    curves,
    circles,
    ellipses,
    svg_attr,
    all_points,
    out_path="svg_geometry_preview.png",
    out_w=2000,
    out_h=2000,
    pad=30,
    min_line_length=0
):
    # Determine coordinate bounds
    viewbox = parse_viewbox(svg_attr)

    if viewbox is not None:
        minx, miny, vb_w, vb_h = viewbox
    else:
        if len(all_points) == 0:
            raise RuntimeError("No geometry points found in SVG.")
        minx, miny, vb_w, vb_h = infer_bounds_from_points(all_points)

    scale_x = (out_w - 2 * pad) / (vb_w if vb_w > 1e-6 else 1.0)
    scale_y = (out_h - 2 * pad) / (vb_h if vb_h > 1e-6 else 1.0)
    scale = min(scale_x, scale_y)

    img = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw lines (BLACK)
    for (x1, y1, x2, y2) in lines:
        if min_line_length > 0 and length_line((x1, y1, x2, y2)) < min_line_length:
            continue
        X1, Y1 = map_xy(x1, y1, minx, miny, scale, pad)
        X2, Y2 = map_xy(x2, y2, minx, miny, scale, pad)
        draw.line((X1, Y1, X2, Y2), fill=(0, 0, 0), width=1)

    # Draw arcs (RED)
    for pts in arcs:
        mapped = [map_xy(x, y, minx, miny, scale, pad) for (x, y) in pts]
        draw.line(mapped, fill=(255, 0, 0), width=2)

    # Draw curves (ORANGE)
    for pts in curves:
        mapped = [map_xy(x, y, minx, miny, scale, pad) for (x, y) in pts]
        draw.line(mapped, fill=(255, 165, 0), width=2)

    # Draw circles (BLUE)
    for c in circles:
        cx, cy, r = c["cx"], c["cy"], c["r"]
        CX, CY = map_xy(cx, cy, minx, miny, scale, pad)
        R = r * scale
        draw.ellipse((CX - R, CY - R, CX + R, CY + R), outline=(0, 0, 255), width=2)

    # Draw ellipses (PURPLE)
    for e in ellipses:
        cx, cy, rx, ry = e["cx"], e["cy"], e["rx"], e["ry"]
        CX, CY = map_xy(cx, cy, minx, miny, scale, pad)
        RX = rx * scale
        RY = ry * scale
        draw.ellipse((CX - RX, CY - RY, CX + RX, CY + RY), outline=(128, 0, 128), width=2)

    img.save(out_path)
    print(f"✅ Saved preview: {out_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    lines, arcs, curves, svg_attr, all_points = extract_from_paths(svg_file)
    circles, ellipses = extract_circles_ellipses(svg_file)

    print("=== Extracted Geometry ===")
    print("Lines:", len(lines))
    print("Arcs:", len(arcs))
    print("Curves (Bezier):", len(curves))
    print("Circles (SVG tags):", len(circles))
    print("Ellipses (SVG tags):", len(ellipses))

    render_preview(
        lines=lines,
        arcs=arcs,
        curves=curves,
        circles=circles,
        ellipses=ellipses,
        svg_attr=svg_attr,
        all_points=all_points,
        out_path="svg_geometry_preview.png",
        out_w=2000,
        out_h=2000,
        pad=30,
        min_line_length=0   # set to 5/10/20 to hide tiny segments
    )
✅ Output legend (in preview image)
Black → Straight lines
Red → True SVG arcs (Arc)
Orange → Bezier curves (approx arcs/curves)
Blue → <circle> elements
Purple → <ellipse> elements
Saved as:
📌 svg_geometry_preview.png
🔧 If your preview looks “too dense”
Because your SVG has many tiny line segments (hatching / arc approximation).
Set:
Copy code
Python
min_line_length=15
or 25 to remove clutter.
If you upload your SVG file, I can also: ✅ filter out hatching automatically
✅ separate “part outline” vs “dimension lines” based on layer/style/stroke width