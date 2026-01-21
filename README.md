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