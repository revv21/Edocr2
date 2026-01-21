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