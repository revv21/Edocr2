Yes 👍 — below is clean, concrete code that annotates extracted dimension lines, arrowheads, and their targets in a single debug preview.
This is meant for verification, not aesthetics.
🎯 What this annotation preview will show
For each extracted dimension:
Color
Meaning
Green triangle
Arrowhead
Red dot
Arrowhead tip
Blue dot
Arrowhead base midpoint
Purple lines
Dimension lines (what you asked for)
Cyan
Target entity if arc / circle
Orange
Target extension line
Gray
All geometry (context only)
Each dimension will also get a dimension ID label (D0, D1, …).
✅ Annotation Renderer: Dimensions + Lines + Targets
📌 This assumes you already have:
dimensions from extract_dimensions(...)
lines, arcs, circles
arrowheads
✅ Code: render_dimension_annotation_preview
Copy code
Python
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_dimension_annotation_preview(
    all_lines,
    arcs,
    circles,
    dimensions,
    out_path="dimension_annotation_preview.png",
    size=2000,
    pad=30
):
    # -------------------------------------------------
    # Collect bounds
    # -------------------------------------------------
    pts = []
    for p1, p2 in all_lines:
        pts.append(p1); pts.append(p2)
    for a in arcs:
        pts.extend(a["points"])
    for c in circles:
        cx, cy = c["center"]
        pts.append(np.array([cx + c["radius"], cy]))
        pts.append(np.array([cx - c["radius"], cy]))

    pts = np.asarray(pts, dtype=np.float32)
    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())

    w = maxx - minx
    h = maxy - miny
    scale = min(
        (size - 2 * pad) / (w + 1e-6),
        (size - 2 * pad) / (h + 1e-6)
    )

    def map_pt(p):
        X = pad + (p[0] - minx) * scale
        Y = pad + (p[1] - miny) * scale
        return (float(X), float(Y))

    # -------------------------------------------------
    # Canvas
    # -------------------------------------------------
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # -------------------------------------------------
    # 1) Draw context geometry (GRAY)
    # -------------------------------------------------
    for p1, p2 in all_lines:
        draw.line((*map_pt(p1), *map_pt(p2)), fill=(220, 220, 220), width=1)

    # arcs (light cyan)
    for a in arcs:
        pts2 = [map_pt(p) for p in a["points"]]
        draw.line(pts2, fill=(180, 240, 240), width=2)

    # circles (light cyan)
    for c in circles:
        cx, cy = c["center"]
        r = c["radius"]
        C = np.array([cx, cy])
        CX, CY = map_pt(C)
        R = r * scale
        draw.ellipse((CX - R, CY - R, CX + R, CY + R),
                     outline=(180, 240, 240), width=2)

    # -------------------------------------------------
    # 2) Draw dimensions
    # -------------------------------------------------
    for i, dim in enumerate(dimensions):
        arrow = dim["arrowhead"]
        dim_lines = dim["dimension_lines"]
        tgt_type = dim["target_type"]
        tgt = dim["target_entity"]

        # ---- Arrowhead
        verts = [np.array(v) for v in arrow["vertices"]]
        tip = np.array(arrow["tip"])

        draw.polygon([map_pt(v) for v in verts],
                     outline=(0, 180, 0), fill=None)

        tx, ty = map_pt(tip)
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4),
                     fill=(255, 0, 0))

        # ---- Base midpoint
        base_pts = [v for v in verts if np.linalg.norm(v - tip) > 1e-3]
        base_mid = 0.5 * (base_pts[0] + base_pts[1])
        bx, by = map_pt(base_mid)
        draw.ellipse((bx - 4, by - 4, bx + 4, by + 4),
                     fill=(0, 0, 255))

        # ---- Dimension lines (PURPLE)
        for p1, p2 in dim_lines:
            draw.line((*map_pt(p1), *map_pt(p2)),
                      fill=(160, 32, 240), width=3)

        # ---- Target entity highlight
        if tgt_type == "circle":
            cx, cy = tgt["center"]
            r = tgt["radius"]
            C = np.array([cx, cy])
            CX, CY = map_pt(C)
            R = r * scale
            draw.ellipse((CX - R, CY - R, CX + R, CY + R),
                         outline=(0, 200, 200), width=4)

        elif tgt_type == "arc":
            pts2 = [map_pt(p) for p in tgt["points"]]
            draw.line(pts2, fill=(0, 200, 200), width=4)

        elif tgt_type == "extension_line":
            p1, p2 = tgt
            draw.line((*map_pt(p1), *map_pt(p2)),
                      fill=(255, 165, 0), width=4)

        # ---- Label
        lx, ly = map_pt(base_mid + np.array([5, -5]))
        draw.text((lx, ly), f"D{i}", fill=(0, 0, 0))

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
    legend = [
        ("Gray : geometry context", (120, 120, 120)),
        ("Green triangle : arrowhead", (0, 180, 0)),
        ("Red dot : arrow tip", (255, 0, 0)),
        ("Blue dot : arrow base", (0, 0, 255)),
        ("Purple : dimension lines", (160, 32, 240)),
        ("Cyan : measured entity", (0, 200, 200)),
        ("Orange : extension line", (255, 165, 0)),
    ]

    y = 10
    for txt, col in legend:
        draw.text((10, y), txt, fill=col)
        y += 22

    img.save(out_path)
    print(f"✅ Saved dimension annotation preview → {out_path}")
✅ How to call it
Copy code
Python
dimensions = extract_dimensions(
    arrowheads=arrowheads,
    lines=final_lines,     # AFTER diagonal merging
    arcs=arcs,
    circles=circles,
    connect_tol=3.0,
    tip_tol=2.5
)

render_dimension_annotation_preview(
    all_lines=final_lines,
    arcs=arcs,
    circles=circles,
    dimensions=dimensions,
    out_path="dimension_annotation_preview.png"
)
🧪 How to validate correctness (important)
Check the output image:
Purple lines must be only dimension lines
Blue dot must sit at arrow base
Red dot must touch the measured entity
Orange lines must be extension lines
No part geometry should be purple
If something is off, tune:
Copy code
Python
connect_tol = 2.0 → 4.0
tip_tol     = 2.0 → 3.5
🚀 Next steps (recommended)
You’re now at a CAD-semantic level. Next logical steps are:
Pair arrowheads → one dimension object
Compute dimension direction & span
Attach OCR text later
Export dimension objects (JSON / DXF)
If you want, I can help with pairing arrowheads into a single dimension next.


Yes 👍 — this is the right cleanup step.
Below is exact code that:
Removes
dimension lines (from arrowhead base)
leader / extension lines (arrowhead tip → straight line)
arrowheads themselves
Keeps
real part geometry: straight edges, merged diagonals, arcs, circles
Shows a clean preview of only the remaining geometry
This is exactly what you want before exporting CAD geometry or matching entities.
🎯 What will be removed
From your already-extracted dimensions:
dimension_lines
target_entity if target_type == "extension_line"
arrowhead triangles (not drawn at all)
✅ Step 1: Utility – segment equality check
We need a tolerant way to remove lines.
Copy code
Python
import numpy as np

def same_segment(seg1, seg2, tol=1e-3):
    (a1, a2), (b1, b2) = seg1, seg2
    return (
        (np.linalg.norm(a1 - b1) < tol and np.linalg.norm(a2 - b2) < tol) or
        (np.linalg.norm(a1 - b2) < tol and np.linalg.norm(a2 - b1) < tol)
    )
✅ Step 2: Collect all lines to remove
Copy code
Python
def collect_dimension_and_leader_lines(dimensions):
    remove = []

    for dim in dimensions:
        # dimension lines
        for ln in dim["dimension_lines"]:
            remove.append(ln)

        # extension / leader line
        if dim["target_type"] == "extension_line":
            remove.append(dim["target_entity"])

    return remove
✅ Step 3: Remove them from geometry
Copy code
Python
def remove_lines_from_geometry(
    all_lines,
    lines_to_remove,
    tol=1e-3
):
    kept = []

    for ln in all_lines:
        discard = False
        for rem in lines_to_remove:
            if same_segment(ln, rem, tol):
                discard = True
                break
        if not discard:
            kept.append(ln)

    return kept
✅ Step 4: Render remaining geometry only
This preview shows only real part geometry.
Color scheme
Black → straight / diagonal part edges
Blue → arcs
Green → circles
Copy code
Python
from PIL import Image, ImageDraw

def render_remaining_geometry_preview(
    lines,
    arcs,
    circles,
    out_path="remaining_geometry_preview.png",
    size=2000,
    pad=30
):
    # -----------------------------
    # Bounds
    # -----------------------------
    pts = []
    for p1, p2 in lines:
        pts.append(p1); pts.append(p2)
    for a in arcs:
        pts.extend(a["points"])
    for c in circles:
        cx, cy = c["center"]
        pts.append(np.array([cx + c["radius"], cy]))
        pts.append(np.array([cx - c["radius"], cy]))

    pts = np.asarray(pts, dtype=np.float32)
    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())

    w = maxx - minx
    h = maxy - miny
    scale = min(
        (size - 2 * pad) / (w + 1e-6),
        (size - 2 * pad) / (h + 1e-6)
    )

    def map_pt(p):
        return (
            pad + (p[0] - minx) * scale,
            pad + (p[1] - miny) * scale
        )

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # -----------------------------
    # Draw part geometry
    # -----------------------------
    for p1, p2 in lines:
        draw.line((*map_pt(p1), *map_pt(p2)),
                  fill=(0, 0, 0), width=2)

    for a in arcs:
        pts2 = [map_pt(p) for p in a["points"]]
        draw.line(pts2, fill=(0, 0, 255), width=3)

    for c in circles:
        cx, cy = c["center"]
        r = c["radius"]
        C = np.array([cx, cy])
        CX, CY = map_pt(C)
        R = r * scale
        draw.ellipse(
            (CX - R, CY - R, CX + R, CY + R),
            outline=(0, 180, 0),
            width=3
        )

    img.save(out_path)
    print(f"✅ Saved remaining geometry → {out_path}")
✅ Step 5: Put it together (exact usage)
Copy code
Python
# 1) extract dimensions (you already have this)
dimensions = extract_dimensions(
    arrowheads=arrowheads,
    lines=final_lines,
    arcs=arcs,
    circles=circles
)

# 2) collect all dimension + leader lines
remove_lines = collect_dimension_and_leader_lines(dimensions)

print("Removing lines:", len(remove_lines))

# 3) remove them from geometry
clean_lines = remove_lines_from_geometry(
    final_lines,
    remove_lines
)

print("Remaining part lines:", len(clean_lines))

# 4) preview clean part geometry
render_remaining_geometry_preview(
    lines=clean_lines,
    arcs=arcs,
    circles=circles,
    out_path="remaining_geometry_preview.png"
)
✅ What “correct” output should look like
✔ No arrowheads
✔ No dimension baselines
✔ No extension / leader lines
✔ Only true part edges + arcs + circles
This image should look like a dimension-free mechanical drawing.
🚀 Next logical steps (optional)
You are now very close to a true CAD reconstruction:
Export clean_lines + arcs + circles → DXF
Compute topology (vertex graph)
Match dimensions to entities (already mostly done)
Attach OCR later
If you want, I can help you export this to DXF next.