Yes ✅ — if you want ZERO diagonal segments left unmerged, we can force it.
What you’re asking:
👉 “If a segment is diagonal (not near horizontal/vertical), it should NOT remain as a raw segment.
It must either become part of a merged line OR be discarded.”
So we will:
✅ keep HV segments as-is
✅ take ALL diagonal segments (small + large) and merge them
✅ do NOT keep diagonal leftovers (drop them)
✅ Code: Merge ALL diagonals, keep NONE
Use this function instead:
Copy code
Python
import numpy as np

def merge_all_diagonals_keep_none(
    segments,
    hv_tol_deg=8,
    join_dist=3.0,
    straight_dev=1.0,
):
    """
    - Keep horizontal/vertical segments as-is
    - Merge ALL diagonal segments (small + large)
    - Drop leftover diagonal segments (so none remain)

    Returns:
      final_lines, stats, categories
    """

    hv_keep = []
    diag_all = []

    for p1, p2 in segments:
        if is_near_horizontal_or_vertical(p1, p2, hv_tol_deg=hv_tol_deg):
            hv_keep.append((p1, p2))
        else:
            diag_all.append((p1, p2))

    # Merge all diagonals
    merged_diag, leftover_diag = merge_remaining_lines(
        diag_all,
        join_dist=join_dist,
        straight_dev=straight_dev
    )

    # IMPORTANT: discard leftover diagonals completely
    final_lines = hv_keep + merged_diag

    stats = {
        "hv_keep": len(hv_keep),
        "diag_all": len(diag_all),
        "merged_diag": len(merged_diag),
        "discarded_leftover_diag": len(leftover_diag),
        "final_lines": len(final_lines),
    }

    categories = {
        "hv_keep": hv_keep,
        "diag_all": diag_all,
        "merged_diag": merged_diag,
        "discarded_leftover_diag": leftover_diag,
        "final_lines": final_lines,
    }

    return final_lines, stats, categories
✅ How to use it
Copy code
Python
final_lines, stats, cats = merge_all_diagonals_keep_none(
    remaining2,
    hv_tol_deg=8,
    join_dist=4.0,
    straight_dev=1.2
)

print(stats)
✅ Preview colors update
Now in preview you should show:
Blue = hv_keep
Black = merged_diag
Red = discarded_leftover_diag (these will NOT be in final)
So you can verify you truly kept no diagonal segments.
⚠️ Warning (important)
This will also remove diagonal things like:
hatch diagonal strokes
chamfer annotations
slanted leader lines
diagonal text strokes (rare)
If you want to avoid deleting useful diagonal geometry, we can add one more rule:
✅ discard leftover diagonals only if they are short
and keep long ones.
But if your goal is clean CAD edges only, your request makes sense.
If you want, I can integrate this into your full pipeline and output:
final_lines.json
final_lines_preview.png

import cv2
import numpy as np

def process_drawing(image_path):
    # 1. Load image and convert to Grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Invert and Threshold (Assuming dark lines on light background)
    # We want white lines on a black background for processing
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # 3. Clean up noise (Morphology)
    # This helps remove small text pixels while keeping long lines
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Extract Lines (Probabilistic Hough Transform)
    # minLineLength: ignores short segments (like parts of letters)
    # maxLineGap: bridges small gaps in the raster line
    lines = cv2.HoughLinesP(clean, 1, np.pi/180, threshold=50, 
                            minLineLength=100, maxLineGap=10)

    # 5. Extract Circles
    # dp: inverse ratio of resolution
    # minDist: distance between centers to avoid double-detecting
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=50, param2=30, minRadius=10, maxRadius=200)

    # Visualization
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for lines

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2) # Blue for circles

    cv2.imshow('Detected Entities', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the function
# process_drawing('your_drawing.png')

import vtracer

# This converts your noisy image into a clean SVG file
vtracer.convert_image_to_svg_py(
    "input_drawing.png", 
    "output_drawing.svg",
    mode="spline",      # Better for curves and engineering lines
    clustering=True,    # Helps group related pixels
    iteration_limit=10
)

import fitz  # PyMuPDF

def annotate_dimensions(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    
    for page in doc:
        # 1. Extract words and their bounding boxes
        words = page.get_text("words") 
        
        for w in words:
            # Coordinates of the word bounding box
            rect = fitz.Rect(w[:4]) 
            text_value = w[4]
            
            # 2. Draw a rectangle annotation around the dimension
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=(0, 1, 0))  # Green border
            annot.update()
            
            # 3. (Optional) Add a "sticky note" or text label next to it
            # This is useful for your 'Checker' to report mistakes
            page.insert_text((rect.x0, rect.y1 + 10), f"Dim: {text_value}", 
                             fontsize=8, color=(1, 0, 0))
            
    doc.save(output_pdf)
    doc.close()

# annotate_dimensions("engineering_drawing.pdf", "annotated_drawing.pdf")


import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line
import math


# -----------------------------
# Helpers
# -----------------------------
def complex_to_xy(z):
    return np.array([float(z.real), float(z.imag)], dtype=np.float32)

def seg_length(p1, p2):
    return float(np.linalg.norm(p2 - p1))


# -----------------------------
# Extract SVG line segments (indexed)
# -----------------------------
def extract_svg_lines(svg_path):
    paths, _, _ = svg2paths2(svg_path)
    segments = []

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                p1 = complex_to_xy(seg.start)
                p2 = complex_to_xy(seg.end)
                if seg_length(p1, p2) > 1e-6:
                    segments.append((p1, p2))

    return segments


# -----------------------------
# Preview long lines only
# -----------------------------
def preview_long_lines(
    all_segments,
    removed_ids,
    min_len=30.0,
    out_path="long_lines_after_removal.png",
    size=2000,
    pad=30
):
    # Keep only remaining segments
    remaining = [
        (p1, p2)
        for sid, (p1, p2) in enumerate(all_segments)
        if sid not in removed_ids
    ]

    # Filter long ones
    long_lines = [
        (p1, p2)
        for (p1, p2) in remaining
        if seg_length(p1, p2) >= min_len
    ]

    print("Total segments           :", len(all_segments))
    print("Removed (arc/arrow)      :", len(removed_ids))
    print("Remaining segments       :", len(remaining))
    print("Long lines (kept)        :", len(long_lines))

    if not long_lines:
        print("⚠️ No long lines found — try reducing min_len")
        return

    # Bounds
    pts = []
    for p1, p2 in long_lines:
        pts.append(p1); pts.append(p2)
    pts = np.asarray(pts, dtype=np.float32)

    minx, miny = float(pts[:, 0].min()), float(pts[:, 1].min())
    maxx, maxy = float(pts[:, 0].max()), float(pts[:, 1].max())
    w = maxx - minx
    h = maxy - miny

    scale = min((size - 2 * pad) / (w + 1e-6),
                (size - 2 * pad) / (h + 1e-6))

    def map_pt(p):
        return (
            pad + (p[0] - minx) * scale,
            pad + (p[1] - miny) * scale
        )

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw all remaining segments faint (context)
    for p1, p2 in remaining:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(220, 220, 220), width=1)

    # Draw long lines bold
    for p1, p2 in long_lines:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=3)

    img.save(out_path)
    print(f"✅ Saved preview: {out_path}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    svg_file = "input/drawing.svg"

    # 1) Extract all segments
    segments = extract_svg_lines(svg_file)

    # 2) Get used IDs from YOUR working detectors
    # Replace these with actual outputs
    used_arc_circle_ids = set()   # from arc/circle detection
    used_arrow_ids = set()        # from arrowhead detection

    removed_ids = used_arc_circle_ids.union(used_arrow_ids)

    # 3) Preview long lines only
    preview_long_lines(
        all_segments=segments,
        removed_ids=removed_ids,
        min_len=30.0,   # 🔧 tune this
        out_path="long_lines_after_removal.png"
    )

