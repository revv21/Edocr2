Yes ✅ — here’s a clean preview renderer that draws each category in a different color, so you can visually verify what is being kept/merged.
Categories shown
Light gray → all original segments (context)
Blue → hv_keep (horizontal/vertical kept)
Orange → other_keep (long diagonal kept)
Purple → diag_small (tiny diagonal candidates)
Black → merged_diag (final merged diagonal lines)
Red → leftover_diag (diagonal small segments that didn’t merge)
✅ Code: Preview all categories in different colors
Add this function:
Copy code
Python
import numpy as np
from PIL import Image, ImageDraw

def render_categories_preview(
    all_segments,
    hv_keep,
    other_keep,
    diag_small,
    merged_diag,
    leftover_diag,
    out_path="categories_preview.png",
    size=2000,
    pad=30
):
    # Collect bounds
    pts = []
    for p1, p2 in all_segments:
        pts.append(p1); pts.append(p2)

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

    # 1) Draw ALL segments in light gray (context)
    for p1, p2 in all_segments:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(230, 230, 230), width=1)

    # Helper for category drawing
    def draw_segments(seg_list, color, width):
        for p1, p2 in seg_list:
            x1, y1 = map_pt(p1)
            x2, y2 = map_pt(p2)
            draw.line((x1, y1, x2, y2), fill=color, width=width)

    # 2) Draw categories
    draw_segments(hv_keep,       color=(0, 0, 255),     width=2)  # Blue
    draw_segments(other_keep,    color=(255, 165, 0),   width=2)  # Orange
    draw_segments(diag_small,    color=(160, 32, 240),  width=1)  # Purple
    draw_segments(leftover_diag, color=(255, 0, 0),     width=2)  # Red
    draw_segments(merged_diag,   color=(0, 0, 0),       width=3)  # Black

    # Add legend text
    legend = [
        ("Blue  : HV keep", (0, 0, 255)),
        ("Orange: other keep", (255, 165, 0)),
        ("Purple: diag_small (merge candidates)", (160, 32, 240)),
        ("Black : merged_diag", (0, 0, 0)),
        ("Red   : leftover_diag", (255, 0, 0)),
        ("Gray  : all segments (context)", (120, 120, 120)),
    ]

    y = 10
    for txt, col in legend:
        draw.text((10, y), txt, fill=col)
        y += 22

    img.save(out_path)
    print(f"✅ Saved category preview: {out_path}")
✅ Modify your merge function to RETURN categories
Update your merge function like this:
Copy code
Python
def merge_only_diagonal_small_segments_return_categories(
    segments,
    small_len=3.0,
    hv_tol_deg=8,
    join_dist=3.0,
    straight_dev=1.0
):
    hv_keep = []
    diag_small = []
    other_keep = []

    for p1, p2 in segments:
        L = float(np.linalg.norm(p2 - p1))

        if is_near_horizontal_or_vertical(p1, p2, hv_tol_deg=hv_tol_deg):
            hv_keep.append((p1, p2))
            continue

        if L <= small_len:
            diag_small.append((p1, p2))
        else:
            other_keep.append((p1, p2))

    merged_diag, leftover_diag = merge_remaining_lines(
        diag_small,
        join_dist=join_dist,
        straight_dev=straight_dev
    )

    final_lines = hv_keep + other_keep + merged_diag + leftover_diag

    stats = {
        "hv_keep": len(hv_keep),
        "other_keep": len(other_keep),
        "diag_small": len(diag_small),
        "merged_diag": len(merged_diag),
        "leftover_diag": len(leftover_diag),
    }

    categories = {
        "hv_keep": hv_keep,
        "other_keep": other_keep,
        "diag_small": diag_small,
        "merged_diag": merged_diag,
        "leftover_diag": leftover_diag,
        "final_lines": final_lines
    }

    return final_lines, stats, categories
✅ Use it in your main code
Copy code
Python
final_lines, stats, cats = merge_only_diagonal_small_segments_return_categories(
    remaining2,
    small_len=3.0,
    hv_tol_deg=8,
    join_dist=3.0,
    straight_dev=1.0
)

print("Diagonal merge stats:", stats)

render_categories_preview(
    all_segments=segments,                 # original segments
    hv_keep=cats["hv_keep"],
    other_keep=cats["other_keep"],
    diag_small=cats["diag_small"],
    merged_diag=cats["merged_diag"],
    leftover_diag=cats["leftover_diag"],
    out_path="categories_preview.png"
)
✅ Output
You’ll get:
📌 categories_preview.png
and it will clearly show if:
HV lines are staying correct (blue)
diagonal broken pieces are merging (black)
remaining clutter is still present (red/purple)
If you want, I can also add a “zoomed crop preview” around each merged diagonal line, so you can inspect each merge result one-by-one.