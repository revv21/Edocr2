def merge_diagonal_segments_strict(
    segments,
    hv_tol_deg=8,
    angle_tol_deg=6,
    join_dist=3.0,
    proj_tol=1.2
):
    """
    Merges ALL diagonal segments into long straight lines.
    Keeps NO diagonal leftovers.
    """

    hv_keep = []
    diagonals = []

    # Separate HV vs diagonal
    for p1, p2 in segments:
        ang = segment_angle(p1, p2)
        if abs(ang) <= hv_tol_deg or abs(ang - 90) <= hv_tol_deg:
            hv_keep.append((p1, p2))
        else:
            diagonals.append((p1, p2))

    used = [False] * len(diagonals)
    merged = []

    for i, (p1, p2) in enumerate(diagonals):
        if used[i]:
            continue

        # Start a new chain
        chain = [p1, p2]
        base_dir = (p2 - p1)
        base_dir /= (np.linalg.norm(base_dir) + 1e-6)
        base_angle = segment_angle(p1, p2)

        used[i] = True

        growing = True
        while growing:
            growing = False
            tip_candidates = [chain[0], chain[-1]]

            for j, (q1, q2) in enumerate(diagonals):
                if used[j]:
                    continue

                ang = segment_angle(q1, q2)
                if abs(ang - base_angle) > angle_tol_deg:
                    continue

                for tip_idx, tip in enumerate(tip_candidates):
                    # choose closest endpoint
                    d1 = np.linalg.norm(q1 - tip)
                    d2 = np.linalg.norm(q2 - tip)

                    if min(d1, d2) > join_dist:
                        continue

                    nxt = q2 if d1 < d2 else q1
                    cur = q1 if d1 < d2 else q2

                    # projection test: point must lie on base line
                    v = nxt - chain[0]
                    perp = v - np.dot(v, base_dir) * base_dir
                    if np.linalg.norm(perp) > proj_tol:
                        continue

                    # accept
                    used[j] = True
                    growing = True

                    if tip_idx == 0:
                        chain.insert(0, nxt)
                    else:
                        chain.append(nxt)
                    break

                if growing:
                    break

        # Merge chain into a single line
        pts = np.asarray(chain)
        best_d = -1
        best_pair = None
        for a in pts:
            for b in pts:
                d = np.linalg.norm(b - a)
                if d > best_d:
                    best_d = d
                    best_pair = (a, b)

        if best_pair is not None:
            merged.append(best_pair)

    return hv_keep + merged


Perfect — below is a clean, final rendering code that matches your corrected diagonal-merging logic and clearly shows exactly what survived and what didn’t.
This renderer shows only geometry categories, no SVG magic, no browser smoothing — purely what your algorithm decided.
✅ What this preview shows (very explicitly)
Color
Meaning
Light gray
All original segments (context only)
Blue
Horizontal / Vertical lines kept as-is
Black (thick)
Merged diagonal lines (final output)
Red
Leftover diagonal segments (should be NONE ideally)
If you still see red, your merging logic still has gaps.
If red = 0 → ✅ your diagonal merging is finally correct.
✅ Rendering Code: render_final_merge_preview.py
Copy code
Python
import numpy as np
from PIL import Image, ImageDraw


def render_final_merge_preview(
    all_segments,
    hv_keep,
    merged_diag,
    leftover_diag,
    out_path="final_merge_preview.png",
    size=2000,
    pad=30
):
    """
    all_segments   : list of (p1,p2)  -> context only
    hv_keep        : horizontal/vertical lines kept
    merged_diag    : merged diagonal lines (FINAL)
    leftover_diag  : diagonal segments NOT merged (should be zero)
    """

    # -----------------------------
    # Compute bounds
    # -----------------------------
    pts = []
    for p1, p2 in all_segments:
        pts.append(p1)
        pts.append(p2)

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

    # -----------------------------
    # Create canvas
    # -----------------------------
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # -----------------------------
    # 1) Draw all segments (context)
    # -----------------------------
    for p1, p2 in all_segments:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(220, 220, 220), width=1)

    # -----------------------------
    # 2) Draw HV kept lines (BLUE)
    # -----------------------------
    for p1, p2 in hv_keep:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=2)

    # -----------------------------
    # 3) Draw leftover diagonals (RED)
    # -----------------------------
    for p1, p2 in leftover_diag:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)

    # -----------------------------
    # 4) Draw merged diagonals (BLACK, thick)
    # -----------------------------
    for p1, p2 in merged_diag:
        x1, y1 = map_pt(p1)
        x2, y2 = map_pt(p2)
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=4)

    # -----------------------------
    # Legend
    # -----------------------------
    legend = [
        ("Gray  : original SVG segments", (120, 120, 120)),
        ("Blue  : HV lines kept", (0, 0, 255)),
        ("Black : merged diagonal lines", (0, 0, 0)),
        ("Red   : leftover diagonals (BAD)", (255, 0, 0)),
    ]

    y = 10
    for text, color in legend:
        draw.text((10, y), text, fill=color)
        y += 22

    img.save(out_path)
    print(f"✅ Saved final merge preview → {out_path}")
✅ How to call it (example)
Assuming you already ran your strict diagonal merging:
Copy code
Python
final_lines, stats, cats = merge_diagonal_segments_strict(
    remaining_segments,
    hv_tol_deg=8,
    angle_tol_deg=6,
    join_dist=3.5,
    proj_tol=1.2
)

render_final_merge_preview(
    all_segments=segments,          # original SVG segments
    hv_keep=cats["hv_keep"],        # blue
    merged_diag=cats["merged_diag"],# black
    leftover_diag=cats["leftover_diag"],  # red
    out_path="final_merge_preview.png"
)
✅ What “correct” output looks like
Black slanted edges: long, clean, single segments
Blue vertical/horizontal edges: unchanged
Red lines: ideally ZERO
Gray clutter: only background context
If you still see red:
either angle tolerance is too strict
or projection tolerance too tight
or arc/arrowhead removal didn’t remove all curve fragments
🔧 Final recommended parameters (work well in practice)
Copy code
Python
hv_tol_deg     = 8
angle_tol_deg  = 6
join_dist      = 3.5
proj_tol       = 1.2
If you want next:
✅ automatic failure heatmap (why a diagonal didn’t merge)
✅ per-merged-line zoom crops
✅ direct DXF export
Just say the word.

import math
import numpy as np


def segment_angle(p1, p2):
    v = p2 - p1
    ang = math.degrees(math.atan2(v[1], v[0]))
    if ang < 0:
        ang += 180.0      # undirected line
    return ang            # [0, 180)


def is_near_horizontal_or_vertical(p1, p2, hv_tol_deg=8):
    ang = segment_angle(p1, p2)
    return (ang <= hv_tol_deg) or (abs(ang - 90) <= hv_tol_deg)


def merge_diagonal_segments_strict(
    segments,
    hv_tol_deg=8,
    angle_tol_deg=6,
    join_dist=3.5,
    proj_tol=1.2
):
    """
    Returns:
      final_lines,
      stats,
      cats
    """

    # -------------------------------------------------
    # 1) Split HV vs diagonal
    # -------------------------------------------------
    hv_keep = []
    diagonals = []

    for p1, p2 in segments:
        if is_near_horizontal_or_vertical(p1, p2, hv_tol_deg):
            hv_keep.append((p1, p2))
        else:
            diagonals.append((p1, p2))

    used = [False] * len(diagonals)
    merged_diag = []
    leftover_diag = []

    # -------------------------------------------------
    # 2) Strict diagonal merging
    # -------------------------------------------------
    for i, (p1, p2) in enumerate(diagonals):
        if used[i]:
            continue

        chain = [p1, p2]
        used[i] = True

        base_dir = p2 - p1
        base_dir /= (np.linalg.norm(base_dir) + 1e-6)
        base_ang = segment_angle(p1, p2)

        growing = True
        while growing:
            growing = False

            for j, (q1, q2) in enumerate(diagonals):
                if used[j]:
                    continue

                ang = segment_angle(q1, q2)
                if abs(ang - base_ang) > angle_tol_deg:
                    continue

                for side in ("start", "end"):
                    tip = chain[0] if side == "start" else chain[-1]

                    d1 = np.linalg.norm(q1 - tip)
                    d2 = np.linalg.norm(q2 - tip)

                    if min(d1, d2) > join_dist:
                        continue

                    nxt = q2 if d1 < d2 else q1

                    # projection / collinearity check
                    v = nxt - chain[0]
                    perp = v - np.dot(v, base_dir) * base_dir
                    if np.linalg.norm(perp) > proj_tol:
                        continue

                    # accept
                    used[j] = True
                    growing = True

                    if side == "start":
                        chain.insert(0, nxt)
                    else:
                        chain.append(nxt)
                    break

                if growing:
                    break

        # -------------------------------------------------
        # 3) Merge chain into a single line
        # -------------------------------------------------
        pts = np.asarray(chain, dtype=np.float32)
        best_d = -1
        best_pair = None

        for a in pts:
            for b in pts:
                d = np.linalg.norm(b - a)
                if d > best_d:
                    best_d = d
                    best_pair = (a, b)

        if best_pair is not None:
            merged_diag.append(best_pair)

    # -------------------------------------------------
    # 4) Collect leftovers (diagonals not used)
    # -------------------------------------------------
    for i, (p1, p2) in enumerate(diagonals):
        if not used[i]:
            leftover_diag.append((p1, p2))

    # -------------------------------------------------
    # 5) Outputs
    # -------------------------------------------------
    final_lines = hv_keep + merged_diag

    stats = {
        "hv_keep": len(hv_keep),
        "diag_total": len(diagonals),
        "merged_diag": len(merged_diag),
        "leftover_diag": len(leftover_diag),
        "final_lines": len(final_lines),
    }

    cats = {
        "hv_keep": hv_keep,
        "diag_all": diagonals,
        "merged_diag": merged_diag,
        "leftover_diag": leftover_diag,
        "final_lines": final_lines,
    }

    return final_lines, stats, cats