Got it ✅ — that’s a very good constraint and it will fix most wrong merges.
🎯 New rule:
👉 Only merge chains made of small segments that are diagonal
(i.e., NOT near-horizontal and NOT near-vertical)
So we leave true long horizontal/vertical edges alone, and we only “repair” the broken slanted edges.
✅ What we’ll do
1) Split segments into:
HV segments (near 0° or 90°) → keep as-is
Diagonal small segments → eligible for merging
Other leftover segments → keep as-is
2) Chain + merge ONLY the diagonal small segments.
✅ Code: merge only diagonal small segments
Add this after you remove arcs/circles + arrowheads:
Copy code
Python
import numpy as np
import math

def segment_angle_deg(p1, p2):
    v = p2 - p1
    ang = math.degrees(math.atan2(v[1], v[0]))
    ang = abs(ang)
    if ang > 90:
        ang = 180 - ang
    return ang  # in [0,90]

def is_near_horizontal_or_vertical(p1, p2, hv_tol_deg=8):
    ang = segment_angle_deg(p1, p2)
    return (ang <= hv_tol_deg) or (abs(90 - ang) <= hv_tol_deg)

def merge_only_diagonal_small_segments(segments,
                                      small_len=3.0,
                                      hv_tol_deg=8,
                                      join_dist=3.0,
                                      straight_dev=1.0):
    """
    segments: list of (p1,p2) after arc/circle + arrowhead removal
    Returns:
      final_lines = kept_segments + merged_diagonal_lines
    """
    hv_keep = []
    diag_small = []
    other_keep = []

    for p1, p2 in segments:
        L = float(np.linalg.norm(p2 - p1))

        if is_near_horizontal_or_vertical(p1, p2, hv_tol_deg=hv_tol_deg):
            # do NOT merge HV lines (keep stable)
            hv_keep.append((p1, p2))
            continue

        # diagonal
        if L <= small_len:
            diag_small.append((p1, p2))
        else:
            other_keep.append((p1, p2))

    # Now merge ONLY diagonal small segments
    merged_diag, leftover_diag = merge_remaining_lines(
        diag_small,
        join_dist=join_dist,
        straight_dev=straight_dev
    )

    final_lines = hv_keep + other_keep + merged_diag + leftover_diag
    return final_lines, {
        "hv_keep": len(hv_keep),
        "other_keep": len(other_keep),
        "diag_small": len(diag_small),
        "merged_diag": len(merged_diag),
        "leftover_diag": len(leftover_diag),
    }
Then in your main pipeline replace the old merging call with:
Copy code
Python
final_lines, stats = merge_only_diagonal_small_segments(
    remaining2,
    small_len=3.0,       # merge only small diagonal segments
    hv_tol_deg=8,        # treat ±8° as horizontal/vertical
    join_dist=3.0,
    straight_dev=1.0
)

print("Diagonal merge stats:", stats)
✅ Why this fixes your issue
Your previous merging was wrong because:
it was merging any connected segments
even horizontal/vertical edges (which are already correct)
and sometimes merging parts of shapes incorrectly
Now we only merge: ✅ broken diagonal edges
and we avoid: ❌ messing up axis-aligned edges
🔧 Tuning (important)
If diagonal edges are still not merging:
increase join_dist → 4.0
increase small_len → 5.0
If wrong diagonal merges happen:
decrease straight_dev → 0.7
decrease join_dist → 2.0
If you want, I can also add one more safety rule:
✅ merge diagonal chains only if the chain length after merge is > some minimum length, so we don’t merge random tiny diagonal clutter (like hatch noise).