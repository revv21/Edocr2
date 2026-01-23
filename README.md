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

https://smailiitmacin-my.sharepoint.com/:u:/g/personal/ed22b063_smail_iitm_ac_in/IQBsDueESb-ASqrfZY3A7Z1nAa9IaMW0rY2x63G4LNe6ljA?e=E59w4E
