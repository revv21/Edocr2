Got it 👍 — thanks for the clarification.
We’ll strictly stick to the logic YOU defined earlier, no graph traversal, no new assumptions.
Let me restate your logic clearly and then give corrected code that follows it exactly.
✅ Your logic (unchanged)
For each arrowhead:
Dimension lines
→ lines connected to the base of the arrowhead triangle
Leader / extension line
→ ONLY IF the arrowhead tip is connected directly to a straight or diagonal line
→ that connected line is the leader/extension line
If arrowhead tip hits arc/circle
→ NO leader line (dimension refers directly to entity)
So:
❌ No chain walking
❌ No BFS / graph logic
❌ No “multiple leader segments”
✅ Only what your earlier extract_dimensions() already classified
🔴 Why leader lines were not removed earlier (real bug)
The issue was not logic, it was implementation detail:
dimension["target_entity"] is the leader line
but removal was done by segment equality
and your final_lines list contains merged / normalized segments
while target_entity might be a different object instance (same geometry, different reference)
So the line was logically identified, but not removed correctly.
✅ Correct fix (using YOUR logic)
Key idea
👉 Remove leader lines by geometric proximity, not object identity.
✅ Step 1: helper – geometric line match (robust)
Copy code
Python
import numpy as np

def line_matches(line, target, tol=1e-3):
    """
    Check if two line segments represent the same geometry
    (order-independent)
    """
    (a1, a2) = line
    (b1, b2) = target

    return (
        (np.linalg.norm(a1 - b1) < tol and np.linalg.norm(a2 - b2) < tol) or
        (np.linalg.norm(a1 - b2) < tol and np.linalg.norm(a2 - b1) < tol)
    )
✅ Step 2: collect lines to remove (dimension + leader)
Copy code
Python
def collect_lines_to_remove_from_dimensions(dimensions):
    """
    Uses ONLY your original logic
    """
    remove_lines = []

    for dim in dimensions:
        # 1) dimension lines (base-connected)
        for ln in dim["dimension_lines"]:
            remove_lines.append(ln)

        # 2) leader / extension line (tip-connected to straight line)
        if dim["target_type"] == "extension_line":
            remove_lines.append(dim["target_entity"])

    return remove_lines
✅ Step 3: remove them from final geometry
Copy code
Python
def remove_dimension_and_leader_lines_simple(
    final_lines,
    dimensions,
    tol=1e-3
):
    lines_to_remove = collect_lines_to_remove_from_dimensions(dimensions)

    cleaned = []
    for ln in final_lines:
        should_remove = False
        for rem in lines_to_remove:
            if line_matches(ln, rem, tol):
                should_remove = True
                break
        if not should_remove:
            cleaned.append(ln)

    return cleaned
✅ Step 4: render remaining geometry (unchanged)
Use the same renderer you already have:
Copy code
Python
render_remaining_geometry_preview(
    lines=clean_lines,
    arcs=arcs,
    circles=circles,
    out_path="remaining_geometry_preview.png"
)
✅ Correct usage (full snippet)
Copy code
Python
# dimensions already extracted using YOUR logic
dimensions = extract_dimensions(
    arrowheads=arrowheads,
    lines=final_lines,
    arcs=arcs,
    circles=circles
)

# remove dimension + leader lines (simple logic)
clean_lines = remove_dimension_and_leader_lines_simple(
    final_lines,
    dimensions,
    tol=1e-3
)

print("Before:", len(final_lines))
print("After :", len(clean_lines))

render_remaining_geometry_preview(
    lines=clean_lines,
    arcs=arcs,
    circles=circles,
    out_path="remaining_geometry_preview.png"
)
✅ What you should see now
✔ Dimension baselines gone
✔ Extension / leader lines gone
✔ Arrowheads gone (already not drawn)
✔ Only true part geometry remains
And importantly:
No new logic was introduced.
This is a pure bug fix in removal.
If the output still shows leader lines after this, then the only remaining possibility is:
the line you think is a “leader” was actually classified as a dimension line (base-connected), not extension_line
If that happens, we can debug one arrowhead case visually and tune connect_tol / tip_tol — but the logic stays exactly yours.

import math

EPS_T = 0.005  # same as paper

def match_front_bottom(ef, eb):
    # shared axis: X
    xf_u, xf_l = extent(ef["points"], "x")
    xb_u, xb_l = extent(eb["points"], "x")

    return math.sqrt((xf_u - xb_u)**2 + (xf_l - xb_l)**2) < EPS_T


def match_front_left(ef, el):
    # shared axis: Y
    yf_u, yf_l = extent(ef["points"], "y")
    yl_u, yl_l = extent(el["points"], "y")

    return math.sqrt((yf_u - yl_u)**2 + (yf_l - yl_l)**2) < EPS_T


def match_bottom_left(eb, el):
    # shared axis: Z
    zb_u, zb_l = extent(eb["points"], "y")   # y == z in bottom view
    zl_u, zl_l = extent(el["points"], "x")   # x == z in left view

    return math.sqrt((zb_u - zl_u)**2 + (zb_l - zl_l)**2) < EPS_T