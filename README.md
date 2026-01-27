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




