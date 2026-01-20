Got it — this screenshot clearly shows the problem ✅
✅ Root cause
Your detect_view.py does NOT support the flag:
Copy code

--save-crop
So when my runner calls:
Copy code
Bash
python detect_view.py ... --save-crop
it crashes with:
detect_view.py: error: unrecognized arguments: --save-crop
That’s why subprocess returns exit status 2.
✅ Fix
We must remove --save-crop and rely on what their script actually supports:
From your screenshot, detect_view.py supports:
--save-txt
--nosave
--view-img
--rotate-result
--save-conf
etc.
So the correct runner should:
Step 1: run view detection using their supported flags
Step 2: read the output labels (labels/*.txt)
Step 3: crop views ourselves using OpenCV (because their script won’t crop)
Same for detect_dim.py and detect_custom.py if they don’t support cropping.
✅ Updated run_cli.py (NO --save-crop, manual cropping)
Create/replace run_cli.py with this:
Copy code
Python
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import cv2


def run_cmd(cmd, cwd=None):
    print("\n================ CMD ================")
    print(" ".join(str(x) for x in cmd))
    print("=====================================\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """
    Reads YOLO format labels:
    class cx cy w h (normalized)
    Returns list of [x1,y1,x2,y2,class_id]
    """
    boxes = []
    if not label_path.exists():
        return boxes

    lines = label_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx = float(parts[1]) * img_w
        cy = float(parts[2]) * img_h
        bw = float(parts[3]) * img_w
        bh = float(parts[4]) * img_h

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        boxes.append([x1, y1, x2, y2, cls])

    return boxes


def crop_and_save(img, boxes, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, (x1, y1, x2, y2, cls) in enumerate(boxes):
        crop = img[y1:y2, x1:x2]
        out_path = out_dir / f"{prefix}_{i:02d}.jpg"
        cv2.imwrite(str(out_path), crop)
        saved.append({"path": str(out_path), "bbox": [x1, y1, x2, y2], "class_id": cls})
    return saved


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img", required=True, help="Input drawing image path")
    ap.add_argument("--out", default=None, help="Output job folder (default auto)")

    ap.add_argument("--view-weights", required=True)
    ap.add_argument("--dim-weights", required=True)
    ap.add_argument("--dimtype-weights", required=True)

    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", default="640")
    ap.add_argument("--conf", default="0.25")
    ap.add_argument("--iou", default="0.45")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    yolov7_dir = repo_root / "yolov7"

    detect_view_py = yolov7_dir / "detect_view.py"
    detect_dim_py = yolov7_dir / "detect_dim.py"
    detect_custom_py = yolov7_dir / "detect_custom.py"

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = repo_root / "runs_cli" / f"job_{ts}"
    else:
        out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    views_dir = out_dir / "views"
    groups_dir = out_dir / "groups"
    dim_parts_dir = out_dir / "dim_parts"

    ensure_empty_dir(views_dir)
    ensure_empty_dir(groups_dir)
    ensure_empty_dir(dim_parts_dir)

    # =======================
    # 1) VIEW DETECTION
    # =======================
    view_project = out_dir / "_yolo_view"
    ensure_empty_dir(view_project)

    run_cmd([
        "python", str(detect_view_py),
        "--weights", str(args.view_weights),
        "--source", str(args.img),
        "--device", str(args.device),
        "--img-size", str(args.imgsz),
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--project", str(view_project),
        "--name", "views",
        "--exist-ok",
        "--nosave",
        "--save-txt",
    ], cwd=str(yolov7_dir))

    # load original drawing image
    drawing = cv2.imread(str(args.img))
    if drawing is None:
        raise RuntimeError(f"Could not read image: {args.img}")
    H, W = drawing.shape[:2]

    # labels expected here:
    # out/_yolo_view/views/labels/<image_name>.txt
    labels_dir = view_project / "views" / "labels"
    if not labels_dir.exists():
        raise RuntimeError(f"Labels not found at: {labels_dir}")

    label_file = labels_dir / (Path(args.img).stem + ".txt")
    view_boxes = read_yolo_labels(label_file, W, H)

    # crop views ourselves
    views = crop_and_save(drawing, view_boxes, views_dir, "view")

    results = {"drawing": str(Path(args.img).resolve()), "views": []}

    # =======================
    # 2) GROUP DETECTION PER VIEW
    # =======================
    for vid, view in enumerate(views):
        view_path = view["path"]
        view_img = cv2.imread(view_path)
        if view_img is None:
            continue
        h, w = view_img.shape[:2]

        group_project = out_dir / "_yolo_groups" / f"view_{vid:02d}"
        ensure_empty_dir(group_project)

        run_cmd([
            "python", str(detect_dim_py),
            "--weights", str(args.dim_weights),
            "--source", str(view_path),
            "--device", str(args.device),
            "--img-size", str(args.imgsz),
            "--conf-thres", str(args.conf),
            "--iou-thres", str(args.iou),
            "--project", str(group_project),
            "--name", "groups",
            "--exist-ok",
            "--nosave",
            "--save-txt",
        ], cwd=str(yolov7_dir))

        g_labels_dir = group_project / "groups" / "labels"
        g_label_file = g_labels_dir / (Path(view_path).stem + ".txt")

        group_boxes = read_yolo_labels(g_label_file, w, h)

        # save group crops
        group_out_dir = groups_dir / f"view_{vid:02d}"
        groups = crop_and_save(view_img, group_boxes, group_out_dir, "group")

        view_rec = {
            "view_id": vid,
            "path": view_path,
            "bbox": view["bbox"],
            "groups": []
        }

        # =======================
        # 3) DIMTYPE DETECTION FOR EACH GROUP
        # =======================
        for gi, g in enumerate(groups):
            group_img = cv2.imread(g["path"])
            if group_img is None:
                continue
            gh, gw = group_img.shape[:2]

            group_rec = {
                "group_id": gi,
                "path": g["path"],
                "bbox": g["bbox"],
                "class_id": g["class_id"],
            }

            # run dimtype on ALL groups (you can filter if you know class_id mapping)
            part_project = out_dir / "_yolo_dimparts" / f"view_{vid:02d}_group_{gi:02d}"
            ensure_empty_dir(part_project)

            run_cmd([
                "python", str(detect_custom_py),
                "--weights", str(args.dimtype_weights),
                "--source", str(g["path"]),
                "--device", str(args.device),
                "--img-size", str(args.imgsz),
                "--conf-thres", str(args.conf),
                "--iou-thres", str(args.iou),
                "--project", str(part_project),
                "--name", "parts",
                "--exist-ok",
                "--nosave",
                "--save-txt",
            ], cwd=str(yolov7_dir))

            p_labels_dir = part_project / "parts" / "labels"
            p_label_file = p_labels_dir / (Path(g["path"]).stem + ".txt")
            part_boxes = read_yolo_labels(p_label_file, gw, gh)

            part_out_dir = dim_parts_dir / f"view_{vid:02d}_group_{gi:02d}"
            parts = crop_and_save(group_img, part_boxes, part_out_dir, "part")

            group_rec["parts"] = parts
            view_rec["groups"].append(group_rec)

        results["views"].append(view_rec)

    out_json = out_dir / "output.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n✅ DONE")
    print("Output folder:", out_dir)
    print("Views:", views_dir)
    print("Groups:", groups_dir)
    print("Dim parts:", dim_parts_dir)
    print("JSON:", out_json)


if __name__ == "__main__":
    main()
✅ Now it will run without that error
Because we removed unsupported args like:
❌ --save-crop
And we do cropping ourselves from the YOLO label text.
How to run
Copy code
Powershell
python run_cli.py `
  --img uploadFiles/example_dwg.jpg `
  --view-weights yolov7/weights/0519_view_best.pt `
  --dim-weights yolov7/weights/0324_dim_and_tol_best.pt `
  --dimtype-weights yolov7/weights/1013_DIMandFCF_best.pt
Next (FCF + Datum)
Yes, we can integrate them too, but they are in:
📌 app/subprocess/detect_fcf.py
📌 app/subprocess/detect_datum.py
Those likely take an image and output crops/text, so we’ll add them as subprocess calls after group crops are created.
Just say: “add FCF + Datum scripts now” and I’ll patch this runner accordingly.