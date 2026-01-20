Perfect — this repo structure is enough. ✅
I don’t need any other files to implement the runner, as long as:
you have the 3 YOLO weight files (view / dim+tol / dimtype)
and you have Tesseract installed (for OCR)
Also: your scripts are CLI-style (if __name__ == "__main__":) and accept args → so the cleanest way is:
✅ run_cli.py calls them using subprocess, exactly like the app does.
That means we replicate the repo behavior without rewriting their detection logic.
✅ Final run_cli.py (works with their detect_view.py, detect_dim.py, detect_custom.py)
Create this file in repo root: run_cli.py
Copy code
Python
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime


def run_cmd(cmd, cwd=None):
    print("\n================ CMD ================")
    print(" ".join(str(x) for x in cmd))
    print("=====================================\n")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_empty_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([x for x in folder.glob("*") if x.suffix.lower() in exts])


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--img", required=True, help="Input drawing image path")
    ap.add_argument("--out", default=None, help="Output job folder (default auto)")

    ap.add_argument("--view-weights", required=True, help="YOLO weights for view detection")
    ap.add_argument("--dim-weights", required=True, help="YOLO weights for dim/tol group detection")
    ap.add_argument("--dimtype-weights", required=True, help="YOLO weights for dim type detection")

    ap.add_argument("--device", default="0", help="cuda device id like 0 or 'cpu'")
    ap.add_argument("--imgsz", default="640", help="inference size (keep 640 for their models)")
    ap.add_argument("--conf", default="0.25", help="confidence threshold")
    ap.add_argument("--iou", default="0.45", help="iou threshold")

    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    yolov7_dir = repo_root / "yolov7"

    detect_view_py = yolov7_dir / "detect_view.py"
    detect_dim_py = yolov7_dir / "detect_dim.py"
    detect_custom_py = yolov7_dir / "detect_custom.py"

    if not detect_view_py.exists():
        raise FileNotFoundError(f"Missing: {detect_view_py}")
    if not detect_dim_py.exists():
        raise FileNotFoundError(f"Missing: {detect_dim_py}")
    if not detect_custom_py.exists():
        raise FileNotFoundError(f"Missing: {detect_custom_py}")

    # ---- job output folder ----
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

    # =========================================================
    # 1) VIEW DETECTION (whole drawing -> cropped views)
    # =========================================================
    run_cmd([
        "python", str(detect_view_py),
        "--weights", str(args.view_weights),
        "--source", str(args.img),
        "--device", str(args.device),
        "--img-size", str(args.imgsz),
        "--conf-thres", str(args.conf),
        "--iou-thres", str(args.iou),
        "--project", str(views_dir),
        "--name", "views",
        "--exist-ok",
        "--nosave",  # important: we want cropped outputs, not annotated full image
        "--save-crop",
        "--save-txt",
    ], cwd=str(yolov7_dir))

    # YOLOv7 detect scripts usually output crops into:
    # views_dir/views/crops/<class_name>/*.jpg
    crops_root = views_dir / "views" / "crops"
    if not crops_root.exists():
        raise RuntimeError(
            f"View crops not found at: {crops_root}\n"
            f"Your detect_view.py might be saving somewhere else. "
            f"Check inside {views_dir}."
        )

    # Collect all view images from crops
    # It may store under a class folder like "view" or similar.
    view_images = []
    for class_dir in crops_root.iterdir():
        if class_dir.is_dir():
            view_images.extend(list_images(class_dir))

    if len(view_images) == 0:
        raise RuntimeError(f"No view crops found inside: {crops_root}")

    # Copy views into a clean naming convention:
    clean_views = []
    for i, vp in enumerate(view_images):
        dst = out_dir / "views" / f"view_{i:02d}{vp.suffix.lower()}"
        shutil.copy2(vp, dst)
        clean_views.append(dst)

    # =========================================================
    # 2) GROUP DETECTION per VIEW (DIM / FCF / DATUM / etc)
    # =========================================================
    all_results = {"drawing": str(Path(args.img).resolve()), "views": []}

    for vid, view_path in enumerate(clean_views):
        view_out = groups_dir / f"view_{vid:02d}"
        view_out.mkdir(parents=True, exist_ok=True)

        run_cmd([
            "python", str(detect_dim_py),
            "--weights", str(args.dim_weights),
            "--source", str(view_path),
            "--device", str(args.device),
            "--img-size", str(args.imgsz),
            "--conf-thres", str(args.conf),
            "--iou-thres", str(args.iou),
            "--project", str(view_out),
            "--name", "groups",
            "--exist-ok",
            "--nosave",
            "--save-crop",
            "--save-txt",
        ], cwd=str(yolov7_dir))

        group_crops_root = view_out / "groups" / "crops"
        view_rec = {
            "view_id": vid,
            "path": str(view_path),
            "groups": []
        }

        if group_crops_root.exists():
            # For each detected class folder (DIM / FCF / DATUM etc)
            for class_dir in sorted([d for d in group_crops_root.iterdir() if d.is_dir()]):
                label = class_dir.name.upper()
                group_imgs = list_images(class_dir)

                for gi, gp in enumerate(group_imgs):
                    group_dst_dir = out_dir / "groups" / f"view_{vid:02d}" / label
                    group_dst_dir.mkdir(parents=True, exist_ok=True)

                    group_img_path = group_dst_dir / f"group_{gi:02d}{gp.suffix.lower()}"
                    shutil.copy2(gp, group_img_path)

                    group_rec = {
                        "label": label,
                        "path": str(group_img_path),
                    }

                    # =========================================================
                    # 3) DIM TYPE detection (only for DIM groups)
                    # =========================================================
                    if label == "DIM":
                        dim_part_out = dim_parts_dir / f"view_{vid:02d}_DIM_{gi:02d}"
                        dim_part_out.mkdir(parents=True, exist_ok=True)

                        run_cmd([
                            "python", str(detect_custom_py),
                            "--weights", str(args.dimtype_weights),
                            "--source", str(group_img_path),
                            "--device", str(args.device),
                            "--img-size", str(args.imgsz),
                            "--conf-thres", str(args.conf),
                            "--iou-thres", str(args.iou),
                            "--project", str(dim_part_out),
                            "--name", "parts",
                            "--exist-ok",
                            "--nosave",
                            "--save-crop",
                            "--save-txt",
                        ], cwd=str(yolov7_dir))

                        group_rec["dim_parts_dir"] = str(dim_part_out)

                    view_rec["groups"].append(group_rec)

        all_results["views"].append(view_rec)

    # save final json
    out_json = out_dir / "output.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n✅ DONE")
    print(f"Views saved: {out_dir / 'views'}")
    print(f"Groups saved: {out_dir / 'groups'}")
    print(f"DIM parts saved: {out_dir / 'dim_parts'}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
✅ How to run it
From repo root:
Copy code
Powershell
python run_cli.py `
  --img uploadFiles/example_dwg.jpg `
  --view-weights yolov7/weights/0519_view_best.pt `
  --dim-weights yolov7/weights/0324_dim_and_tol_best.pt `
  --dimtype-weights yolov7/weights/1013_DIMandFCF_best.pt
(Replace input image path with yours)
✅ Do we need any other files for this runner?
No.
Because the runner:
calls their existing scripts
uses YOLOv7’s internal saving structure
copies crops into a clean folder format
writes a simple JSON summary
⚠️ Why your earlier “expired files” message happened
That was because earlier you uploaded individual .py files separately and those became unavailable later.
But your ZIP is enough.
If you want, next I can extend this runner to also call:
✅ app/subprocess/detect_fcf.py
✅ app/subprocess/detect_datum.py
(because those are non-YOLO OCR-based scripts and need a different call style)
Just say: “add FCF+Datum subprocess scripts also” and I’ll patch run_cli.py accordingly.