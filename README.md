Got it — and your screenshot confirms two separate issues:
Not all DIM groups are getting “parts” detected
OCR fails badly on rotated/slanted DIM text (very common)
Below is the real fix (not just “try better OCR”).
✅ Problem 1: “Parts not generated for every dimension”
Why it happens
Your type model (0324_dim_and_tol_best.pt) is trained to detect parts inside a clean DIM crop.
But in many drawings:
DIM crops include dimension lines + arrows + extension lines
DIM text is small / low contrast
DIM crop is too tight or too large
DIM crop is rotated
So YOLO type detector returns 0 boxes → no parts saved.
Fix: add padding + upscale + confidence tuning
We should:
pad the DIM crop before running the type detector
upscale small DIM crops (OCR and YOLO both improve)
lower conf threshold slightly for the type detector
✅ Problem 2: OCR is bad for rotated parts
Why it happens
Your current rotation correction is “global deskew”, but rotated dimension text can be:
vertical (90°)
upside-down (180°)
mixed orientation
Fix: do multi-angle OCR voting
For each crop, run OCR on:
0°
90°
180°
270°
Pick the best result based on:
number of valid characters
confidence / length
contains digits (0-9) or symbols (Ø, ±, R)
This works much better than relying on Tesseract OSD.
✅ What I want you to change (exact code)
1) Upgrade OCR: multi-rotation OCR
Replace your recognize_text() with this.
pipeline/ocr/tess_ocr_best.py (UPDATED)
Copy code
Python
import cv2
import numpy as np
import pytesseract
import re


def pad_to_square(image, scale=1.4):
    h, w = image.shape[:2]
    side = int(max(h, w) * scale)
    canvas = np.ones((side, side, 3), dtype=np.uint8) * 255
    y, x = (side - h) // 2, (side - w) // 2
    canvas[y:y+h, x:x+w] = image
    return canvas


def clean_for_ocr(img):
    # Strong binarization to remove lines/noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def score_text(text: str) -> int:
    if not text:
        return 0
    text = text.strip()

    # Reward digits/symbols typically found in dimensions
    digit_count = len(re.findall(r"\d", text))
    sym_count = len(re.findall(r"[Ø±RrMm]", text))

    # Penalize pure junk
    junk = len(re.findall(r"[^0-9A-Za-zØ±\.\-\+\s]", text))

    return 5 * digit_count + 2 * sym_count + len(text) - 2 * junk


def ocr_once(img, lang="eng_math"):
    data = pytesseract.image_to_data(
        img,
        config=f"--oem 3 --psm 6 -l {lang}",
        output_type=pytesseract.Output.DICT
    )
    tokens = []
    for t, conf in zip(data["text"], data["conf"]):
        t = t.strip()
        if t and conf != "-1" and int(float(conf)) > 35:
            tokens.append(t)
    return " ".join(tokens)


def recognize_text_best(image, lang="eng_math"):
    """
    OCR with rotation voting:
    tries 0/90/180/270 and returns best scored result
    """
    img = pad_to_square(image, scale=1.4)

    # Upscale helps small text a LOT
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    img = clean_for_ocr(img)

    candidates = []
    for k, angle in enumerate([0, 90, 180, 270]):
        if angle == 0:
            rot = img
        elif angle == 90:
            rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rot = cv2.rotate(img, cv2.ROTATE_180)
        else:
            rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        text = ocr_once(rot, lang=lang)
        candidates.append((score_text(text), angle, text))

    candidates.sort(reverse=True, key=lambda x: x[0])
    best = candidates[0]
    return best[2]
✅ This will massively improve rotated OCR.
2) Upgrade DIM type detection: pad + upscale + fallback OCR
Key idea
If YOLO type detector returns 0 parts, don’t fail.
Instead:
save the DIM crop anyway
run OCR directly on the DIM crop as a fallback
pipeline/extractors/dim_extractor.py (UPDATED)
Copy code
Python
import cv2
from pathlib import Path
from pipeline.processing.image_ops import crop, save_image
from pipeline.ocr.tess_ocr_best import recognize_text_best


def pad_image(img, pad=20):
    h, w = img.shape[:2]
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255,255,255))


class DIMExtractor:
    def __init__(self, type_detector):
        self.type_detector = type_detector

    def extract(self, dim_image_path: Path, out_parts_dir: Path):
        dim_img = cv2.imread(str(dim_image_path))
        if dim_img is None:
            return []

        # Improve detection stability
        padded = pad_image(dim_img, pad=25)

        # Overwrite the file temporarily for YOLO (path-based)
        tmp_path = out_parts_dir / "_tmp_dim_for_type.jpg"
        save_image(tmp_path, padded)

        # Lower conf a bit for part detection
        part_dets = self.type_detector.detect(str(tmp_path), conf=0.15, iou=0.45)
        part_dets = sorted(part_dets, key=lambda d: d["xyxy"][0])

        results = []

        # --------------------------
        # Case A: parts detected
        # --------------------------
        if len(part_dets) > 0:
            for pi, p in enumerate(part_dets):
                part_label = p["label"]
                part_crop = crop(padded, p["xyxy"])

                # Upscale small crops
                part_crop = cv2.resize(part_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                part_path = out_parts_dir / f"part_{pi:02d}_{part_label}.jpg"
                save_image(part_path, part_crop)

                text = recognize_text_best(part_crop)

                results.append({
                    "part_id": pi,
                    "label": part_label,
                    "bbox": p["xyxy"],
                    "confidence": p["confidence"],
                    "path": str(part_path),
                    "ocr": text
                })

        # --------------------------
        # Case B: fallback if 0 parts
        # --------------------------
        else:
            fallback_path = out_parts_dir / "fallback_full_dim.jpg"
            save_image(fallback_path, padded)

            text = recognize_text_best(padded)

            results.append({
                "part_id": 0,
                "label": "FULL_DIM_FALLBACK",
                "bbox": [0, 0, padded.shape[1], padded.shape[0]],
                "confidence": 0.0,
                "path": str(fallback_path),
                "ocr": text
            })

        # cleanup temp
        try:
            tmp_path.unlink()
        except:
            pass

        return results
✅ Now you will always get something saved and OCR’d per DIM.
3) Small change in runner: ensure every DIM produces folder + output
Your runner is mostly fine, but ensure you always create the parts folder.
In runner.py, inside if label == "DIM":
Copy code
Python
part_folder = dim_parts_dir / f"view_{vi:02d}_DIM_{gi:02d}"
part_folder.mkdir(parents=True, exist_ok=True)

parts = self.dim_extractor.extract(group_path, part_folder)
group_record["parts"] = parts
(you already do this, good)
✅ Why this fixes your screenshot problem
“Not generating parts for every dimension”
Now:
If type model detects parts → saved as part_XX_*.jpg
If it detects nothing → still saved as fallback_full_dim.jpg
So every DIM will have at least 1 output crop.
✅ Why OCR improves for rotated parts
Because we do:
pad → text not clipped
upscale → OCR sees clearer characters
binarize → removes line noise
0/90/180/270 OCR voting → handles rotated dims reliably
This is the biggest improvement you can do without changing models.
🔥 Bonus: Want “best OCR image” saved too?
If you want, I can also save the preprocessed OCR image (after binarize + rotation) for debugging, like:
Copy code

part_03_dimension_ocr_input.jpg
That helps you visually confirm why OCR fails.
If you apply these two updates (tess_ocr_best.py + dim_extractor.py) and run again, you should see:
✅ parts generated for almost all dimensions
✅ rotated text OCR becomes readable
If you want, share 1–2 sample cropped part_*.jpg images and I can tune the preprocessing further (line removal + dilation rules).