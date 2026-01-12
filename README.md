import fitz  # PyMuPDF

def extract_paths(pdf_path, page_number=0):
    doc = fitz.open(pdf_path)
    page = doc[page_number]

    drawings = page.get_drawings()
    paths = []

    for d in drawings:
        for item in d["items"]:
            cmd = item[0]

            if cmd == "l":  # line
                x1, y1, x2, y2 = item[1]
                paths.append(("LINE", (x1, y1, x2, y2)))

            elif cmd == "re":  # rectangle
                x, y, w, h = item[1]
                paths.extend([
                    ("LINE", (x, y, x+w, y)),
                    ("LINE", (x+w, y, x+w, y+h)),
                    ("LINE", (x+w, y+h, x, y+h)),
                    ("LINE", (x, y+h, x, y))
                ])

            elif cmd == "c":  # cubic Bezier
                paths.append(("BEZIER", item[1]))

    return paths
def pdf_to_dxf_coords(x, y, page_height):
    return x, page_height - y
import ezdxf

def write_dxf(paths, pdf_path, output_dxf):
    doc = fitz.open(pdf_path)
    page = doc[0]
    page_height = page.rect.height

    dxf = ezdxf.new(setup=True)
    msp = dxf.modelspace()

    for ptype, data in paths:
        if ptype == "LINE":
            x1, y1, x2, y2 = data
            x1, y1 = pdf_to_dxf_coords(x1, y1, page_height)
            x2, y2 = pdf_to_dxf_coords(x2, y2, page_height)
            msp.add_line((x1, y1), (x2, y2))

    dxf.saveas(output_dxf)
pdf_file = "input.pdf"
dxf_file = "output.dxf"

paths = extract_paths(pdf_file)
print(f"Extracted {len(paths)} vector segments")

write_dxf(paths, pdf_file, dxf_file)
print("DXF written successfully")
