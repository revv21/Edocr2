"""
Standalone DFMEA test — Body Control Module (BCM)
Parses the structured data, builds the form dict, calls Claude,
and writes output to dfmea_bcm.xlsx

Run:
    pip install anthropic openpyxl
    set ANTHROPIC_API_KEY=sk-ant-...
    python test_dfmea.py
"""

import json, io, os, sys
import anthropic
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────────────────────
# YOUR DATA — parsed exactly as provided
# ─────────────────────────────────────────────────────────────────────────────

elements = [
    {"element_id": 1, "name": "Body Control Module",                        "level": "focus"},
    {"element_id": 2, "name": "Processing Unit (Microcontroller & I/O pins)","level": "lower"},
    {"element_id": 3, "name": "Communications (CAN/LIN transceivers)",       "level": "lower"},
    {"element_id": 4, "name": "Dashboard",                                   "level": "higher"},
    {"element_id": 5, "name": "Diagnostic Tool",                             "level": "higher"},
    {"element_id": 6, "name": "Electrical Appliances",                       "level": "higher"},
    {"element_id": 7, "name": "Sensors / Switches / Ajar Inputs",            "level": "higher"},
    {"element_id": 8, "name": "Other ECUs",                                  "level": "higher"},
]

focus_functions_raw = [
    {"function_id": 201, "element_id": 1, "name": "Read sensor input signals"},
    {"function_id": 202, "element_id": 1, "name": "Process sensor data"},
    {"function_id": 203, "element_id": 1, "name": "Generate warning and status messages"},
    {"function_id": 204, "element_id": 1, "name": "Generate gateway communication messages"},
    {"function_id": 205, "element_id": 1, "name": "Generate hardwire control outputs"},
    {"function_id": 206, "element_id": 1, "name": "Generate diagnostic trouble code information"},
    {"function_id": 207, "element_id": 1, "name": "Generate vehicle configuration messages"},
]

higher_functions_raw = [
    {"function_id": 301, "element_id": 7, "name": "Provide vehicle state measurements"},
    {"function_id": 302, "element_id": 4, "name": "Display warnings and status information"},
    {"function_id": 303, "element_id": 6, "name": "Perform electrical actuation"},
    {"function_id": 304, "element_id": 5, "name": "Retrieve diagnostic trouble codes"},
    {"function_id": 305, "element_id": 8, "name": "Receive gateway communication messages"},
    {"function_id": 306, "element_id": 8, "name": "Receive vehicle configuration information"},
]

lower_functions_raw = [
    {"function_id": 101, "element_id": 2, "name": "Read sensor input signals"},
    {"function_id": 102, "element_id": 2, "name": "Generate hardwire control outputs"},
    {"function_id": 103, "element_id": 2, "name": "Detect electrical system malfunctions"},
    {"function_id": 104, "element_id": 2, "name": "Manage internal power supply"},
    {"function_id": 105, "element_id": 2, "name": "Store diagnostic trouble codes"},
    {"function_id": 106, "element_id": 3, "name": "Transmit network messages"},
    {"function_id": 107, "element_id": 3, "name": "Receive network messages"},
]

flows = [
    # Lower → Focus
    {"from_function": 101, "to_function": 201},
    {"from_function": 102, "to_function": 205},
    {"from_function": 103, "to_function": 203},
    {"from_function": 104, "to_function": 202},
    {"from_function": 105, "to_function": 206},
    {"from_function": 106, "to_function": 204},
    {"from_function": 107, "to_function": 204},
    # Focus internal
    {"from_function": 201, "to_function": 202},
    {"from_function": 202, "to_function": 203},
    # Focus → Higher
    {"from_function": 201, "to_function": 301},
    {"from_function": 203, "to_function": 302},
    {"from_function": 205, "to_function": 303},
    {"from_function": 206, "to_function": 304},
    {"from_function": 204, "to_function": 305},
    {"from_function": 207, "to_function": 306},
]

# Noise factors — categorised dict, shared across all elements
# Format: { "Category": ["factor1", "factor2", ...], ... }
NOISE_FACTORS = {
    "Customer Usage": [
        "Continuous high-duty-cycle operation (long-haul trucking)",
        "Frequent cold starts in sub-zero temperatures",
        "Operation in high-vibration off-road or construction environments",
    ],
    "Environmental Factors": [
        "Thermal cycling (-40°C to +85°C operating, storage to +125°C)",
        "Moisture and condensation ingress",
        "Electromagnetic interference (EMI) from ignition, motors, relays",
        "Voltage supply fluctuations (9V–16V, load dump spikes up to 40V)",
        "Mechanical vibration (truck frame PSD loads, road shock)",
    ],
    "Piece to Piece Variation": [
        "PCB trace width and layer stack-up tolerances",
        "Connector contact plating thickness variation",
        "Component parameter spread (ADC reference voltage, pull-up resistor tolerance)",
        "Solder joint quality variation across production batches",
    ],
    "Changes Over Time": [
        "Connector corrosion and fretting at harness interfaces",
        "Aging and component wear over vehicle service life (10+ years)",
        "Electrolytic capacitor ESR increase with age",
        "Software calibration drift after repeated flash memory rewrites",
    ],
    "System Interactions": [
        "CAN/LIN bus signal corruption or loss due to other ECU activity",
        "Sensor signal noise or open/short circuit on input lines",
        "Ground offset shifts from high-current loads sharing ground path",
        "Back-EMF transients from inductive loads (motors, solenoids)",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# BUILD LOOKUP MAPS
# ─────────────────────────────────────────────────────────────────────────────

# function_id → function record
all_fns = {f["function_id"]: f for f in focus_functions_raw + higher_functions_raw + lower_functions_raw}

# element_id → element record
elem_map = {e["element_id"]: e for e in elements}

# focus function_id → list of lower function_ids that feed INTO it
lower_to_focus = {}   # focus_fn_id → [lower_fn_id, ...]
focus_to_higher = {}  # focus_fn_id → [higher_fn_id, ...]

lower_fn_ids  = {f["function_id"] for f in lower_functions_raw}
focus_fn_ids  = {f["function_id"] for f in focus_functions_raw}
higher_fn_ids = {f["function_id"] for f in higher_functions_raw}

for flow in flows:
    src = flow["from_function"]
    dst = flow["to_function"]

    # Lower → Focus
    if src in lower_fn_ids and dst in focus_fn_ids:
        lower_to_focus.setdefault(dst, []).append(src)

    # Focus → Higher
    if src in focus_fn_ids and dst in higher_fn_ids:
        focus_to_higher.setdefault(src, []).append(dst)

# ─────────────────────────────────────────────────────────────────────────────
# BUILD form DICT  (matches generate_dfmea() expected structure)
# ─────────────────────────────────────────────────────────────────────────────

focus_element_name = "Body Control Module"

# Collect unique higher element names for the "higher_element" field
# (multiple higher elements exist — we'll use their actual element names per effect)

# Build focus_functions list
focus_functions_form = []
for ff in focus_functions_raw:
    ff_id = ff["function_id"]

    # Lower connections for this focus function
    lower_conns = []
    for lf_id in lower_to_focus.get(ff_id, []):
        lf = all_fns[lf_id]
        le = elem_map[lf["element_id"]]
        lower_conns.append({
            "lower_element":  le["name"],
            "lower_function": lf["name"],
        })

    # Higher connections for this focus function
    higher_conns = []
    for hf_id in focus_to_higher.get(ff_id, []):
        hf = all_fns[hf_id]
        he = elem_map[hf["element_id"]]
        # Store as "ElementName — FunctionName" so effects are specific
        higher_conns.append(f"{he['name']} — {hf['name']}")

    focus_functions_form.append({
        "desc":               ff["name"],
        "requirement":        "",          # no requirements specified in data
        "lower_connections":  lower_conns,
        "higher_connections": higher_conns,
    })

# Lower elements summary (for context in prompts)
lower_elements_form = []
for elem in elements:
    if elem["level"] == "lower":
        fns = [{"desc": f["name"]} for f in lower_functions_raw if f["element_id"] == elem["element_id"]]
        lower_elements_form.append({"name": elem["name"], "functions": fns})

form = {
    "focus_element":    focus_element_name,
    "higher_element":   "Vehicle Systems",   # umbrella — actual names are in higher_conns strings
    "higher_functions": [f["name"] for f in higher_functions_raw],
    "lower_elements":   lower_elements_form,
    "focus_functions":  focus_functions_form,
    "noise_factors":    NOISE_FACTORS,
}

# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONNECTION MAP (sanity check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CONNECTION MAP")
print("="*70)
for ff in focus_functions_form:
    print(f"\n  FOCUS: {ff['desc']}")
    if ff["lower_connections"]:
        for lc in ff["lower_connections"]:
            print(f"    ← [{lc['lower_element']}] {lc['lower_function']}")
    else:
        print(f"    ← (no lower connections)")
    if ff["higher_connections"]:
        for hc in ff["higher_connections"]:
            print(f"    → {hc}")
    else:
        print(f"    → (no higher connections)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS  (copied from dfmea_app.py)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert DFMEA engineer specializing in automotive electronic control unit (ECU) systems on heavy-duty trucks, following the AIAG-VDA DFMEA methodology (2019).

Rules:
- Return ONLY valid JSON. No markdown fences, no commentary.
- Failure modes must describe HOW the function fails — not WHY.
- Causes must be physical or electrical root-level mechanisms — never restate the failure mode.
- Effects must describe the impact on the connected higher-level function and end consequence to the vehicle/driver.
- Be specific to automotive ECU / truck operating conditions:
  voltage transients, EMI, thermal cycling, vibration, software faults, connector corrosion, bus signal errors.
"""

MAX_TOKENS = 3000
MODEL = "claude-opus-4-5"   # swap to your model string as needed


def prompt_failure_modes(focus_element, focus_function, requirement):
    return f"""Generate an exhaustive list of failure modes for the following focus element function.

Focus Element: {focus_element}
Function: {focus_function}
Requirement: {requirement or "Not specified"}

Generate failure modes across all four archetypes:
1. Complete loss of function (fails to perform at all)
2. Partial / degraded function (too little, too slow, incorrect output)
3. Excessive / unintended function (performs when not commanded, wrong value)
4. Intermittent function (works sometimes, fails unpredictably)

Return JSON:
{{
  "failure_modes": [
    {{
      "id": "FM-01",
      "description": "Specific failure mode statement",
      "archetype": "Loss | Degraded | Excessive | Intermittent"
    }}
  ]
}}"""


def prompt_causes(focus_element, focus_function, failure_mode,
                  lower_element, lower_function, noise_factors):
    # Format categorised noise factors dict into readable prompt block
    noise_str = ""
    for category, factors in noise_factors.items():
        if factors:
            noise_str += f"  [{category}]\n"
            noise_str += "\n".join(f"    - {f}" for f in factors) + "\n"

    return f"""Generate all plausible root causes for the following failure mode.
Causes must originate from the lower-level element and its function, triggered by the noise factors.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Lower-Level Element: {lower_element}
Lower-Level Function: {lower_function}

Noise Factors (grouped by category):
{noise_str}
Rules:
- Each cause must name a specific physical/electrical mechanism
  (e.g. solder joint fatigue, MOSFET gate oxide breakdown, contact fretting, CAN bus dominant stuck fault,
   ADC reference voltage drift, flash memory bit flip, connector pin corrosion)
- State which noise factor triggers it AND its category
- State which design characteristic is inadequate
- Do NOT restate the failure mode as a cause

Return JSON:
{{
  "causes": [
    {{
      "id": "C-01",
      "description": "Specific root cause",
      "noise_category": "Customer Usage | Environmental Factors | Piece to Piece Variation | Changes Over Time | System Interactions",
      "noise_factor": "Specific triggering noise factor",
      "failure_mechanism": "e.g. fatigue | corrosion | EMI | thermal drift | software fault | etc.",
      "design_characteristic": "e.g. material | PCB trace width | connector type | firmware | tolerance"
    }}
  ]
}}"""


def prompt_effects(focus_element, focus_function, failure_mode,
                   higher_element_and_function):
    # higher_element_and_function is "ElementName — FunctionName"
    return f"""Generate all plausible failure effects for the following failure mode.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Connected Higher-Level Element & Function: {higher_element_and_function}

Rules:
- Local effect: how the higher-level element's function is impaired
- End effect: worst-case impact on the truck, driver, or regulatory compliance
- Be specific — avoid generic phrases like "system does not work"
- Set safety_critical: true if this could cause loss of vehicle control, driver injury, or regulatory violation

Return JSON:
{{
  "effects": [
    {{
      "id": "E-01",
      "local_effect": "Impact on the higher-level element function",
      "end_effect": "Impact on vehicle / driver / safety / regulation",
      "safety_critical": false
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE CALLER
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client, user_prompt):
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_dfmea(client, form):
    rows = []
    focus_element  = form["focus_element"]
    focus_functions = form["focus_functions"]
    noise_factors  = form["noise_factors"]

    total_fns = len(focus_functions)

    for fi, ff in enumerate(focus_functions):
        ff_desc      = ff["desc"]
        ff_req       = ff.get("requirement", "")
        lower_conns  = ff.get("lower_connections", [])
        higher_conns = ff.get("higher_connections", [])

        print(f"\n[{fi+1}/{total_fns}] Function: {ff_desc}")

        # ── 1. Failure modes ──────────────────────────────────────────────
        print("  → Generating failure modes...")
        try:
            fm_result = call_claude(client, prompt_failure_modes(focus_element, ff_desc, ff_req))
            failure_modes = fm_result.get("failure_modes", [])
            print(f"  → {len(failure_modes)} failure modes")
        except Exception as e:
            print(f"  ✗ Failure mode generation failed: {e}")
            continue

        for fm in failure_modes:
            fm_desc      = fm["description"]
            fm_archetype = fm.get("archetype", "")

            # ── 2. Causes ─────────────────────────────────────────────────
            all_causes = []
            if lower_conns:
                for conn in lower_conns:
                    le_name = conn["lower_element"]
                    lf_desc = conn["lower_function"]
                    print(f"  → Causes: [{le_name}] {lf_desc}")
                    try:
                        c_result = call_claude(client, prompt_causes(
                            focus_element, ff_desc, fm_desc,
                            le_name, lf_desc, noise_factors
                        ))
                        causes = c_result.get("causes", [])
                        for c in causes:
                            c["lower_element"] = le_name
                            c["lower_function"] = lf_desc
                        all_causes.extend(causes)
                        print(f"     {len(causes)} causes")
                    except Exception as e:
                        print(f"     ✗ {e}")
            else:
                all_causes = [{"id": "C-00", "description": "No lower-level connection defined",
                               "noise_factor": "", "failure_mechanism": "",
                               "design_characteristic": "", "lower_element": "", "lower_function": ""}]

            # ── 3. Effects ────────────────────────────────────────────────
            all_effects = []
            if higher_conns:
                for hf_str in higher_conns:
                    print(f"  → Effects on: {hf_str}")
                    try:
                        e_result = call_claude(client, prompt_effects(
                            focus_element, ff_desc, fm_desc, hf_str
                        ))
                        effects = e_result.get("effects", [])
                        for e in effects:
                            e["higher_function"] = hf_str
                        all_effects.extend(effects)
                        print(f"     {len(effects)} effects")
                    except Exception as e:
                        print(f"     ✗ {e}")
            else:
                all_effects = [{"id": "E-00", "local_effect": "No higher-level connection defined",
                                "end_effect": "", "safety_critical": False, "higher_function": ""}]

            # ── Flatten ───────────────────────────────────────────────────
            for cause in all_causes:
                for effect in all_effects:
                    # Parse higher element name from "ElementName — FunctionName"
                    hf_full = effect.get("higher_function", "")
                    if " — " in hf_full:
                        he_name, hf_name = hf_full.split(" — ", 1)
                    else:
                        he_name, hf_name = "", hf_full

                    rows.append({
                        "Focus Element":          focus_element,
                        "Focus Function":         ff_desc,
                        "Requirement":            ff_req,
                        "Failure Mode ID":        fm.get("id", ""),
                        "Failure Mode":           fm_desc,
                        "Archetype":              fm_archetype,
                        "Cause ID":               cause.get("id", ""),
                        "Cause":                  cause.get("description", ""),
                        "Lower-Level Element":    cause.get("lower_element", ""),
                        "Lower-Level Function":   cause.get("lower_function", ""),
                        "Noise Category":         cause.get("noise_category", ""),
                        "Noise Factor":           cause.get("noise_factor", ""),
                        "Failure Mechanism":      cause.get("failure_mechanism", ""),
                        "Design Characteristic":  cause.get("design_characteristic", ""),
                        "Effect ID":              effect.get("id", ""),
                        "Higher-Level Element":   he_name,
                        "Higher-Level Function":  hf_name,
                        "Local Effect":           effect.get("local_effect", ""),
                        "End Effect":             effect.get("end_effect", ""),
                        "Safety Critical":        "YES" if effect.get("safety_critical") else "NO",
                        "Severity (S)":           "",
                        "Occurrence (O)":         "",
                        "Detection (D)":          "",
                        "Action Priority (AP)":   "",
                        "Recommended Actions":    "",
                    })

    print(f"\n✅ Generation complete — {len(rows)} total rows")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT  (same styling as dfmea_app.py)
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    "Loss":         "FFCCCC",
    "Degraded":     "FFE5CC",
    "Excessive":    "E5CCFF",
    "Intermittent": "CCE5FF",
}

HEADER_GROUPS = {
    "STRUCTURE":       ["Focus Element", "Focus Function", "Requirement"],
    "FAILURE MODE":    ["Failure Mode ID", "Failure Mode", "Archetype"],
    "CAUSE":           ["Cause ID", "Cause", "Lower-Level Element", "Lower-Level Function",
                        "Noise Category", "Noise Factor", "Failure Mechanism", "Design Characteristic"],
    "EFFECT":          ["Effect ID", "Higher-Level Element", "Higher-Level Function",
                        "Local Effect", "End Effect", "Safety Critical"],
    "RISK ASSESSMENT": ["Severity (S)", "Occurrence (O)", "Detection (D)",
                        "Action Priority (AP)", "Recommended Actions"],
}

GROUP_COLORS = {
    "STRUCTURE":       "1A1A2E",
    "FAILURE MODE":    "C0392B",
    "CAUSE":           "1A5276",
    "EFFECT":          "1E8449",
    "RISK ASSESSMENT": "784212",
}


def build_excel(rows, focus_element):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DFMEA"

    all_cols = []
    for group_cols in HEADER_GROUPS.values():
        all_cols.extend(group_cols)

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Row 1 — title
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(all_cols))
    cell = ws.cell(row=1, column=1,
                   value=f"DFMEA — {focus_element}   |   AIAG-VDA 2019   |   Heavy-Duty Truck / BCM")
    cell.font      = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    cell.fill      = PatternFill("solid", fgColor="1A1A2E")
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # Row 2 — group headers
    col_idx = 1
    for group, cols in HEADER_GROUPS.items():
        start, end = col_idx, col_idx + len(cols) - 1
        if start != end:
            ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
        cell = ws.cell(row=2, column=start, value=group)
        cell.font      = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor=GROUP_COLORS[group])
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border    = border
        col_idx = end + 1
    ws.row_dimensions[2].height = 22

    # Row 3 — column headers
    for ci, col in enumerate(all_cols, 1):
        cell = ws.cell(row=3, column=ci, value=col)
        cell.font      = Font(name="Calibri", bold=True, size=9, color="1A1A2E")
        cell.fill      = PatternFill("solid", fgColor="F0F0F0")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border
    ws.row_dimensions[3].height = 32

    # Data rows
    for ri, row in enumerate(rows, 4):
        archetype  = row.get("Archetype", "")
        row_color  = next(
            (color for key, color in ARCHETYPE_COLORS.items() if key.lower() in archetype.lower()),
            None
        )
        for ci, col in enumerate(all_cols, 1):
            val  = row.get(col, "")
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font      = Font(name="Calibri", size=9)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border    = border
            if col == "Safety Critical" and val == "YES":
                cell.font = Font(name="Calibri", size=9, bold=True, color="CC0000")
            if row_color:
                cell.fill = PatternFill("solid", fgColor=row_color)
        ws.row_dimensions[ri].height = 40

    # Column widths
    widths = {
        "Focus Element": 20, "Focus Function": 28, "Requirement": 20,
        "Failure Mode ID": 12, "Failure Mode": 34, "Archetype": 14,
        "Cause ID": 10, "Cause": 36, "Lower-Level Element": 22,
        "Lower-Level Function": 26, "Noise Category": 20, "Noise Factor": 22,
        "Failure Mechanism": 18, "Design Characteristic": 22,
        "Effect ID": 10, "Higher-Level Element": 22, "Higher-Level Function": 26,
        "Local Effect": 34, "End Effect": 34, "Safety Critical": 12,
        "Severity (S)": 10, "Occurrence (O)": 10, "Detection (D)": 10,
        "Action Priority (AP)": 14, "Recommended Actions": 28,
    }
    for ci, col in enumerate(all_cols, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths.get(col, 14)

    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:{get_column_letter(len(all_cols))}3"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    rows = generate_dfmea(client, form)

    out_path = "dfmea_bcm.xlsx"
    excel_bytes = build_excel(rows, "Body Control Module")
    with open(out_path, "wb") as f:
        f.write(excel_bytes)

    print(f"\n📄 Excel saved → {out_path}")
    print(f"   {len(rows)} rows | {len(focus_functions_raw)} focus functions")

https://smailiitmacin-my.sharepoint.com/:x:/g/personal/ed22b063_smail_iitm_ac_in/IQD6efn3-0TtQrB7xEm0dNJCAZGgKGcCG2RQhZ9YImZDSP4?e=eL1Xwa
