"""
DFMEA Generator — Streamlit App
Approach:
  1. User enters element names + functions at all 3 levels (typed manually)
  2. User connects focus functions → lower-level functions (typed) and
     focus functions → higher-level functions (typed)
  3. App traverses each focus function:
       a. Generates failure modes for the focus function
       b. For each failure mode → generates causes from connected lower-level
          functions + noise factors of that lower-level element
       c. For each failure mode → generates effects on connected higher-level functions
  4. Output: downloadable Excel file (AIAG-VDA table format)

Run:
    pip install streamlit anthropic openpyxl
    streamlit run dfmea_app.py
"""

import json
import io
import streamlit as st
import anthropic
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 3000

SYSTEM_PROMPT = """You are an expert DFMEA engineer specializing in heavy-duty truck systems,
following the AIAG-VDA DFMEA methodology (2019).

Rules:
- Return ONLY valid JSON. No markdown fences, no commentary.
- All output must be grounded in the specific element, function, and conditions provided.
- Failure modes must directly describe HOW the function fails — not WHY.
- Causes must be physical root-level mechanisms, never a restatement of the failure mode.
- Effects must describe the impact on the connected higher-level function, not just repeat the failure mode.
- Be specific to heavy-duty truck operating conditions where relevant:
  road salt, vibration, thermal cycling (-40°C to +125°C), high duty cycles, contamination.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client: anthropic.Anthropic, user_prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def prompt_failure_modes(
    focus_element: str,
    focus_function: str,
    focus_function_requirement: str,
) -> str:
    return f"""Generate an exhaustive list of failure modes for the following focus element function.

Focus Element: {focus_element}
Function: {focus_function}
Requirement / Specification: {focus_function_requirement or "Not specified"}

A failure mode is HOW the function fails. Generate failure modes across all four archetypes:
1. Complete loss of function (fails to perform at all)
2. Partial / degraded function (performs insufficiently — too little, too slow, too weak)
3. Excessive / unintended function (performs too much, at wrong time, or when not commanded)
4. Intermittent function (performs sometimes but not consistently)

Return JSON:
{{
  "failure_modes": [
    {{
      "id": "FM-01",
      "description": "Clear, specific failure mode statement",
      "archetype": "Loss | Degraded | Excessive | Intermittent"
    }}
  ]
}}"""


def prompt_causes(
    focus_element: str,
    focus_function: str,
    failure_mode: str,
    lower_element: str,
    lower_function: str,
    noise_factors: list[str],
) -> str:
    noise_str = "\n".join(f"  - {n}" for n in noise_factors) if noise_factors else "  - None specified"
    return f"""Generate all plausible root causes for the following failure mode.
Causes must come from the lower-level element and its function, triggered by the noise factors.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Lower-Level Element: {lower_element}
Lower-Level Function: {lower_function}

Noise Factors acting on {lower_element}:
{noise_str}

Rules for causes:
- Each cause must be a specific physical mechanism (e.g. fatigue cracking, corrosion pitting,
  compression set, dimensional creep, fretting wear, dielectric breakdown)
- State which noise factor triggers it
- State which design characteristic of the lower-level element is inadequate
- Do NOT restate the failure mode as a cause

Return JSON:
{{
  "causes": [
    {{
      "id": "C-01",
      "description": "Physical root cause statement",
      "noise_factor": "Which noise factor triggers this",
      "failure_mechanism": "e.g. fatigue | corrosion | wear | deformation | contamination | etc.",
      "design_characteristic": "Which characteristic is inadequate (material | geometry | tolerance | coating | surface finish | etc.)"
    }}
  ]
}}"""


def prompt_effects(
    focus_element: str,
    focus_function: str,
    failure_mode: str,
    higher_element: str,
    higher_function: str,
) -> str:
    return f"""Generate all plausible failure effects for the following failure mode.
Effects must describe the impact on the connected higher-level element function,
and the resulting impact on the vehicle and driver.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Higher-Level Element: {higher_element}
Connected Higher-Level Function: {higher_function}

Rules for effects:
- Local effect: how the higher-level element's function is impaired
- End effect: worst-case impact on the truck, driver, or regulatory compliance
- Be specific — avoid generic statements like "system does not work"
- Flag if this failure could cause loss of vehicle control, driver injury, or FMVSS violation

Return JSON:
{{
  "effects": [
    {{
      "id": "E-01",
      "local_effect": "Impact on {higher_element} — {higher_function}",
      "end_effect": "Impact on vehicle / driver / safety / regulation",
      "safety_critical": true
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_dfmea(client: anthropic.Anthropic, form: dict) -> list[dict]:
    """
    Traverse each focus function:
      1. Generate failure modes
      2. For each FM × connected lower function → generate causes
      3. For each FM × connected higher function → generate effects
    Returns flat list of DFMEA rows.
    """
    rows = []

    focus_element = form["focus_element"]
    higher_element = form["higher_element"]
    lower_elements = form["lower_elements"]        # list of {name, functions: [{desc, noise_factors}]}
    focus_functions = form["focus_functions"]      # list of {desc, requirement, lower_connections, higher_connections}

    # Build lookup: lower element name → {function_desc → noise_factors}
    lower_lookup: dict[str, dict] = {}
    for le in lower_elements:
        lower_lookup[le["name"]] = {}
        for lf in le["functions"]:
            lower_lookup[le["name"]][lf["desc"]] = lf.get("noise_factors", [])

    # Build lookup: higher function desc (we only have one higher element)
    higher_functions = form["higher_functions"]  # list of str

    status = st.status("Generating DFMEA...", expanded=True)

    for ff in focus_functions:
        ff_desc = ff["desc"]
        ff_req  = ff.get("requirement", "")
        lower_conns  = ff.get("lower_connections", [])   # list of {lower_element, lower_function}
        higher_conns = ff.get("higher_connections", [])  # list of str (higher function descs)

        status.write(f"**Function:** {ff_desc}")

        # ── Step 1: Failure modes ──────────────────────────────────────────
        status.write("  → Generating failure modes...")
        try:
            fm_result = call_claude(client, prompt_failure_modes(focus_element, ff_desc, ff_req))
            failure_modes = fm_result.get("failure_modes", [])
        except Exception as e:
            st.warning(f"Failed to generate failure modes for '{ff_desc}': {e}")
            continue

        status.write(f"  → {len(failure_modes)} failure modes found")

        for fm in failure_modes:
            fm_desc     = fm["description"]
            fm_archetype = fm.get("archetype", "")

            # ── Step 2: Causes (per connected lower function) ──────────────
            all_causes = []
            if lower_conns:
                for conn in lower_conns:
                    le_name = conn["lower_element"]
                    lf_desc = conn["lower_function"]
                    noise   = lower_lookup.get(le_name, {}).get(lf_desc, [])

                    status.write(f"  → Causes from [{le_name}] {lf_desc}")
                    try:
                        c_result = call_claude(client, prompt_causes(
                            focus_element, ff_desc, fm_desc,
                            le_name, lf_desc, noise
                        ))
                        causes = c_result.get("causes", [])
                        # Tag each cause with its source element
                        for c in causes:
                            c["lower_element"] = le_name
                            c["lower_function"] = lf_desc
                        all_causes.extend(causes)
                    except Exception as e:
                        st.warning(f"Cause generation failed: {e}")
            else:
                all_causes = [{"id": "C-00", "description": "No lower-level connection defined",
                               "noise_factor": "", "failure_mechanism": "", "design_characteristic": "",
                               "lower_element": "", "lower_function": ""}]

            # ── Step 3: Effects (per connected higher function) ────────────
            all_effects = []
            if higher_conns:
                for hf_desc in higher_conns:
                    status.write(f"  → Effects on [{higher_element}] {hf_desc}")
                    try:
                        e_result = call_claude(client, prompt_effects(
                            focus_element, ff_desc, fm_desc,
                            higher_element, hf_desc
                        ))
                        effects = e_result.get("effects", [])
                        for e in effects:
                            e["higher_function"] = hf_desc
                        all_effects.extend(effects)
                    except Exception as e:
                        st.warning(f"Effect generation failed: {e}")
            else:
                all_effects = [{"id": "E-00", "local_effect": "No higher-level connection defined",
                                "end_effect": "", "safety_critical": False, "higher_function": ""}]

            # ── Flatten: each row = one cause × one effect ─────────────────
            for cause in all_causes:
                for effect in all_effects:
                    rows.append({
                        # Structure
                        "Focus Element":          focus_element,
                        "Focus Function":         ff_desc,
                        "Requirement":            ff_req,
                        # Failure mode
                        "Failure Mode ID":        fm.get("id", ""),
                        "Failure Mode":           fm_desc,
                        "Archetype":              fm_archetype,
                        # Cause
                        "Cause ID":               cause.get("id", ""),
                        "Cause":                  cause.get("description", ""),
                        "Lower-Level Element":    cause.get("lower_element", ""),
                        "Lower-Level Function":   cause.get("lower_function", ""),
                        "Noise Factor":           cause.get("noise_factor", ""),
                        "Failure Mechanism":      cause.get("failure_mechanism", ""),
                        "Design Characteristic":  cause.get("design_characteristic", ""),
                        # Effect
                        "Effect ID":              effect.get("id", ""),
                        "Higher-Level Element":   higher_element,
                        "Higher-Level Function":  effect.get("higher_function", ""),
                        "Local Effect":           effect.get("local_effect", ""),
                        "End Effect":             effect.get("end_effect", ""),
                        "Safety Critical":        "YES" if effect.get("safety_critical") else "NO",
                        # Engineer fills these in
                        "Severity (S)":           "",
                        "Occurrence (O)":         "",
                        "Detection (D)":          "",
                        "Action Priority (AP)":   "",
                        "Recommended Actions":    "",
                    })

    status.update(label=f"✅ Done — {len(rows)} DFMEA rows generated", state="complete")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    "Loss":         "FFCCCC",
    "Degraded":     "FFE5CC",
    "Excessive":    "E5CCFF",
    "Intermittent": "CCE5FF",
}

HEADER_GROUPS = {
    "STRUCTURE":      ["Focus Element", "Focus Function", "Requirement"],
    "FAILURE MODE":   ["Failure Mode ID", "Failure Mode", "Archetype"],
    "CAUSE":          ["Cause ID", "Cause", "Lower-Level Element", "Lower-Level Function",
                       "Noise Factor", "Failure Mechanism", "Design Characteristic"],
    "EFFECT":         ["Effect ID", "Higher-Level Element", "Higher-Level Function",
                       "Local Effect", "End Effect", "Safety Critical"],
    "RISK ASSESSMENT":["Severity (S)", "Occurrence (O)", "Detection (D)",
                       "Action Priority (AP)", "Recommended Actions"],
}

GROUP_COLORS = {
    "STRUCTURE":       "1A1A2E",
    "FAILURE MODE":    "C0392B",
    "CAUSE":           "1A5276",
    "EFFECT":          "1E8449",
    "RISK ASSESSMENT": "784212",
}


def build_excel(rows: list[dict], focus_element: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DFMEA"

    # All column names in order
    all_cols = []
    for group_cols in HEADER_GROUPS.values():
        all_cols.extend(group_cols)

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Row 1: Title ──
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(all_cols))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = f"DFMEA — {focus_element}   |   AIAG-VDA 2019   |   Heavy-Duty Truck"
    title_cell.font = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    title_cell.fill = PatternFill("solid", fgColor="1A1A2E")
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # ── Row 2: Group headers ──
    col_idx = 1
    for group, cols in HEADER_GROUPS.items():
        start = col_idx
        end   = col_idx + len(cols) - 1
        if start == end:
            cell = ws.cell(row=2, column=start, value=group)
        else:
            ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
            cell = ws.cell(row=2, column=start, value=group)
        cell.font      = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor=GROUP_COLORS[group])
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border    = border
        col_idx = end + 1
    ws.row_dimensions[2].height = 22

    # ── Row 3: Column headers ──
    for ci, col in enumerate(all_cols, start=1):
        cell = ws.cell(row=3, column=ci, value=col)
        cell.font      = Font(name="Calibri", bold=True, size=9, color="1A1A2E")
        cell.fill      = PatternFill("solid", fgColor="F0F0F0")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border
    ws.row_dimensions[3].height = 32

    # ── Data rows ──
    for ri, row in enumerate(rows, start=4):
        archetype = row.get("Archetype", "")
        # Pick row tint based on archetype keyword
        row_color = None
        for key, color in ARCHETYPE_COLORS.items():
            if key.lower() in archetype.lower():
                row_color = color
                break

        for ci, col in enumerate(all_cols, start=1):
            val  = row.get(col, "")
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font      = Font(name="Calibri", size=9)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border    = border

            # Safety critical → red text
            if col == "Safety Critical" and val == "YES":
                cell.font = Font(name="Calibri", size=9, bold=True, color="CC0000")

            # Archetype tint
            if row_color:
                cell.fill = PatternFill("solid", fgColor=row_color)

        ws.row_dimensions[ri].height = 40

    # ── Column widths ──
    col_widths = {
        "Focus Element": 18, "Focus Function": 28, "Requirement": 24,
        "Failure Mode ID": 12, "Failure Mode": 32, "Archetype": 14,
        "Cause ID": 10, "Cause": 34, "Lower-Level Element": 18,
        "Lower-Level Function": 24, "Noise Factor": 20,
        "Failure Mechanism": 18, "Design Characteristic": 20,
        "Effect ID": 10, "Higher-Level Element": 18, "Higher-Level Function": 24,
        "Local Effect": 34, "End Effect": 34, "Safety Critical": 12,
        "Severity (S)": 10, "Occurrence (O)": 10, "Detection (D)": 10,
        "Action Priority (AP)": 14, "Recommended Actions": 28,
    }
    for ci, col in enumerate(all_cols, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = col_widths.get(col, 14)

    # ── Freeze panes ──
    ws.freeze_panes = "A4"

    # ── Auto-filter ──
    ws.auto_filter.ref = f"A3:{get_column_letter(len(all_cols))}3"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="DFMEA Generator", layout="wide", page_icon="⚙️")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stButton > button {
        background: #1A1A2E; color: #E6A817;
        border: 1.5px solid #E6A817; border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace; font-weight: 700;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover { background: #E6A817; color: #1A1A2E; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ DFMEA Generator")
    st.caption("AIAG-VDA 2019 · Heavy-Duty Truck Systems")
    st.divider()

    # ── API Key ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔑 API Key")
        api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        st.divider()
        st.markdown("### How it works")
        st.markdown("""
1. **Define** element names
2. **Enter functions** at all 3 levels
3. **Connect** focus functions to relevant lower and higher functions
4. **Generate** — the app traverses each focus function:
   - Generates failure modes
   - Generates causes from connected lower functions + noise factors
   - Generates effects on connected higher functions
5. **Download** Excel file
        """)

    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar to get started.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Element Names
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Element Names")
    col1, col2, col3 = st.columns(3)
    with col1:
        higher_element = st.text_input("Higher-Level Element", placeholder="e.g. Air Brake Actuator Assembly")
    with col2:
        focus_element = st.text_input("Focus Element", placeholder="e.g. Brake Diaphragm")
    with col3:
        n_lower = st.number_input("Number of Lower-Level Elements", min_value=1, max_value=8, value=2, step=1)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Higher-Level Functions
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"### Step 2 — Higher-Level Functions  ·  *{higher_element or 'Higher Element'}*")
    n_hf = st.number_input("Number of higher-level functions", min_value=1, max_value=10, value=2, step=1, key="n_hf")
    higher_functions = []
    for i in range(int(n_hf)):
        hf = st.text_input(f"HF-{i+1}", placeholder="e.g. Convert air pressure to clamping force", key=f"hf_{i}")
        higher_functions.append(hf)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Lower-Level Elements + Functions + Noise Factors
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 3 — Lower-Level Elements, Functions & Noise Factors")
    lower_elements = []
    for li in range(int(n_lower)):
        with st.expander(f"Lower-Level Element {li+1}", expanded=(li == 0)):
            le_name = st.text_input("Element Name", placeholder="e.g. Diaphragm Bead", key=f"le_name_{li}")
            n_lf = st.number_input("Number of functions", min_value=1, max_value=8, value=2, step=1, key=f"n_lf_{li}")
            le_functions = []
            for fi in range(int(n_lf)):
                c1, c2 = st.columns([2, 3])
                with c1:
                    lf_desc = st.text_input(
                        f"Function {fi+1}",
                        placeholder="e.g. Maintain sealed interface with housing",
                        key=f"lf_{li}_{fi}"
                    )
                with c2:
                    noise_raw = st.text_input(
                        f"Noise Factors (comma-separated)",
                        placeholder="e.g. road salt, thermal cycling, vibration",
                        key=f"noise_{li}_{fi}"
                    )
                    noise_factors = [n.strip() for n in noise_raw.split(",") if n.strip()]
                le_functions.append({"desc": lf_desc, "noise_factors": noise_factors})
            lower_elements.append({"name": le_name, "functions": le_functions})

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Focus Element Functions + Connections
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"### Step 4 — Focus Functions & Connections  ·  *{focus_element or 'Focus Element'}*")
    n_ff = st.number_input("Number of focus functions", min_value=1, max_value=10, value=2, step=1, key="n_ff")

    focus_functions = []

    for fi in range(int(n_ff)):
        with st.expander(f"Focus Function {fi+1}", expanded=True):
            ff_desc = st.text_input(
                "Function description",
                placeholder="e.g. Transmit air pressure force to push rod",
                key=f"ff_desc_{fi}"
            )
            ff_req = st.text_input(
                "Requirement / Specification",
                placeholder="e.g. Withstand 120 psi max operating pressure",
                key=f"ff_req_{fi}"
            )

            st.markdown("**↑ Connect to Higher-Level Functions**")
            st.caption("Type the higher-level function descriptions this focus function supports (one per line)")
            hf_conn_raw = st.text_area(
                "Higher connections",
                placeholder="\n".join(higher_functions[:2]) if higher_functions else "Paste higher-level function here",
                height=80,
                key=f"hf_conn_{fi}",
                label_visibility="collapsed"
            )
            higher_connections = [l.strip() for l in hf_conn_raw.splitlines() if l.strip()]

            st.markdown("**↓ Connect to Lower-Level Functions**")
            st.caption("For each connection, select the lower-level element and type the function description")

            n_lc = st.number_input(
                "Number of lower connections",
                min_value=0, max_value=10, value=1, step=1,
                key=f"n_lc_{fi}"
            )
            lower_connections = []
            le_names = [le["name"] for le in lower_elements if le["name"]]
            for lci in range(int(n_lc)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    le_choice = st.selectbox(
                        "Lower-Level Element",
                        options=le_names if le_names else ["(define elements above)"],
                        key=f"lc_elem_{fi}_{lci}"
                    )
                with c2:
                    # Show available functions for selected element as hint
                    hint_fns = []
                    for le in lower_elements:
                        if le["name"] == le_choice:
                            hint_fns = [lf["desc"] for lf in le["functions"] if lf["desc"]]
                    lf_choice = st.selectbox(
                        "Lower-Level Function",
                        options=hint_fns if hint_fns else ["(define functions above)"],
                        key=f"lc_fn_{fi}_{lci}"
                    )
                lower_connections.append({"lower_element": le_choice, "lower_function": lf_choice})

            focus_functions.append({
                "desc": ff_desc,
                "requirement": ff_req,
                "higher_connections": higher_connections,
                "lower_connections": lower_connections,
            })

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 5 — Generate DFMEA")

    # Validation
    errors = []
    if not focus_element:
        errors.append("Focus element name is required")
    if not higher_element:
        errors.append("Higher-level element name is required")
    if not any(ff["desc"] for ff in focus_functions):
        errors.append("At least one focus function is required")

    if errors:
        for e in errors:
            st.warning(e)
    else:
        if st.button("⚙️  GENERATE DFMEA", use_container_width=True):
            form = {
                "focus_element":   focus_element,
                "higher_element":  higher_element,
                "higher_functions": [hf for hf in higher_functions if hf],
                "lower_elements":  lower_elements,
                "focus_functions": [ff for ff in focus_functions if ff["desc"]],
            }

            with st.container():
                rows = generate_dfmea(client, form)

            if rows:
                st.success(f"Generated **{len(rows)}** DFMEA rows across **{len([ff for ff in focus_functions if ff['desc']])}** focus functions.")

                # Preview table
                st.markdown("#### Preview (first 20 rows)")
                preview_cols = ["Focus Function", "Failure Mode", "Archetype",
                                "Cause", "Noise Factor", "Local Effect", "End Effect", "Safety Critical"]
                preview = [{c: r[c] for c in preview_cols} for r in rows[:20]]
                st.dataframe(preview, use_container_width=True)

                # Download
                excel_bytes = build_excel(rows, focus_element)
                st.download_button(
                    label="📥  Download DFMEA Excel",
                    data=excel_bytes,
                    file_name=f"DFMEA_{focus_element.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.error("No rows were generated. Check that functions and connections are filled in.")


if __name__ == "__main__":
    main()



"""
DFMEA Generator — Streamlit App
Approach:
  1. User enters element names + functions at all 3 levels (typed manually)
  2. User connects focus functions → lower-level functions (typed) and
     focus functions → higher-level functions (typed)
  3. App traverses each focus function:
       a. Generates failure modes for the focus function
       b. For each failure mode → generates causes from connected lower-level
          functions + noise factors of that lower-level element
       c. For each failure mode → generates effects on connected higher-level functions
  4. Output: downloadable Excel file (AIAG-VDA table format)

Run:
    pip install streamlit anthropic openpyxl
    streamlit run dfmea_app.py
"""

import json
import io
import streamlit as st
import anthropic
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 3000

SYSTEM_PROMPT = """You are an expert DFMEA engineer specializing in heavy-duty truck systems,
following the AIAG-VDA DFMEA methodology (2019).

Rules:
- Return ONLY valid JSON. No markdown fences, no commentary.
- All output must be grounded in the specific element, function, and conditions provided.
- Failure modes must directly describe HOW the function fails — not WHY.
- Causes must be physical root-level mechanisms, never a restatement of the failure mode.
- Effects must describe the impact on the connected higher-level function, not just repeat the failure mode.
- Be specific to heavy-duty truck operating conditions where relevant:
  road salt, vibration, thermal cycling (-40°C to +125°C), high duty cycles, contamination.
"""


# ─────────────────────────────────────────────────────────────────────────────
# ANTHROPIC CALL
# ─────────────────────────────────────────────────────────────────────────────
def call_claude(client: anthropic.Anthropic, user_prompt: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def prompt_failure_modes(
    focus_element: str,
    focus_function: str,
    focus_function_requirement: str,
) -> str:
    return f"""Generate an exhaustive list of failure modes for the following focus element function.

Focus Element: {focus_element}
Function: {focus_function}
Requirement / Specification: {focus_function_requirement or "Not specified"}

A failure mode is HOW the function fails. Generate failure modes across all four archetypes:
1. Complete loss of function (fails to perform at all)
2. Partial / degraded function (performs insufficiently — too little, too slow, too weak)
3. Excessive / unintended function (performs too much, at wrong time, or when not commanded)
4. Intermittent function (performs sometimes but not consistently)

Return JSON:
{{
  "failure_modes": [
    {{
      "id": "FM-01",
      "description": "Clear, specific failure mode statement",
      "archetype": "Loss | Degraded | Excessive | Intermittent"
    }}
  ]
}}"""


def prompt_causes(
    focus_element: str,
    focus_function: str,
    failure_mode: str,
    lower_element: str,
    lower_function: str,
    noise_factors: list[str],
) -> str:
    noise_str = "\n".join(f"  - {n}" for n in noise_factors) if noise_factors else "  - None specified"
    return f"""Generate all plausible root causes for the following failure mode.
Causes must come from the lower-level element and its function, triggered by the noise factors.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Lower-Level Element: {lower_element}
Lower-Level Function: {lower_function}

Noise Factors acting on {lower_element}:
{noise_str}

Rules for causes:
- Each cause must be a specific physical mechanism (e.g. fatigue cracking, corrosion pitting,
  compression set, dimensional creep, fretting wear, dielectric breakdown)
- State which noise factor triggers it
- State which design characteristic of the lower-level element is inadequate
- Do NOT restate the failure mode as a cause

Return JSON:
{{
  "causes": [
    {{
      "id": "C-01",
      "description": "Physical root cause statement",
      "noise_factor": "Which noise factor triggers this",
      "failure_mechanism": "e.g. fatigue | corrosion | wear | deformation | contamination | etc.",
      "design_characteristic": "Which characteristic is inadequate (material | geometry | tolerance | coating | surface finish | etc.)"
    }}
  ]
}}"""


def prompt_effects(
    focus_element: str,
    focus_function: str,
    failure_mode: str,
    higher_element: str,
    higher_function: str,
) -> str:
    return f"""Generate all plausible failure effects for the following failure mode.
Effects must describe the impact on the connected higher-level element function,
and the resulting impact on the vehicle and driver.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Higher-Level Element: {higher_element}
Connected Higher-Level Function: {higher_function}

Rules for effects:
- Local effect: how the higher-level element's function is impaired
- End effect: worst-case impact on the truck, driver, or regulatory compliance
- Be specific — avoid generic statements like "system does not work"
- Flag if this failure could cause loss of vehicle control, driver injury, or FMVSS violation

Return JSON:
{{
  "effects": [
    {{
      "id": "E-01",
      "local_effect": "Impact on {higher_element} — {higher_function}",
      "end_effect": "Impact on vehicle / driver / safety / regulation",
      "safety_critical": true
    }}
  ]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# CORE GENERATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_dfmea(client: anthropic.Anthropic, form: dict) -> list[dict]:
    """
    Traverse each focus function:
      1. Generate failure modes
      2. For each FM × connected lower function → generate causes
      3. For each FM × connected higher function → generate effects
    Returns flat list of DFMEA rows.
    """
    rows = []

    focus_element = form["focus_element"]
    higher_element = form["higher_element"]
    lower_elements = form["lower_elements"]        # list of {name, functions: [{desc, noise_factors}]}
    focus_functions = form["focus_functions"]      # list of {desc, requirement, lower_connections, higher_connections}

    # Build lookup: lower element name → {function_desc → noise_factors}
    lower_lookup: dict[str, dict] = {}
    for le in lower_elements:
        lower_lookup[le["name"]] = {}
        for lf in le["functions"]:
            lower_lookup[le["name"]][lf["desc"]] = lf.get("noise_factors", [])

    # Build lookup: higher function desc (we only have one higher element)
    higher_functions = form["higher_functions"]  # list of str

    status = st.status("Generating DFMEA...", expanded=True)

    for ff in focus_functions:
        ff_desc = ff["desc"]
        ff_req  = ff.get("requirement", "")
        lower_conns  = ff.get("lower_connections", [])   # list of {lower_element, lower_function}
        higher_conns = ff.get("higher_connections", [])  # list of str (higher function descs)

        status.write(f"**Function:** {ff_desc}")

        # ── Step 1: Failure modes ──────────────────────────────────────────
        status.write("  → Generating failure modes...")
        try:
            fm_result = call_claude(client, prompt_failure_modes(focus_element, ff_desc, ff_req))
            failure_modes = fm_result.get("failure_modes", [])
        except Exception as e:
            st.warning(f"Failed to generate failure modes for '{ff_desc}': {e}")
            continue

        status.write(f"  → {len(failure_modes)} failure modes found")

        for fm in failure_modes:
            fm_desc     = fm["description"]
            fm_archetype = fm.get("archetype", "")

            # ── Step 2: Causes (per connected lower function) ──────────────
            all_causes = []
            if lower_conns:
                for conn in lower_conns:
                    le_name = conn["lower_element"]
                    lf_desc = conn["lower_function"]
                    noise   = lower_lookup.get(le_name, {}).get(lf_desc, [])

                    status.write(f"  → Causes from [{le_name}] {lf_desc}")
                    try:
                        c_result = call_claude(client, prompt_causes(
                            focus_element, ff_desc, fm_desc,
                            le_name, lf_desc, noise
                        ))
                        causes = c_result.get("causes", [])
                        # Tag each cause with its source element
                        for c in causes:
                            c["lower_element"] = le_name
                            c["lower_function"] = lf_desc
                        all_causes.extend(causes)
                    except Exception as e:
                        st.warning(f"Cause generation failed: {e}")
            else:
                all_causes = [{"id": "C-00", "description": "No lower-level connection defined",
                               "noise_factor": "", "failure_mechanism": "", "design_characteristic": "",
                               "lower_element": "", "lower_function": ""}]

            # ── Step 3: Effects (per connected higher function) ────────────
            all_effects = []
            if higher_conns:
                for hf_desc in higher_conns:
                    status.write(f"  → Effects on [{higher_element}] {hf_desc}")
                    try:
                        e_result = call_claude(client, prompt_effects(
                            focus_element, ff_desc, fm_desc,
                            higher_element, hf_desc
                        ))
                        effects = e_result.get("effects", [])
                        for e in effects:
                            e["higher_function"] = hf_desc
                        all_effects.extend(effects)
                    except Exception as e:
                        st.warning(f"Effect generation failed: {e}")
            else:
                all_effects = [{"id": "E-00", "local_effect": "No higher-level connection defined",
                                "end_effect": "", "safety_critical": False, "higher_function": ""}]

            # ── Flatten: each row = one cause × one effect ─────────────────
            for cause in all_causes:
                for effect in all_effects:
                    rows.append({
                        # Structure
                        "Focus Element":          focus_element,
                        "Focus Function":         ff_desc,
                        "Requirement":            ff_req,
                        # Failure mode
                        "Failure Mode ID":        fm.get("id", ""),
                        "Failure Mode":           fm_desc,
                        "Archetype":              fm_archetype,
                        # Cause
                        "Cause ID":               cause.get("id", ""),
                        "Cause":                  cause.get("description", ""),
                        "Lower-Level Element":    cause.get("lower_element", ""),
                        "Lower-Level Function":   cause.get("lower_function", ""),
                        "Noise Factor":           cause.get("noise_factor", ""),
                        "Failure Mechanism":      cause.get("failure_mechanism", ""),
                        "Design Characteristic":  cause.get("design_characteristic", ""),
                        # Effect
                        "Effect ID":              effect.get("id", ""),
                        "Higher-Level Element":   higher_element,
                        "Higher-Level Function":  effect.get("higher_function", ""),
                        "Local Effect":           effect.get("local_effect", ""),
                        "End Effect":             effect.get("end_effect", ""),
                        "Safety Critical":        "YES" if effect.get("safety_critical") else "NO",
                        # Engineer fills these in
                        "Severity (S)":           "",
                        "Occurrence (O)":         "",
                        "Detection (D)":          "",
                        "Action Priority (AP)":   "",
                        "Recommended Actions":    "",
                    })

    status.update(label=f"✅ Done — {len(rows)} DFMEA rows generated", state="complete")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPE_COLORS = {
    "Loss":         "FFCCCC",
    "Degraded":     "FFE5CC",
    "Excessive":    "E5CCFF",
    "Intermittent": "CCE5FF",
}

HEADER_GROUPS = {
    "STRUCTURE":      ["Focus Element", "Focus Function", "Requirement"],
    "FAILURE MODE":   ["Failure Mode ID", "Failure Mode", "Archetype"],
    "CAUSE":          ["Cause ID", "Cause", "Lower-Level Element", "Lower-Level Function",
                       "Noise Factor", "Failure Mechanism", "Design Characteristic"],
    "EFFECT":         ["Effect ID", "Higher-Level Element", "Higher-Level Function",
                       "Local Effect", "End Effect", "Safety Critical"],
    "RISK ASSESSMENT":["Severity (S)", "Occurrence (O)", "Detection (D)",
                       "Action Priority (AP)", "Recommended Actions"],
}

GROUP_COLORS = {
    "STRUCTURE":       "1A1A2E",
    "FAILURE MODE":    "C0392B",
    "CAUSE":           "1A5276",
    "EFFECT":          "1E8449",
    "RISK ASSESSMENT": "784212",
}


def build_excel(rows: list[dict], focus_element: str) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "DFMEA"

    # All column names in order
    all_cols = []
    for group_cols in HEADER_GROUPS.values():
        all_cols.extend(group_cols)

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Row 1: Title ──
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(all_cols))
    title_cell = ws.cell(row=1, column=1)
    title_cell.value = f"DFMEA — {focus_element}   |   AIAG-VDA 2019   |   Heavy-Duty Truck"
    title_cell.font = Font(name="Calibri", bold=True, size=13, color="FFFFFF")
    title_cell.fill = PatternFill("solid", fgColor="1A1A2E")
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    # ── Row 2: Group headers ──
    col_idx = 1
    for group, cols in HEADER_GROUPS.items():
        start = col_idx
        end   = col_idx + len(cols) - 1
        if start == end:
            cell = ws.cell(row=2, column=start, value=group)
        else:
            ws.merge_cells(start_row=2, start_column=start, end_row=2, end_column=end)
            cell = ws.cell(row=2, column=start, value=group)
        cell.font      = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor=GROUP_COLORS[group])
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border    = border
        col_idx = end + 1
    ws.row_dimensions[2].height = 22

    # ── Row 3: Column headers ──
    for ci, col in enumerate(all_cols, start=1):
        cell = ws.cell(row=3, column=ci, value=col)
        cell.font      = Font(name="Calibri", bold=True, size=9, color="1A1A2E")
        cell.fill      = PatternFill("solid", fgColor="F0F0F0")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border
    ws.row_dimensions[3].height = 32

    # ── Data rows ──
    for ri, row in enumerate(rows, start=4):
        archetype = row.get("Archetype", "")
        # Pick row tint based on archetype keyword
        row_color = None
        for key, color in ARCHETYPE_COLORS.items():
            if key.lower() in archetype.lower():
                row_color = color
                break

        for ci, col in enumerate(all_cols, start=1):
            val  = row.get(col, "")
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.font      = Font(name="Calibri", size=9)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border    = border

            # Safety critical → red text
            if col == "Safety Critical" and val == "YES":
                cell.font = Font(name="Calibri", size=9, bold=True, color="CC0000")

            # Archetype tint
            if row_color:
                cell.fill = PatternFill("solid", fgColor=row_color)

        ws.row_dimensions[ri].height = 40

    # ── Column widths ──
    col_widths = {
        "Focus Element": 18, "Focus Function": 28, "Requirement": 24,
        "Failure Mode ID": 12, "Failure Mode": 32, "Archetype": 14,
        "Cause ID": 10, "Cause": 34, "Lower-Level Element": 18,
        "Lower-Level Function": 24, "Noise Factor": 20,
        "Failure Mechanism": 18, "Design Characteristic": 20,
        "Effect ID": 10, "Higher-Level Element": 18, "Higher-Level Function": 24,
        "Local Effect": 34, "End Effect": 34, "Safety Critical": 12,
        "Severity (S)": 10, "Occurrence (O)": 10, "Detection (D)": 10,
        "Action Priority (AP)": 14, "Recommended Actions": 28,
    }
    for ci, col in enumerate(all_cols, start=1):
        ws.column_dimensions[get_column_letter(ci)].width = col_widths.get(col, 14)

    # ── Freeze panes ──
    ws.freeze_panes = "A4"

    # ── Auto-filter ──
    ws.auto_filter.ref = f"A3:{get_column_letter(len(all_cols))}3"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="DFMEA Generator", layout="wide", page_icon="⚙️")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stButton > button {
        background: #1A1A2E; color: #E6A817;
        border: 1.5px solid #E6A817; border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace; font-weight: 700;
        letter-spacing: 0.05em;
    }
    .stButton > button:hover { background: #E6A817; color: #1A1A2E; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ DFMEA Generator")
    st.caption("AIAG-VDA 2019 · Heavy-Duty Truck Systems")
    st.divider()

    # ── API Key ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔑 API Key")
        api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
        st.divider()
        st.markdown("### How it works")
        st.markdown("""
1. **Define** element names
2. **Enter functions** at all 3 levels
3. **Connect** focus functions to relevant lower and higher functions
4. **Generate** — the app traverses each focus function:
   - Generates failure modes
   - Generates causes from connected lower functions + noise factors
   - Generates effects on connected higher functions
5. **Download** Excel file
        """)

    if not api_key:
        st.info("Enter your Anthropic API key in the sidebar to get started.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Element Names
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Element Names")
    col1, col2, col3 = st.columns(3)
    with col1:
        higher_element = st.text_input("Higher-Level Element", placeholder="e.g. Air Brake Actuator Assembly")
    with col2:
        focus_element = st.text_input("Focus Element", placeholder="e.g. Brake Diaphragm")
    with col3:
        n_lower = st.number_input("Number of Lower-Level Elements", min_value=1, max_value=8, value=2, step=1)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Higher-Level Functions
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"### Step 2 — Higher-Level Functions  ·  *{higher_element or 'Higher Element'}*")
    n_hf = st.number_input("Number of higher-level functions", min_value=1, max_value=10, value=2, step=1, key="n_hf")
    higher_functions = []
    for i in range(int(n_hf)):
        hf = st.text_input(f"HF-{i+1}", placeholder="e.g. Convert air pressure to clamping force", key=f"hf_{i}")
        higher_functions.append(hf)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Lower-Level Elements + Functions + Noise Factors
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 3 — Lower-Level Elements, Functions & Noise Factors")
    lower_elements = []
    for li in range(int(n_lower)):
        with st.expander(f"Lower-Level Element {li+1}", expanded=(li == 0)):
            le_name = st.text_input("Element Name", placeholder="e.g. Diaphragm Bead", key=f"le_name_{li}")
            n_lf = st.number_input("Number of functions", min_value=1, max_value=8, value=2, step=1, key=f"n_lf_{li}")
            le_functions = []
            for fi in range(int(n_lf)):
                c1, c2 = st.columns([2, 3])
                with c1:
                    lf_desc = st.text_input(
                        f"Function {fi+1}",
                        placeholder="e.g. Maintain sealed interface with housing",
                        key=f"lf_{li}_{fi}"
                    )
                with c2:
                    noise_raw = st.text_input(
                        f"Noise Factors (comma-separated)",
                        placeholder="e.g. road salt, thermal cycling, vibration",
                        key=f"noise_{li}_{fi}"
                    )
                    noise_factors = [n.strip() for n in noise_raw.split(",") if n.strip()]
                le_functions.append({"desc": lf_desc, "noise_factors": noise_factors})
            lower_elements.append({"name": le_name, "functions": le_functions})

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Focus Element Functions + Connections
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown(f"### Step 4 — Focus Functions & Connections  ·  *{focus_element or 'Focus Element'}*")
    n_ff = st.number_input("Number of focus functions", min_value=1, max_value=10, value=2, step=1, key="n_ff")

    focus_functions = []

    for fi in range(int(n_ff)):
        with st.expander(f"Focus Function {fi+1}", expanded=True):
            ff_desc = st.text_input(
                "Function description",
                placeholder="e.g. Transmit air pressure force to push rod",
                key=f"ff_desc_{fi}"
            )
            ff_req = st.text_input(
                "Requirement / Specification",
                placeholder="e.g. Withstand 120 psi max operating pressure",
                key=f"ff_req_{fi}"
            )

            st.markdown("**↑ Connect to Higher-Level Functions**")
            st.caption("Type the higher-level function descriptions this focus function supports (one per line)")
            hf_conn_raw = st.text_area(
                "Higher connections",
                placeholder="\n".join(higher_functions[:2]) if higher_functions else "Paste higher-level function here",
                height=80,
                key=f"hf_conn_{fi}",
                label_visibility="collapsed"
            )
            higher_connections = [l.strip() for l in hf_conn_raw.splitlines() if l.strip()]

            st.markdown("**↓ Connect to Lower-Level Functions**")
            st.caption("For each connection, select the lower-level element and type the function description")

            n_lc = st.number_input(
                "Number of lower connections",
                min_value=0, max_value=10, value=1, step=1,
                key=f"n_lc_{fi}"
            )
            lower_connections = []
            le_names = [le["name"] for le in lower_elements if le["name"]]
            for lci in range(int(n_lc)):
                c1, c2 = st.columns([1, 2])
                with c1:
                    le_choice = st.selectbox(
                        "Lower-Level Element",
                        options=le_names if le_names else ["(define elements above)"],
                        key=f"lc_elem_{fi}_{lci}"
                    )
                with c2:
                    # Show available functions for selected element as hint
                    hint_fns = []
                    for le in lower_elements:
                        if le["name"] == le_choice:
                            hint_fns = [lf["desc"] for lf in le["functions"] if lf["desc"]]
                    lf_choice = st.selectbox(
                        "Lower-Level Function",
                        options=hint_fns if hint_fns else ["(define functions above)"],
                        key=f"lc_fn_{fi}_{lci}"
                    )
                lower_connections.append({"lower_element": le_choice, "lower_function": lf_choice})

            focus_functions.append({
                "desc": ff_desc,
                "requirement": ff_req,
                "higher_connections": higher_connections,
                "lower_connections": lower_connections,
            })

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Step 5 — Generate DFMEA")

    # Validation
    errors = []
    if not focus_element:
        errors.append("Focus element name is required")
    if not higher_element:
        errors.append("Higher-level element name is required")
    if not any(ff["desc"] for ff in focus_functions):
        errors.append("At least one focus function is required")

    if errors:
        for e in errors:
            st.warning(e)
    else:
        if st.button("⚙️  GENERATE DFMEA", use_container_width=True):
            form = {
                "focus_element":   focus_element,
                "higher_element":  higher_element,
                "higher_functions": [hf for hf in higher_functions if hf],
                "lower_elements":  lower_elements,
                "focus_functions": [ff for ff in focus_functions if ff["desc"]],
            }

            with st.container():
                rows = generate_dfmea(client, form)

            if rows:
                st.success(f"Generated **{len(rows)}** DFMEA rows across **{len([ff for ff in focus_functions if ff['desc']])}** focus functions.")

                # Preview table
                st.markdown("#### Preview (first 20 rows)")
                preview_cols = ["Focus Function", "Failure Mode", "Archetype",
                                "Cause", "Noise Factor", "Local Effect", "End Effect", "Safety Critical"]
                preview = [{c: r[c] for c in preview_cols} for r in rows[:20]]
                st.dataframe(preview, use_container_width=True)

                # Download
                excel_bytes = build_excel(rows, focus_element)
                st.download_button(
                    label="📥  Download DFMEA Excel",
                    data=excel_bytes,
                    file_name=f"DFMEA_{focus_element.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            else:
                st.error("No rows were generated. Check that functions and connections are filled in.")


if __name__ == "__main__":
    main()
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

# Noise factors — shared across all elements (BCM / truck context)
NOISE_FACTORS = [
    "Voltage supply fluctuations (9V–16V, load dump spikes up to 40V)",
    "Thermal cycling (-40°C to +85°C operating, storage to -40°C/+125°C)",
    "Mechanical vibration (truck frame PSD loads, road shock)",
    "Electromagnetic interference (EMI) from ignition, motors, relays",
    "Moisture and condensation ingress",
    "CAN/LIN bus signal corruption or loss",
    "Sensor signal noise or open/short circuit on input lines",
    "Software faults / memory corruption",
    "Aging and component wear over vehicle service life (10+ years)",
    "Connector corrosion and fretting at harness interfaces",
]

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
    noise_str = "\n".join(f"  - {n}" for n in noise_factors)
    return f"""Generate all plausible root causes for the following failure mode.
Causes must originate from the lower-level element and its function, triggered by the noise factors.

Focus Element: {focus_element}
Focus Function: {focus_function}
Failure Mode: {failure_mode}

Lower-Level Element: {lower_element}
Lower-Level Function: {lower_function}

Noise Factors:
{noise_str}

Rules:
- Each cause must name a specific physical/electrical mechanism
  (e.g. solder joint fatigue, MOSFET gate oxide breakdown, contact fretting, CAN bus dominant stuck fault,
   ADC reference voltage drift, flash memory bit flip, connector pin corrosion)
- State which noise factor triggers it
- State which design characteristic is inadequate
- Do NOT restate the failure mode as a cause

Return JSON:
{{
  "causes": [
    {{
      "id": "C-01",
      "description": "Specific root cause",
      "noise_factor": "Triggering noise factor",
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
                        "Noise Factor", "Failure Mechanism", "Design Characteristic"],
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
        "Lower-Level Function": 26, "Noise Factor": 22,
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
