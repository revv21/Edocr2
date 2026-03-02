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
