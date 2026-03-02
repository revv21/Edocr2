"""
DFMEA Generator - Core Generation Logic
Follows AIAG-VDA methodology for heavy-duty truck systems
"""

import json
import os
from typing import Optional
from anthropic import Anthropic

client = Anthropic()

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert DFMEA engineer specialized in heavy-duty truck systems,
following the AIAG-VDA DFMEA methodology (2019).

You generate exhaustive failure modes, effects, and causes for truck components.

Domain context:
- Heavy-duty trucks face: road salt/corrosion, severe vibration (PSD loads on frame),
  wide thermal cycling (-40°C to +125°C), high duty cycles, long service life (10+ years / 1M+ cycles),
  contamination (oil, mud, brake dust, diesel exhaust).
- Safety-critical systems must comply with FMVSS regulations.
- Failures can cause loss of vehicle control, driver injury, cargo loss, or regulatory violation.

Output rules:
- Return ONLY valid JSON — no commentary, no markdown fences.
- Causes must be physical root-level mechanisms — never restate the failure mode.
- Effects must describe BOTH local impact (on higher-level element) AND end effect (vehicle/driver/safety).
- Flag safety_critical = true if failure could lead to loss of vehicle control, injury, or FMVSS violation.
"""

# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

def build_failure_mode_prompt(
    focus_fn: dict,
    focus_element: dict,
    higher_element: dict,
    lower_elements: list,
    interfaces: list,
    noise_factors: list,
    archetype: str,
    archetype_instruction: str,
) -> str:
    return f"""
Generate failure modes for ONE specific function of the focus element.

=== VEHICLE CONTEXT ===
System: {focus_element.get('system', 'N/A')} > {focus_element.get('subsystem', 'N/A')}
Vehicle: Heavy-duty truck / Class 8

=== FOCUS ELEMENT ===
Name: {focus_element['name']}
Type: {focus_element.get('type', 'N/A')}
Material: {focus_element.get('material', 'N/A')}
Operating Conditions: Temp {focus_element.get('temp_min', -40)}°C to {focus_element.get('temp_max', 125)}°C
Life Target: {focus_element.get('life_target', '10 years / 1M cycles')}
Exposure: {', '.join(focus_element.get('exposure', []))}

=== HIGHER-LEVEL ELEMENT ===
Name: {higher_element['name']}
Functions: {json.dumps(higher_element.get('functions', []))}

=== LOWER-LEVEL ELEMENTS ===
{json.dumps(lower_elements, indent=2)}

=== INTERFACES ===
{json.dumps(interfaces, indent=2)}

=== NOISE FACTORS ===
{json.dumps(noise_factors, indent=2)}

=== FUNCTION TO ANALYZE ===
Function ID: {focus_fn['id']}
Function: "{focus_fn['description']}"
Requirement: "{focus_fn.get('requirement', 'N/A')}"
Connected Lower-Level Functions: {json.dumps(focus_fn.get('connected_lower_functions', []))}
Connected Higher-Level Functions: {json.dumps(focus_fn.get('connected_higher_functions', []))}

=== ARCHETYPE TO GENERATE ===
Archetype: {archetype}
Instruction: {archetype_instruction}

Generate ALL plausible failure modes for this archetype only.
For each failure mode provide effects AND causes.

Return this exact JSON structure:
{{
  "function_id": "{focus_fn['id']}",
  "function": "{focus_fn['description']}",
  "archetype": "{archetype}",
  "failure_modes": [
    {{
      "id": "FM-{focus_fn['id']}-{archetype[:3].upper()}-01",
      "description": "...",
      "archetype": "{archetype}",
      "safety_critical": true,
      "effects": [
        {{
          "local_effect": "Impact on {higher_element['name']} function",
          "end_effect": "Impact on vehicle / driver / safety / regulation"
        }}
      ],
      "causes": [
        {{
          "id": "C-{focus_fn['id']}-{archetype[:3].upper()}-01-01",
          "description": "Physical root cause statement",
          "noise_factor": "Which noise factor triggers this",
          "failure_mechanism": "Physical process (corrosion / fatigue / wear / deformation / etc.)",
          "lower_level_element": "Which lower-level element or interface",
          "design_characteristic": "Which design characteristic is inadequate (geometry / material / tolerance / coating)"
        }}
      ]
    }}
  ]
}}
"""


def build_connection_suggestion_prompt(
    focus_element: dict,
    higher_element: dict,
    lower_elements: list,
    user_connections: dict,
) -> str:
    return f"""
A DFMEA engineer has manually connected functions across levels.
Review these connections and suggest any they may have missed.

=== FOCUS ELEMENT: {focus_element['name']} ===
Functions: {json.dumps([f['description'] for f in focus_element.get('functions', [])], indent=2)}

=== HIGHER-LEVEL ELEMENT: {higher_element['name']} ===
Functions: {json.dumps(higher_element.get('functions', []), indent=2)}

=== LOWER-LEVEL ELEMENTS ===
{json.dumps([{{'name': e['name'], 'functions': e.get('functions', [])}} for e in lower_elements], indent=2)}

=== USER-DEFINED CONNECTIONS ===
{json.dumps(user_connections, indent=2)}

Consider: physical interfaces, shared materials, thermal/mechanical coupling,
common failure paths in truck environments (vibration, corrosion, thermal cycling).

Return JSON:
{{
  "suggested_additional_connections": [
    {{
      "focus_function_id": "...",
      "connects_to_level": "higher | lower",
      "element_name": "...",
      "function": "...",
      "reason": "Why this connection matters for failure analysis"
    }}
  ],
  "reasoning": "Overall assessment of connection completeness"
}}
"""


def build_combination_pass_prompt(
    focus_element: dict,
    higher_element: dict,
    generated_fmea: list,
) -> str:
    # Summarize existing FMs to avoid huge prompt
    fm_summary = []
    for fn_data in generated_fmea:
        for fm in fn_data.get("failure_modes", []):
            fm_summary.append({
                "function": fn_data["function"],
                "failure_mode": fm["description"],
                "primary_cause": fm["causes"][0]["description"] if fm.get("causes") else "N/A"
            })

    return f"""
Review this DFMEA for a {focus_element['name']} in a heavy-duty truck.
Identify failure modes that could be caused by a SINGLE root cause affecting MULTIPLE functions simultaneously.

=== EXISTING FAILURE MODES ===
{json.dumps(fm_summary, indent=2)}

Common multi-function failures in trucks:
- Structural fracture affects both load-bearing AND sealing functions
- Corrosion affects both electrical continuity AND mechanical strength
- Thermal distortion affects both dimensional accuracy AND assembly interfaces
- Fatigue crack propagation affects multiple load paths simultaneously

Return JSON:
{{
  "combination_failure_modes": [
    {{
      "id": "CFM-01",
      "description": "Combined failure mode description",
      "affected_functions": ["Function 1", "Function 2"],
      "single_root_cause": "The shared physical root cause",
      "failure_mechanism": "Physical mechanism",
      "effects": [
        {{
          "local_effect": "Combined impact on {higher_element['name']}",
          "end_effect": "Vehicle / driver / safety impact"
        }}
      ],
      "safety_critical": true
    }}
  ]
}}
"""


def build_consistency_check_prompt(generated_fmea: list) -> str:
    return f"""
Audit this DFMEA for logical consistency and completeness.

=== DFMEA DATA ===
{json.dumps(generated_fmea, indent=2)}

Check each failure mode:
1. Is the cause physically plausible given the stated failure mode?
2. Is the effect a logical consequence of the failure mode?
3. Are there additional causes that could produce the same effect?
4. Should safety_critical be true but is marked false?
5. Are any causes just restating the failure mode (not a root cause)?

Also check overall coverage:
- Corrosion / road salt attack covered?
- Vibration fatigue covered?
- Thermal cycling embrittlement covered?
- Contamination (oil, mud) covered?
- Assembly error / improper installation covered?
- Wear at interfaces over high mileage covered?

Return JSON:
{{
  "issues": [
    {{
      "failure_mode_id": "...",
      "issue_type": "inconsistent_cause | missing_cause | wrong_safety_flag | restatement | missing_end_effect",
      "description": "What's wrong",
      "suggested_fix": "How to correct it"
    }}
  ],
  "missing_coverage": [
    {{
      "mechanism": "e.g., Road salt corrosion",
      "suggested_failure_mode": "...",
      "affected_function": "..."
    }}
  ],
  "overall_completeness_score": 85,
  "summary": "Brief overall assessment"
}}
"""


# ─────────────────────────────────────────────
# ARCHETYPES
# ─────────────────────────────────────────────
ARCHETYPES = [
    ("Loss of Function",       "In what ways could the element COMPLETELY FAIL to perform the function? Consider sudden, catastrophic failure."),
    ("Degraded Function",      "In what ways could the element perform the function PARTIALLY or INSUFFICIENTLY (too little, too slow, too weak)?"),
    ("Excessive Function",     "In what ways could the element perform the function EXCESSIVELY or UNINTENTIONALLY (too much, at wrong time, stuck on)?"),
    ("Intermittent Function",  "In what ways could the element perform the function INTERMITTENTLY (works sometimes, fails unpredictably)?"),
]


# ─────────────────────────────────────────────
# CORE GENERATOR
# ─────────────────────────────────────────────
def call_claude(prompt: str, system: str = SYSTEM_PROMPT) -> dict:
    """Call Claude API and parse JSON response."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def suggest_missing_connections(
    focus_element: dict,
    higher_element: dict,
    lower_elements: list,
    user_connections: dict,
) -> dict:
    """Step 1: Suggest connections the user may have missed."""
    print("\n🔗 Checking for missed function connections...")
    prompt = build_connection_suggestion_prompt(
        focus_element, higher_element, lower_elements, user_connections
    )
    return call_claude(prompt)


def generate_failure_modes_for_function(
    focus_fn: dict,
    focus_element: dict,
    higher_element: dict,
    lower_elements: list,
    interfaces: list,
    noise_factors: list,
) -> dict:
    """Step 2: Generate failure modes for one focus function — all 4 archetypes."""
    print(f"\n⚙️  Generating failure modes for: {focus_fn['description']}")
    
    all_failure_modes = []

    for archetype, instruction in ARCHETYPES:
        print(f"   → Archetype: {archetype}")
        prompt = build_failure_mode_prompt(
            focus_fn, focus_element, higher_element,
            lower_elements, interfaces, noise_factors,
            archetype, instruction
        )
        try:
            result = call_claude(prompt)
            all_failure_modes.extend(result.get("failure_modes", []))
        except Exception as e:
            print(f"   ⚠️  Error on archetype {archetype}: {e}")

    return {
        "function_id": focus_fn["id"],
        "function": focus_fn["description"],
        "requirement": focus_fn.get("requirement", ""),
        "failure_modes": all_failure_modes
    }


def run_combination_pass(
    focus_element: dict,
    higher_element: dict,
    generated_fmea: list,
) -> dict:
    """Step 3: Find single-root-cause failures affecting multiple functions."""
    print("\n🔄 Running multi-function combination pass...")
    prompt = build_combination_pass_prompt(focus_element, higher_element, generated_fmea)
    return call_claude(prompt)


def run_consistency_check(generated_fmea: list) -> dict:
    """Step 4: Audit logical consistency and coverage gaps."""
    print("\n🔍 Running consistency & completeness audit...")
    prompt = build_consistency_check_prompt(generated_fmea)
    return call_claude(prompt)


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────
def generate_dfmea(input_data: dict) -> dict:
    """
    Main DFMEA generation pipeline.
    
    input_data structure:
    {
        "focus_element": {
            "name": str,
            "type": str,
            "material": str,
            "system": str,
            "subsystem": str,
            "temp_min": int,
            "temp_max": int,
            "exposure": [str],
            "life_target": str,
            "functions": [
                {
                    "id": str,
                    "description": str,
                    "requirement": str,
                    "connected_lower_functions": [str],   # lower-level function descriptions
                    "connected_higher_functions": [str]   # higher-level function descriptions
                }
            ]
        },
        "higher_level_element": {
            "name": str,
            "functions": [str]
        },
        "lower_level_elements": [
            {
                "name": str,
                "type": str,
                "functions": [str],
                "characteristics": [str]
            }
        ],
        "interfaces": [
            {
                "mating_element": str,
                "interface_type": str,
                "characteristics": [str]
            }
        ],
        "noise_factors": [str],
        "user_connections": {}   # optional: existing connections for gap check
    }
    """
    focus_element   = input_data["focus_element"]
    higher_element  = input_data["higher_level_element"]
    lower_elements  = input_data.get("lower_level_elements", [])
    interfaces      = input_data.get("interfaces", [])
    noise_factors   = input_data.get("noise_factors", [])
    user_connections = input_data.get("user_connections", {})

    results = {
        "metadata": {
            "focus_element": focus_element["name"],
            "higher_level_element": higher_element["name"],
            "system": focus_element.get("system", ""),
            "subsystem": focus_element.get("subsystem", ""),
        },
        "suggested_connections": {},
        "fmea_by_function": [],
        "combination_failures": {},
        "consistency_audit": {},
        "summary": {}
    }

    # ── Step 1: Suggest missed connections ──
    if user_connections:
        results["suggested_connections"] = suggest_missing_connections(
            focus_element, higher_element, lower_elements, user_connections
        )

    # ── Step 2: Generate FMs per function (all 4 archetypes) ──
    for focus_fn in focus_element.get("functions", []):
        fn_result = generate_failure_modes_for_function(
            focus_fn, focus_element, higher_element,
            lower_elements, interfaces, noise_factors
        )
        results["fmea_by_function"].append(fn_result)

    # ── Step 3: Combination pass ──
    results["combination_failures"] = run_combination_pass(
        focus_element, higher_element, results["fmea_by_function"]
    )

    # ── Step 4: Consistency check ──
    results["consistency_audit"] = run_consistency_check(results["fmea_by_function"])

    # ── Summary stats ──
    total_fms = sum(len(f["failure_modes"]) for f in results["fmea_by_function"])
    total_combo = len(results["combination_failures"].get("combination_failure_modes", []))
    safety_critical = sum(
        1 for f in results["fmea_by_function"]
        for fm in f["failure_modes"]
        if fm.get("safety_critical")
    )
    issues = len(results["consistency_audit"].get("issues", []))
    missing = len(results["consistency_audit"].get("missing_coverage", []))

    results["summary"] = {
        "total_failure_modes": total_fms,
        "combination_failure_modes": total_combo,
        "safety_critical_count": safety_critical,
        "consistency_issues": issues,
        "coverage_gaps": missing,
        "completeness_score": results["consistency_audit"].get("overall_completeness_score", "N/A")
    }

    print(f"\n✅ DFMEA generation complete!")
    print(f"   Total FMs: {total_fms} | Combo FMs: {total_combo} | Safety Critical: {safety_critical}")
    print(f"   Issues found: {issues} | Coverage gaps: {missing}")

    return results


# ─────────────────────────────────────────────
# FLAT TABLE EXPORTER
# ─────────────────────────────────────────────
def export_to_flat_table(dfmea_results: dict) -> list[dict]:
    """
    Convert nested DFMEA output to flat rows suitable for Excel/CSV.
    Each row = one (function, failure_mode, effect, cause) combination.
    """
    rows = []
    meta = dfmea_results["metadata"]

    for fn_data in dfmea_results["fmea_by_function"]:
        for fm in fn_data["failure_modes"]:
            effects = fm.get("effects", [{}])
            causes  = fm.get("causes", [{}])

            # Cross-join effects × causes for full traceability
            for effect in effects:
                for cause in causes:
                    rows.append({
                        "System":             meta.get("system", ""),
                        "Subsystem":          meta.get("subsystem", ""),
                        "Focus Element":      meta["focus_element"],
                        "Function ID":        fn_data["function_id"],
                        "Function":           fn_data["function"],
                        "Requirement":        fn_data.get("requirement", ""),
                        "Failure Mode ID":    fm.get("id", ""),
                        "Failure Mode":       fm.get("description", ""),
                        "Archetype":          fm.get("archetype", ""),
                        "Safety Critical":    "YES" if fm.get("safety_critical") else "NO",
                        "Local Effect":       effect.get("local_effect", ""),
                        "End Effect":         effect.get("end_effect", ""),
                        "Cause ID":           cause.get("id", ""),
                        "Cause":              cause.get("description", ""),
                        "Noise Factor":       cause.get("noise_factor", ""),
                        "Failure Mechanism":  cause.get("failure_mechanism", ""),
                        "Lower-Level Element":cause.get("lower_level_element", ""),
                        "Design Characteristic": cause.get("design_characteristic", ""),
                        "Severity (S)":       "",   # To be filled by engineer
                        "Occurrence (O)":     "",
                        "Detection (D)":      "",
                        "Action Priority":    "",
                        "Recommended Actions": "",
                    })

    # Add combination failures
    for cfm in dfmea_results.get("combination_failures", {}).get("combination_failure_modes", []):
        for effect in cfm.get("effects", [{}]):
            rows.append({
                "System":             meta.get("system", ""),
                "Subsystem":          meta.get("subsystem", ""),
                "Focus Element":      meta["focus_element"],
                "Function ID":        "COMBO",
                "Function":           " + ".join(cfm.get("affected_functions", [])),
                "Requirement":        "",
                "Failure Mode ID":    cfm.get("id", ""),
                "Failure Mode":       cfm.get("description", ""),
                "Archetype":          "Multi-Function",
                "Safety Critical":    "YES" if cfm.get("safety_critical") else "NO",
                "Local Effect":       effect.get("local_effect", ""),
                "End Effect":         effect.get("end_effect", ""),
                "Cause ID":           "",
                "Cause":              cfm.get("single_root_cause", ""),
                "Noise Factor":       "",
                "Failure Mechanism":  cfm.get("failure_mechanism", ""),
                "Lower-Level Element": "",
                "Design Characteristic": "",
                "Severity (S)":       "",
                "Occurrence (O)":     "",
                "Detection (D)":      "",
                "Action Priority":    "",
                "Recommended Actions": "",
            })

    return rows


if __name__ == "__main__":
    # ── Example: Brake Diaphragm ──
    example_input = {
        "focus_element": {
            "name": "Brake Diaphragm",
            "type": "Sealing / Force Transmission",
            "material": "EPDM rubber",
            "system": "Braking System",
            "subsystem": "Air Brake Actuator",
            "temp_min": -40,
            "temp_max": 125,
            "exposure": ["road salt", "moisture", "oils", "vibration", "brake dust"],
            "life_target": "1,000,000 cycles / 10 years",
            "functions": [
                {
                    "id": "F1",
                    "description": "Transmit air pressure force to push rod",
                    "requirement": "Transmit full force at 120 psi max operating pressure",
                    "connected_lower_functions": [
                        "Diaphragm body maintains structural integrity under pressure",
                        "Bead maintains clamped seal in housing groove"
                    ],
                    "connected_higher_functions": [
                        "Actuator assembly converts air pressure to clamping force on brake pad"
                    ]
                },
                {
                    "id": "F2",
                    "description": "Seal pressure chamber to prevent air leakage",
                    "requirement": "Zero leakage at 100 psi sustained for 60 seconds",
                    "connected_lower_functions": [
                        "Bead maintains clamped seal in housing groove",
                        "Diaphragm body maintains thickness uniformity"
                    ],
                    "connected_higher_functions": [
                        "Actuator maintains consistent brake application force"
                    ]
                }
            ]
        },
        "higher_level_element": {
            "name": "Air Brake Actuator Assembly",
            "functions": [
                "Convert air pressure into mechanical clamping force on brake pad",
                "Maintain consistent braking force under sustained application",
                "Release braking force completely when air pressure is released"
            ]
        },
        "lower_level_elements": [
            {
                "name": "Diaphragm Bead",
                "type": "Sealing Interface Feature",
                "functions": ["Maintain sealed clamping interface with housing groove"],
                "characteristics": ["bead geometry", "Shore A hardness 60±5", "compression set resistance", "surface finish"]
            },
            {
                "name": "Diaphragm Body",
                "type": "Flexible Membrane",
                "functions": ["Flex under pressure cycling", "Transmit load to push rod contact"],
                "characteristics": ["thickness uniformity ±0.2mm", "tensile strength", "fatigue life", "chemical resistance"]
            }
        ],
        "interfaces": [
            {
                "mating_element": "Push Rod",
                "interface_type": "Force Transmission",
                "characteristics": ["contact area", "alignment", "surface finish Ra 1.6"]
            },
            {
                "mating_element": "Housing Groove",
                "interface_type": "Sealing / Clamping",
                "characteristics": ["clamping bolt torque 45±5 Nm", "groove dimensions", "surface roughness"]
            }
        ],
        "noise_factors": [
            "Thermal cycling -40°C to +125°C",
            "Road salt and moisture exposure",
            "Hydraulic oil contamination",
            "Mechanical vibration (truck frame PSD loads)",
            "Repeated pressure cycling (1M+ cycles)",
            "Assembly variation (improper clamping torque)",
            "Aging / oxidation over 10-year service life"
        ],
        "user_connections": {
            "F1_to_lower": ["Diaphragm body integrity", "Bead clamping"],
            "F1_to_higher": ["Actuator force output"],
            "F2_to_lower": ["Bead seal", "Diaphragm thickness"],
            "F2_to_higher": ["Actuator pressure maintenance"]
        }
    }

    results = generate_dfmea(example_input)

    # Save full JSON output
    with open("dfmea_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📄 Full results saved to dfmea_output.json")

    # Save flat table
    flat = export_to_flat_table(results)
    with open("dfmea_flat_table.json", "w") as f:
        json.dump(flat, f, indent=2)
    print(f"📊 Flat table ({len(flat)} rows) saved to dfmea_flat_table.json")
