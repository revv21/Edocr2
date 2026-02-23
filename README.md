Yes — this is a much more DFMEA-correct causal direction, and honestly this is closer to how experienced teams think.
You’re moving from
function-driven enumeration → cause-driven reasoning, which is exactly what DFMEA intends.
Below I’ll do this cleanly:
Restate the new causal logic precisely
Show the LLM prompt contracts for each step
Give you generic, plug-and-play Python code that implements exactly this logic
Explain why this avoids over-generation and stays audit-safe
No hard-coding. No domain assumptions.
1️⃣ New DFMEA logic (formalized)
Your requested logic is:
Copy code

Noise factor
   ↓
Failure cause (lower-level origin)
   ↓
Which focus element functions could be affected?
   ↓
Failure modes of those focus functions
   ↓
Relevant higher-level functions
   ↓
Failure effects
Key differences from earlier versions:
Failure cause comes FIRST
Focus functions are filtered, not enumerated
Failure modes are generated only for affected functions
Higher-level functions are contextually relevant, not cartesian
This is the right direction.
2️⃣ LLM prompt contracts (generic, DFMEA-safe)
These are the only places where “intelligence” lives.
A) Generate failure causes from noise (lower-level origin)
Plain text
Copy code
Given:
- Lower-level element name
- ONE specific lower-level function
- ONE noise factor from the P-diagram

Task:
List all realistic failure causes that could originate in the lower-level element due to the given noise factor.

Rules:
- Causes must originate in the lower-level element
- Causes must affect the stated lower-level function
- Do NOT include failure modes or effects
- Each cause must be ONE sentence
- Return an empty list if no realistic failure cause exists
- Return as a bullet list
B) Identify affected focus element functions (filter step)
Plain text
Copy code
Given:
- Focus element name
- List of focus element functions
- ONE failure cause originating in a lower-level element

Task:
From the list of focus element functions, identify which functions could realistically be affected by this failure cause.

Rules:
- Return ONLY functions from the provided list
- Return an empty list if none are affected
- Do NOT explain reasoning
- Return as a bullet list
This step is critical — it prevents garbage mappings.
C) Generate failure modes from affected focus functions
Plain text
Copy code
Given:
- Focus element name
- ONE specific focus element function
- ONE failure cause

Task:
List all realistic failure modes of the focus element function that could result from the given failure cause.

Rules:
- Describe incorrect behavior of the function
- Do NOT include causes or effects
- Do NOT mention failure mode classification
- Each failure mode must be ONE sentence
- Return as a bullet list
D) Identify relevant higher-level functions
Plain text
Copy code
Given:
- Higher-level element name
- List of higher-level functions
- ONE focus element function
- ONE focus element failure mode

Task:
From the list of higher-level functions, identify which functions could be impacted by this failure mode.

Rules:
- Return ONLY functions from the provided list
- Return an empty list if none are impacted
- Do NOT explain reasoning
- Return as a bullet list
E) Generate failure effects
Plain text
Copy code
Given:
- Higher-level element name
- ONE higher-level function
- ONE focus element failure mode

Task:
Describe ONE realistic failure effect on the higher-level function.

Rules:
- Describe loss or degradation of the function
- Do NOT include causes
- ONE sentence only
3️⃣ Generic DFMEA generator (updated logic)
This is fully generic, LLM-first, and matches your structure.
✅ Python code (cause-driven DFMEA engine)
Python
Copy code
import pandas as pd


class DFMEAGeneratorLLM:

    def __init__(self, llm_client):
        self.llm = llm_client

    # ---------- helpers ----------

    def _parse_bullets(self, text):
        lines = [l.strip("-• ").strip() for l in text.split("\n")]
        return [l for l in lines if len(l) > 5]

    # ---------- Stage A: failure causes ----------

    def generate_failure_causes(self, lower_element, lower_function, noise):
        prompt = f"""
Lower-level element: {lower_element}
Lower-level function: {lower_function}
Noise factor: {noise}

Task:
List all realistic failure causes that could originate in the lower-level element due to the given noise factor.

Rules:
- Causes must originate in the lower-level element
- Causes must affect the stated lower-level function
- Do NOT include failure modes or effects
- Each cause must be ONE sentence
- Return an empty list if no realistic failure cause exists
- Return as a bullet list
"""
        return self._parse_bullets(self.llm.generate(prompt))

    # ---------- Stage B: affected focus functions ----------

    def affected_focus_functions(self, focus_element, focus_functions, failure_cause):
        prompt = f"""
Focus element: {focus_element}
Focus element functions:
{chr(10).join('- ' + f for f in focus_functions)}

Failure cause:
{failure_cause}

Task:
From the list of focus element functions, identify which functions could realistically be affected by this failure cause.

Rules:
- Return ONLY functions from the provided list
- Return an empty list if none are affected
- Return as a bullet list
"""
        return self._parse_bullets(self.llm.generate(prompt))

    # ---------- Stage C: failure modes ----------

    def generate_failure_modes(self, focus_element, focus_function, failure_cause):
        prompt = f"""
Focus element: {focus_element}
Focus element function: {focus_function}
Failure cause: {failure_cause}

Task:
List all realistic failure modes of the focus element function that could result from the given failure cause.

Rules:
- Describe incorrect behavior of the function
- Do NOT include causes or effects
- Do NOT mention failure mode classification
- Each failure mode must be ONE sentence
- Return as a bullet list
"""
        return self._parse_bullets(self.llm.generate(prompt))

    # ---------- Stage D: relevant higher-level functions ----------

    def affected_higher_functions(self, higher_element, higher_functions, focus_function, failure_mode):
        prompt = f"""
Higher-level element: {higher_element}
Higher-level functions:
{chr(10).join('- ' + f for f in higher_functions)}

Focus element function: {focus_function}
Failure mode: {failure_mode}

Task:
From the list of higher-level functions, identify which functions could be impacted by this failure mode.

Rules:
- Return ONLY functions from the provided list
- Return an empty list if none are impacted
- Return as a bullet list
"""
        return self._parse_bullets(self.llm.generate(prompt))

    # ---------- Stage E: failure effects ----------

    def generate_failure_effect(self, higher_element, higher_function, failure_mode):
        prompt = f"""
Higher-level element: {higher_element}
Higher-level function: {higher_function}
Focus element failure mode: {failure_mode}

Task:
Describe ONE realistic failure effect on the higher-level function.

Rules:
- Describe loss or degradation of the function
- Do NOT include causes
- ONE sentence only
"""
        return self.llm.generate(prompt).strip()

    # ---------- MAIN PIPELINE ----------

    def generate_dfmea(self, system_data):
        rows = []

        focus = system_data["focus_element"]
        lower_elements = system_data["lower_level_elements"]
        higher_elements = system_data["higher_level_elements"]

        noises = [
            f"{cat}: {n}"
            for cat, factors in system_data["noise_factors"].items()
            for n in factors
        ]

        for lower in lower_elements:
            for lower_fn in lower["Function to focus element"]:
                for noise in noises:

                    causes = self.generate_failure_causes(
                        lower["Name"], lower_fn, noise
                    )

                    for cause in causes:

                        affected_focus_fns = self.affected_focus_functions(
                            focus["Name"], focus["Functions"], cause
                        )

                        for focus_fn in affected_focus_fns:

                            modes = self.generate_failure_modes(
                                focus["Name"], focus_fn, cause
                            )

                            for mode in modes:

                                for higher in higher_elements:
                                    affected_higher_fns = self.affected_higher_functions(
                                        higher["Name"],
                                        higher["Function from focus element"],
                                        focus_fn,
                                        mode
                                    )

                                    for hf in affected_higher_fns:
                                        effect = self.generate_failure_effect(
                                            higher["Name"], hf, mode
                                        )

                                        rows.append({
                                            "Lower Level Element": lower["Name"],
                                            "Lower Level Function": lower_fn,
                                            "Noise Factor": noise,
                                            "Failure Cause": cause,
                                            "Focus Element": focus["Name"],
                                            "Focus Element Function": focus_fn,
                                            "Failure Mode": mode,
                                            "Higher Level Element": higher["Name"],
                                            "Higher Level Function": hf,
                                            "Failure Effect": effect
                                        })

        return pd.DataFrame(rows)
4️⃣ Why this logic is better than before
Aspect
Old
This version
Starting point
Functions
Causes (correct)
Noise handling
Filtered
Causal driver
Focus functions
Enumerated
Affected only
Failure modes
Broad
Cause-specific
Higher-level effects
Cartesian
Relevant only
DFMEA quality
Medium
High
This is now very close to how expert DFMEA workshops run.
5️⃣ What I’d recommend next (optional but powerful)
🔒 Add LLM self-check: “Is this DFMEA row realistic?”
🔁 Add caching (causes, affected functions, modes)
🌲 Auto-generate FTA from the cause→mode→effect graph
📊 Add Severity / Occurrence / Detection estimation
If you want, tell me which one to do next and I’ll build it cleanly on top of this.