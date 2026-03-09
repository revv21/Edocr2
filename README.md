import pandas as pd
import numpy as np
import json
import requests
import os
from network_build import classify_functions, get_affected_focus_functions, get_affected_higher_functions


# ─────────────────────────────────────────────────────────────────────────────
# TITAN EMBEDDING CLIENT
# ─────────────────────────────────────────────────────────────────────────────

REGION        = "ap-south-1"
TITAN_MODEL   = "amazon.titan-embed-text-v2:0"
TITAN_URL     = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{TITAN_MODEL}/invoke"
BEDROCK_API_KEY = os.environ.get("BEDROCK_API_KEY", "")


def get_embedding(text: str) -> np.ndarray:
    """Call AWS Titan embedding model and return embedding vector."""
    headers = {
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputText": text,
        "dimensions": 512,        # Titan v2 supports 256 / 512 / 1024
        "normalize": True,        # Returns unit vectors → cosine sim = dot product
    }
    response = requests.post(TITAN_URL, headers=headers, json=payload).json()
    return np.array(response["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (already normalised by Titan)."""
    return float(np.dot(a, b))


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING CACHE
# Avoids re-embedding the same function text multiple times across focus fn loop
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    def __init__(self):
        self._cache: dict[str, np.ndarray] = {}

    def get(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self._cache[text] = get_embedding(text)
        return self._cache[text]


# ─────────────────────────────────────────────────────────────────────────────
# SIMILARITY-BASED FUNCTION MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def find_relevant_lower_functions(
    focus_fn: int,
    G,
    lower_fns: list[int],
    cache: EmbeddingCache,
    top_k: int = 3,
    threshold: float = 0.55,
) -> list[tuple[int, float]]:
    """
    Find the most semantically similar lower-level functions to a focus function
    using pure embedding similarity — no graph traversal.

    Returns top_k lower functions with similarity >= threshold,
    sorted by score descending.
    """
    focus_name = G.nodes[focus_fn]["name"]
    focus_emb  = cache.get(focus_name)

    scored = []
    for lf in lower_fns:
        lf_name = G.nodes[lf]["name"]
        lf_emb  = cache.get(lf_name)
        sim     = cosine_similarity(focus_emb, lf_emb)
        if sim >= threshold:
            scored.append((lf, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def find_relevant_higher_functions(
    focus_fn: int,
    G,
    higher_fns: list[int],
    cache: EmbeddingCache,
    top_k: int = 2,
    threshold: float = 0.55,
) -> list[tuple[int, float]]:
    """
    Find the most semantically similar higher-level functions to a focus function
    using pure embedding similarity — no graph traversal.

    Returns top_k higher functions with similarity >= threshold,
    sorted by score descending.
    """
    focus_name = G.nodes[focus_fn]["name"]
    focus_emb  = cache.get(focus_name)

    scored = []
    for hf in higher_fns:
        hf_name = G.nodes[hf]["name"]
        hf_emb  = cache.get(hf_name)
        sim     = cosine_similarity(focus_emb, hf_emb)
        if sim >= threshold:
            scored.append((hf, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DFMEA GENERATOR CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DFMEAGeneratorLLM:

    def __init__(self, llm_client, similarity_top_k_lower: int = 3,
                 similarity_top_k_higher: int = 2,
                 similarity_threshold: float = 0.55):
        """
        Args:
            llm_client:               Your existing LLM client with .generate(prompt) method
            similarity_top_k_lower:   Max extra lower functions to add via similarity (beyond graph connections)
            similarity_top_k_higher:  Max extra higher functions to add via similarity (beyond graph connections)
            similarity_threshold:     Minimum cosine similarity to consider a function relevant (0–1)
        """
        self.llm               = llm_client
        self.top_k_lower       = similarity_top_k_lower
        self.top_k_higher      = similarity_top_k_higher
        self.threshold         = similarity_threshold
        self._emb_cache        = EmbeddingCache()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parse_bullets(self, text):
        lines = [l.strip("–• ").strip() for l in text.split("\n")]
        return [l for l in lines if len(l) > 5]

    # ── prompt methods ───────────────────────────────────────────────────────

    def generate_failure_modes_for_focus(self, focus_function_name, focus_element):
        prompt = f"""
TASK: Generate failure modes for the focus element function
Focus element function:
{focus_function_name}
Focus element:
{focus_element}
Rules:
- Generate these 4 cases first (Only generate the failure, not the type of failure and DO NOT include the cause for the failure):
  - Function not happening
  - Function happening partially
  - Function happening with a delay
  - Function happening incorrectly
- Among these cases, only return technically most relevant cases

OUTPUT RULES:
- Output in bullet points only.
- If no realistic failure mode exists, output exactly: NONE
- Each failure mode must be ONE sentence
OUTPUT FORMAT:
- <failure mode sentence>
- <failure mode sentence>
"""
        return self._parse_bullets(self.llm.generate(prompt))

    def generate_failure_causes_for_mode(self, lower_function_name, focus_function_name,
                                          failure_mode, noise_factors, lower_element):
        causes = []
        for category, factors in noise_factors.items():
            for factor in factors:
                noise = f"{category}: {factor}"
                prompt = f"""
TASK: Generate Failure Cause

Lower-level element:
{lower_element}
Lower-level function:
{lower_function_name}
Focus Function:
{focus_function_name}
Assumed failure mode of focus element:
"{failure_mode}"
Noise factor:
{noise}
Rules:
- Cause must originate in the lower-level function
- The cause MUST be triggered by the given noise factor.
- Must be specific, measurable, and actionable
- Focus on design-influenced causes and avoid impossible causes
- Do NOT include effects
OUTPUT RULES:
- If NO realistic failure cause exists, output exactly: NONE
- Otherwise, output 1 bullet point only.
- The bullet point must contain exactly one failure cause.
- Do not include explanations or extra text.
OUTPUT FORMAT:
- <failure cause sentence>
"""
                response = self.llm.generate(prompt).strip()
                if response and response.upper() != "NONE":
                    causes.append((response, noise))
        return causes

    def generate_failure_effect(self, higher_function_name, focus_function_name,
                                 failure_mode, higher_element):
        prompt = f"""
TASK: Generate Failure Effect

Higher-level element:
{higher_element}

Higher-level function:
{higher_function_name}

Focus element failure mode:
"{failure_mode}"

DEFINITION:
A failure effect describes the loss or degradation of the higher-level function caused by the failure mode.
REQUIREMENTS:
- The effect MUST describe impact on the higher-level function only.
- The effect MUST NOT include causes or focus element behavior.
- The effect MUST be ONE sentence.
OUTPUT RULES:
- Output exactly ONE sentence.
- Do NOT include explanations.

OUTPUT FORMAT:
<failure effect sentence>
"""
        return self.llm.generate(prompt).strip()

    # ── main pipeline ────────────────────────────────────────────────────────

    def generate_dfmea(self, G, elements, functions, noise_factors):
        rows = []

        lower_fns, focus_fns, higher_fns = classify_functions(functions, elements)

        # Pre-embed ALL function names upfront in one pass
        # (avoids redundant API calls during the nested loop)
        print("Pre-computing embeddings for all functions...")
        all_fn_ids = list(lower_fns) + list(focus_fns) + list(higher_fns)
        for fn_id in all_fn_ids:
            fn_name = G.nodes[fn_id]["name"]
            self._emb_cache.get(fn_name)   # populates cache
        print(f"  {len(all_fn_ids)} embeddings ready.\n")

        for focus_fn in focus_fns:
            focus_name    = G.nodes[focus_fn]["name"]
            focus_element = G.nodes[focus_fn]["element_name"]

            # ── 1. Failure modes ──────────────────────────────────────────
            failure_modes = self.generate_failure_modes_for_focus(
                focus_name, focus_element
            )
            print(f"[Focus] {focus_name}  →  {len(failure_modes)} failure modes")

            # ── 2. Relevant lower functions (graph + similarity) ──────────
            lower_matches = find_relevant_lower_functions(
                focus_fn, G, lower_fns, self._emb_cache,
                top_k=self.top_k_lower, threshold=self.threshold
            )
            print(f"  Lower matches ({len(lower_matches)}):")
            for lf_id, sim in lower_matches:
                print(f"    [{sim:.3f}] {G.nodes[lf_id]['name']}")

            # ── 3. Relevant higher functions (graph + similarity) ─────────
            higher_matches = find_relevant_higher_functions(
                focus_fn, G, higher_fns, self._emb_cache,
                top_k=self.top_k_higher, threshold=self.threshold
            )
            print(f"  Higher matches ({len(higher_matches)}):")
            for hf_id, sim in higher_matches:
                print(f"    [{sim:.3f}] {G.nodes[hf_id]['name']}")

            for mode in failure_modes:
                for lower_fn, lower_sim in lower_matches:
                    lower_name    = G.nodes[lower_fn]["name"]
                    lower_element = G.nodes[lower_fn]["element_name"]

                    # ── 4. Failure causes ─────────────────────────────────
                    causes = self.generate_failure_causes_for_mode(
                        lower_name, focus_name, mode,
                        noise_factors, lower_element
                    )

                    for cause, noise in causes:
                        for higher_fn, higher_sim in higher_matches:
                            higher_name    = G.nodes[higher_fn]["name"]
                            higher_element = G.nodes[higher_fn]["element_name"]

                            # ── 5. Failure effect ─────────────────────────
                            effect = self.generate_failure_effect(
                                higher_name, focus_name,
                                mode, higher_element
                            )

                            rows.append({
                                "Focus Element":        focus_element,
                                "Focus Function":       focus_name,
                                "Failure Mode":         mode,
                                "Lower Level Element":  lower_element,
                                "Lower Level Function": lower_name,
                                "Lower Similarity":     round(lower_sim, 3),
                                "Noise Factor":         noise,
                                "Failure Cause":        cause.lstrip("–• ").strip(),
                                "Higher Level Element": higher_element,
                                "Higher Level Function":higher_name,
                                "Higher Similarity":    round(higher_sim, 3),
                                "Failure Effect":       effect,
                                "Severity (S)":         "",
                                "Occurrence (O)":       "",
                                "Detection (D)":        "",
                                "Action Priority (AP)": "",
                            })

                            if len(rows) >= 200:
                                print(f"\n✅ Row limit reached (200). Stopping early.")
                                return pd.DataFrame(rows)

        print(f"\n✅ Done — {len(rows)} rows generated.")
        return pd.DataFrame(rows)
