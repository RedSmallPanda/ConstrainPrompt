# stage1_extraction.py
import os
import json
import argparse
from typing import List, Dict, Any

import openai
from tqdm import tqdm

from typing import List, Dict

# Keep the same API key line; only translate the comment to English.
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your key

# ----------------------------- Tool Schemas (UNCHANGED) -----------------------------

CATEGORY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_constraints_from_prompt",
        "description": "Extract and classify constraints from a single prompt template.",
        "parameters": {
            "type": "object",
            "properties": {
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "constraint": {"type": "string"},
                            "application_type": {
                                "type": "string",
                                "enum": ["unconditional", "conditional"]
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "Output → Specific format constraint",
                                    "Output → Numerical constraint",
                                    "Output → Lexical matching constraint",
                                    "Output → Lexical exclusion constraint",
                                    "Output → Semantic inclusion constraint",
                                    "Output → Semantic exclusion constraint",
                                    "Output → Qualitative constraint",
                                    "Others"
                                ]
                            },
                            "reason": {"type": "string"},
                            "source": {"type": "string"}
                        },
                        "required": ["constraint", "application_type", "category", "reason", "source"]
                    }
                }
            },
            "required": ["constraints"]
        }
    }
}

SINGLE_CONDITION_ASSESS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "assess_single_conditional_condition",
        "description": "Assess whether ONE conditional constraint's condition is code-verifiable (semantics-agnostic).",
        "parameters": {
            "type": "object",
            "properties": {
                "assessment": {
                    "type": "object",
                    "properties": {
                        "constraint": {"type": "string"},
                        "condition_verifiable": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "suggested_check": {
                            "type": "string",
                            "description": "If verifiable, a brief suggestion for how to check it in code using only the raw input string (e.g., regex/keyword/length)."
                        }
                    },
                    "required": ["constraint", "condition_verifiable", "reason"]
                }
            },
            "required": ["assessment"]
        }
    }
}

# ----------------------------- System Prompts (UNCHANGED) -----------------------------

SYSTEM_PROMPT = """
You are a precision prompt constraint analyzer.

Given a prompt template, extract **all constraints** that specify what the model output must or must not do. For each constraint:
- Determine whether it's **unconditional** or **conditional**:
  - **unconditional**: This constraint applies universally to all outputs, regardless of the input content.
  - **conditional**: The prompt specifies a trigger condition that determines whether the constraint applies, and the trigger is either:
    - Expressed using clear indicators like “if …”, “when …”, “only if …”, “in case …”, or
    - Explicitly tied to detectable input features (e.g., contains specific keywords, matches a language code, exceeds a given length).
- Assign one of the following categories:
  1. Output → Specific format constraint: The output must conform to a specific file or data format (e.g., JSON, Markdown, HTML, key–value pairs, defined data structures, source code in a specific language).
  2. Output → Numerical constraint: Restrictions on output length, counts, or numerical ranges (e.g., character/word/token/sentence/paragraph counts, score values).
  3. Output → Lexical matching constraint: The output must contain, match, or adhere to a specific string pattern (e.g., selection from a predefined list, exact string match, lowercase requirement).
  4. Output → Lexical exclusion constraint: Certain words, phrases, string or character patterns are explicitly prohibited in the output.
  5. Output → Semantic inclusion constraint: The output must semantically include certain concepts, entities, or topics. (not verifable by code)
  6. Output → Semantic exclusion constraint: The output must not semantically mention certain concepts, entities, or topics. (not verifable by code)
  7. Output → Qualitative constraint: The output must exhibit specific non-quantitative qualities or styles (e.g., concise, academic tone, persuasive, language). (not verifable by code)
  8. Others
- For each constraint, extract the **exact sentence** from the prompt that expresses it as `source`
- Give a short justification for the category.

Return all constraints in a structured list using the function tool.
"""

SYSTEM_PROMPT_CONDITION_ASSESS_SINGLE = """
You assess ONE conditional constraint at a time.

Goal: Decide if the constraint's **condition** is objectively code-verifiable from the raw input string only.

A condition is code-verifiable iff it can be checked deterministically with string match/regex/length/keyword/numeric tests over the input text alone, requiring no semantic interpretation, intent understanding, topic inference without explicit keywords, or external knowledge.

Return ONLY via the tool call with:
- constraint (verbatim),
- condition_verifiable: true/false,
- reason,
- suggested_check (optional if verifiable).

Do not return free text.
"""

# ----------------------------- Core Functions -----------------------------

def classify_prompt_constraints(prompt: str, model: str = "gpt-4o") -> List[Dict[str, str]]:
    """Call the LLM to extract and classify constraints from the prompt template."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Prompt template:\n{prompt}"}
            ],
            tools=[CATEGORY_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "classify_constraints_from_prompt"}},
            temperature=0
        )
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return arguments.get("constraints", [])
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return []

def assess_single_conditional_bool(constraint_text: str, source_text: str, prompt_text: str, model: str = "gpt-4o") -> bool:
    """
    Use SINGLE_CONDITION_ASSESS_TOOL_SCHEMA and SYSTEM_PROMPT_CONDITION_ASSESS_SINGLE
    to assess ONE conditional constraint. Return True if its condition is code-verifiable.
    """
    user_payload = {
        "prompt": prompt_text,
        "constraint": constraint_text,
        "source": source_text
    }
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CONDITION_ASSESS_SINGLE.strip()},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            tools=[SINGLE_CONDITION_ASSESS_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "assess_single_conditional_condition"}},
            temperature=0
        )
        args = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
        return bool(args.get("condition_verifiable", False))
    except Exception as e:
        print(f"[ERROR] Condition verifiability assessment failed: {e}")
        return False

def filter_code_verifiable_constraints(constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only constraints that can be checked by code."""
    code_verifiable_categories = {
        "Output → Specific format constraint",
        "Output → Numerical constraint",
        "Output → Lexical matching constraint",
        "Output → Lexical exclusion constraint"
    }
    return [c for c in constraints if c.get("category") in code_verifiable_categories]

def filter_code_verifiable_conditionals(constraints: List[Dict[str, Any]], prompt_text: str, model: str = "gpt-4o") -> List[Dict[str, Any]]:
    """
    Filtering rules:
    - Keep all 'unconditional' constraints.
    - For 'conditional' constraints, keep only those whose condition is code-verifiable.
    """
    kept = []
    for c in tqdm(constraints, desc="Filtering conditional constraints", disable=True):
        if c.get("application_type") == "unconditional":
            kept.append(c)
        elif c.get("application_type") == "conditional":
            ok = assess_single_conditional_bool(
                constraint_text=c.get("constraint", ""),
                source_text=c.get("source", ""),
                prompt_text=prompt_text,
                model=model
            )
            if ok:
                kept.append(c)
    return kept

# ----------------------------- CLI -----------------------------

def _read_text(maybe_path: str) -> str:
    """Read from a file path if exists; otherwise treat input as inline text."""
    if os.path.exists(maybe_path):
        with open(maybe_path, "r", encoding="utf-8") as f:
            return f.read()
    return maybe_path

def main():
    parser = argparse.ArgumentParser(description="Stage I - Constraint Extraction")
    parser.add_argument("--prompt", required=True, help="Path to a prompt file, or inline prompt text")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--out", default="constraints.stage1.json", help="Output file for filtered constraints JSON")
    parser.add_argument("--save-raw", default=None, help="Optional path to save raw (unfiltered) constraints JSON")
    args = parser.parse_args()

    prompt_text = _read_text(args.prompt)

    print("[Stage I] Extracting constraints...")
    raw_constraints = classify_prompt_constraints(prompt_text, model=args.model)

    if args.save_raw:
        with open(args.save_raw, "w", encoding="utf-8") as f:
            json.dump(raw_constraints, f, ensure_ascii=False, indent=2)
        print(f"[Stage I] Saved raw constraints to: {args.save_raw}")

    filtered = filter_code_verifiable_constraints(raw_constraints)
    filtered = filter_code_verifiable_conditionals(filtered, prompt_text, model=args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f"[Stage I] Wrote code-verifiable constraints to: {args.out}")

if __name__ == "__main__":
    main()
