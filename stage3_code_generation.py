# stage3_code_generation.py
import os
import json
import argparse
from typing import Dict, Any

import openai

# Keep the same API key line; only translate the comment to English.
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your key

# ----------------------------- Tool Schema (UNCHANGED) -----------------------------

GENERATE_CODE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_output_checker_code",
        "description": "Generate a Python function that returns (bool, reason, violation) when verifying output using a constraint evaluation tree.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python code defining is_valid_output(output: str, input_text: str) -> Tuple[bool, Optional[str], Optional[str]]"
                }
            },
            "required": ["code"]
        }
    }
}

# ----------------------------- System Prompt (UNCHANGED) -----------------------------

SYSTEM_PROMPT_CODE_GEN = """
You are a Python code generation agent specialized in logical validation.

Your task is to generate a Python function `is_valid_output(output: str, input_text: str) -> Tuple[bool, Optional[str], Optional[str]]` that checks whether the model output satisfies a set of constraints, which are organized as a decision tree.

The decision tree follows this format:
- Each node contains:
  - `conditional`: whether this is a conditional constraint (True/False)
  - `parent_ok`: whether the condition of its parent node was satisfied (True/False)
  - `constraint_category`: the category of the constraint or 'result' if leaf node
  - `constraint`: a human-readable string that describes the check
  - `source`: the source constraint (the original sentence in prompt) that represents the check in evaluation tree
  - `scope`: the part of the output this constraint applies to
  - `children`: list of child nodes

### Rules:
1. Before any validation, normalize the `output` string to prevent false negatives from insignificant whitespace:
   - Strip leading and trailing blank lines
   - Collapse multiple consecutive blank lines into a single blank line
   
2. Traverse the tree starting from the root node. At each node:
   - If `constraint_category == 'result'`, return `(True, None, None)` if `constraint == 'yes'`, else return `(False, <short reason>, <violation>)`, where <reason> is a concise diagnostic describing what failed and <violation> is the constraint's `source` value in evaluation tree (the original prompt constraint that was violated).
   - If `conditional == True`, evaluate the *condition part only* of the constraint at this level
       - If the condition is **not directly verifiable in code**, return False by default
       - If the condition is verifiable, use an `if` to decide which branch of `children` to check
   - If `conditional == False` and `parent_ok == True`, validate the constraint against the output
       - If it passes, recurse to `children[0]`; else recurse to `children[1]`
       - If `parent_ok == False`, directly recurse to `children[1]`

3. The `output` is always the raw string returned by a language model. Any structural checks (e.g., JSON parsing) or type checks (e.g., numeric value) must first convert or parse this string appropriately.

4. The generated code must handle malformed output robustly (e.g., invalid JSON)

5. You may define helper functions for checking common patterns (e.g., word count, JSON keys, exact match)

6. The constraint string should be used as a comment to make clear what each check is doing.

7. Only use standard Python libraries (no external dependencies).

Output the complete Python function only. Do not include explanation or comments outside the code.
"""

# ----------------------------- Core Function -----------------------------

def generate_checker_code(prompt: str, tree: Dict[str, Any], model: str = "gpt-4o") -> str:
    """Compile the evaluation tree into an executable Python validator source."""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CODE_GEN},
                {"role": "user", "content": f"Prompt: {prompt}\n\nEvaluation Tree:\n{json.dumps(tree, indent=2)}"}
            ],
            temperature=0,
            tools=[GENERATE_CODE_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "generate_output_checker_code"}},
        )
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return arguments["code"]
    except Exception as e:
        print(f"[ERROR] GPT code generation failed: {e}")
        return ""

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage III - Code Generation")
    parser.add_argument("--prompt", required=True, help="Path to a prompt file, or inline prompt text")
    parser.add_argument("--tree", required=True, help="Path to Stage II output JSON (evaluation tree)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--out", default="checker.stage3.py", help="Output file for generated Python validator")
    args = parser.parse_args()

    prompt_text = open(args.prompt, "r", encoding="utf-8").read() if os.path.exists(args.prompt) else args.prompt
    with open(args.tree, "r", encoding="utf-8") as f:
        tree = json.load(f)

    print("[Stage III] Generating Python validator code...")
    code_str = generate_checker_code(prompt_text, tree, model=args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(code_str)
    print(f"[Stage III] Wrote validator code to: {args.out}")

if __name__ == "__main__":
    main()
