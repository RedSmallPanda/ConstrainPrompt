# stage2_tree_generation.py
import os
import json
import argparse
from typing import List, Dict, Any

import openai

# Keep the same API key line; only translate the comment to English.
openai.api_key = os.getenv("OPENAI_API_KEY")  # Replace with your key

# ----------------------------- Tool Schema (UNCHANGED) -----------------------------

EVAL_TREE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "generate_constraint_check_tree",
        "description": "Generate a logic tree to verify the output against a list of constraints.",
        "parameters": {
            "type": "object",
            "properties": {
                "tree": {
                    "type": "object",
                    "properties": {
                        "conditional": {"type": "boolean"},
                        "parent_ok": {"type": "boolean"},
                        "constraint_category": {"type": "string"},
                        "constraint": {"type": "string"},
                        "source": {"type": "string"},
                        "scope": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/function/parameters/properties/tree"}
                        }
                    },
                    "required": ["conditional", "parent_ok", "constraint_category", "constraint", "source", "scope", "children"]
                }
            },
            "required": ["tree"]
        }
    }
}

# ----------------------------- System Prompt (UNCHANGED) -----------------------------

EVAL_TREE_SYSTEM_PROMPT = """
You are a constraint-checking logic tree generator.

Your task is to construct a single decision tree for validating model output based on a list of constraints.

Each node must include the following fields:
- conditional: true or false
- parent_ok: true or false
- constraint_category: one of the following: 'Output → Specific format constraint', 'Output → Numerical constraint', 'Output → Lexical matching constraint', 'Output → Lexical exclusion constraint', or 'result' (used in leaf nodes)
- constraint: a concise human-readable description of what is being checked. **Also clearly indicate whether this applies to the entire output or to a specific field/section.** For example:  
  - "output must be valid JSON object"  
  - "output['queries'] must be a list of 1 to 5 unique strings"
- source: exactly copy the `source` field from the corresponding constraint object provided in the corresponding constraint in the input constraint list.
- scope: A description of **which part of the output** this constraint applies to.
  - Use "entire output" if the constraint applies to the whole response (e.g., length, general formatting, string content).
  - Use specific references (e.g., "JSON field 'questions'", "markdown header", "list elements", "first sentence") when the constraint targets a **subsection or component** of a structured output.
- children: a list of exactly two children unless this is a leaf node; children must describe what happens when the constraint **is met** and **is not met**

Rules:
1. The tree must evaluate constraints **in this order**:
   - First: all **conditional** constraints  
   - Then: all **unconditional** constraints  
   - Within each group: order by granularity — **format → type/field → value**
   - This reflects a macro-to-micro validation order: check overall structure first (e.g., JSON), then expected output type, then more detailed content or length.

2. Each conditional constraint must have two child branches:
   - If **condition is met** (`parent_ok=True`): its expected output behavior must be explicitly checked as a child node. Only apply the behavior required by that conditional constraint
   - If **condition is not met** (`parent_ok=False`): evaluate all unconditional constraints in order

3. For **every constraint node**, generate exactly **two children**:
   - One where `parent_ok = true` (constraint is satisfied)
   - One where `parent_ok = false` (constraint is not satisfied)

4. All **leaf nodes** must be of the form:
   - `conditional`: false
   - `parent_ok`: true or false
   - `constraint_category`: `'result'`
   - `constraint`: `'yes'` or `'no'`
   - `source`: `None`
   - `scope`: `None`
   - `children`: empty list

5. For conditional constraints, only evaluate the condition at the current node.
   - If the condition is met (parent_ok = true), generate a child node to check the required action or constraint specified by the condition.
   - If the condition is not met (parent_ok = false), proceed to check other related constraints.
"""

# ----------------------------- Core Functions -----------------------------

def generate_evaluation_tree(prompt: str, constraints: List[Dict[str, Any]], model: str = "gpt-4o") -> Dict[str, Any]:
    """Generate the evaluation tree from constraints."""
    user_prompt = f"""Prompt:\n{prompt}\n\nConstraints:\n{json.dumps(constraints, indent=2)}"""
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVAL_TREE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            tools=[EVAL_TREE_TOOL_SCHEMA],
            tool_choice={"type": "function", "function": {"name": "generate_constraint_check_tree"}},
            temperature=0
        )
        arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return arguments["tree"]
    except Exception as e:
        print(f"[ERROR] Evaluation tree generation failed: {e}")
        return {}

def pretty_print_tree(node: Dict[str, Any], prefix: str = "", is_last: bool = True):
    """Pretty-print the tree using the fields defined by the tool schema."""
    branch = "└── " if is_last else "├── "
    condition_tag = "[COND]" if node['conditional'] else "[UNCOND]"
    line = (
        f"{prefix}{branch}{condition_tag} "
        f"(parent_met={node['parent_ok']}) | "
        f"constraint_category: {node['constraint_category']} | scope: {node['scope']} | constraint: {node['constraint']} | source: {node['source']}"
    )
    print(line)

    children = node.get("children", [])
    new_prefix = prefix + ("    " if is_last else "│   ")
    for idx, child in enumerate(children):
        pretty_print_tree(child, new_prefix, idx == len(children) - 1)

# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage II - Tree Generation")
    parser.add_argument("--prompt", required=True, help="Path to a prompt file, or inline prompt text")
    parser.add_argument("--constraints", required=True, help="Path to Stage I output JSON (filtered constraints)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--out", default="tree.stage2.json", help="Output file for evaluation tree JSON")
    args = parser.parse_args()

    prompt_text = open(args.prompt, "r", encoding="utf-8").read() if os.path.exists(args.prompt) else args.prompt
    with open(args.constraints, "r", encoding="utf-8") as f:
        constraints = json.load(f)

    print("[Stage II] Generating evaluation tree...")
    tree = generate_evaluation_tree(prompt_text, constraints, model=args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)
    print(f"[Stage II] Wrote evaluation tree to: {args.out}")

    if tree:
        print("\n[Stage II] Tree preview:")
        pretty_print_tree(tree)

if __name__ == "__main__":
    main()
