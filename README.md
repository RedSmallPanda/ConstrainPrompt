# Paper Pipeline (Three Stages)

This repository implements the main pipeline of the paper "ConstrainPrompt: Code-Based Assurance of Prompt-Defined Constraints":

1. **Constraint Extraction** → `stage1_extraction.py`  
2. **Tree Generation** → `stage2_tree_generation.py`  
3. **Code Generation** → `stage3_code_generation.py`

---

## Environment & Setup

**Requirements**
- Python 3.9+
- Packages: `openai`, `tqdm`

**Install**
```bash
pip install openai tqdm
```

**API Key**
```bash
export OPENAI_API_KEY="sk-..."   # Recommended for security
```
Override via `OPENAI_API_KEY` in practice.

---

## Repository Structure

```
stage1_extraction.py        # Stage I: constraint extraction (with main())
stage2_tree_generation.py   # Stage II: evaluation tree synthesis (with main())
stage3_code_generation.py   # Stage III: Python validator code generation (with main())
run_checker.py              # (Optional) your runner to execute generated code
calculate_bleu.py           # (Optional) compute sentence-level BLEU for model-selected span and gold label
prompt.txt                  # (Example) your prompt template (you provide)
```

---

## Data Contracts (I/O Between Stages)

### Stage I → Stage II
- **Input:** Prompt text (file path or inline string).
- **Output (JSON, e.g., `constraints.stage1.json`):** list of constraint objects:
  - `constraint` (str)
  - `application_type` ∈ {`unconditional`, `conditional`}
  - `category` ∈ {
    `Output → Specific format constraint`,
    `Output → Numerical constraint`,
    `Output → Lexical matching constraint`,
    `Output → Lexical exclusion constraint`,
    `Output → Semantic inclusion constraint`,
    `Output → Semantic exclusion constraint`,
    `Output → Qualitative constraint`,
    `Others`
    }
  - `reason` (str)
  - `source` (str)

> Stage I keeps only **code-verifiable** categories and filters **conditional** constraints to those whose **condition is code-verifiable**.

### Stage II → Stage III
- **Input:** The prompt text and Stage I JSON.
- **Output (JSON, e.g., `tree.stage2.json`):** evaluation tree object:
  - `conditional` (bool)  
  - `parent_ok` (bool)  
  - `constraint_category` (str)  
  - `constraint` (str)  
  - `source` (str)  
  - `scope` (str)  
  - `children` (list of nodes with the same schema)

### Stage III Output
- **Input:** The prompt text and Stage II JSON.
- **Output (Python file, e.g., `checker.stage3.py`):** source code defining:
```python
is_valid_output(output: str, input_text: str) -> Tuple[bool, Optional[str], Optional[str]]
```
which returns `(passed, reason, violation)`.

---

## Stage I — Constraint Extraction

**Script:** `stage1_extraction.py`

**What it does**
- Calls the LLM to extract constraints.
- Keeps **code-verifiable** categories only.
- Check and retain only conditional constraints with **code-verifiable** conditions.

**Run**
```bash
python stage1_extraction.py \
  --prompt prompt.txt \
  --model gpt-4o \
  --out constraints.stage1.json \
  --save-raw raw_constraints.json
```

**Arguments**
- `--prompt` (required): path to a prompt file **or** inline prompt text.
- `--model` (optional, default: `gpt-4o`)
- `--out` (optional, default: `constraints.stage1.json`)
- `--save-raw` (optional): path to save raw (unfiltered) constraints.

**Result**
- Writes `constraints.stage1.json`.  
Example (shape only):
```json
[
  {
    "constraint": "...",
    "application_type": "unconditional",
    "category": "Output → Numerical constraint",
    "reason": "...",
    "source": "..."
  }
]
```

---

## Stage II — Tree Generation

**Script:** `stage2_tree_generation.py`

**What it does**
- Synthesizes a **guard-first, coarse-to-fine** evaluation tree from Stage I constraints.
- Pretty-prints the tree to console.

**Run**
```bash
python stage2_tree_generation.py \
  --prompt prompt.txt \
  --constraints constraints.stage1.json \
  --model gpt-4o \
  --out tree.stage2.json
```

**Arguments**
- `--prompt` (required): path to a prompt file **or** inline prompt text.
- `--constraints` (required): path to Stage I output JSON.
- `--model` (optional, default: `gpt-4o`)
- `--out` (optional, default: `tree.stage2.json`)

**Result**
- Writes `tree.stage2.json` and prints a textual tree preview to the console.

---

## Stage III — Code Generation

**Script:** `stage3_code_generation.py`

**What it does**
- Compiles the evaluation tree into a Python file that exports:
```python
is_valid_output(output: str, input_text: str) -> Tuple[bool, Optional[str], Optional[str]]
```
The generated function normalizes whitespace, traverses the tree, handles malformed structures (e.g., JSON), and returns a concise failure reason plus the violated `source` on failure.

**Run**
```bash
python stage3_code_generation.py \
  --prompt prompt.txt \
  --tree tree.stage2.json \
  --model gpt-4o \
  --out checker.stage3.py
```

**Arguments**
- `--prompt` (required): path to a prompt file **or** inline prompt text.
- `--tree` (required): path to Stage II output JSON.
- `--model` (optional, default: `gpt-4o`)
- `--out` (optional, default: `checker.stage3.py`)

**Result**
- Writes `checker.stage3.py` containing the complete `is_valid_output(...)` validator.

---

## End-to-End Example

```bash
# 1) Extract constraints
python stage1_extraction.py \
  --prompt prompt.txt \
  --out constraints.stage1.json

# 2) Build evaluation tree
python stage2_tree_generation.py \
  --prompt prompt.txt \
  --constraints constraints.stage1.json \
  --out tree.stage2.json

# 3) Generate validator code
python stage3_code_generation.py \
  --prompt prompt.txt \
  --tree tree.stage2.json \
  --out checker.stage3.py
```

(Optional) Execute the generated validator:
- Use your runner (e.g., `run_checker.py`) to import the generated file and call:
```python
from checker.stage3 import is_valid_output
passed, reason, violation = is_valid_output(output_text, input_text)
```
- Or use an existing helper such as:
```python
run_checker.run_generated_checker(code_str=code_str, input_text=input_text, output_text=output_text)
```

## Evaluate Generated Data (BLEU, single pair)

**Script:** `calculate_bleu.py` — compute sentence-level BLEU for a **single** pair: one predicted *reason* vs one *gold* label.

### Install (optional)
```bash
pip install nltk
```
If `nltk` is not installed, the script falls back to a simple token-overlap approximation.

### CLI
```bash
python calculate_bleu.py \
  --gold "the gold reference reason text" \
  --reason "the model-generated reason text" \
  --verbose
```
- Prints a single floating-point BLEU score in `[0, 1]`.
- `--verbose` also prints normalized texts.

**Examples**
```bash
# Minimal (prints BLEU only)
python calculate_bleu.py --gold "missing required key" --reason "missing required key"