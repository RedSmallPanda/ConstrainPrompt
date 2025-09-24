import sys
import types
import json
from typing import Optional, Tuple

def run_generated_checker(code_str: str, input_text: str, output_text: str) -> bool:
    checker_module = types.ModuleType("checker_module")
    try:
        exec(code_str, checker_module.__dict__)
        if "is_valid_output" not in checker_module.__dict__:
            raise ValueError("❌ Generated code does not contain `is_valid_output` function.")

        result = checker_module.is_valid_output(output_text, input_text)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("❌ `is_valid_output` must return a tuple: (bool, Optional[str], Optional[str])")

        passed, reason, violation = result
        if passed:
            # print("✅ Output passed all checks.")
            output = {
                "satisfied": True,
                "reason": None,
                "violation": None
            }
            print(json.dumps(output, ensure_ascii=False))
        else:
            # print(f"❌ Output failed.\n")
            output = {
                "satisfied": False,
                "reason": reason,
                "violation": violation
            }
            print(json.dumps(output, ensure_ascii=False))
        return passed

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run generated constraint-checking code.")
    parser.add_argument("--code", type=str, required=True, help="Path to the generated Python code file.")
    parser.add_argument("--input", type=str, required=True, help="Input text string.")
    parser.add_argument("--output", type=str, required=True, help="Model output string.")

    args = parser.parse_args()

    with open(args.code, "r", encoding="utf-8") as f:
        code_string = f.read()

    print("\n▶️ Running constraint check...\n")
    result = run_generated_checker(code_string, args.input, args.output)
    print(f"\n🎯 Final Result: {'PASS ✅' if result else 'FAIL ❌'}")
