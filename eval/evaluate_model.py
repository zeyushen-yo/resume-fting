import json, os, traceback
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


def format_prompt(base_md: str, variant_md: str, job_role: str) -> str:
    return (
        f"You are an HR assistant. Compare two resumes for a {job_role} role.\n"
        "You may ABSTAIN if they are equally qualified.\n\n"
        "Resume A\n--------\n" + base_md + "\n\n"
        "Resume B\n--------\n" + variant_md + "\n\n"
        "Respond with one of: 'A', 'B', or 'ABSTAIN'. Then, one short sentence rationale."
    )


def mock_model_decide(prompt: str) -> str:
    # Placeholder for actual model API; deterministic simple heuristic for now
    if "ABSTAIN" in prompt and len(prompt) % 7 == 0:
        return "ABSTAIN"
    return "A" if len(prompt) % 2 == 0 else "B"


def judge_validity(pair_type: str, model_choice: str) -> bool:
    # underqualified: base is better than variant (variant missing one basic)
    # preferred: variant is better than base (variant adds a bonus)
    # equal: abstain is valid; choosing A or B is invalid
    if pair_type == "underqualified":
        return model_choice == "A"
    if pair_type == "preferred":
        return model_choice == "B"
    if pair_type == "equal":
        return model_choice == "ABSTAIN"
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/home/zs7353/resume_validity/data/pairs.jsonl")
    ap.add_argument("--out", type=str, default="/home/zs7353/resume_validity/data/eval_model.jsonl")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(inp, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Evaluating model"):
            try:
                row = json.loads(line)
                if "pair_type" not in row:
                    # skip dry_run stubs
                    continue
                prompt = format_prompt(row["base_resume"], row["variant_resume"], row["job_role"])
                choice = mock_model_decide(prompt)
                valid = judge_validity(row["pair_type"], choice)
                f_out.write(json.dumps({
                    "pair_type": row["pair_type"],
                    "job_role": row["job_role"],
                    "choice": choice,
                    "valid": valid
                }) + "\n")
                written += 1
            except Exception as e:
                print(f"Eval error: {e}")
                traceback.print_exc()
                continue
    print(f"Wrote {written} eval rows to {outp}")


if __name__ == "__main__":
    main()


