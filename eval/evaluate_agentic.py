import json, traceback
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm


def agentic_decide(base_md: str, variant_md: str, job_role: str) -> str:
    # Placeholder: in real agentic flow we'd use tools + multiple calls; keep interface
    # Return 'A', 'B', or 'ABSTAIN'
    if len(base_md) == len(variant_md):
        return "ABSTAIN"
    return "A" if len(base_md) > len(variant_md) else "B"


def judge_validity(pair_type: str, choice: str) -> bool:
    if pair_type == "underqualified":
        return choice == "A"
    if pair_type == "preferred":
        return choice == "B"
    if pair_type == "equal":
        return choice == "ABSTAIN"
    return False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="/home/zs7353/resume_validity/data/pairs.jsonl")
    ap.add_argument("--out", type=str, default="/home/zs7353/resume_validity/data/eval_agentic.jsonl")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(inp, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Evaluating agentic"):
            try:
                row = json.loads(line)
                if "pair_type" not in row:
                    continue
                choice = agentic_decide(row["base_resume"], row["variant_resume"], row["job_role"])
                valid = judge_validity(row["pair_type"], choice)
                f_out.write(json.dumps({
                    "pair_type": row["pair_type"],
                    "job_role": row["job_role"],
                    "choice": choice,
                    "valid": valid
                }) + "\n")
                written += 1
            except Exception as e:
                print(f"Agentic eval error: {e}")
                traceback.print_exc()
                continue
    print(f"Wrote {written} agentic eval rows to {outp}")


if __name__ == "__main__":
    main()


