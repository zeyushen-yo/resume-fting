#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import inspect

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:
    print("[ERROR] PyTorch is required to run fine-tuning.")
    print(traceback.format_exc())
    raise

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        PreTrainedTokenizerBase,
    )
    from transformers.trainer_utils import get_last_checkpoint
except Exception:
    print("[ERROR] transformers is required.")
    print(traceback.format_exc())
    raise

# Full fine-tuning only; no PEFT/LoRA


SYSTEM_PROMPT = (
    "You are an impartial hiring assistant. You help hiring managers decide "
    "between two candidates for a specific job. Carefully evaluate both "
    "candidates equally – do not favour the first candidate simply because "
    "they appear first."
)


def name_from_resume(md: str) -> str:
    for line in str(md).splitlines():
        if line.strip().lower().startswith("name:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_inputs_block(base_md: str, var_md: str) -> str:
    return (
        "Resume 1\n---------\n" + str(base_md).strip() + "\n\n"
        + "Resume 2\n---------\n" + str(var_md).strip()
    )


def build_user_message(example: Dict[str, Any]) -> str:
    base = example["inputs"].rstrip()
    jd = str(example.get("job_description") or "").strip()
    if jd and "Job Description:" not in base:
        base = f"Job Description:\n{jd}\n\n" + base
    user_msg = base + (
        "\n\nFirst, briefly justify your choice in 1-2 sentences. Then output your final decision "
        "ONLY inside <answer>...</answer> tags. Respond with exactly ONE of: 'first', 'second', "
        "or 'ABSTAIN'. If the two resumes are equally qualified, output <answer>ABSTAIN</answer>. "
        "Definition: 'first' means choose Resume 1; 'second' means choose Resume 2 as presented above."
    )
    return user_msg


def compute_gold_answer(row: Dict[str, Any]) -> str:
    pt = str(row.get("pair_type") or "").lower()
    if pt == "underqualified":
        return "<answer>first</answer>"
    elif pt == "preferred":
        return "<answer>second</answer>"
    elif pt in {"equal", "reworded", "equivalent"}:
        return "<answer>ABSTAIN</answer>"
    else:
        # Unknown type: skip by returning empty string
        return ""


def read_pairs_build_examples_from_files(
    paths: List[Path],
    allowed_experiment_types: List[str],
    exclude_roles: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Read JSONL pairs from one or more files and construct input/output examples for SFT.

    Filters by experiment_type if allowed_experiment_types is not empty.
    Input text mirrors evaluate_model.py: system guidance + user content.
    Output text is <answer>...</answer> or <answer>ABSTAIN</answer>.
    """
    allowed: List[str] = [s.strip().lower() for s in allowed_experiment_types or [] if str(s).strip()]
    examples: List[Dict[str, Any]] = []
    excluded = set([r.strip() for r in (exclude_roles or []) if str(r).strip()])
    try:
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        print("[WARN] Skipping unparsable line.")
                        continue
                    role_val = str(row.get("role") or "").strip()
                    if role_val and role_val in excluded:
                        continue
                    if allowed:
                        et = str(row.get("experiment_type") or "").strip().lower()
                        if et not in allowed:
                            continue
                    base = row.get("base_resume", "")
                    var = row.get("variant_resume", "")
                    if not base or not var:
                        # Accept pre-packed records containing "inputs"
                        inputs = row.get("inputs")
                        if not inputs:
                            continue
                        gold = compute_gold_answer(row)
                        if not gold:
                            continue
                        ex = {
                            **row,
                            "input_text": (
                                "System: " + SYSTEM_PROMPT + "\n\n" + build_user_message(row)
                            ),
                            "target_text": gold,
                        }
                        examples.append(ex)
                        continue

                    inputs = build_inputs_block(base, var)
                    row["inputs"] = inputs
                    gold = compute_gold_answer(row)
                    if not gold:
                        continue
                    ex = {
                        **row,
                        "input_text": (
                            "System: " + SYSTEM_PROMPT + "\n\n" + build_user_message(row)
                        ),
                        "target_text": gold,
                    }
                    examples.append(ex)
    except Exception:
        print("[ERROR] Failed to read/prepare pairs examples.")
        print(traceback.format_exc())
        raise

    return examples
def compute_job_group_key(rec: Dict[str, Any]) -> str:
    """Return a stable group key for a job posting used for disjoint splits.

    Preference order:
    1) Explicit id fields if present (job_id, job_posting_id, posting_id)
    2) Normalized job_description text hash (case/whitespace agnostic)
    3) Fallback to pair content hash to at least keep pairs together
    """
    for key in ("job_id", "job_posting_id", "posting_id"):
        val = rec.get(key)
        if val is not None and str(val).strip():
            return f"id:{str(val).strip()}"
    jd = str(rec.get("job_description") or "").strip()
    if jd:
        norm = " ".join(jd.split()).lower()
        return "jd:" + hashlib.md5(norm.encode("utf-8")).hexdigest()
    base_md = str(rec.get("base_resume") or "")
    var_md = str(rec.get("variant_resume") or "")
    h = hashlib.md5((base_md + "||" + var_md).encode("utf-8")).hexdigest()
    return "pair:" + h



class PairSFTDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        prompt = rec["input_text"].rstrip() + "\n\nFinal decision:"  # small cue
        answer = rec["target_text"].strip()

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer,
            truncation=True,
            max_length=max(8, min(64, self.max_length // 8)),
            add_special_tokens=False,
        )["input_ids"]
        input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
        labels = ([-100] * len(prompt_ids)) + answer_ids + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


@dataclass
class DataCollatorWithPaddingMask:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Find max length
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"]) 
            input_ids.append(
                torch.nn.functional.pad(
                    f["input_ids"], (0, pad_len), value=self.tokenizer.pad_token_id
                )
            )
            attention_mask.append(
                torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0)
            )
            labels.append(
                torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100)
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def extract_answer(text: str) -> str:
    """Extract decision from model text, preferring content inside <answer> tags.

    Returns one of: 'first', 'second', 'ABSTAIN', or '' if not found.
    """
    if not isinstance(text, str):
        return ""
    import re
    # Prefer tagged answer (case-insensitive)
    matches = list(re.finditer(r"<\s*answer\s*>(.*?)</\s*answer\s*>", text, flags=re.I | re.S))
    if matches:
        val = matches[-1].group(1).strip().strip('"').strip("'")
        lv = val.lower()
        if lv in {"first", "second", "abstain"}:
            return "ABSTAIN" if lv == "abstain" else lv
        return val
    # Fallback: plain-text detection; prefer explicit first/second over abstain
    lower = text.lower()
    if "<answer>" in lower:  # malformed closing; try naive split
        try:
            val = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
            lv = val.lower()
            if lv in {"first", "second", "abstain"}:
                return "ABSTAIN" if lv == "abstain" else lv
        except Exception:
            pass
    if "first" in lower:
        return "first"
    if "second" in lower:
        return "second"
    if "abstain" in lower:
        return "ABSTAIN"
    return ""


def is_valid_record(rec: Dict[str, Any], pred_answer: str) -> Tuple[bool, bool]:
    """Return (is_correct_label, is_valid) using first/second/ABSTAIN labels."""
    pt = str(rec.get("pair_type") or "").lower()
    label = str(pred_answer or "").strip().lower()
    abstained = (label == "abstain")
    if pt in {"equal", "reworded", "equivalent"}:
        return (abstained, abstained)
    if pt == "underqualified":
        return (label == "first", label == "first")
    if pt == "preferred":
        return (label == "second", label == "second")
    return (False, False)


def run_inference(
    model, tokenizer, records: List[Dict[str, Any]], out_csv: Path, batch_size: int = 2, max_new_tokens: int = 64
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    device = next(model.parameters()).device
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = []
        for rec in batch:
            msg = build_user_message({"inputs": build_inputs_block(rec.get("base_resume", ""), rec.get("variant_resume", "")), "job_description": rec.get("job_description", "")})
            prompts.append("System: " + SYSTEM_PROMPT + "\n\n" + msg + "\n\nFinal decision:")

        with torch.no_grad():
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=tokenizer.model_max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for rec, full_text in zip(batch, decoded):
            pred = extract_answer(full_text)
            is_correct, is_valid = is_valid_record(rec, pred)
            rows.append({
                **rec,
                "generated": full_text,
                "pred_answer": pred,
                "is_correct": is_correct,
                "is_valid": is_valid,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser("SFT on resume pairs for final decision tagging")
    parser.add_argument("--model_path", type=str, default="/scratch/gpfs/zs7353/Llama-3.1-8B-Instruct")
    parser.add_argument("--data_path", type=str, default="/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_rel6_with_jd.jsonl")
    parser.add_argument("--extra_data", type=str, nargs="*", default=None, help="Optional additional JSONL files or directories to include.")
    parser.add_argument("--experiment_types", type=str, default="validity_demographics", help="Comma-separated experiment types to include (e.g., validity,validity_demographics)")
    parser.add_argument("--exclude_roles", type=str, default="", help="Comma-separated role directory names to exclude from training")
    parser.add_argument("--exclude_roles_file", type=str, default="", help="Optional file with one role per line to exclude from training")
    parser.add_argument("--eval_external", type=str, default="", help="Optional JSONL file or directory for external evaluation set (bypasses internal test eval)")
    parser.add_argument("--output_dir", type=str, default="/home/zs7353/resume_validity/evaluations/ft/llama_sft")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--disjoint_by_job", action="store_true", help="If set, split train/test by job posting groups so no job overlaps across splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=4096)
    # No LoRA/PEFT; full fine-tuning
    parser.add_argument("--use_4bit", action="store_true", default=False)
    parser.add_argument("--eval_max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Collect data files (file or directory)
    def collect_jsonl(p: Path) -> List[Path]:
        if p.is_dir():
            return sorted([q for q in p.glob("*.jsonl") if q.is_file()])
        if p.is_file():
            return [p]
        return []

    paths: List[Path] = []
    primary = Path(args.data_path)
    paths.extend(collect_jsonl(primary))
    if args.extra_data:
        for item in args.extra_data:
            paths.extend(collect_jsonl(Path(item)))
    paths = [p for p in paths if p.exists()]
    if not paths:
        print(f"[ERROR] No data files found from inputs: {[args.data_path] + (args.extra_data or [])}")
        sys.exit(1)

    allowed_types = [s.strip() for s in str(args.experiment_types or "").split(",") if s.strip()]
    # Collect excluded roles
    excluded_roles: List[str] = []
    if args.exclude_roles:
        excluded_roles.extend([s.strip() for s in args.exclude_roles.split(",") if s.strip()])
    if args.exclude_roles_file and Path(args.exclude_roles_file).exists():
        try:
            with open(args.exclude_roles_file, "r", encoding="utf-8") as f:
                excluded_roles.extend([ln.strip() for ln in f if ln.strip()])
        except Exception:
            print("[WARN] Failed to read exclude_roles_file; continuing without it.")

    # Prepare examples
    examples = read_pairs_build_examples_from_files(paths, allowed_types, exclude_roles=excluded_roles)
    if not examples:
        print("[ERROR] No training examples prepared.")
        sys.exit(1)

    # Train/test split (optionally disjoint by job posting)
    if args.disjoint_by_job:
        rng = np.random.RandomState(args.seed)
        group_to_indices: Dict[str, List[int]] = {}
        for i, rec in enumerate(examples):
            gid = compute_job_group_key(rec)
            group_to_indices.setdefault(gid, []).append(i)
        group_ids = list(group_to_indices.keys())
        rng.shuffle(group_ids)
        total = len(examples)
        target_test = max(1, int(total * args.test_size))
        acc = 0
        chosen_test_groups: List[str] = []
        for gid in group_ids:
            if acc >= target_test:
                break
            chosen_test_groups.append(gid)
            acc += len(group_to_indices[gid])
        test_idx = set()
        for gid in chosen_test_groups:
            test_idx.update(group_to_indices[gid])
        train_records = [examples[i] for i in range(len(examples)) if i not in test_idx]
        test_records = [examples[i] for i in range(len(examples)) if i in test_idx]
        print(
            f"Prepared {len(train_records)} train / {len(test_records)} test examples "
            f"(group-split by job: {len(group_ids) - len(chosen_test_groups)} train groups / {len(chosen_test_groups)} test groups)."
        )
    else:
        rng = np.random.RandomState(args.seed)
        idx = np.arange(len(examples))
        rng.shuffle(idx)
        test_n = max(1, int(len(idx) * args.test_size))
        test_idx = set(idx[:test_n].tolist())
        train_records = [examples[i] for i in range(len(examples)) if i not in test_idx]
        test_records = [examples[i] for i in range(len(examples)) if i in test_idx]
        print(f"Prepared {len(train_records)} train / {len(test_records)} test examples.")

    # Load tokenizer/model
    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For decoder-only models, use left padding during training/eval
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    tokenizer.model_max_length = args.max_length

    load_kwargs = {}
    if args.use_4bit:
        try:
            import bitsandbytes as bnb  # noqa: F401
            load_kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "device_map": "auto",
            }
        except Exception:
            print("[WARN] bitsandbytes not available; proceeding without 4-bit.")
            load_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        low_cpu_mem_usage=True,
        **load_kwargs,
    )

    # Full fine-tuning path: train all parameters

    # Build datasets
    train_ds = PairSFTDataset(train_records, tokenizer, args.max_length)
    test_ds = PairSFTDataset(test_records, tokenizer, args.max_length)

    collator = DataCollatorWithPaddingMask(tokenizer)

    # Training
    print("Starting training...")
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    # Build TrainingArguments with backward compatibility for older transformers
    base_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        save_steps=500,
        logging_steps=50,
    )
    optional_kwargs = dict(
        evaluation_strategy="steps",
        eval_steps=500,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=["none"],
    )
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in {**base_kwargs, **optional_kwargs}.items() if k in allowed}
        train_args = TrainingArguments(**filtered)
    except Exception as e:
        print("[WARN] TrainingArguments signature filtering failed; falling back to minimal args.")
        print(traceback.format_exc())
        train_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
        )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    try:
        trainer.train(resume_from_checkpoint=last_ckpt)
    except Exception:
        print("[ERROR] Trainer failed during training.")
        print(traceback.format_exc())
        sys.exit(1)

    try:
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
    except Exception:
        print("[WARN] Failed to save full model/tokenizer; continuing.")
        print(traceback.format_exc())

    # Inference: if external eval is provided, use it; otherwise fall back to the internal test split
    def _run_eval_on_records(records: List[Dict[str, Any]], out_name: str) -> None:
        try:
            preds_csv = Path(args.output_dir) / out_name
            df = run_inference(
                model,
                tokenizer,
                [{
                    "base_resume": r.get("base_resume", ""),
                    "variant_resume": r.get("variant_resume", ""),
                    "job_description": r.get("job_description", ""),
                    "pair_type": r.get("pair_type", ""),
                } for r in records],
                preds_csv,
                batch_size=max(1, args.batch_size // 2),
                max_new_tokens=args.eval_max_new_tokens,
            )
            overall_valid = float(df["is_valid"].mean()) if len(df) else 0.0
            by_pt = df.groupby(df["pair_type"].astype(str).str.lower())["is_valid"].mean().to_dict() if len(df) else {}
            print(f"Eval validity ({out_name}): {overall_valid:.3f}")
            print("By pair_type validity:", by_pt)
            with open(Path(args.output_dir) / (out_name.replace(".csv", "_metrics.json")), "w", encoding="utf-8") as f:
                json.dump({"overall_valid": overall_valid, "by_pair_type_valid": by_pt}, f, indent=2)
        except Exception:
            print("[ERROR] Evaluation failed for:", out_name)
            print(traceback.format_exc())
            sys.exit(1)

    if args.eval_external:
        # Load external eval JSONL(s)
        ext_paths: List[Path] = []
        p = Path(args.eval_external)
        if p.is_dir():
            ext_paths = sorted([q for q in p.glob("*.jsonl") if q.is_file()])
        elif p.is_file():
            ext_paths = [p]
        else:
            print("[ERROR] eval_external path not found:", str(p))
            sys.exit(1)

        external_records: List[Dict[str, Any]] = []
        for fp in ext_paths:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    base = row.get("base_resume", "")
                    var  = row.get("variant_resume", "")
                    if not base or not var:
                        continue
                    external_records.append({
                        "base_resume": base,
                        "variant_resume": var,
                        "job_description": row.get("job_description", ""),
                        "pair_type": row.get("pair_type", ""),
                    })
        print(f"Loaded {len(external_records)} external eval pairs from {str(p)}")
        _run_eval_on_records(external_records, out_name="external_test_predictions.csv")
    else:
        print("Running inference on internal test split...")
        _run_eval_on_records(test_records, out_name="test_predictions.csv")


if __name__ == "__main__":
    main()


