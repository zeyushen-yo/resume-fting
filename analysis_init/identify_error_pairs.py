#!/usr/bin/env python3
"""
Identify resume pairs with generation errors (truncation, missing qualifications, etc.)
and create a blacklist for filtering evaluation results.
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Set, Tuple


def parse_args():
    p = argparse.ArgumentParser("Identify error pairs in dataset")
    p.add_argument("--pairs_dir", type=str, required=True, help="Directory containing pairs_shard_*.jsonl files")
    p.add_argument("--output", type=str, default="analysis/error_pairs_blacklist.json", help="Output JSON with error pair hashes")
    return p.parse_args()


def pair_hash(base_resume: str, variant_resume: str) -> str:
    """Create a unique hash for a resume pair."""
    combined = base_resume + "|||" + variant_resume
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def check_truncation(text: str) -> bool:
    """Check if resume appears truncated."""
    # Look for truncation markers
    if '........' in text or '......' in text:
        return True
    # Check if ends abruptly (very short or ends mid-sentence)
    if len(text) < 200:
        return True
    return False


def check_qualification_present(resume: str, qualification: str, fuzzy: bool = True) -> bool:
    """Check if a qualification is present in the resume."""
    resume_lower = resume.lower()
    qual_lower = qualification.lower()
    
    if fuzzy:
        # Check if first 30 chars of qualification appear
        check_str = qual_lower[:30] if len(qual_lower) > 30 else qual_lower
        return check_str in resume_lower
    else:
        return qual_lower in resume_lower


def identify_errors(pairs_dir: Path) -> Set[str]:
    """Identify all error pairs and return set of their hashes."""
    error_hashes = set()
    
    for fp in sorted(pairs_dir.glob("pairs_shard_*.jsonl")):
        with open(fp, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                except:
                    continue
                
                base = obj.get('base_resume', '')
                variant = obj.get('variant_resume', '')
                pair_type = obj.get('pair_type', '')
                differed = obj.get('differed_qualifications', [])
                num_differed = obj.get('num_differed', 0)
                
                # Check for truncation
                if check_truncation(base) or check_truncation(variant):
                    h = pair_hash(base, variant)
                    error_hashes.add(h)
                    print(f"[TRUNCATION] {fp.name}:{line_num} - {obj.get('job_title')} - {pair_type} k={num_differed}")
                    continue
                
                # Check underqualified pairs: removed qualifications should NOT be in variant
                if pair_type == 'underqualified' and differed:
                    # For underqualified, we expect qualifications to be removed
                    # Check if they're still present (error)
                    still_present = sum(1 for q in differed if check_qualification_present(variant, q))
                    if still_present > len(differed) * 0.3:  # More than 30% still present = error
                        h = pair_hash(base, variant)
                        error_hashes.add(h)
                        print(f"[UNDERQUAL-ERROR] {fp.name}:{line_num} - {obj.get('job_title')} - {still_present}/{len(differed)} quals still present")
                
                # Check preferred pairs: added qualifications SHOULD be in variant but NOT in base
                if pair_type == 'preferred' and differed:
                    missing_from_variant = sum(1 for q in differed if not check_qualification_present(variant, q))
                    if missing_from_variant > 0:  # Any missing = error
                        h = pair_hash(base, variant)
                        error_hashes.add(h)
                        print(f"[PREFERRED-ERROR] {fp.name}:{line_num} - {obj.get('job_title')} - {missing_from_variant}/{len(differed)} quals missing from variant")
    
    return error_hashes


def main():
    args = parse_args()
    pairs_dir = Path(args.pairs_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    error_hashes = identify_errors(pairs_dir)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(list(error_hashes), f, indent=2)
    
    print(f"\nFound {len(error_hashes)} error pairs")
    print(f"Saved blacklist to {output_path}")


if __name__ == "__main__":
    main()


