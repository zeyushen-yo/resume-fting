Resume Validity Pipeline
========================

This project builds a new pipeline for constructing resume pairs from real job postings, generating realistic resumes via Gemini 2.5 Pro, evaluating models with abstention, and analyzing results.

Directories
- `scrape/`: job scraping and role selection
- `llm/`: Gemini utilities for qualification extraction and resume generation
- `build/`: pair construction and demographic pairing
- `eval/`: evaluation scripts for standard and agentic flows (with abstention)
- `analysis/`: result analysis
- `data/`: cache and outputs

Quick Start
1) Install requirements: `pip install -r requirements.txt`
2) Export credentials: `export GOOGLE_API_KEY=...`
3) Scrape and build pairs (dry run): `python -m resume_validity.build.build_pairs --dry_run`
4) Evaluate model: `python -m resume_validity.eval.evaluate_model --input data/pairs.jsonl`
5) Analyze: `python -m resume_validity.analysis.analyze_utility --inputs data/eval_results.jsonl`

Notes
- Names for demographics are sourced from `/home/zs7353/resume-fting/data/names.json`.
- Resume content includes a placeholder `{{CANDIDATE_NAME}}` near the top for name injection.
- Evaluation supports abstention: the model may answer that two resumes are equally qualified.


