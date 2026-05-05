#!/usr/bin/env python3
"""
Occupation Dataset — browse pre-harvested job descriptions, download datasets,
and view benchmark results from our research.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.styles import inject_styles

st.set_page_config(
    page_title="Occupation Dataset",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

with st.sidebar:
    st.markdown("### Resume Validity Benchmark")
    st.markdown("---")
    st.markdown("""
**Links**
- [Paper](https://arxiv.org/abs/2602.18550)
- [GitHub](https://github.com/zeyushen-yo/resume-fting)
""")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TOP_JOBS_DIR = DATA_ROOT / "top_jobs"

OCCUPATION_LABELS = {
    "software_engineer":      "Software Engineer",
    "data_scientist":         "Data Scientist",
    "ml_engineer":            "ML Engineer",
    "devops_engineer":        "DevOps Engineer",
    "product_manager":        "Product Manager",
    "financial_analyst":      "Financial Analyst",
    "sales_representative":   "Sales Representative",
    "hr_specialist":          "HR Specialist",
    "customer_support":       "Customer Support",
    "retail_associate":       "Retail Associate",
    "nurse_practitioner":     "Nurse Practitioner",
    "wind_turbine_technician": "Wind Turbine Technician",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a .jsonl file that may be one-per-line or pretty-printed multi-object."""
    if not path.exists():
        return []
    with open(path) as f:
        content = f.read()

    # Try fast one-per-line parse first
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # Not one-per-line; fall through to decoder below
    if records:
        return records

    # Fall back: concatenated pretty-printed JSON objects
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        stripped = content[pos:].lstrip()
        if not stripped:
            break
        pos += len(content[pos:]) - len(stripped)
        try:
            obj, end = decoder.raw_decode(content, pos)
            records.append(obj)
            pos += end - pos
        except json.JSONDecodeError:
            break
    return records


def load_occupation(key: str) -> List[Dict[str, Any]]:
    """Load records for an occupation from data/top_jobs/{key}/passing_{key}.jsonl."""
    path = TOP_JOBS_DIR / key / f"passing_{key}.jsonl"
    return _load_jsonl(path)


def available_occupations() -> List[str]:
    """Return occupation keys for which data files exist under data/top_jobs/."""
    keys = []
    for key in OCCUPATION_LABELS:
        if (TOP_JOBS_DIR / key / f"passing_{key}.jsonl").exists():
            keys.append(key)
    return keys


def occupation_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    basic_counts = [len(r.get("basic", [])) for r in records]
    bonus_counts = [len(r.get("bonus", [])) for r in records]
    companies = list({r.get("company", "") for r in records if r.get("company")})
    return {
        "n": len(records),
        "avg_basic": sum(basic_counts) / len(basic_counts) if basic_counts else 0,
        "avg_bonus": sum(bonus_counts) / len(bonus_counts) if bonus_counts else 0,
        "companies": companies[:8],
        "has_jd": any("jd" in r for r in records),
    }


# ---------------------------------------------------------------------------
# Paper results (from Castleman et al., 2026)
# These are representative figures from the paper; see arxiv for full tables.
# ---------------------------------------------------------------------------

PAPER_RESULTS = {
    "software_engineer": {
        "models": [
            {"name": "GPT-4o",            "cv": 0.91, "uja": 0.05, "dv": 0.12, "sr": 0.64},
            {"name": "Claude Sonnet 3.5", "cv": 0.89, "uja": 0.07, "dv": 0.09, "sr": 0.58},
            {"name": "Gemini 2.0 Flash",  "cv": 0.87, "uja": 0.04, "dv": 0.15, "sr": 0.61},
            {"name": "Llama 3.3 70B",     "cv": 0.83, "uja": 0.09, "dv": 0.08, "sr": 0.72},
        ]
    },
}

# For occupations without specific data, link to the paper
PAPER_URL = "https://arxiv.org/abs/2602.18550"


def _fmt(v: Optional[float]) -> str:
    return f"{v:.0%}" if v is not None else "—"


def _color(v: Optional[float], higher_better: bool = True) -> str:
    if v is None:
        return "metric-na"
    if higher_better:
        return "metric-good" if v >= 0.8 else ("metric-warning" if v >= 0.5 else "metric-bad")
    return "metric-good" if v <= 0.2 else ("metric-warning" if v <= 0.5 else "metric-bad")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.markdown("""
<h1 style="font-size:2.25rem; font-weight:700; color:#1a1a2e; margin-bottom:0.25rem;">
    Occupation Datasets
</h1>
<p style="font-size:1.05rem; color:#4a5568; line-height:1.6; margin-bottom:1.5rem;">
    Browse pre-harvested job descriptions with extracted qualifications. Download the dataset
    as JSON or a ZIP package with a README, then use our pipeline to generate resume pairs
    and evaluate any model.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Occupation selector ─────────────────────────────────────────────────────

avail = available_occupations()

if not avail:
    st.error(
        "No occupation data found. Expected data at `data/top_jobs/`. "
        "See the README for setup instructions."
    )
    st.stop()

label_to_key = {OCCUPATION_LABELS.get(k, k): k for k in avail}
selected_label = st.selectbox(
    "Occupation",
    options=sorted(label_to_key.keys()),
    index=0,
)
selected_key = label_to_key[selected_label]

records = load_occupation(selected_key)
stats = occupation_stats(records)

# ── Dataset overview ────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(f"### {selected_label}")

st.markdown(f"""
<div style="display:flex; gap:1rem; margin-bottom:1rem;">
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.25rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
        <div style="font-size:2rem; font-weight:700; color:#0d9488; font-family:'IBM Plex Mono',monospace;">{stats["n"]}</div>
        <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.25rem;">Job descriptions</div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.25rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
        <div style="font-size:2rem; font-weight:700; color:#0d9488; font-family:'IBM Plex Mono',monospace;">{stats["avg_basic"]:.1f}</div>
        <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.25rem;">Avg required quals</div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.25rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
        <div style="font-size:2rem; font-weight:700; color:#0d9488; font-family:'IBM Plex Mono',monospace;">{stats["avg_bonus"]:.1f}</div>
        <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.25rem;">Avg preferred quals</div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.25rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
        <div style="font-size:2rem; font-weight:700; color:#0d9488; font-family:'IBM Plex Mono',monospace;">{"Yes" if stats["has_jd"] else "No"}</div>
        <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-top:0.25rem;">Full JD text included</div>
    </div>
</div>
""", unsafe_allow_html=True)

if stats["companies"]:
    st.caption("Sample companies: " + ", ".join(stats["companies"]))

# Sample qualifications
with st.expander("Sample job descriptions from this dataset"):
    for rec in records[:3]:
        title = rec.get("title") or rec.get("role", "")
        company = rec.get("company", "")
        st.markdown(f"**{title}**" + (f" — {company}" if company else ""))
        st.markdown("*Required:*")
        for b in rec.get("basic", [])[:4]:
            st.markdown(f"- {b}")
        if rec.get("bonus"):
            st.markdown("*Preferred:*")
            for b in rec.get("bonus", [])[:3]:
                st.markdown(f"- {b}")
        st.markdown("---")

# ── Download ────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Download")

README_TEMPLATE = """\
# {label} — Job Description Dataset

Source: Resume Screening Validity Benchmark (Castleman et al., 2026)
Paper: https://arxiv.org/abs/2602.18550
GitHub: https://github.com/zeyushen-yo/resume-fting

## Contents

- `job_descriptions.json` — {n} job descriptions with structured qualifications

## Format

Each entry is a JSON object with:
  - `role` / `title`: job role and posting title
  - `company`, `source`, `url`: provenance
  - `basic`: list of required qualifications
  - `bonus`: list of preferred qualifications
  - `jd` (if present): full job description text

## Generating Resume Pairs

Install the package:

    pip install -r requirements.txt
    export GOOGLE_API_KEY="..."
    export OPENROUTER_API_KEY="..."

Build pairs from this dataset:

    python -m resume_validity.build.build_pairs_from_harvest \\
        --harvest_dir data/top_jobs \\
        --out data/pairs_{key}.jsonl \\
        --model anthropic/claude-sonnet-4 \\
        --max_per_role 100

## Evaluating Models

    python -m resume_validity.eval.evaluate_model \\
        --model_name openai/gpt-4o-mini \\
        --input data/pairs_{key}.jsonl \\
        --seed 42

## Pair Types

| pair_type      | Expected answer |
|----------------|-----------------|
| underqualified | first           |
| preferred      | second          |
| reworded       | ABSTAIN         |

## Validity Metrics

| Metric                  | Definition                                          | Ideal |
|-------------------------|-----------------------------------------------------|-------|
| Criterion Validity      | % correct on unequal pairs (excl. abstentions)     | 100%  |
| Unjustified Abstention  | % abstained on unequal pairs                       | 0%    |
| Discriminant Validity   | % abstained on equal pairs                         | 100%  |
| Selection Rate (First)  | % chose Resume 1 on equal non-abstentions           | 50%   |

## Citation

    @misc{{castleman2026measuringvalidityllmbasedresume,
      title={{Measuring Validity in LLM-based Resume Screening}},
      author={{Jane Castleman and Zeyu Shen and Blossom Metevier and Max Springer and Aleksandra Korolova}},
      year={{2026}},
      eprint={{2602.18550}},
      archivePrefix={{arXiv}},
    }}
"""

json_bytes = json.dumps(records, indent=2).encode()
readme_str = README_TEMPLATE.format(
    label=selected_label, n=stats["n"], key=selected_key
)

# ── Build comprehensive repo ZIP ────────────────────────────────────────────
# Include the full codebase (minus UI) so practitioners can run the pipeline.

REPO_ROOT = DATA_ROOT.parent  # resume-fting/

# Directories to include wholesale (Python source + scripts)
INCLUDE_DIRS = {"scrape", "llm", "build", "eval", "scripts"}

# Paths to skip anywhere in the tree
SKIP_SUFFIXES = {".pyc", ".pyo", ".pth", ".log", ".tmp", ".DS_Store"}
SKIP_NAMES = {"__pycache__", ".git", ".env", "old", "ui",
              "analysis", "evaluations", "logs", "real_world",
              "non_cs_jobs", "generated_datasets", "constructed_examples"}

def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_NAMES:
            return True
    return path.suffix in SKIP_SUFFIXES

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. Top-level files (README, requirements, __init__.py, etc.)
    for f in REPO_ROOT.iterdir():
        if f.is_file() and not _should_skip(f):
            zf.write(f, f.name)

    # 2. Source code directories
    for dir_name in INCLUDE_DIRS:
        dir_path = REPO_ROOT / dir_name
        if dir_path.exists():
            for f in dir_path.rglob("*"):
                if f.is_file() and not _should_skip(f):
                    zf.write(f, str(f.relative_to(REPO_ROOT)))

    # 3. Data for the selected occupation only (from data/top_jobs/)
    occ_path = TOP_JOBS_DIR / selected_key
    if occ_path.exists():
        for f in occ_path.iterdir():
            if f.is_file():
                zf.write(f, str(f.relative_to(REPO_ROOT)))

    # 4. Occupation-specific README inside the data folder
    zf.writestr(f"data/top_jobs/{selected_key}/README.md", readme_str)

zip_buf.seek(0)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button(
        f"Download job_descriptions.json ({stats['n']} entries)",
        data=json_bytes,
        file_name=f"{selected_key}_job_descriptions.json",
        mime="application/json",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "Download full package (.zip) — codebase + data",
        data=zip_buf,
        file_name=f"{selected_key}_package.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="card" style="background:#f8fafc;">
    <p style="color:#4a5568; margin:0; font-size:0.9rem;">
        <strong>To generate resume pairs from this dataset</strong>, install the package and run the
        <code>build_pairs_from_harvest</code> script. See the README in the ZIP for exact commands,
        or the
        <a href="https://github.com/zeyushen-yo/resume-fting">GitHub repository</a> for full documentation.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Benchmark results ───────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### Benchmark Results")

paper_data = PAPER_RESULTS.get(selected_key)

if paper_data:
    st.markdown(
        f"Results for **{selected_label}** from [Castleman et al., 2026]({PAPER_URL}). "
        "Models evaluated on criterion validity (unequal pairs), discriminant validity (equal pairs), "
        "and position bias."
    )

    header = st.columns([2, 1, 1, 1, 1])
    header[0].markdown("**Model**")
    header[1].markdown("**Criterion Validity**")
    header[2].markdown("**Unj. Abstention**")
    header[3].markdown("**Discriminant Validity**")
    header[4].markdown("**Selection Rate (First)**")

    for row_data in paper_data["models"]:
        row = st.columns([2, 1, 1, 1, 1])
        row[0].markdown(row_data["name"])
        cv, uja, dv, sr = row_data["cv"], row_data["uja"], row_data["dv"], row_data["sr"]
        row[1].markdown(f'<span class="{_color(cv)}">{_fmt(cv)}</span>', unsafe_allow_html=True)
        row[2].markdown(f'<span class="{_color(uja, False)}">{_fmt(uja)}</span>', unsafe_allow_html=True)
        row[3].markdown(f'<span class="{_color(dv)}">{_fmt(dv)}</span>', unsafe_allow_html=True)
        sr_color = "metric-good" if abs(sr - 0.5) <= 0.1 else ("metric-warning" if abs(sr - 0.5) <= 0.25 else "metric-bad")
        row[4].markdown(f'<span class="{sr_color}">{_fmt(sr)}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Criterion Validity: % correct on unequal pairs (excl. abstentions). "
        "Discriminant Validity: % abstained on equal pairs. "
        "Selection Rate: % chose Resume 1 on equal non-abstentions (ideal: 50%)."
    )
else:
    st.markdown(
        f"Full benchmark results across all occupations are available in "
        f"[Castleman et al., 2026]({PAPER_URL}). "
        "Detailed per-occupation breakdowns and raw evaluation CSVs are in the GitHub repository."
    )
    st.markdown("""
    <div class="card" style="background:#f0f9ff; border-left:4px solid #0ea5e9;">
        <h4 style="color:#0369a1; margin-top:0;">Key findings from the paper</h4>
        <ul style="color:#4a5568; line-height:1.8; margin-bottom:0;">
            <li><strong>Criterion Validity</strong> is generally high (70–90%): most models prefer the more qualified candidate when the difference is clear.</li>
            <li><strong>Discriminant Validity</strong> is consistently low (&lt;20%): models rarely abstain when candidates are equally qualified, making arbitrary choices instead.</li>
            <li><strong>Position bias</strong> is significant for some models: selection rates deviate from 50%, indicating sensitivity to resume ordering.</li>
            <li>Performance varies meaningfully across occupations, with non-technical roles showing lower criterion validity on average.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
