#!/usr/bin/env python3
"""
Upload Job Description — select from our dataset or paste your own, configure
qualifications, choose a resume source, generate pairs, and evaluate LLMs.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.config import (
    EVALUATION_MODELS,
    QUICK_MODELS,
    SYSTEM_PROMPT,
    check_bypass_password,
    get_bypass_api_config,
    get_openrouter_key,
)
from ui.stress_test import (
    MAX_RETRIES,
    RETRY_DELAY_BASE,
    Qualification,
    clean_resume_to_markdown,
    extract_answer,
    extract_qualifications,
    generate_preferred_variant,
    generate_reworded_variant,
    generate_underqualified_variant,
)
from ui.styles import inject_styles
from ui.utils import extract_text_from_file, extract_texts_from_files

st.set_page_config(
    page_title="Upload Job Description",
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
# Dataset helpers (mirrors 2_Occupation_Dataset.py)
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent.parent.parent / "data"
TOP_JOBS_DIR = DATA_ROOT / "top_jobs"

OCCUPATION_LABELS = {
    "software_engineer":       "Software Engineer",
    "data_scientist":          "Data Scientist",
    "ml_engineer":             "ML Engineer",
    "devops_engineer":         "DevOps Engineer",
    "product_manager":         "Product Manager",
    "financial_analyst":       "Financial Analyst",
    "sales_representative":    "Sales Representative",
    "hr_specialist":           "HR Specialist",
    "customer_support":        "Customer Support",
    "retail_associate":        "Retail Associate",
    "nurse_practitioner":      "Nurse Practitioner",
    "wind_turbine_technician": "Wind Turbine Technician",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        content = f.read()
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    if records:
        return records
    # fallback: pretty-printed multi-object
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


def available_occupations() -> List[str]:
    return [k for k in OCCUPATION_LABELS
            if (TOP_JOBS_DIR / k / f"passing_{k}.jsonl").exists()]


def load_occupation_records(key: str) -> List[Dict[str, Any]]:
    return _load_jsonl(TOP_JOBS_DIR / key / f"passing_{key}.jsonl")


# ---------------------------------------------------------------------------
# API / LLM helpers
# ---------------------------------------------------------------------------

def _call_api(api_base: str, api_key: str, model: str,
              messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
    api_base = api_base.strip().rstrip("/")
    if not api_base.endswith("/v1"):
        api_base += "/v1"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(f"{api_base}/chat/completions",
                                 headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
        except requests.exceptions.HTTPError as e:
            if resp.status_code in [429, 500, 502, 503, 504]:
                last_error = e
                time.sleep(RETRY_DELAY_BASE * (2 ** attempt))
            else:
                raise
    raise last_error or RuntimeError("Max retries exhausted")


def build_base_resume(api_base: str, api_key: str, model: str,
                      role_title: str, basic_quals: List[str]) -> str:
    system = "You write realistic professional resumes in Markdown. Use concise, credible content."
    user = (
        f"Write a realistic resume in Markdown for the role: {role_title}.\n"
        f"Include ALL of these required qualifications and no others beyond reasonable elaboration:\n"
        + "\n".join(f"- {q}" for q in basic_quals)
        + "\n\nRules:\n"
        "- Start with 'Name: {{CANDIDATE_NAME}}'.\n"
        "- Replace company names with {{COMPANY_NAME}} and schools with {{SCHOOL_NAME}}.\n"
        "- No contact information. One page. Sections: Summary, Experience, Education, Skills.\n"
    )
    return _call_api(api_base, api_key, model,
                     [{"role": "system", "content": system},
                      {"role": "user", "content": user}], max_tokens=2048)


def evaluate_pair(api_base: str, api_key: str, model: str,
                  resume1: str, resume2: str, jd: str, expected: str) -> Dict[str, Any]:
    user_msg = (
        f"Job Description:\n{jd}\n\n"
        f"Resume 1\n---------\n{resume1.strip()}\n\n"
        f"Resume 2\n---------\n{resume2.strip()}\n\n"
        "Briefly justify your choice in 1-2 sentences, then output your final decision "
        "ONLY inside <answer>...</answer> tags. Respond with exactly ONE of: "
        "'first', 'second', or 'ABSTAIN'."
    )
    try:
        response = _call_api(api_base, api_key, model,
                             [{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}], max_tokens=512)
        decision = extract_answer(response)
        is_correct = True if expected == "either" else (decision == expected)
        return {"decision": decision, "is_correct": is_correct,
                "raw_response": response, "error": None}
    except Exception as e:
        return {"decision": "", "is_correct": False, "raw_response": "", "error": str(e)}


@dataclass
class BenchmarkResult:
    model_name: str
    n_strict: int = 0
    strict_correct: int = 0
    strict_abstained: int = 0
    n_equal: int = 0
    equal_abstained: int = 0
    equal_selected_first: int = 0
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def criterion_validity(self) -> Optional[float]:
        n = self.n_strict - self.strict_abstained
        return self.strict_correct / n if n else None

    @property
    def unjustified_abstention(self) -> Optional[float]:
        return self.strict_abstained / self.n_strict if self.n_strict else None

    @property
    def discriminant_validity(self) -> Optional[float]:
        return self.equal_abstained / self.n_equal if self.n_equal else None

    @property
    def selection_rate_first(self) -> Optional[float]:
        n = self.n_equal - self.equal_abstained
        return self.equal_selected_first / n if n else None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _api_config() -> Optional[Dict[str, str]]:
    env_key = get_openrouter_key()
    if env_key:
        return {"api_base": "https://openrouter.ai/api", "api_key": env_key,
                "model": "google/gemini-2.0-flash-001"}
    with st.expander("API Access", expanded=True):
        st.markdown(
            "An OpenRouter API key is required to parse qualifications and generate resumes. "
            "Set `OPENROUTER_API_KEY` in your environment, or enter credentials below."
        )
        col1, col2 = st.columns(2)
        with col1:
            use_demo = st.checkbox("Use demo mode",
                                   help="Use our API credits via a shared password.")
            if use_demo:
                pw = st.text_input("Demo password", type="password")
                if pw:
                    if check_bypass_password(pw):
                        st.success("Demo mode active.")
                        cfg = get_bypass_api_config()
                        return {"api_base": cfg["api_base"], "api_key": cfg["api_key"],
                                "model": "google/gemini-2.0-flash-001"}
                    else:
                        st.error("Incorrect password.")
        with col2:
            if not use_demo:
                custom_key = st.text_input("OpenRouter API key", type="password",
                                           placeholder="sk-or-...")
                if custom_key:
                    return {"api_base": "https://openrouter.ai/api", "api_key": custom_key,
                            "model": "google/gemini-2.0-flash-001"}
    return None


def _fmt(v: Optional[float]) -> str:
    return f"{v:.0%}" if v is not None else "N/A"


def _color(v: Optional[float], higher_better: bool = True) -> str:
    if v is None:
        return "metric-na"
    if higher_better:
        return "metric-good" if v >= 0.8 else ("metric-warning" if v >= 0.5 else "metric-bad")
    return "metric-good" if v <= 0.2 else ("metric-warning" if v <= 0.5 else "metric-bad")


def _stat_row(items: List[tuple]) -> None:
    """Render a row of stat boxes. items = [(label, value), ...]"""
    cols = "".join(f"""
        <div style="flex:1; background:#fff; border:1px solid #e2e8f0; border-radius:12px;
                    padding:1.25rem; text-align:center; box-shadow:0 2px 6px rgba(0,0,0,.04);">
            <div style="font-size:2rem; font-weight:700; color:#0d9488;
                        font-family:'IBM Plex Mono',monospace;">{v}</div>
            <div style="font-size:0.8rem; color:#64748b; text-transform:uppercase;
                        letter-spacing:.05em; margin-top:.25rem;">{l}</div>
        </div>""" for l, v in items)
    st.markdown(f'<div style="display:flex; gap:1rem; margin-bottom:1rem;">{cols}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown("""
<h1 style="font-size:2.25rem; font-weight:700; color:#1a1a2e; margin-bottom:.25rem;">
    Upload Job Description
</h1>
<p style="font-size:1.05rem; color:#4a5568; line-height:1.6; margin-bottom:1.5rem;">
    Select one of our pre-parsed job descriptions or paste your own. Then choose to generate
    a synthetic base resume or upload your own resumes. Generate controlled pairs, download
    them, and optionally evaluate a suite of LLMs.
</p>
""", unsafe_allow_html=True)

api_cfg = _api_config()

st.markdown("---")

# ── Section 1: Job Description source ──────────────────────────────────────

st.markdown("### Job Description")
jd_tab_dataset, jd_tab_custom = st.tabs(["Select from our dataset", "Paste / upload your own"])

with jd_tab_dataset:
    avail = available_occupations()
    if not avail:
        st.warning("No occupation data found at `data/top_jobs/`.")
    else:
        label_to_key = {OCCUPATION_LABELS.get(k, k): k for k in avail}
        occ_label = st.selectbox("Occupation", sorted(label_to_key.keys()),
                                 key="jd_occ_select")
        occ_key = label_to_key[occ_label]
        records = load_occupation_records(occ_key)

        # Build display options: "Title — Company"
        def _rec_label(r: Dict) -> str:
            title = r.get("title") or r.get("original_role") or r.get("role", "Unknown")
            company = r.get("company", "")
            return f"{title} — {company}" if company else title

        options = [_rec_label(r) for r in records]
        chosen_idx = st.selectbox("Role / company", range(len(options)),
                                  format_func=lambda i: options[i],
                                  key="jd_role_select")
        chosen_rec = records[chosen_idx]

        # Show a preview of extracted qualifications
        with st.expander("Preview qualifications", expanded=True):
            c_basic, c_bonus = st.columns(2)
            with c_basic:
                st.markdown(f"**Required ({len(chosen_rec.get('basic', []))})**")
                for q in chosen_rec.get("basic", []):
                    st.markdown(f"- {q}")
            with c_bonus:
                st.markdown(f"**Preferred ({len(chosen_rec.get('bonus', []))})**")
                for q in chosen_rec.get("bonus", []):
                    st.markdown(f"- {q}")

        if st.button("Use this job description", key="use_dataset_jd"):
            basic = [Qualification(text=q, kind="basic")
                     for q in chosen_rec.get("basic", [])]
            bonus = [Qualification(text=q, kind="bonus")
                     for q in chosen_rec.get("bonus", [])]
            jd_text = chosen_rec.get("jd", "")
            if not jd_text:
                role = chosen_rec.get("title") or chosen_rec.get("role", "")
                jd_text = f"Role: {role}\n\nRequired Qualifications:\n"
                jd_text += "\n".join(f"- {q}" for q in chosen_rec.get("basic", []))
                if chosen_rec.get("bonus"):
                    jd_text += "\n\nPreferred Qualifications:\n"
                    jd_text += "\n".join(f"- {q}" for q in chosen_rec.get("bonus", []))
            st.session_state["parsed_quals"] = {"basic": basic, "bonus": bonus}
            st.session_state["parsed_jd"] = jd_text
            st.session_state["parsed_role_title"] = (
                chosen_rec.get("title") or chosen_rec.get("role", ""))
            st.session_state.pop("generated_pairs", None)
            st.session_state.pop("eval_results", None)
            st.success("Job description loaded.")

with jd_tab_custom:
    col_upload, col_text = st.columns([1, 2])
    with col_upload:
        jd_file = st.file_uploader("Upload (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"],
                                   key="jd_file")
        file_text = ""
        if jd_file:
            file_text = extract_text_from_file(jd_file)
            if file_text:
                st.success(f"Loaded {len(file_text):,} characters from {jd_file.name}")
    with col_text:
        jd_text_input = st.text_area(
            "Paste job description",
            value=file_text, height=220,
            placeholder="Paste the full job description here.",
            label_visibility="collapsed",
        )
    if st.button("Parse Job Description", disabled=not api_cfg,
                 help="Requires API access"):
        if not jd_text_input.strip():
            st.error("Please provide a job description.")
        else:
            with st.spinner("Extracting qualifications..."):
                try:
                    quals = extract_qualifications(jd_text_input)
                    st.session_state["parsed_quals"] = quals
                    st.session_state["parsed_jd"] = jd_text_input
                    st.session_state.pop("parsed_role_title", None)
                    st.session_state.pop("generated_pairs", None)
                    st.session_state.pop("eval_results", None)
                except Exception as e:
                    st.error(f"Parsing failed: {e}")
                    st.code(traceback.format_exc())

# ── Section 2: Configure qualifications ────────────────────────────────────

if "parsed_quals" in st.session_state:
    quals: Dict = st.session_state["parsed_quals"]
    basic: List[Qualification] = quals.get("basic", [])
    bonus: List[Qualification] = quals.get("bonus", [])

    st.markdown("---")
    st.markdown("### Qualifications")

    col_basic, col_bonus = st.columns(2)
    priority_set: set = set()

    with col_basic:
        st.markdown(f"**Required ({len(basic)})**")
        st.caption("Uncheck to exclude from pair generation.")
        for q in basic:
            if st.checkbox(q.text, key=f"basic_{hash(q.text)}", value=True):
                priority_set.add(q.text)
    with col_bonus:
        st.markdown(f"**Preferred ({len(bonus)})**")
        st.caption("Preferred qualifications are used to generate 'preferred' (unequal) pairs.")
        for q in bonus:
            st.markdown(f"- {q.text}")

    with st.expander("Demographics of interest (optional)"):
        st.markdown(
            "Select demographic groups to record as metadata. Use `build/assign_names.py` "
            "to inject names after download."
        )
        selected_demos = st.multiselect(
            "Demographic groups",
            ["White / Male (W_M)", "White / Female (W_W)",
             "Black / Male (B_M)", "Black / Female (B_W)"],
            default=[],
        )

    # ── Section 3: Resume source ────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### Resume Source")

    res_tab_generate, res_tab_upload = st.tabs([
        "Generate a synthetic resume",
        "Upload your own resumes",
    ])

    with res_tab_generate:
        st.markdown(
            "We will use an LLM to build a base resume that satisfies the selected "
            "required qualifications, then generate variants from it."
        )
        default_title = st.session_state.get("parsed_role_title", "")
        role_title = st.text_input("Role title (optional — inferred if blank)",
                                   value=default_title, key="role_title_gen")
        num_reworded_gen = st.slider("Equal (reworded) pairs", 1, 5, 3,
                                     key="num_reworded_gen")
        resume_source = "generate"

    with res_tab_upload:
        st.markdown(
            "Upload one or more resumes. We will generate underqualified, preferred, "
            "and reworded variants from each one."
        )
        resume_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_upload_own",
        )
        uploaded_resumes: List[str] = []
        if resume_files:
            uploaded_resumes = extract_texts_from_files(resume_files)
            st.success(f"{len(uploaded_resumes)} resume(s) loaded.")

        paste_resume = st.text_area("Or paste a single resume", height=160,
                                    key="paste_resume_own",
                                    placeholder="Paste resume text here...")
        if paste_resume.strip():
            uploaded_resumes = [paste_resume.strip()] + uploaded_resumes

        num_reworded_upload = st.slider("Equal (reworded) pairs per resume", 1, 5, 3,
                                        key="num_reworded_upload")
        resume_source = "upload"  # last-written wins; actual branching below

    # ── Section 4: Generate pairs ────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### Generate Pairs")

    if st.button("Generate Pairs", disabled=not api_cfg, key="gen_pairs_btn"):
        basic_texts = [q.text for q in basic if q.text in priority_set]
        bonus_texts = [q.text for q in bonus]
        jd = st.session_state["parsed_jd"]
        demo_codes = [d.split("(")[1].rstrip(")") for d in selected_demos] \
                     if selected_demos else []

        # Determine which tab was active last (Streamlit can't detect this directly,
        # so we infer from whether uploaded_resumes is populated)
        use_upload = bool(uploaded_resumes)
        num_reworded = num_reworded_upload if use_upload else num_reworded_gen

        progress = st.progress(0)
        status = st.empty()
        pairs: List[Dict[str, Any]] = []

        def _make_pair(base: str, variant: str, pair_type: str,
                       differed: List[str], better: str,
                       resume_label: str = "") -> Dict[str, Any]:
            p = {
                "job_title": st.session_state.get("parsed_role_title", ""),
                "job_source": {"source": "custom"},
                "job_description": jd,
                "base_resume": base,
                "variant_resume": variant,
                "pair_type": pair_type,
                "differed_qualifications": differed,
                "num_differed": len(differed),
                "better": better,
                "demographics": ["placeholder", "placeholder"],
                "experiment_type": "validity",
            }
            if demo_codes:
                p["requested_demographics"] = demo_codes
            if resume_label:
                p["resume_label"] = resume_label
            return p

        try:
            if use_upload:
                total = len(uploaded_resumes) * (1 + len(basic_texts) + len(bonus_texts) + num_reworded)
                step = [0]

                def _tick(msg: str):
                    step[0] += 1
                    progress.progress(min(step[0] / max(total, 1), 1.0))
                    status.text(msg)

                for r_idx, raw_resume in enumerate(uploaded_resumes):
                    label = f"Resume {r_idx + 1}"
                    _tick(f"{label}: cleaning...")
                    try:
                        base = clean_resume_to_markdown(raw_resume)
                    except Exception:
                        base = raw_resume  # fall back to raw text

                    for qt in basic_texts:
                        _tick(f"{label}: underqualified variant — {qt[:35]}...")
                        pairs.append(_make_pair(
                            base, generate_underqualified_variant(base, qt),
                            "underqualified", [qt], "first", label))

                    for qt in bonus_texts:
                        _tick(f"{label}: preferred variant — {qt[:35]}...")
                        pairs.append(_make_pair(
                            base, generate_preferred_variant(base, qt),
                            "preferred", [qt], "second", label))

                    for i in range(num_reworded):
                        _tick(f"{label}: equal pair {i + 1}/{num_reworded}...")
                        pairs.append(_make_pair(
                            base, generate_reworded_variant(base),
                            "reworded", [], "either", label))

            else:
                inferred_title = role_title.strip() or \
                    st.session_state.get("parsed_role_title", "the described role")
                total = 1 + len(basic_texts) + len(bonus_texts) + num_reworded
                step = [0]

                def _tick(msg: str):
                    step[0] += 1
                    progress.progress(min(step[0] / max(total, 1), 1.0))
                    status.text(msg)

                _tick("Building synthetic base resume...")
                base = build_base_resume(
                    api_cfg["api_base"], api_cfg["api_key"], api_cfg["model"],
                    inferred_title, basic_texts)

                for qt in basic_texts:
                    _tick(f"Underqualified variant — {qt[:40]}...")
                    pairs.append(_make_pair(
                        base, generate_underqualified_variant(base, qt),
                        "underqualified", [qt], "first"))

                for qt in bonus_texts:
                    _tick(f"Preferred variant — {qt[:40]}...")
                    pairs.append(_make_pair(
                        base, generate_preferred_variant(base, qt),
                        "preferred", [qt], "second"))

                for i in range(num_reworded):
                    _tick(f"Equal pair {i + 1}/{num_reworded}...")
                    pairs.append(_make_pair(
                        base, generate_reworded_variant(base),
                        "reworded", [], "either"))

            progress.progress(1.0)
            status.text(f"Done — {len(pairs)} pairs generated.")
            st.session_state["generated_pairs"] = pairs
            st.session_state.pop("eval_results", None)

        except Exception as e:
            st.error(f"Pair generation failed: {e}")
            st.code(traceback.format_exc())

# ── Section 5: Download ──────────────────────────────────────────────────────

if "generated_pairs" in st.session_state:
    pairs = st.session_state["generated_pairs"]
    unequal = [p for p in pairs if p["pair_type"] != "reworded"]
    equal   = [p for p in pairs if p["pair_type"] == "reworded"]
    n_quals = len(set(q for p in unequal for q in p["differed_qualifications"]))

    st.markdown("---")
    st.markdown("### Download Dataset")
    _stat_row([("Total pairs", len(pairs)), ("Unequal pairs", len(unequal)),
               ("Equal pairs", len(equal)), ("Qualifications tested", n_quals)])

    json_bytes = json.dumps(pairs, indent=2).encode()
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("pairs.json", json.dumps(pairs, indent=2))
        zf.writestr("README.md", """# Resume Pairs Dataset

Generated by the Resume Screening Validity Benchmark.

## Format
Each entry: job_description, base_resume, variant_resume, pair_type,
differed_qualifications, better ("first"/"second"/"either"), experiment_type.

## Pair types
| pair_type      | Expected answer |
|----------------|-----------------|
| underqualified | first           |
| preferred      | second          |
| reworded       | ABSTAIN         |

## Evaluate
    python -m resume_validity.eval.evaluate_model \\
        --model_name openai/gpt-4o-mini \\
        --input pairs.json --seed 42

## Add demographic names
    python -m resume_validity.build.assign_names \\
        --input pairs.json \\
        --same_out pairs_same_group.jsonl \\
        --cross_out pairs_cross_group.jsonl
""")
    zip_buf.seek(0)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download pairs.json", data=json_bytes,
                           file_name="pairs.json", mime="application/json",
                           use_container_width=True)
    with dl2:
        st.download_button("Download dataset.zip (pairs + README)", data=zip_buf,
                           file_name="dataset.zip", mime="application/zip",
                           use_container_width=True)

    # ── Section 6: LLM Evaluation ────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### LLM Evaluation Suite")
    st.markdown(
        "Run the generated pairs through a suite of LLMs to measure criterion validity, "
        "discriminant validity, and position bias."
    )

    suite = st.radio("Model suite",
                     ["Small (3 models — faster)", "Large (5 models)"],
                     horizontal=True, key="eval_suite")
    models_to_run = QUICK_MODELS if suite.startswith("Small") else EVALUATION_MODELS
    st.caption("Models: " + ", ".join(m["name"] for m in models_to_run))

    if st.button("Run LLM Evaluation", disabled=not api_cfg, key="run_eval_btn"):
        all_results: List[BenchmarkResult] = []
        eval_progress = st.progress(0)
        eval_status = st.empty()
        total = len(models_to_run) * len(pairs)
        done = [0]

        for model_cfg in models_to_run:
            result = BenchmarkResult(model_name=model_cfg["name"])
            for pair in pairs:
                eval_status.text(
                    f"{model_cfg['name']}: pair {done[0] % len(pairs) + 1}/{len(pairs)}")
                er = evaluate_pair(
                    api_cfg["api_base"], api_cfg["api_key"], model_cfg["id"],
                    pair["base_resume"], pair["variant_resume"],
                    pair["job_description"], pair["better"])
                decision = (er["decision"] or "").lower()
                is_abstain = decision == "abstain"

                if pair["pair_type"] in ("underqualified", "preferred"):
                    result.n_strict += 1
                    if is_abstain:
                        result.strict_abstained += 1
                    elif er["is_correct"]:
                        result.strict_correct += 1
                else:
                    result.n_equal += 1
                    if is_abstain:
                        result.equal_abstained += 1
                    elif decision == "first":
                        result.equal_selected_first += 1

                result.detailed_results.append({
                    "pair_type": pair["pair_type"],
                    "qualification": pair["differed_qualifications"][0]
                                     if pair["differed_qualifications"] else "",
                    "decision": decision, "is_correct": er["is_correct"],
                    "abstained": is_abstain,
                })
                if er["error"]:
                    result.errors.append(f"{model_cfg['name']}: {er['error']}")

                done[0] += 1
                eval_progress.progress(min(done[0] / max(total, 1), 1.0))

            all_results.append(result)

        eval_progress.progress(1.0)
        eval_status.text("Evaluation complete.")
        st.session_state["eval_results"] = all_results

    if "eval_results" in st.session_state:
        results: List[BenchmarkResult] = st.session_state["eval_results"]
        st.markdown("#### Results")

        header = st.columns([2, 1, 1, 1, 1])
        header[0].markdown("**Model**")
        header[1].markdown("**Criterion Validity**")
        header[2].markdown("**Unj. Abstention**")
        header[3].markdown("**Discriminant Validity**")
        header[4].markdown("**Selection Rate (First)**")

        for r in results:
            row = st.columns([2, 1, 1, 1, 1])
            row[0].markdown(r.model_name)
            row[1].markdown(f'<span class="{_color(r.criterion_validity)}">'
                            f'{_fmt(r.criterion_validity)}</span>',
                            unsafe_allow_html=True)
            row[2].markdown(f'<span class="{_color(r.unjustified_abstention, False)}">'
                            f'{_fmt(r.unjustified_abstention)}</span>',
                            unsafe_allow_html=True)
            row[3].markdown(f'<span class="{_color(r.discriminant_validity)}">'
                            f'{_fmt(r.discriminant_validity)}</span>',
                            unsafe_allow_html=True)
            sr = r.selection_rate_first
            sr_color = ("metric-good" if sr is not None and abs(sr - 0.5) <= 0.1
                        else "metric-warning" if sr is not None and abs(sr - 0.5) <= 0.25
                        else "metric-bad") if sr is not None else "metric-na"
            row[4].markdown(f'<span class="{sr_color}">{_fmt(sr)}</span>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Criterion Validity: % correct on unequal pairs (excl. abstentions). "
            "Discriminant Validity: % abstained on equal pairs. "
            "Selection Rate: % chose Resume 1 on equal non-abstentions (ideal: 50%)."
        )

        with st.expander("Export results"):
            st.download_button(
                "Download results.json",
                data=json.dumps([asdict(r) for r in results], indent=2).encode(),
                file_name="eval_results.json", mime="application/json")
