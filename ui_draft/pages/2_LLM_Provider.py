#!/usr/bin/env python3
"""
LLM Provider View - Benchmark your LLM's hiring decision quality.
"""
import streamlit as st
import os
import sys
import json
import time
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.styles import inject_styles
from ui.utils import extract_text_from_file, extract_texts_from_files
from ui.stress_test import (
    extract_qualifications, 
    clean_resume_to_markdown,
    generate_underqualified_variant,
    generate_preferred_variant,
    generate_reworded_variant,
    extract_answer,
    Qualification,
    MAX_RETRIES,
    RETRY_DELAY_BASE,
)
from ui.config import SYSTEM_PROMPT

import requests
from pathlib import Path
import random

# Page config
st.set_page_config(
    page_title="LLM Benchmark - Provider View",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()


# ============================================================================
# LOCAL JOB DESCRIPTION LOADING
# ============================================================================

# Job categories organized by type for diverse sampling
JOB_CATEGORIES = {
    "tech": [
        "software_engineer",
        "data_scientist", 
        "ml_engineer",
        "devops_engineer",
    ],
    "business": [
        "product_manager",
        "financial_analyst",
        "sales_representative",
    ],
    "operations": [
        "hr_specialist",
        "customer_support",
        "retail_associate",
    ],
}

ALL_CATEGORIES = [cat for cats in JOB_CATEGORIES.values() for cat in cats]


def get_local_job_descriptions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load job descriptions from local data/harvest_top10 directory.
    
    Uses round-robin selection across categories to ensure diversity
    (both tech and non-tech jobs are represented).
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "harvest_top10"
    
    # Load jobs by category
    jobs_by_category: Dict[str, List[Dict[str, Any]]] = {}
    
    for category in ALL_CATEGORIES:
        jsonl_path = data_dir / category / f"passing_{category}.jsonl"
        if jsonl_path.exists():
            jobs_by_category[category] = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            job = json.loads(line)
                            job["category"] = category
                            # Add human-readable category type
                            if category in JOB_CATEGORIES["tech"]:
                                job["category_type"] = "tech"
                            elif category in JOB_CATEGORIES["business"]:
                                job["category_type"] = "business"
                            else:
                                job["category_type"] = "operations"
                            jobs_by_category[category].append(job)
                        except json.JSONDecodeError:
                            continue
            # Shuffle within each category
            random.shuffle(jobs_by_category[category])
    
    # Round-robin selection across categories for diversity
    selected_jobs = []
    category_indices = {cat: 0 for cat in jobs_by_category}
    categories_with_jobs = [cat for cat in ALL_CATEGORIES if cat in jobs_by_category and jobs_by_category[cat]]
    
    # Shuffle category order for variety
    random.shuffle(categories_with_jobs)
    
    while len(selected_jobs) < limit and categories_with_jobs:
        # Try each category in order
        exhausted_categories = []
        for category in categories_with_jobs:
            if len(selected_jobs) >= limit:
                break
            
            idx = category_indices[category]
            if idx < len(jobs_by_category[category]):
                selected_jobs.append(jobs_by_category[category][idx])
                category_indices[category] += 1
            else:
                exhausted_categories.append(category)
        
        # Remove exhausted categories
        for cat in exhausted_categories:
            categories_with_jobs.remove(cat)
    
    return selected_jobs


@dataclass
class BenchmarkResult:
    """Result of benchmarking an LLM using the validity framework metrics."""
    model_name: str
    
    # Strict pairs (S): pairs where one resume is objectively better (k >= 1)
    # Used for: Criterion Validity, Unjustified Abstention
    n_strict: int = 0
    strict_correct: int = 0      # Chose the correct (better) resume
    strict_abstained: int = 0    # Abstained when shouldn't have
    
    # Equal pairs (E): pairs where both resumes are equally qualified (k == 0)  
    # Used for: Discriminant Validity, Selection Rate
    n_equal: int = 0
    equal_abstained: int = 0     # Correctly abstained
    equal_selected_first: int = 0  # Selected first when forced to choose
    
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def criterion_validity(self) -> Optional[float]:
        """CV = correct decisions / strict pairs (where model didn't abstain)"""
        non_abstained = self.n_strict - self.strict_abstained
        if non_abstained == 0:
            return None
        return self.strict_correct / non_abstained
    
    @property
    def unjustified_abstention(self) -> Optional[float]:
        """UJA = abstentions / strict pairs"""
        if self.n_strict == 0:
            return None
        return self.strict_abstained / self.n_strict
    
    @property
    def discriminant_validity(self) -> Optional[float]:
        """DV = abstentions / equal pairs"""
        if self.n_equal == 0:
            return None
        return self.equal_abstained / self.n_equal
    
    @property
    def selection_rate_first(self) -> Optional[float]:
        """SR = first selected / (equal pairs that didn't abstain)"""
        non_abstained = self.n_equal - self.equal_abstained
        if non_abstained == 0:
            return None
        return self.equal_selected_first / non_abstained


def call_custom_api(
    api_base: str,
    api_key: str,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
) -> str:
    """Call a custom OpenAI-compatible API with retry logic."""
    # Normalize API base URL - strip whitespace and trailing slashes
    api_base = api_base.strip().rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"
    
    url = f"{api_base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                print(f"API error response (attempt {attempt + 1}): {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_error = e
            wait_time = RETRY_DELAY_BASE * (2 ** attempt)
            print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            time.sleep(wait_time)
        except requests.exceptions.HTTPError as e:
            if resp.status_code in [429, 500, 502, 503, 504]:
                last_error = e
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"Server error {resp.status_code} (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            raise
    
    raise last_error or RuntimeError("Max retries exhausted")


def evaluate_pair_custom(
    api_base: str,
    api_key: str,
    model_name: str,
    resume1: str,
    resume2: str,
    job_description: str,
    expected_winner: str,
) -> Dict[str, Any]:
    """Evaluate a resume pair using a custom API."""
    
    user_msg = (
        f"Job Description:\n{job_description}\n\n"
        f"Resume 1\n---------\n{resume1.strip()}\n\n"
        f"Resume 2\n---------\n{resume2.strip()}\n\n"
        "First, briefly justify your choice in 1-2 sentences. Then output your final decision "
        "ONLY inside <answer>...</answer> tags. Respond with exactly ONE of: 'first', 'second', "
        "or 'ABSTAIN'. If the two resumes are equally qualified, output <answer>ABSTAIN</answer>. "
        "Definition: 'first' means choose Resume 1; 'second' means choose Resume 2."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    
    try:
        response = call_custom_api(api_base, api_key, model_name, messages)
        decision = extract_answer(response)
        
        # Determine if correct
        if expected_winner == "either":
            is_correct = True  # Any answer is fine for reworded
        elif expected_winner == "first":
            is_correct = (decision == "first")
        elif expected_winner == "second":
            is_correct = (decision == "second")
        else:
            is_correct = False
        
        return {
            "decision": decision,
            "is_correct": is_correct,
            "expected": expected_winner,
            "raw_response": response,
            "error": None,
        }
    except Exception as e:
        return {
            "decision": "",
            "is_correct": False,
            "expected": expected_winner,
            "raw_response": "",
            "error": str(e),
        }


# ============================================================================
# QUICK BENCHMARK FUNCTIONS
# ============================================================================

def build_base_resume_from_jd(
    api_base: str,
    api_key: str,
    model_name: str,
    role_title: str,
    basic_qualifications: List[str],
) -> str:
    """Build a base resume from job qualifications using the custom LLM."""
    placeholder = "{{CANDIDATE_NAME}}"
    system_prompt = "You write realistic professional resumes in Markdown. Use concise, credible content with clean, readable formatting."
    
    user_prompt = (
        f"Construct a realistic-looking resume in Markdown for the role: {role_title}.\n"
        f"Include ALL of these required qualifications and do not include ANY other qualifications beyond reasonable elaborations.\n"
        + "\n".join([f"- {q}" for q in basic_qualifications]) + "\n\n"
        "Rules:\n"
        f"- Begin with 'Name: {placeholder}'.\n"
        "- Replace company names with {{COMPANY_NAME}} and school names with {{SCHOOL_NAME}}.\n"
        "- Do NOT include any contact information.\n"
        "- Keep within one page, concise. Prefer 1–2 roles in Experience; 2–3 bullets per role.\n"
        "- Use clean Markdown formatting: section headers, bullets.\n"
        "- Sections: Summary, Experience, Education, Skills.\n"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    return call_custom_api(api_base, api_key, model_name, messages, max_tokens=2048)


def run_quick_benchmark(
    api_base: str,
    api_key: str, 
    model_name: str,
    num_jobs: int = 10,
    progress_callback: Optional[callable] = None,
) -> BenchmarkResult:
    """
    Run a quick benchmark using local job descriptions.
    
    For each job:
    1. Build a base resume from the JD's basic qualifications
    2. Generate underqualified/preferred/reworded variants
    3. Evaluate the model's decisions
    """
    result = BenchmarkResult(model_name=model_name)
    
    # Load local job descriptions
    jobs = get_local_job_descriptions(limit=num_jobs)
    if not jobs:
        result.errors.append("No local job descriptions found in data/harvest_top10/")
        return result
    
    # Calculate total steps
    # Per job: 1 (build resume) + len(basic) + len(bonus) + 3 (reworded)
    # Estimate average of 5 basic + 3 bonus per job
    NUM_REWORDED_VARIANTS = 3
    estimated_steps = len(jobs) * (1 + 8 + NUM_REWORDED_VARIANTS)
    current_step = 0
    
    def update_progress(message: str):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            clamped = min(current_step, estimated_steps)
            progress_callback(clamped, estimated_steps, message)
    
    for job_idx, job in enumerate(jobs):
        role_title = job.get("title", job.get("role", "Unknown Role"))
        basic_quals = job.get("basic", [])
        bonus_quals = job.get("bonus", [])
        category = job.get("category", "unknown")
        
        if not basic_quals:
            result.errors.append(f"Job {job_idx + 1}: No basic qualifications found")
            continue
        
        # Build job description text
        job_description = f"Role: {role_title}\n\nRequired Qualifications:\n"
        job_description += "\n".join(f"- {q}" for q in basic_quals)
        if bonus_quals:
            job_description += "\n\nPreferred Qualifications:\n"
            job_description += "\n".join(f"- {q}" for q in bonus_quals)
        
        update_progress(f"Job {job_idx + 1}/{len(jobs)} [{category}]: Building base resume...")
        
        try:
            # Build base resume
            base_resume = build_base_resume_from_jd(
                api_base, api_key, model_name,
                role_title, basic_quals
            )
        except Exception as e:
            result.errors.append(f"Job {job_idx + 1}: Failed to build resume - {e}")
            continue
        
        # Test underqualified variants (sample up to 2 basic quals to keep it fast)
        quals_to_test = basic_quals[:2] if len(basic_quals) > 2 else basic_quals
        for qual in quals_to_test:
            update_progress(f"Job {job_idx + 1}: Testing without '{qual[:30]}...'")
            try:
                variant = generate_underqualified_variant(base_resume, qual)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="first"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_correct = (decision == "first")
                
                result.n_strict += 1
                if is_abstain:
                    result.strict_abstained += 1
                elif is_correct:
                    result.strict_correct += 1
                
                result.detailed_results.append({
                    "job_idx": job_idx,
                    "job_title": role_title,
                    "category": category,
                    "test_type": "strict",
                    "pair_type": "underqualified",
                    "qualification": qual,
                    "decision": decision,
                    "is_correct": is_correct,
                    "abstained": is_abstain,
                })
            except Exception as e:
                result.errors.append(f"Job {job_idx + 1}, underqualified: {e}")
        
        # Test preferred variants (sample up to 2 bonus quals)
        bonus_to_test = bonus_quals[:2] if len(bonus_quals) > 2 else bonus_quals
        for qual in bonus_to_test:
            update_progress(f"Job {job_idx + 1}: Testing with '{qual[:30]}...'")
            try:
                variant = generate_preferred_variant(base_resume, qual)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="second"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_correct = (decision == "second")
                
                result.n_strict += 1
                if is_abstain:
                    result.strict_abstained += 1
                elif is_correct:
                    result.strict_correct += 1
                
                result.detailed_results.append({
                    "job_idx": job_idx,
                    "job_title": role_title,
                    "category": category,
                    "test_type": "strict",
                    "pair_type": "preferred",
                    "qualification": qual,
                    "decision": decision,
                    "is_correct": is_correct,
                    "abstained": is_abstain,
                })
            except Exception as e:
                result.errors.append(f"Job {job_idx + 1}, preferred: {e}")
        
        # Test reworded variants (3 per job)
        for reword_idx in range(NUM_REWORDED_VARIANTS):
            update_progress(f"Job {job_idx + 1}: Testing reworded variant {reword_idx + 1}/{NUM_REWORDED_VARIANTS}...")
            try:
                variant = generate_reworded_variant(base_resume)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="either"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_first = (decision == "first")
                
                result.n_equal += 1
                if is_abstain:
                    result.equal_abstained += 1
                elif is_first:
                    result.equal_selected_first += 1
                
                result.detailed_results.append({
                    "job_idx": job_idx,
                    "job_title": role_title,
                    "category": category,
                    "test_type": "equal",
                    "pair_type": f"reworded_{reword_idx + 1}",
                    "decision": decision,
                    "is_correct": is_abstain,
                    "abstained": is_abstain,
                })
            except Exception as e:
                result.errors.append(f"Job {job_idx + 1}, reworded {reword_idx + 1}: {e}")
    
    return result


def run_benchmark(
    api_base: str,
    api_key: str,
    model_name: str,
    job_description: str,
    resumes: List[str],
    progress_callback: Optional[callable] = None,
) -> BenchmarkResult:
    """
    Run a full benchmark on a custom LLM using the validity framework.
    
    Generates two types of pairs:
    - Strict pairs (S): One resume is objectively better (k >= 1 qualifications differ)
      - Used for: Criterion Validity, Unjustified Abstention
    - Equal pairs (E): Both resumes are equally qualified (k == 0)
      - Used for: Discriminant Validity, Selection Rate
    """
    result = BenchmarkResult(model_name=model_name)
    
    # Extract qualifications from job description
    qualifications = extract_qualifications(job_description)
    basic_quals = qualifications.get("basic", [])
    bonus_quals = qualifications.get("bonus", [])
    
    # Calculate total steps
    # For each resume: clean + (len(basic) underqualified) + (len(bonus) preferred) + 3 reworded
    NUM_REWORDED_VARIANTS = 3
    tests_per_resume = len(basic_quals) + len(bonus_quals) + NUM_REWORDED_VARIANTS
    total_steps = len(resumes) * (1 + tests_per_resume)  # 1 for cleaning
    current_step = 0
    
    def update_progress(message: str):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, message)
    
    for resume_idx, raw_resume in enumerate(resumes):
        # Clean resume
        update_progress(f"Resume {resume_idx + 1}/{len(resumes)}: Cleaning...")
        try:
            base_resume = clean_resume_to_markdown(raw_resume)
        except Exception as e:
            result.errors.append(f"Resume {resume_idx + 1}: Failed to clean - {e}")
            continue
        
        # STRICT PAIRS: Test underqualified variants (remove each basic qualification)
        # Expected: model should pick base_resume (first) since it has the qualification
        for qual in basic_quals:
            update_progress(f"Resume {resume_idx + 1}: Testing without '{qual.text[:30]}...'")
            try:
                variant = generate_underqualified_variant(base_resume, qual.text)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="first"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_correct = (decision == "first")  # Correct if chose the better resume
                
                result.n_strict += 1
                if is_abstain:
                    result.strict_abstained += 1
                elif is_correct:
                    result.strict_correct += 1
                
                result.detailed_results.append({
                    "resume_idx": resume_idx,
                    "test_type": "strict",
                    "pair_type": "underqualified",
                    "qualification": qual.text,
                    "expected": "first",
                    "decision": decision,
                    "is_correct": is_correct,
                    "abstained": is_abstain,
                    **eval_result
                })
                
                if eval_result["error"]:
                    result.errors.append(f"Resume {resume_idx + 1}, underqualified '{qual.text[:30]}': {eval_result['error']}")
            except Exception as e:
                result.errors.append(f"Resume {resume_idx + 1}, underqualified: {e}")
        
        # STRICT PAIRS: Test preferred variants (add each bonus qualification)
        # Expected: model should pick variant (second) since it has the extra qualification
        for qual in bonus_quals:
            update_progress(f"Resume {resume_idx + 1}: Testing with '{qual.text[:30]}...'")
            try:
                variant = generate_preferred_variant(base_resume, qual.text)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="second"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_correct = (decision == "second")  # Correct if chose the better resume
                
                result.n_strict += 1
                if is_abstain:
                    result.strict_abstained += 1
                elif is_correct:
                    result.strict_correct += 1
                
                result.detailed_results.append({
                    "resume_idx": resume_idx,
                    "test_type": "strict",
                    "pair_type": "preferred",
                    "qualification": qual.text,
                    "expected": "second",
                    "decision": decision,
                    "is_correct": is_correct,
                    "abstained": is_abstain,
                    **eval_result
                })
                
                if eval_result["error"]:
                    result.errors.append(f"Resume {resume_idx + 1}, preferred '{qual.text[:30]}': {eval_result['error']}")
            except Exception as e:
                result.errors.append(f"Resume {resume_idx + 1}, preferred: {e}")
        
        # EQUAL PAIRS: Test 3 reworded variants (same qualifications, different phrasing)
        # Expected: model should abstain since both are equally qualified
        NUM_REWORDED_VARIANTS = 3
        for reword_idx in range(NUM_REWORDED_VARIANTS):
            update_progress(f"Resume {resume_idx + 1}: Testing reworded variant {reword_idx + 1}/{NUM_REWORDED_VARIANTS}...")
            try:
                variant = generate_reworded_variant(base_resume)
                eval_result = evaluate_pair_custom(
                    api_base, api_key, model_name,
                    base_resume, variant,
                    job_description,
                    expected_winner="either"
                )
                
                decision = eval_result["decision"].lower() if eval_result["decision"] else ""
                is_abstain = (decision == "abstain")
                is_first = (decision == "first")
                
                result.n_equal += 1
                if is_abstain:
                    result.equal_abstained += 1
                elif is_first:
                    result.equal_selected_first += 1
                # If "second", it's counted in n_equal but not equal_selected_first
                
                result.detailed_results.append({
                    "resume_idx": resume_idx,
                    "test_type": "equal",
                    "pair_type": f"reworded_{reword_idx + 1}",
                    "qualification": f"(phrasing variant {reword_idx + 1})",
                    "expected": "abstain",
                    "decision": decision,
                    "is_correct": is_abstain,  # Correct if abstained
                    "abstained": is_abstain,
                    **eval_result
                })
            except Exception as e:
                result.errors.append(f"Resume {resume_idx + 1}, reworded {reword_idx + 1}: {e}")
    
    return result


def render_header():
    """Render the hero header with explanation."""
    st.markdown("""
    <h1 style="font-size: 2.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem;">
        🔬 LLM Provider Benchmark
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 1.1rem; color: #4a5568; line-height: 1.6; margin-bottom: 1rem;">
        Evaluate your LLM's hiring decision quality using our research framework. 
        We measure <strong>criterion validity</strong> (does it pick the better candidate?) and 
        <strong>discriminant validity</strong> (does it correctly abstain on equal pairs?).
    </p>
    """, unsafe_allow_html=True)
    
    # Reproduce our paper callout
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #f0fdf4 0%, #f0f9ff 100%); border-left: 4px solid #0d9488;">
        <h4 style="color: #0d9488; margin-top: 0;">📄 Reproduce Our Paper Results</h4>
        <p style="color: #4a5568; margin-bottom: 0; line-height: 1.6;">
            The <strong>Quick Benchmark</strong> mode uses the same job descriptions and methodology from our research paper. 
            Connect your model and run the benchmark to see how it compares to GPT-4, Claude, and other frontier models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Privacy notice
    with st.expander("🔒 Privacy & Data Handling", expanded=False):
        st.markdown("""
        **Your data stays secure:**
        
        - ✅ **No storage**: Resumes and job descriptions are NOT stored after your session
        - ✅ **Automated**: All processing is fully automated — no humans review your data
        - ✅ **Your API**: When you connect your own model, data goes directly to your server
        - ✅ **Demo Mode**: Uses our API key for convenience — same privacy guarantees apply
        - 💡 **Local option**: Connect to a locally-hosted LLM (Ollama, vLLM) for maximum privacy
        """)


def render_api_config():
    """Render API configuration section with demo mode option."""
    st.markdown("### 🔌 Connect Your LLM")
    
    # Demo mode toggle
    st.markdown("#### Access Mode")
    col_demo1, col_demo2 = st.columns([1, 2])
    
    with col_demo1:
        use_demo = st.checkbox(
            "🎯 Use Demo Mode",
            value=False,
            help="Demo mode uses our API credits via OpenRouter - no API key needed!"
        )
    
    demo_active = False
    demo_password = ""
    
    if use_demo:
        with col_demo2:
            demo_password = st.text_input(
                "Demo Password",
                type="password",
                help="Enter the demo password to activate",
                placeholder="Enter password...",
                key="demo_password_provider"
            )
            
            from ui.config import check_bypass_password, get_bypass_api_config
            
            if demo_password:
                if check_bypass_password(demo_password):
                    st.success("✅ Demo mode activated! Using OpenRouter with preset models.")
                    demo_active = True
                else:
                    st.error("❌ Invalid password")
    
    if demo_active:
        # In demo mode, use our OpenRouter config
        config = get_bypass_api_config()
        api_base = config["api_base"]
        api_key = config["api_key"]
        
        st.markdown("""
        <div class="card" style="background: #f0fdf4; border: 1px solid #86efac;">
            <p style="color: #166534; margin: 0;">
                <strong>Demo Mode Active</strong> — Using OpenRouter API. 
                Select a model below to benchmark.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection for demo mode
        demo_models = [
            ("GPT-4o Mini", "openai/gpt-4o-mini"),
            ("Claude 3.5 Haiku", "anthropic/claude-3-5-haiku-20241022"),
            ("Gemini 2.0 Flash", "google/gemini-2.0-flash-001"),
            ("Llama 3.1 8B", "meta-llama/llama-3.1-8b-instruct"),
            ("DeepSeek V3", "deepseek/deepseek-chat-v3-0324"),
        ]
        
        model_choice = st.selectbox(
            "Select Model to Benchmark",
            options=[name for name, _ in demo_models],
            index=0
        )
        
        model_name = next((id for name, id in demo_models if name == model_choice), demo_models[0][1])
        
    else:
        # Custom API mode
        st.markdown("""
        <p style="color: #4a5568; font-size: 0.95rem;">
            Connect your own model via OpenAI-compatible API:
        </p>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 Supported Endpoints", expanded=False):
            st.markdown("""
            | Provider | Base URL | Needs API Key |
            |----------|----------|---------------|
            | **vLLM** | `http://localhost:8000` | No |
            | **TGI** | `http://localhost:8080` | No |
            | **Ollama** | `http://localhost:11434` | No |
            | **OpenRouter** | `https://openrouter.ai/api` | Yes |
            | **Together AI** | `https://api.together.xyz` | Yes |
            | **Anyscale** | `https://api.endpoints.anyscale.com` | Yes |
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_base = st.text_input(
                "API Base URL",
                value="http://localhost:8000",
                help="The base URL of your OpenAI-compatible API server",
                placeholder="http://localhost:8000"
            )
            
            model_name = st.text_input(
                "Model Name",
                value="",
                help="The model name/ID to use (as expected by your server)",
                placeholder="e.g., meta-llama/Llama-3.1-8B-Instruct"
            )
        
        with col2:
            api_key = st.text_input(
                "API Key (optional)",
                value="",
                type="password",
                help="API key if required by your server",
                placeholder="sk-..."
            )
            
            # Test connection button
            if st.button("🔗 Test Connection"):
                if not api_base or not model_name:
                    st.error("Please provide API base URL and model name.")
                else:
                    try:
                        with st.spinner("Testing connection..."):
                            response = call_custom_api(
                                api_base, api_key, model_name,
                                [{"role": "user", "content": "Say 'Connection successful!' in exactly those words."}],
                                max_tokens=50
                            )
                        st.success(f"✅ Connection successful! Response: {response[:100]}...")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {e}")
    
    return api_base, api_key, model_name


def render_data_input():
    """Render data input section."""
    st.markdown("### 📁 Upload Benchmark Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Job Description")
        jd_file = st.file_uploader(
            "Upload job description (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="provider_jd_upload",
        )
        
        jd_text = ""
        if jd_file:
            jd_text = extract_text_from_file(jd_file)
            if jd_text:
                st.success(f"✓ Loaded {len(jd_text)} characters")
        
        job_desc = st.text_area(
            "Or paste job description",
            value=jd_text,
            height=200,
            placeholder="Paste job description text here...",
        )
    
    with col2:
        st.markdown("#### Resumes (Multiple)")
        resume_files = st.file_uploader(
            "Upload resumes (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="provider_resume_upload",
            accept_multiple_files=True,
            help="Upload multiple resume files for comprehensive benchmarking"
        )
        
        resumes = []
        if resume_files:
            resumes = extract_texts_from_files(resume_files)
            st.success(f"✓ Loaded {len(resumes)} resumes")
        
        # Option to paste a single resume
        paste_resume = st.text_area(
            "Or paste a single resume",
            height=200,
            placeholder="Paste resume text here...",
        )
        
        if paste_resume.strip() and paste_resume.strip() not in resumes:
            resumes.append(paste_resume.strip())
    
    return job_desc, resumes


def format_metric(value: Optional[float], denominator_text: str = "") -> str:
    """Format a metric value, handling None (N/A) cases."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def get_metric_color(value: Optional[float], higher_is_better: bool = True) -> str:
    """Get CSS color class based on metric value."""
    if value is None:
        return "metric-na"  # Special style for N/A
    if higher_is_better:
        if value >= 0.8:
            return "metric-good"
        elif value >= 0.5:
            return "metric-warning"
        else:
            return "metric-bad"
    else:  # Lower is better (e.g., unjustified abstention)
        if value <= 0.2:
            return "metric-good"
        elif value <= 0.5:
            return "metric-warning"
        else:
            return "metric-bad"


def render_benchmark_results(result: BenchmarkResult):
    """Render benchmark results with validity framework metrics."""
    st.markdown("---")
    st.markdown("## 📊 Benchmark Results")
    
    # Get metrics from the BenchmarkResult properties
    cv = result.criterion_validity
    uja = result.unjustified_abstention
    dv = result.discriminant_validity
    sr = result.selection_rate_first
    
    # Display model name
    st.markdown(f"### Model: `{result.model_name}`")
    
    # Primary metrics row
    st.markdown("#### Validity Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cv_color = get_metric_color(cv, higher_is_better=True)
        cv_display = format_metric(cv)
        non_abstained = result.n_strict - result.strict_abstained
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {cv_color}">{cv_display}</div>
            <div class="metric-label">Criterion Validity</div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {result.strict_correct}/{non_abstained} correct (when not abstained)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        uja_color = get_metric_color(uja, higher_is_better=False)
        uja_display = format_metric(uja)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {uja_color}">{uja_display}</div>
            <div class="metric-label">Unjustified Abstention</div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {result.strict_abstained}/{result.n_strict} abstained (on strict pairs)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        dv_color = get_metric_color(dv, higher_is_better=True)
        dv_display = format_metric(dv)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {dv_color}">{dv_display}</div>
            <div class="metric-label">Discriminant Validity</div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {result.equal_abstained}/{result.n_equal} abstained (on equal pairs)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Selection rate: 0.5 (50%) is ideal - no bias
        sr_ideal_diff = abs(sr - 0.5) if sr is not None else None
        sr_color = "metric-good" if sr_ideal_diff is not None and sr_ideal_diff <= 0.1 else (
            "metric-warning" if sr_ideal_diff is not None and sr_ideal_diff <= 0.25 else "metric-bad"
        ) if sr is not None else "metric-na"
        sr_display = format_metric(sr)
        non_abstained_equal = result.n_equal - result.equal_abstained
        # Combine the detail lines to match other cards' height
        sr_detail = f"{result.equal_selected_first}/{non_abstained_equal} first (ideal: 50%)"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {sr_color}">{sr_display}</div>
            <div class="metric-label">Selection Rate (First)</div>
            <div style="color: #64748b; font-size: 0.8rem;">
                {sr_detail}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary statistics row
    st.markdown("#### Test Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(145deg, #1e3a5f 0%, #1e40af 100%);">
            <div class="metric-label" style="color: #93c5fd;">Strict Pairs (S)</div>
            <div style="color: #e2e8f0; font-size: 1.5rem; font-weight: 600;">{result.n_strict}</div>
            <div style="color: #93c5fd; font-size: 0.85rem;">
                Pairs where one candidate is objectively better (k ≥ 1)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(145deg, #065f46 0%, #047857 100%);">
            <div class="metric-label" style="color: #6ee7b7;">Equal Pairs (E)</div>
            <div style="color: #e2e8f0; font-size: 1.5rem; font-weight: 600;">{result.n_equal}</div>
            <div style="color: #6ee7b7; font-size: 0.85rem;">
                Pairs where both candidates are equally qualified (k = 0)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Interpretation
    st.markdown("### 📝 Interpretation")
    
    interpretations = []
    
    # Criterion Validity interpretation
    if cv is not None:
        if cv >= 0.8:
            interpretations.append({
                "type": "success",
                "icon": "✅",
                "message": f"**Criterion Validity ({cv:.0%})**: Excellent. When the model makes a decision, it correctly identifies the better candidate."
            })
        elif cv >= 0.5:
            interpretations.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"**Criterion Validity ({cv:.0%})**: Moderate. The model sometimes picks the wrong candidate when making decisions."
            })
        else:
            interpretations.append({
                "type": "warning",
                "icon": "❌",
                "message": f"**Criterion Validity ({cv:.0%})**: Poor. The model frequently selects the less qualified candidate."
            })
    else:
        interpretations.append({
            "type": "info",
            "icon": "ℹ️",
            "message": "**Criterion Validity**: N/A - No decisions made on strict pairs (all abstained)."
        })
    
    # Unjustified Abstention interpretation
    if uja is not None:
        if uja <= 0.2:
            interpretations.append({
                "type": "success",
                "icon": "✅",
                "message": f"**Unjustified Abstention ({uja:.0%})**: Low. The model rarely abstains when there's a clear better candidate."
            })
        elif uja <= 0.5:
            interpretations.append({
                "type": "info",
                "icon": "📊",
                "message": f"**Unjustified Abstention ({uja:.0%})**: Moderate. The model sometimes abstains when it should pick a candidate."
            })
        else:
            interpretations.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"**Unjustified Abstention ({uja:.0%})**: High. The model frequently abstains even when one candidate is clearly better."
            })
    else:
        interpretations.append({
            "type": "info",
            "icon": "ℹ️",
            "message": "**Unjustified Abstention**: N/A - No strict pairs tested."
        })
    
    # Discriminant Validity interpretation
    if dv is not None:
        if dv >= 0.8:
            interpretations.append({
                "type": "success",
                "icon": "✅",
                "message": f"**Discriminant Validity ({dv:.0%})**: Excellent. The model correctly abstains when candidates are equally qualified."
            })
        elif dv >= 0.5:
            interpretations.append({
                "type": "info",
                "icon": "📊",
                "message": f"**Discriminant Validity ({dv:.0%})**: Moderate. The model sometimes makes arbitrary choices between equal candidates."
            })
        else:
            interpretations.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"**Discriminant Validity ({dv:.0%})**: Low. The model rarely abstains even when candidates are identical in qualification."
            })
    else:
        interpretations.append({
            "type": "info",
            "icon": "ℹ️",
            "message": "**Discriminant Validity**: N/A - No equal pairs tested."
        })
    
    # Selection Rate interpretation
    if sr is not None:
        bias_direction = "first" if sr > 0.5 else "second"
        bias_magnitude = abs(sr - 0.5)
        if bias_magnitude <= 0.1:
            interpretations.append({
                "type": "success",
                "icon": "✅",
                "message": f"**Selection Rate ({sr:.0%} first)**: Balanced. No significant ordering bias detected."
            })
        elif bias_magnitude <= 0.25:
            interpretations.append({
                "type": "info",
                "icon": "📊",
                "message": f"**Selection Rate ({sr:.0%} first)**: Slight bias towards {bias_direction} position. "
                          f"Consider if order affects decisions."
            })
        else:
            interpretations.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"**Selection Rate ({sr:.0%} first)**: Strong bias towards {bias_direction} position. "
                          f"The model may be influenced by resume ordering."
            })
    
    for interp in interpretations:
        css_class = f"insight-{interp['type']}"
        st.markdown(f"""
        <div class="card {css_class}">
            <span style="font-size: 1.5rem;">{interp['icon']}</span>
            <span style="margin-left: 0.5rem; font-size: 1rem;">{interp['message']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Errors if any
    if result.errors:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander(f"⚠️ {len(result.errors)} Errors/Warnings"):
            for err in result.errors:
                st.warning(err)
    
    # Detailed results
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 Detailed Test Results"):
        # Group by test type
        strict_results = [r for r in result.detailed_results if r.get("test_type") == "strict"]
        equal_results = [r for r in result.detailed_results if r.get("test_type") == "equal"]
        
        if strict_results:
            st.markdown("**Strict Pairs (k ≥ 1):**")
            for r in strict_results:
                abstained = r.get("abstained", False)
                if abstained:
                    icon = "⏸️"  # Abstained (unjustified)
                    status = "abstained"
                elif r["is_correct"]:
                    icon = "✅"
                    status = "correct"
                else:
                    icon = "❌"
                    status = "incorrect"
                
                pair_type = r.get("pair_type", "unknown")
                qual_snippet = r.get("qualification", "")[:60] if r.get("qualification") else ""
                
                # Handle both resume_idx (custom benchmark) and job_idx (quick benchmark)
                if "resume_idx" in r:
                    item_label = f"Resume {r['resume_idx'] + 1}"
                elif "job_idx" in r:
                    job_title = r.get("job_title", "Unknown")[:30]
                    item_label = f"Job {r['job_idx'] + 1} ({job_title})"
                else:
                    item_label = "Test"
                
                st.markdown(f"- {icon} {item_label} | {pair_type} | `{r['decision']}` ({status})")
                if qual_snippet:
                    st.caption(f"    Qualification: {qual_snippet}...")
        
        if equal_results:
            st.markdown("**Equal Pairs (k = 0):**")
            for r in equal_results:
                abstained = r.get("abstained", False)
                if abstained:
                    icon = "✅"  # Abstained (correct for equal pairs)
                    status = "correctly abstained"
                else:
                    icon = "⚠️"
                    status = f"chose {r['decision']} (should abstain)"
                
                # Handle both resume_idx and job_idx
                if "resume_idx" in r:
                    item_label = f"Resume {r['resume_idx'] + 1}"
                elif "job_idx" in r:
                    job_title = r.get("job_title", "Unknown")[:30]
                    item_label = f"Job {r['job_idx'] + 1} ({job_title})"
                else:
                    item_label = "Test"
                
                pair_type = r.get("pair_type", "reworded")
                st.markdown(f"- {icon} {item_label} | {pair_type} | {status}")
    
    # Export results
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📥 Export Results as JSON"):
        result_dict = asdict(result)
        st.download_button(
            "Download JSON",
            data=json.dumps(result_dict, indent=2),
            file_name=f"benchmark_{result.model_name.replace('/', '_')}.json",
            mime="application/json"
        )


def main():
    render_header()
    
    # API configuration
    api_base, api_key, model_name = render_api_config()
    
    st.markdown("---")
    
    # Benchmark mode selection
    st.markdown("### 📊 Choose Benchmark Mode")
    
    tab1, tab2 = st.tabs(["⚡ Quick Benchmark (Local JDs)", "📁 Custom Benchmark (Upload Data)"])
    
    with tab1:
        st.markdown("""
        **Quick Benchmark** uses job descriptions from our curated dataset spanning **10 diverse job categories**:
        
        🖥️ **Tech**: Software Engineer, Data Scientist, ML Engineer, DevOps Engineer  
        💼 **Business**: Product Manager, Financial Analyst, Sales Representative  
        🏢 **Operations**: HR Specialist, Customer Support, Retail Associate
        
        Jobs are selected using **round-robin sampling** across categories to ensure diversity 
        (both tech and non-tech roles are tested).
        
        For each job, we:
        1. Generate a base resume matching the required qualifications
        2. Create underqualified variants (remove qualifications)
        3. Create preferred variants (add bonus qualifications)  
        4. Create 3 reworded variants (same qualifications, different phrasing)
        5. Evaluate your model's decisions on all pairs
        """)
        
        num_jobs = st.slider(
            "Number of job descriptions to test",
            min_value=1,
            max_value=50,
            value=10,
            help="More jobs = more comprehensive benchmark, but takes longer"
        )
        
        st.info(f"Estimated tests: ~{num_jobs * 7} pairs (2 underqualified + 2 preferred + 3 reworded per job)")
        
        quick_run_button = st.button("⚡ Run Quick Benchmark", use_container_width=True, key="quick_run")
        
        if quick_run_button:
            if not api_base:
                st.error("Please provide an API base URL.")
            elif not model_name:
                st.error("Please provide a model name.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(step: int, total: int, message: str):
                    progress_bar.progress(min(step / total, 1.0))
                    status_text.text(message)
                
                try:
                    with st.spinner("Running quick benchmark..."):
                        result = run_quick_benchmark(
                            api_base=api_base,
                            api_key=api_key,
                            model_name=model_name,
                            num_jobs=num_jobs,
                            progress_callback=progress_callback,
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Quick benchmark complete!")
                    st.session_state["provider_result"] = result
                    
                except Exception as e:
                    st.error(f"Error running quick benchmark: {e}")
                    st.code(traceback.format_exc())
    
    with tab2:
        # Custom data input
        job_desc, resumes = render_data_input()
        
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Configuration:** {len(resumes)} resume(s) against `{model_name or '(not set)'}`")
        
        with col2:
            run_button = st.button("🔬 Run Custom Benchmark", use_container_width=True, key="custom_run")
        
        if run_button:
            if not api_base:
                st.error("Please provide an API base URL.")
            elif not model_name:
                st.error("Please provide a model name.")
            elif not job_desc.strip():
                st.error("Please provide a job description.")
            elif not resumes:
                st.error("Please upload at least one resume.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(step: int, total: int, message: str):
                    progress_bar.progress(min(step / total, 1.0))
                    status_text.text(message)
                
                try:
                    with st.spinner("Running benchmark..."):
                        result = run_benchmark(
                            api_base=api_base,
                            api_key=api_key,
                            model_name=model_name,
                            job_description=job_desc,
                            resumes=resumes,
                            progress_callback=progress_callback,
                        )
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Benchmark complete!")
                    st.session_state["provider_result"] = result
                    
                except Exception as e:
                    st.error(f"Error running benchmark: {e}")
                    st.code(traceback.format_exc())
    
    # Show results if available
    if "provider_result" in st.session_state:
        render_benchmark_results(st.session_state["provider_result"])


if __name__ == "__main__":
    main()

