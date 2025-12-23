"""
Core logic for resume stress testing.
Generates variants and evaluates against multiple AI models.
"""
from __future__ import annotations
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import SYSTEM_PROMPT, get_openrouter_key, get_google_key


# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds, will be multiplied by 2^attempt


@dataclass
class Qualification:
    text: str
    kind: str  # "basic" or "bonus"


@dataclass
class StressTestResult:
    """Result of stress testing a single resume against AI models."""
    original_resume: str
    job_description: str
    qualifications: Dict[str, List[Qualification]]
    variants: Dict[str, str]  # variant_type -> variant_resume
    model_results: List[Dict[str, Any]] = field(default_factory=list)
    qualification_insights: List[Dict[str, Any]] = field(default_factory=list)


def _make_api_request(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    """Make an API request with retry logic."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(f"API error response (attempt {attempt + 1}): {resp.text}")
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.ChunkedEncodingError, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_error = e
            wait_time = RETRY_DELAY_BASE * (2 ** attempt)
            print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
        except requests.exceptions.HTTPError as e:
            # Don't retry on HTTP errors like 400, 401, 403, etc.
            if resp.status_code in [429, 500, 502, 503, 504]:
                last_error = e
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"Server error {resp.status_code} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            raise
    
    # All retries exhausted
    raise last_error or RuntimeError("Max retries exhausted")


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Call Gemini via OpenRouter with retry logic."""
    api_key = get_openrouter_key()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    
    model_id = "google/gemini-2.0-flash-001"
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        data = _make_api_request(url, headers, payload)
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        traceback.print_exc()
        raise


def call_openrouter(model_id: str, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
    """Call OpenRouter API with retry logic."""
    api_key = get_openrouter_key()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    
    try:
        data = _make_api_request(url, headers, payload)
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenRouter API error for {model_id}: {e}")
        traceback.print_exc()
        raise


def extract_qualifications(job_description: str) -> Dict[str, List[Qualification]]:
    """Extract basic and bonus qualifications from a job description using Gemini."""
    system = (
        "Extract qualifications from a job description.\n"
        "Classify each as 'basic' (required) or 'bonus' (preferred).\n"
        "Return JSON with 'basic' and 'bonus' lists. Each item: {\"text\": <requirement>, \"kind\": \"basic\"|\"bonus\"}.\n"
        "Return ONLY valid JSON. No prose, no markdown fences."
    )
    user = f"Job Description:\n{job_description}\n\nReturn only JSON with keys 'basic' and 'bonus'."
    
    import json
    text = call_gemini(system, user, temperature=0.1)
    
    # Clean up response
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline+1:]
        text = text.rstrip("`\n ")
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse qualifications JSON: {e}")
        print(f"Raw response: {text[:500]}")
        return {"basic": [], "bonus": []}
    
    basics = [Qualification(text=i.get('text', '').strip(), kind='basic') 
              for i in data.get('basic', []) if i.get('text')]
    bonuses = [Qualification(text=i.get('text', '').strip(), kind='bonus') 
               for i in data.get('bonus', []) if i.get('text')]
    
    return {"basic": basics, "bonus": bonuses}


def extract_resume_qualifications(resume_text: str) -> List[Qualification]:
    """Extract qualifications/skills from a resume using Gemini."""
    system = (
        "Extract key qualifications, skills, and experiences from a resume.\n"
        "Focus on concrete, testable qualifications (skills, technologies, years of experience, degrees, certifications).\n"
        "Return JSON with a 'qualifications' list. Each item: {\"text\": <qualification>}.\n"
        "Return ONLY valid JSON. No prose, no markdown fences."
    )
    user = f"Resume:\n{resume_text}\n\nReturn only JSON with key 'qualifications'."
    
    import json
    text = call_gemini(system, user, temperature=0.1)
    
    # Clean up response
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline+1:]
        text = text.rstrip("`\n ")
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse resume qualifications JSON: {e}")
        print(f"Raw response: {text[:500]}")
        return []
    
    quals = [Qualification(text=i.get('text', '').strip(), kind='resume') 
             for i in data.get('qualifications', []) if i.get('text')]
    
    return quals


def extract_jd_qualifications(job_description: str) -> List[Qualification]:
    """Extract qualifications from a job description that the candidate might want to add."""
    system = (
        "Extract required and preferred qualifications from a job description.\n"
        "Focus on concrete, testable qualifications (skills, technologies, years of experience, degrees, certifications).\n"
        "Return JSON with a 'qualifications' list. Each item: {\"text\": <qualification>}.\n"
        "Return ONLY valid JSON. No prose, no markdown fences."
    )
    user = f"Job Description:\n{job_description}\n\nReturn only JSON with key 'qualifications'."
    
    import json
    text = call_gemini(system, user, temperature=0.1)
    
    # Clean up response
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline+1:]
        text = text.rstrip("`\n ")
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JD qualifications JSON: {e}")
        print(f"Raw response: {text[:500]}")
        return []
    
    quals = [Qualification(text=i.get('text', '').strip(), kind='jd') 
             for i in data.get('qualifications', []) if i.get('text')]
    
    return quals


def clean_resume_to_markdown(raw_resume: str) -> str:
    """Clean up a raw resume into standardized markdown format."""
    system = """
You clean and format raw resume text into standardized Markdown.
Preserve all original content. Do NOT add or remove qualifications.
""".strip()
    
    user = (
        "Convert this resume into clean, professional Markdown:\n"
        "- Start with 'Name: {{CANDIDATE_NAME}}'\n"
        "- Replace company names with {{COMPANY_NAME}}\n"
        "- Replace school names with {{SCHOOL_NAME}}\n"
        "- Use clear section headers (Summary, Experience, Education, Skills)\n"
        "- Remove contact info (email, phone, LinkedIn)\n"
        "- Keep ALL original qualifications and experience\n\n"
        f"Resume:\n{raw_resume}"
    )
    
    return call_gemini(system, user, temperature=0.2)


def generate_underqualified_variant(base_resume: str, qualification_to_remove: str) -> str:
    """Generate a resume variant with one qualification removed."""
    system = "You write realistic professional resumes in Markdown. Use concise, credible content."
    
    user = (
        f"Given this resume, create a version that REMOVES this qualification: '{qualification_to_remove}'.\n"
        "Do not add new qualifications. Keep the same structure and style.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and {{COMPANY_NAME}}/{{SCHOOL_NAME}} placeholders.\n"
        "No contact info.\n\n"
        f"Resume:\n{base_resume}"
    )
    
    return call_gemini(system, user, temperature=0.3)


def generate_preferred_variant(base_resume: str, qualification_to_add: str) -> str:
    """Generate a resume variant with one bonus qualification added."""
    system = "You write realistic professional resumes in Markdown. Use concise, credible content."
    
    user = (
        f"Given this resume, create a version that ADDS this preferred qualification: '{qualification_to_add}'.\n"
        "Do not add any other new qualifications.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and {{COMPANY_NAME}}/{{SCHOOL_NAME}} placeholders.\n"
        "No contact info.\n\n"
        f"Resume:\n{base_resume}"
    )
    
    return call_gemini(system, user, temperature=0.35)


def generate_reworded_variant(base_resume: str) -> str:
    """Generate a resume variant with same qualifications but different phrasing."""
    system = "You write realistic professional resumes in Markdown. Use concise, credible content."
    
    user = (
        "Reword and restructure this resume while keeping ALL qualifications equivalent.\n"
        "Do not add or remove qualifications beyond rephrasing.\n"
        "Keep 'Name: {{CANDIDATE_NAME}}' and {{COMPANY_NAME}}/{{SCHOOL_NAME}} placeholders.\n"
        "No contact info.\n\n"
        f"Resume:\n{base_resume}"
    )
    
    return call_gemini(system, user, temperature=0.5)


def extract_answer(text: str) -> str:
    """Extract the model's decision from the response."""
    if not isinstance(text, str):
        return ""
    
    # Prefer the last <answer>...</answer> tag
    matches = list(re.finditer(r"<\s*answer\s*>(.*?)</\s*answer\s*>", text, flags=re.I | re.S))
    if matches:
        val = matches[-1].group(1).strip().strip('"').strip("'")
        lv = val.lower()
        if lv in {"first", "second", "abstain"}:
            return "ABSTAIN" if lv == "abstain" else lv
        return val
    
    # Fallback
    lower = text.lower()
    if "first" in lower:
        return "first"
    if "second" in lower:
        return "second"
    if "abstain" in lower:
        return "ABSTAIN"
    return ""


def evaluate_pair(
    model_id: str,
    resume1: str,
    resume2: str,
    job_description: str,
    expected_winner: str,  # "first", "second", or "either" (for reworded)
) -> Dict[str, Any]:
    """Evaluate a resume pair using a single model."""
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
        response = call_openrouter(model_id, messages)
        decision = extract_answer(response)
        
        # Determine if correct
        if expected_winner == "either":
            # For reworded, any decision is fine (abstain preferred but not required)
            is_correct = True
        elif expected_winner == "first":
            is_correct = (decision == "first")
        elif expected_winner == "second":
            is_correct = (decision == "second")
        else:
            is_correct = False
        
        return {
            "model_id": model_id,
            "decision": decision,
            "is_correct": is_correct,
            "expected": expected_winner,
            "raw_response": response,
            "error": None,
        }
    except Exception as e:
        return {
            "model_id": model_id,
            "decision": "",
            "is_correct": False,
            "expected": expected_winner,
            "raw_response": "",
            "error": str(e),
        }


def run_stress_test(
    resume_text: str,
    job_description: str,
    models: List[Dict[str, str]],
    progress_callback: Optional[callable] = None,
) -> StressTestResult:
    """
    Run a full stress test on a resume.
    
    New logic:
    - For each qualification IN THE RESUME: remove it and test if models notice
    - For each qualification IN THE JD: add it to resume and test if models value it
    - One reworded test for phrasing sensitivity
    
    Args:
        resume_text: The user's resume (raw or markdown)
        job_description: The target job description
        models: List of models to evaluate against
        progress_callback: Optional callback for progress updates (step, total, message)
    
    Returns:
        StressTestResult with all variants and evaluation results
    """
    result = StressTestResult(
        original_resume=resume_text,
        job_description=job_description,
        qualifications={},
        variants={},
    )
    
    # Step 1: Clean up resume first (needed for qualification extraction)
    current_step = 0
    
    def update_progress(message: str):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            # We'll update total_steps once we know the counts
            total = getattr(update_progress, 'total', 100)
            # Clamp step to not exceed total to avoid Streamlit progress bar errors
            clamped_step = min(current_step, total)
            progress_callback(clamped_step, total, message)
    
    update_progress("Cleaning and formatting your resume...")
    base_resume = clean_resume_to_markdown(resume_text)
    result.variants["original"] = base_resume
    
    # Step 2: Extract qualifications from RESUME (things to potentially remove)
    update_progress("Extracting qualifications from your resume...")
    resume_quals = extract_resume_qualifications(base_resume)
    
    # Step 3: Extract qualifications from JD (things to potentially add)
    update_progress("Extracting qualifications from job description...")
    jd_quals = extract_jd_qualifications(job_description)
    
    # Store qualifications in result
    result.qualifications = {
        "resume": resume_quals,  # From resume - will be removed to test
        "jd": jd_quals,          # From JD - will be added to test
    }
    
    num_resume_quals = len(resume_quals)
    num_jd_quals = len(jd_quals)
    num_models = len(models)
    
    # Calculate total steps now that we know the counts
    # Steps: 
    #   - 3 initial (cleanup + 2 extractions)
    #   - num_resume_quals (removed variants)
    #   - num_jd_quals (added variants)
    #   - 3 (reworded variants)
    #   - (num_resume_quals + num_jd_quals + 3) * num_models (evaluations)
    NUM_REWORDED_VARIANTS = 3
    total_variants = num_resume_quals + num_jd_quals + NUM_REWORDED_VARIANTS
    update_progress.total = 3 + total_variants + (total_variants * num_models)
    
    # Step 4: Generate "removed" variants - remove each qualification FROM RESUME
    removed_variants = []  # List of (qual_text, variant_resume)
    for i, qual in enumerate(resume_quals):
        update_progress(f"Testing removal {i+1}/{num_resume_quals}: '{qual.text[:40]}...'")
        variant = generate_underqualified_variant(base_resume, qual.text)
        removed_variants.append((qual.text, variant))
    result.variants["removed_list"] = removed_variants
    
    # Step 5: Generate "added" variants - add each qualification FROM JD
    added_variants = []  # List of (qual_text, variant_resume)
    for i, qual in enumerate(jd_quals):
        update_progress(f"Testing addition {i+1}/{num_jd_quals}: '{qual.text[:40]}...'")
        variant = generate_preferred_variant(base_resume, qual.text)
        added_variants.append((qual.text, variant))
    result.variants["added_list"] = added_variants
    
    # Step 6: Generate 3 reworded variants (for equal pairs / discriminant validity)
    NUM_REWORDED_VARIANTS = 3
    reworded_variants = []
    for i in range(NUM_REWORDED_VARIANTS):
        update_progress(f"Generating reworded variant {i+1}/{NUM_REWORDED_VARIANTS}...")
        reworded_variants.append(generate_reworded_variant(base_resume))
    result.variants["reworded_list"] = reworded_variants
    
    # Step 7: Evaluate ALL variants with each model
    for model in models:
        model_name = model["name"]
        model_id = model["id"]
        
        # Test all "removed" variants - original should win (first)
        # Because we're comparing: original (with qual) vs variant (without qual)
        for qual_text, variant in removed_variants:
            update_progress(f"{model_name}: Testing without '{qual_text[:30]}...'")
            eval_result = evaluate_pair(
                model_id,
                base_resume,  # Original with the qualification
                variant,      # Variant without the qualification
                job_description,
                expected_winner="first"  # Original should be preferred
            )
            eval_result["test_type"] = "removed"
            eval_result["qualification"] = qual_text
            eval_result["model_name"] = model_name
            result.model_results.append(eval_result)
        
        # Test all "added" variants - enhanced should win (second)
        # Because we're comparing: original (without qual) vs variant (with qual)
        for qual_text, variant in added_variants:
            update_progress(f"{model_name}: Testing with '{qual_text[:30]}...'")
            eval_result = evaluate_pair(
                model_id,
                base_resume,  # Original without the JD qualification
                variant,      # Variant with the JD qualification added
                job_description,
                expected_winner="second"  # Enhanced should be preferred
            )
            eval_result["test_type"] = "added"
            eval_result["qualification"] = qual_text
            eval_result["model_name"] = model_name
            result.model_results.append(eval_result)
        
        # Test all reworded variants (should be roughly equal - for discriminant validity)
        for i, reworded_variant in enumerate(reworded_variants):
            update_progress(f"{model_name}: Testing reworded version {i+1}/{NUM_REWORDED_VARIANTS}...")
            eval_reword = evaluate_pair(
                model_id,
                base_resume,
                reworded_variant,
                job_description,
                expected_winner="either"
            )
            eval_reword["test_type"] = "reworded"
            eval_reword["qualification"] = f"(phrasing variant {i+1})"
            eval_reword["model_name"] = model_name
            result.model_results.append(eval_reword)
    
    # Generate insights
    result.qualification_insights = generate_insights(result)
    
    return result


def generate_insights(result: StressTestResult) -> List[Dict[str, Any]]:
    """Generate human-readable insights from the stress test results."""
    insights = []
    
    # Use new test types: "removed" (from resume) and "added" (from JD)
    removed_results = [r for r in result.model_results if r["test_type"] == "removed"]
    added_results = [r for r in result.model_results if r["test_type"] == "added"]
    reworded_results = [r for r in result.model_results if r["test_type"] == "reworded"]
    
    # Analyze "removed" tests - which of YOUR qualifications matter?
    if removed_results:
        qual_stats = {}
        for r in removed_results:
            qual = r.get("qualification", "unknown")
            if qual not in qual_stats:
                qual_stats[qual] = {"correct": 0, "total": 0}
            qual_stats[qual]["total"] += 1
            if r["is_correct"]:
                qual_stats[qual]["correct"] += 1
        
        # Find qualifications that ALL models noticed vs NONE noticed
        all_noticed = [q for q, s in qual_stats.items() if s["correct"] == s["total"]]
        none_noticed = [q for q, s in qual_stats.items() if s["correct"] == 0]
        
        if all_noticed:
            insights.append({
                "type": "success",
                "icon": "✅",
                "message": f"Essential resume skills: {len(all_noticed)} of your qualifications were noticed by ALL models when removed. These are your strongest assets!"
            })
        
        if none_noticed:
            insights.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"Undervalued skills: {len(none_noticed)} of your qualifications weren't noticed when removed. Consider emphasizing them more or they may not matter for this role."
            })
        
        overall_correct = sum(1 for r in removed_results if r["is_correct"])
        overall_total = len(removed_results)
        insights.append({
            "type": "info",
            "icon": "📊",
            "message": f"Resume qualification impact: {overall_correct}/{overall_total} ({int(overall_correct/max(overall_total,1)*100)}%) of your qualifications are valued by AI systems."
        })
    
    # Analyze "added" tests - which JD requirements would help if added?
    if added_results:
        qual_stats = {}
        for r in added_results:
            qual = r.get("qualification", "unknown")
            if qual not in qual_stats:
                qual_stats[qual] = {"correct": 0, "total": 0}
            qual_stats[qual]["total"] += 1
            if r["is_correct"]:
                qual_stats[qual]["correct"] += 1
        
        # Find high-value vs low-value JD qualifications
        all_valued = [q for q, s in qual_stats.items() if s["correct"] == s["total"]]
        none_valued = [q for q, s in qual_stats.items() if s["correct"] == 0]
        
        if all_valued:
            insights.append({
                "type": "success",
                "icon": "🌟",
                "message": f"High-impact skills to add: {len(all_valued)} JD requirements would be valued by ALL models if added to your resume!"
            })
        
        if none_valued:
            insights.append({
                "type": "info",
                "icon": "💡",
                "message": f"Lower priority: {len(none_valued)} 'preferred' qualifications weren't valued by any models. May not be worth the effort."
            })
    
    # Analyze reworded tests
    if reworded_results:
        decisions = [r["decision"] for r in reworded_results]
        abstain_count = sum(1 for d in decisions if d == "ABSTAIN")
        total = len(decisions)
        
        if abstain_count == total:
            insights.append({
                "type": "success",
                "icon": "⚖️",
                "message": "Phrasing-neutral: All models correctly ignored cosmetic differences. Your resume's substance matters more than exact wording."
            })
        elif abstain_count == 0:
            insights.append({
                "type": "warning",
                "icon": "🎲",
                "message": "Phrasing-sensitive: All models were influenced by wording changes. Consider A/B testing different phrasings."
            })
        else:
            insights.append({
                "type": "info",
                "icon": "📝",
                "message": f"Mixed phrasing sensitivity: {abstain_count}/{total} models ignored phrasing, {total-abstain_count} were influenced by it."
            })
    
    return insights

