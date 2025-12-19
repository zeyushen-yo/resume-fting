"""
Core logic for resume stress testing.
Generates variants and evaluates against multiple AI models.
"""
from __future__ import annotations
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from .config import SYSTEM_PROMPT, get_openrouter_key, get_google_key


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


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Call Gemini API directly."""
    api_key = get_google_key()
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY")
    
    model = "models/gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent"
    
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
        ],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}
    }
    
    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        traceback.print_exc()
        raise


def call_openrouter(model_id: str, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
    """Call OpenRouter API."""
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
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
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
    
    total_steps = 6 + len(models) * 3  # cleanup + quals + 3 variants + 3 evals per model
    current_step = 0
    
    def update_progress(message: str):
        nonlocal current_step
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, message)
    
    # Step 1: Clean up resume
    update_progress("Cleaning and formatting your resume...")
    base_resume = clean_resume_to_markdown(resume_text)
    result.variants["original"] = base_resume
    
    # Step 2: Extract qualifications from job description
    update_progress("Extracting qualifications from job description...")
    qualifications = extract_qualifications(job_description)
    result.qualifications = qualifications
    
    # Step 3: Generate underqualified variant (remove first basic qualification)
    if qualifications["basic"]:
        removed_qual = qualifications["basic"][0]
        update_progress(f"Generating underqualified variant (removing: {removed_qual.text[:50]}...)")
        result.variants["underqualified"] = generate_underqualified_variant(base_resume, removed_qual.text)
        result.variants["underqualified_removed"] = removed_qual.text
    else:
        update_progress("Skipping underqualified variant (no basic qualifications found)")
        result.variants["underqualified"] = base_resume
        result.variants["underqualified_removed"] = ""
    
    # Step 4: Generate preferred variant (add first bonus qualification)
    if qualifications["bonus"]:
        added_qual = qualifications["bonus"][0]
        update_progress(f"Generating preferred variant (adding: {added_qual.text[:50]}...)")
        result.variants["preferred"] = generate_preferred_variant(base_resume, added_qual.text)
        result.variants["preferred_added"] = added_qual.text
    else:
        update_progress("Skipping preferred variant (no bonus qualifications found)")
        result.variants["preferred"] = base_resume
        result.variants["preferred_added"] = ""
    
    # Step 5: Generate reworded variant
    update_progress("Generating reworded variant...")
    result.variants["reworded"] = generate_reworded_variant(base_resume)
    
    # Step 6: Evaluate with each model
    for model in models:
        model_name = model["name"]
        model_id = model["id"]
        
        # Test 1: Original vs Underqualified (should pick original = "first")
        update_progress(f"{model_name}: Testing original vs underqualified...")
        eval_under = evaluate_pair(
            model_id,
            base_resume,
            result.variants["underqualified"],
            job_description,
            expected_winner="first"
        )
        eval_under["test_type"] = "underqualified"
        eval_under["model_name"] = model_name
        result.model_results.append(eval_under)
        
        # Test 2: Original vs Preferred (should pick preferred = "second")
        update_progress(f"{model_name}: Testing original vs preferred...")
        eval_pref = evaluate_pair(
            model_id,
            base_resume,
            result.variants["preferred"],
            job_description,
            expected_winner="second"
        )
        eval_pref["test_type"] = "preferred"
        eval_pref["model_name"] = model_name
        result.model_results.append(eval_pref)
        
        # Test 3: Original vs Reworded (should be roughly equal)
        update_progress(f"{model_name}: Testing original vs reworded...")
        eval_reword = evaluate_pair(
            model_id,
            base_resume,
            result.variants["reworded"],
            job_description,
            expected_winner="either"
        )
        eval_reword["test_type"] = "reworded"
        eval_reword["model_name"] = model_name
        result.model_results.append(eval_reword)
    
    # Generate insights
    result.qualification_insights = generate_insights(result)
    
    return result


def generate_insights(result: StressTestResult) -> List[Dict[str, Any]]:
    """Generate human-readable insights from the stress test results."""
    insights = []
    
    # Count correct decisions per test type
    test_types = ["underqualified", "preferred", "reworded"]
    for test_type in test_types:
        relevant = [r for r in result.model_results if r["test_type"] == test_type]
        if not relevant:
            continue
        
        correct = sum(1 for r in relevant if r["is_correct"])
        total = len(relevant)
        
        if test_type == "underqualified":
            removed = result.variants.get("underqualified_removed", "a basic qualification")
            if correct == total:
                insights.append({
                    "type": "success",
                    "icon": "✅",
                    "message": f"All {total} models correctly identified that removing '{removed[:60]}...' would hurt your application."
                })
            elif correct == 0:
                insights.append({
                    "type": "warning",
                    "icon": "⚠️",
                    "message": f"No models noticed when we removed '{removed[:60]}...' — it may not be as important as expected."
                })
            else:
                insights.append({
                    "type": "info",
                    "icon": "📊",
                    "message": f"{correct}/{total} models noticed when we removed '{removed[:60]}...'."
                })
        
        elif test_type == "preferred":
            added = result.variants.get("preferred_added", "a bonus qualification")
            if correct == total:
                insights.append({
                    "type": "success",
                    "icon": "✅",
                    "message": f"All {total} models recognized that adding '{added[:60]}...' would strengthen your application."
                })
            elif correct == 0:
                insights.append({
                    "type": "warning",
                    "icon": "⚠️",
                    "message": f"No models valued adding '{added[:60]}...' — it might not matter as much as you think."
                })
            else:
                insights.append({
                    "type": "info",
                    "icon": "📊",
                    "message": f"{correct}/{total} models valued adding '{added[:60]}...'."
                })
        
        elif test_type == "reworded":
            # For reworded, check if models gave different answers
            decisions = [r["decision"] for r in relevant]
            unique = set(decisions)
            if len(unique) == 1 and "ABSTAIN" in unique:
                insights.append({
                    "type": "success",
                    "icon": "✅",
                    "message": "All models correctly recognized that a reworded resume is equally qualified."
                })
            elif len(unique) > 1:
                insights.append({
                    "type": "warning",
                    "icon": "⚠️",
                    "message": f"Models disagreed on the reworded version: {dict((d, decisions.count(d)) for d in unique)}. Phrasing matters!"
                })
    
    return insights

