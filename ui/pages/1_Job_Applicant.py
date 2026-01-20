#!/usr/bin/env python3
"""
Job Applicant View - Test your resume against AI hiring systems.
"""
import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui.styles import inject_styles
from ui.utils import extract_text_from_file
from ui.config import (
    EVALUATION_MODELS, QUICK_MODELS, get_openrouter_key,
    check_bypass_password, get_bypass_api_config
)
from ui.stress_test import run_stress_test, StressTestResult

# Page config
st.set_page_config(
    page_title="Resume Stress Test - Job Applicant",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()


def render_header():
    """Render the hero header with explanation."""
    st.markdown("""
    <h1 style="font-size: 2.5rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem;">
        👤 Resume Stress Test
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style="font-size: 1.1rem; color: #4a5568; line-height: 1.6; margin-bottom: 1rem;">
        Understand how AI hiring systems evaluate your resume. We test what happens when 
        qualifications are added or removed to reveal which skills AI actually values.
    </p>
    """, unsafe_allow_html=True)
    
    # Privacy notice
    with st.expander("🔒 Privacy & Data Handling", expanded=False):
        st.markdown("""
        **Your privacy matters to us:**
        
        - ✅ **No storage**: Your resume is processed only to generate results and is NOT stored after your session
        - ✅ **Automated**: All processing is fully automated — no humans review your data
        - ✅ **Third-party APIs**: When testing against AI models, your resume is sent to those providers 
          (OpenAI, Anthropic, Google, etc.). Their data handling follows their policies.
        - 💡 **Recommendation**: Consider anonymizing sensitive info (exact addresses, phone numbers) before testing
        
        For maximum privacy, you can [benchmark with a locally-hosted LLM](/LLM_Provider).
        """)


def render_demo_mode():
    """Render demo mode toggle for users without API key."""
    st.markdown("### 🔑 Access Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_demo = st.checkbox(
            "Use Demo Mode (no API key needed)",
            value=False,
            help="Demo mode uses our API key for quick testing. For heavy usage, please provide your own key."
        )
    
    demo_password = ""
    if use_demo:
        with col2:
            demo_password = st.text_input(
                "Demo Password",
                type="password",
                help="Enter the demo password to use our API credits",
                placeholder="Enter password..."
            )
            
            if demo_password:
                if check_bypass_password(demo_password):
                    st.success("✅ Demo mode activated!")
                else:
                    st.error("❌ Invalid password")
                    demo_password = ""
    
    return use_demo, demo_password


def check_api_keys(use_demo: bool, demo_password: str):
    """Check if required API keys are available."""
    if use_demo and check_bypass_password(demo_password):
        return True
    
    openrouter_key = get_openrouter_key()
    
    if not openrouter_key:
        st.error("⚠️ Missing API access. Either enable Demo Mode or set OPENROUTER_API_KEY.")
        return False
    return True


def render_input_section():
    """Render the resume and job description input section."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Resume")
        
        # File upload option
        resume_file = st.file_uploader(
            "Upload resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_upload",
            help="Upload your resume file, or paste text below"
        )
        
        # Extract text from file if uploaded
        file_resume_text = ""
        if resume_file:
            file_resume_text = extract_text_from_file(resume_file)
            if file_resume_text:
                st.success(f"✓ Loaded {len(file_resume_text)} characters from {resume_file.name}")
        
        resume = st.text_area(
            "Paste your resume here",
            value=file_resume_text,
            height=350,
            placeholder="Paste your resume text here...\n\nYou can paste raw text, markdown, or even HTML — we'll clean it up.",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 💼 Target Job Description")
        
        # File upload option
        jd_file = st.file_uploader(
            "Upload job description (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_upload",
            help="Upload job description file, or paste text below"
        )
        
        # Extract text from file if uploaded
        file_jd_text = ""
        if jd_file:
            file_jd_text = extract_text_from_file(jd_file)
            if file_jd_text:
                st.success(f"✓ Loaded {len(file_jd_text)} characters from {jd_file.name}")
        
        job_desc = st.text_area(
            "Paste the job description here",
            value=file_jd_text,
            height=350,
            placeholder="Paste the full job description here...\n\nWe'll extract the required and preferred qualifications automatically.",
            label_visibility="collapsed"
        )
    
    return resume, job_desc


def render_model_selection():
    """Render model selection options."""
    st.markdown("### ⚙️ Model Selection")
    
    st.markdown("""
    <p style="color: #4a5568; font-size: 0.95rem;">
        Choose which AI models to test your resume against. Different models may value 
        different qualifications.
    </p>
    """, unsafe_allow_html=True)
    
    # Preset options
    preset = st.radio(
        "Choose evaluation preset",
        ["Quick (3 models)", "Full (5 models)", "Custom"],
        horizontal=True,
        help="Quick mode uses fewer models for faster results. Full mode provides more comprehensive coverage."
    )
    
    if preset == "Quick (3 models)":
        selected_models = QUICK_MODELS
        st.write(f"**Models:** {', '.join(m['name'] for m in selected_models)}")
    elif preset == "Full (5 models)":
        selected_models = EVALUATION_MODELS
        st.write(f"**Models:** {', '.join(m['name'] for m in selected_models)}")
    else:
        # Custom selection
        all_models = EVALUATION_MODELS + [
            {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku", "provider": "Anthropic"},
            {"id": "openai/gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
            {"id": "mistralai/mistral-large", "name": "Mistral Large", "provider": "Mistral"},
        ]
        
        # Remove duplicates by id
        seen_ids = set()
        unique_models = []
        for m in all_models:
            if m["id"] not in seen_ids:
                seen_ids.add(m["id"])
                unique_models.append(m)
        
        selected_names = st.multiselect(
            "Select models to evaluate",
            options=[m["name"] for m in unique_models],
            default=[unique_models[0]["name"]] if unique_models else []
        )
        
        selected_models = [m for m in unique_models if m["name"] in selected_names]
        
        if not selected_models:
            st.warning("Please select at least one model.")
    
    return selected_models


def render_results(result: StressTestResult):
    """Render the stress test results."""
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Calculate stats using new test types
    removed_results = [r for r in result.model_results if r["test_type"] == "removed"]
    added_results = [r for r in result.model_results if r["test_type"] == "added"]
    reworded_results = [r for r in result.model_results if r["test_type"] == "reworded"]
    
    num_resume_quals = len(result.qualifications.get("resume", []))
    num_jd_quals = len(result.qualifications.get("jd", []))
    num_models = len(set(r.get("model_name", "") for r in result.model_results))
    
    removed_correct = sum(1 for r in removed_results if r["is_correct"])
    added_correct = sum(1 for r in added_results if r["is_correct"])
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{num_resume_quals + num_jd_quals}</div>
            <div class="stat-label">Qualifications Tested</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_removed = len(removed_results)
        pct = int(removed_correct / max(total_removed, 1) * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{pct}%</div>
            <div class="stat-label">Your Skills Noticed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_added = len(added_results)
        pct = int(added_correct / max(total_added, 1) * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{pct}%</div>
            <div class="stat-label">JD Skills Help</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        abstain_count = sum(1 for r in reworded_results if r["decision"] == "ABSTAIN")
        total_reworded = len(reworded_results)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{abstain_count}/{total_reworded}</div>
            <div class="stat-label">Phrasing Neutral</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Insights
    st.markdown("### 💡 Key Insights")
    
    for insight in result.qualification_insights:
        css_class = f"insight-{insight['type']}"
        st.markdown(f"""
        <div class="card {css_class}">
            <span style="font-size: 1.5rem;">{insight['icon']}</span>
            <span style="margin-left: 0.5rem; font-size: 1rem; color: #1a1a2e;">{insight['message']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed results per test
    st.markdown("### 📋 Detailed Results by Qualification")
    
    tab1, tab2, tab3 = st.tabs(["📄 Your Resume Skills", "📋 JD Requirements", "🔄 Phrasing Test"])
    
    with tab1:
        st.markdown("""
        **Test:** What happens when each of YOUR qualifications is removed?  
        **Expected:** AI should prefer your original resume (with the qualification)
        """)
        st.markdown("---")
        
        if not removed_results:
            st.info("No resume qualifications were tested.")
        else:
            # Group results by qualification
            qual_groups = {}
            for r in removed_results:
                qual = r.get("qualification", "Unknown")
                if qual not in qual_groups:
                    qual_groups[qual] = []
                qual_groups[qual].append(r)
            
            for qual, results in qual_groups.items():
                correct_count = sum(1 for r in results if r["is_correct"])
                total = len(results)
                
                if correct_count == total:
                    emoji = "✅"
                    status = "All models noticed"
                elif correct_count == 0:
                    emoji = "⚠️"
                    status = "No models noticed"
                else:
                    emoji = "📊"
                    status = f"{correct_count}/{total} noticed"
                
                with st.expander(f"{emoji} **{qual[:60]}{'...' if len(qual) > 60 else ''}** — {status}"):
                    for r in results:
                        icon = "✅" if r["is_correct"] else "❌"
                        decision = r["decision"] or "no answer"
                        st.markdown(f"- {icon} **{r['model_name']}**: chose `{decision}` (expected: `first`)")
                        
                        # Show reasoning snippet
                        raw_response = r.get("raw_response", "")
                        if raw_response:
                            # Extract first few sentences of reasoning
                            snippet = raw_response[:400].replace('\n', ' ')
                            st.caption(f"Reasoning: {snippet}...")
    
    with tab2:
        st.markdown("""
        **Test:** What happens when each JD requirement is added to your resume?  
        **Expected:** AI should prefer the enhanced resume (with the added qualification)
        """)
        st.markdown("---")
        
        if not added_results:
            st.info("No JD qualifications were tested.")
        else:
            # Group results by qualification
            qual_groups = {}
            for r in added_results:
                qual = r.get("qualification", "Unknown")
                if qual not in qual_groups:
                    qual_groups[qual] = []
                qual_groups[qual].append(r)
            
            for qual, results in qual_groups.items():
                correct_count = sum(1 for r in results if r["is_correct"])
                total = len(results)
                
                if correct_count == total:
                    emoji = "✅"
                    status = "All models valued it"
                elif correct_count == 0:
                    emoji = "⚠️"
                    status = "No models valued it"
                else:
                    emoji = "📊"
                    status = f"{correct_count}/{total} valued it"
                
                with st.expander(f"{emoji} **{qual[:60]}{'...' if len(qual) > 60 else ''}** — {status}"):
                    for r in results:
                        icon = "✅" if r["is_correct"] else "❌"
                        decision = r["decision"] or "no answer"
                        st.markdown(f"- {icon} **{r['model_name']}**: chose `{decision}` (expected: `second`)")
                        
                        # Show reasoning snippet
                        raw_response = r.get("raw_response", "")
                        if raw_response:
                            snippet = raw_response[:400].replace('\n', ' ')
                            st.caption(f"Reasoning: {snippet}...")
    
    with tab3:
        st.markdown("""
        **Test:** Same qualifications, just reworded  
        **Expected:** AI should abstain or recognize equivalence
        """)
        st.markdown("---")
        
        for r in reworded_results:
            decision = r["decision"] or "no answer"
            if decision == "ABSTAIN":
                st.markdown(f"- ⚖️ **{r['model_name']}**: correctly abstained")
            else:
                st.markdown(f"- 🎲 **{r['model_name']}**: chose `{decision}` (phrasing influenced decision)")
            
            with st.expander(f"See {r['model_name']}'s reasoning"):
                st.text(r.get("raw_response", "No response")[:1000])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show qualifications extracted
    st.markdown("### 📝 Qualifications Extracted")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**From Your Resume ({num_resume_quals}):**")
        for q in result.qualifications.get("resume", []):
            st.markdown(f"- {q.text}")
    
    with col2:
        st.markdown(f"**From Job Description ({num_jd_quals}):**")
        for q in result.qualifications.get("jd", []):
            st.markdown(f"- {q.text}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show generated variants
    with st.expander("🔍 View Generated Resume Variants"):
        st.markdown("#### Original (Cleaned)")
        st.markdown(result.variants.get("original", ""))
        
        # Show reworded variants
        reworded_list = result.variants.get("reworded_list", [])
        if reworded_list:
            st.markdown("---")
            st.markdown("#### Reworded Variants")
            for i, variant in enumerate(reworded_list):
                with st.expander(f"Reworded Version {i+1}"):
                    st.markdown(variant)
        
        # Show removed variants
        removed_list = result.variants.get("removed_list", [])
        if removed_list:
            st.markdown("---")
            st.markdown("#### Variants with Your Skills Removed")
            for qual, variant in removed_list:
                with st.expander(f"Without: {qual[:50]}..."):
                    st.markdown(variant)
        
        # Show added variants
        added_list = result.variants.get("added_list", [])
        if added_list:
            st.markdown("---")
            st.markdown("#### Variants with JD Skills Added")
            for qual, variant in added_list:
                with st.expander(f"With: {qual[:50]}..."):
                    st.markdown(variant)


def main():
    render_header()
    
    # Demo mode toggle
    use_demo, demo_password = render_demo_mode()
    
    if not check_api_keys(use_demo, demo_password):
        st.stop()
    
    st.markdown("---")
    
    # Input section
    resume, job_desc = render_input_section()
    
    st.markdown("---")
    
    # Model selection
    selected_models = render_model_selection()
    
    # Run button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        run_button = st.button("🚀 Run Stress Test", use_container_width=True)
    
    # Run stress test
    if run_button:
        if not resume.strip():
            st.error("Please paste your resume first.")
            return
        if not job_desc.strip():
            st.error("Please paste the job description first.")
            return
        if not selected_models:
            st.error("Please select at least one model.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(step: int, total: int, message: str):
            progress_bar.progress(min(step / total, 1.0))
            status_text.text(message)
        
        try:
            with st.spinner("Running stress test..."):
                result = run_stress_test(
                    resume_text=resume,
                    job_description=job_desc,
                    models=selected_models,
                    progress_callback=progress_callback,
                )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Complete!")
            
            # Store result in session state
            st.session_state["applicant_result"] = result
            
        except Exception as e:
            st.error(f"Error running stress test: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Show results if available
    if "applicant_result" in st.session_state:
        render_results(st.session_state["applicant_result"])


if __name__ == "__main__":
    main()
