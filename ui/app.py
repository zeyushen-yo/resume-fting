#!/usr/bin/env python3
"""
Resume Stress Test UI
Test your resume against AI screening systems.
"""
import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.config import EVALUATION_MODELS, QUICK_MODELS, get_openrouter_key, get_google_key
from ui.stress_test import run_stress_test, StressTestResult

# Page config
st.set_page_config(
    page_title="Resume Stress Test",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Import Sora font */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Sora', sans-serif;
    }
    
    /* Hero header */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.15);
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Insight cards */
    .insight-success {
        background: linear-gradient(145deg, #064e3b 0%, #065f46 100%);
        border-left: 4px solid #10b981;
    }
    
    .insight-warning {
        background: linear-gradient(145deg, #78350f 0%, #92400e 100%);
        border-left: 4px solid #f59e0b;
    }
    
    .insight-info {
        background: linear-gradient(145deg, #1e3a5f 0%, #1e40af 100%);
        border-left: 4px solid #3b82f6;
    }
    
    /* Results grid */
    .model-result {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .result-correct {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .result-incorrect {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .result-neutral {
        background: rgba(156, 163, 175, 0.2);
        color: #9ca3af;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }
    
    /* Stats */
    .stat-box {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f1f5f9;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Textarea styling */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        background-color: #1e1e2e !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Sora', sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the hero header."""
    st.markdown('<h1 class="hero-title">🎯 Resume Stress Test</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Discover how AI hiring systems evaluate your resume — '
        'and what actually matters to them.</p>',
        unsafe_allow_html=True
    )


def check_api_keys():
    """Check if required API keys are available."""
    google_key = get_google_key()
    openrouter_key = get_openrouter_key()
    
    missing = []
    if not google_key:
        missing.append("GOOGLE_API_KEY (for resume processing)")
    if not openrouter_key:
        missing.append("OPENROUTER_API_KEY (for model evaluation)")
    
    if missing:
        st.error("⚠️ Missing API keys:")
        for key in missing:
            st.write(f"  • {key}")
        st.info("Set these environment variables before running the app.")
        return False
    return True


def render_input_section():
    """Render the resume and job description input section."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Your Resume")
        resume = st.text_area(
            "Paste your resume here",
            height=400,
            placeholder="Paste your resume text here...\n\nYou can paste raw text, markdown, or even HTML — we'll clean it up.",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### 💼 Target Job Description")
        job_desc = st.text_area(
            "Paste the job description here",
            height=400,
            placeholder="Paste the full job description here...\n\nWe'll extract the required and preferred qualifications automatically.",
            label_visibility="collapsed"
        )
    
    return resume, job_desc


def render_model_selection():
    """Render model selection options."""
    st.markdown("### ⚙️ Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_set = st.radio(
            "Choose evaluation speed",
            ["Quick (3 models, ~1 min)", "Full (5 models, ~2 min)"],
            horizontal=True,
            help="Quick mode uses fewer models for faster results. Full mode provides more comprehensive coverage."
        )
    
    is_quick = "Quick" in model_set
    selected_models = QUICK_MODELS if is_quick else EVALUATION_MODELS
    
    with col2:
        st.write(f"**Models:** {', '.join(m['name'] for m in selected_models)}")
    
    return selected_models


def render_results(result: StressTestResult):
    """Render the stress test results."""
    st.markdown("---")
    st.markdown("## 📊 Results")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate stats
    underqualified_results = [r for r in result.model_results if r["test_type"] == "underqualified"]
    preferred_results = [r for r in result.model_results if r["test_type"] == "preferred"]
    reworded_results = [r for r in result.model_results if r["test_type"] == "reworded"]
    
    underqualified_correct = sum(1 for r in underqualified_results if r["is_correct"])
    preferred_correct = sum(1 for r in preferred_results if r["is_correct"])
    total_models = len(underqualified_results)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total_models}</div>
            <div class="stat-label">Models Tested</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pct = int(underqualified_correct / max(total_models, 1) * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{pct}%</div>
            <div class="stat-label">Notice Missing Skills</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pct = int(preferred_correct / max(total_models, 1) * 100)
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{pct}%</div>
            <div class="stat-label">Value Added Skills</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        abstain_count = sum(1 for r in reworded_results if r["decision"] == "ABSTAIN")
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{abstain_count}/{total_models}</div>
            <div class="stat-label">Ignore Phrasing</div>
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
            <span style="margin-left: 0.5rem; font-size: 1rem;">{insight['message']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed results per test
    st.markdown("### 📋 Detailed Results")
    
    tab1, tab2, tab3 = st.tabs(["🔻 Underqualified Test", "🔺 Preferred Test", "🔄 Reworded Test"])
    
    with tab1:
        removed = result.variants.get("underqualified_removed", "N/A")
        st.markdown(f"**Test:** What happens if we remove: *{removed}*")
        st.markdown("**Expected:** AI should prefer your original resume (first)")
        
        for r in underqualified_results:
            icon = "✅" if r["is_correct"] else "❌"
            decision = r["decision"] or "no answer"
            st.markdown(f"- {icon} **{r['model_name']}**: chose `{decision}`")
            
            with st.expander(f"See {r['model_name']}'s reasoning"):
                st.text(r.get("raw_response", "No response")[:1000])
    
    with tab2:
        added = result.variants.get("preferred_added", "N/A")
        st.markdown(f"**Test:** What happens if we add: *{added}*")
        st.markdown("**Expected:** AI should prefer the enhanced resume (second)")
        
        for r in preferred_results:
            icon = "✅" if r["is_correct"] else "❌"
            decision = r["decision"] or "no answer"
            st.markdown(f"- {icon} **{r['model_name']}**: chose `{decision}`")
            
            with st.expander(f"See {r['model_name']}'s reasoning"):
                st.text(r.get("raw_response", "No response")[:1000])
    
    with tab3:
        st.markdown("**Test:** Same qualifications, just reworded")
        st.markdown("**Expected:** AI should abstain or be split (no clear winner)")
        
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
    st.markdown("### 📝 Qualifications We Found")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Required (Basic):**")
        for q in result.qualifications.get("basic", []):
            st.markdown(f"- {q.text}")
    
    with col2:
        st.markdown("**Preferred (Bonus):**")
        for q in result.qualifications.get("bonus", []):
            st.markdown(f"- {q.text}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show generated variants
    with st.expander("🔍 View Generated Resume Variants"):
        var_tab1, var_tab2, var_tab3, var_tab4 = st.tabs(
            ["Original (Cleaned)", "Underqualified", "Preferred", "Reworded"]
        )
        
        with var_tab1:
            st.markdown(result.variants.get("original", ""))
        
        with var_tab2:
            st.markdown(result.variants.get("underqualified", ""))
        
        with var_tab3:
            st.markdown(result.variants.get("preferred", ""))
        
        with var_tab4:
            st.markdown(result.variants.get("reworded", ""))


def main():
    render_header()
    
    if not check_api_keys():
        return
    
    # Input section
    resume, job_desc = render_input_section()
    
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
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(step: int, total: int, message: str):
            progress_bar.progress(step / total)
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
            st.session_state["result"] = result
            
        except Exception as e:
            st.error(f"Error running stress test: {e}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Show results if available
    if "result" in st.session_state:
        render_results(st.session_state["result"])


if __name__ == "__main__":
    main()

