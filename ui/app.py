#!/usr/bin/env python3
"""
Resume Validity Benchmark
Main landing page for the multi-page Streamlit app.
"""
import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.styles import inject_styles

# Page config
st.set_page_config(
    page_title="Resume Screening Validity Benchmark",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# Sidebar with quick links and info
with st.sidebar:
    st.markdown("### 📋 Resume Validity")
    st.markdown("---")
    
    st.markdown("""
    **Quick Links**
    - [📄 Test Your Resume](#)
    - [🔬 Benchmark Your LLM](#)
    - [📊 Paper Results](#results)
    - [❓ FAQ](#faq)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    **Resources**
    - [📖 Full Paper](https://arxiv.org/abs/)
    - [💻 GitHub Repo](https://github.com/)
    - [📧 Contact Us](mailto:)
    """)
    
    st.markdown("---")
    st.caption("© 2025 PEAS Lab")

# Main content
st.markdown("""
<h1 style="font-size: 2.75rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem;">
    📋 Resume Screening Validity Benchmark
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 1.15rem; color: #4a5568; line-height: 1.6; margin-bottom: 2rem;">
    An open-source tool for evaluating whether AI hiring systems make <strong>valid</strong> decisions — 
    do they actually prefer more qualified candidates, or are they influenced by irrelevant factors?
</p>
""", unsafe_allow_html=True)

# Brief motivation section
st.markdown("""
<div class="card" style="background: linear-gradient(135deg, #f0fdf4 0%, #f0f9ff 100%); border-left: 4px solid #0d9488;">
    <h3 style="color: #0d9488; margin-top: 0;">🎯 Why This Matters</h3>
    <p style="color: #4a5568; margin-bottom: 0; line-height: 1.7;">
        AI-powered resume screening is increasingly used in hiring, affecting millions of job seekers. 
        But how do we know these systems are actually identifying the best candidates? 
        <strong>Validity</strong> — the degree to which a hiring tool measures what it's supposed to measure — 
        is a foundational concept in employment testing law and industrial psychology. 
        Our tool helps you test whether AI screeners meet this standard.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Two main feature cards
st.markdown("## Choose Your Path")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">👤</div>
        <div class="feature-title">For Job Applicants</div>
        <div class="feature-text">
            <p><strong>Understand how AI sees your resume.</strong></p>
            <p>Test your resume against multiple AI hiring systems to discover:</p>
            <ul style="color: #4a5568; margin-left: 1rem;">
                <li>Which of your qualifications AI systems actually notice</li>
                <li>Which job requirements would help if added</li>
                <li>How sensitive the AI is to wording changes</li>
            </ul>
            <p style="margin-top: 1rem; font-size: 0.9rem; color: #64748b;">
                <em>No AI experience required. Just paste your resume and job description.</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/1_Job_Applicant.py", label="🚀 Test My Resume", use_container_width=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-title">For LLM Providers & Researchers</div>
        <div class="feature-text">
            <p><strong>Benchmark your model's hiring decision quality.</strong></p>
            <p>Evaluate your LLM against our validity framework:</p>
            <ul style="color: #4a5568; margin-left: 1rem;">
                <li><strong>Criterion Validity:</strong> Does it pick the better candidate?</li>
                <li><strong>Discriminant Validity:</strong> Does it abstain on equal pairs?</li>
                <li><strong>Selection Rate:</strong> Is there ordering bias?</li>
            </ul>
            <p style="margin-top: 1rem; font-size: 0.9rem; color: #64748b;">
                <em>Connect via OpenAI-compatible API. Reproduce our paper's results in minutes.</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/2_LLM_Provider.py", label="📊 Benchmark My LLM", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# How it works section
st.markdown("## How It Works")

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1rem;">
    <div class="card">
        <h4 style="color: #0d9488;">1️⃣ Extract Qualifications</h4>
        <p style="color: #4a5568;">
            We analyze job descriptions to identify <strong>required</strong> qualifications 
            (must-haves) and <strong>preferred</strong> qualifications (nice-to-haves).
        </p>
    </div>
    <div class="card">
        <h4 style="color: #0d9488;">2️⃣ Generate Test Pairs</h4>
        <p style="color: #4a5568;">
            We create resume variants that differ by exactly one qualification, 
            creating pairs where one candidate is objectively better.
        </p>
    </div>
    <div class="card">
        <h4 style="color: #0d9488;">3️⃣ Evaluate & Measure</h4>
        <p style="color: #4a5568;">
            We test if AI systems correctly identify the better candidate and 
            calculate validity metrics based on their decisions.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Key findings teaser
st.markdown('<a name="results"></a>', unsafe_allow_html=True)
st.markdown("## 📊 Key Findings from Our Research")

st.markdown("""
<div class="paper-citation">
    <div class="paper-title">Validity of AI Resume Screening: How Well Do LLMs Identify Qualified Candidates?</div>
    <div class="paper-authors">Research paper (2025) • Available on arXiv</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <h4 style="color: #1a1a2e;">🔍 What We Found</h4>
        <ul style="color: #4a5568; line-height: 1.8;">
            <li><strong>High Criterion Validity:</strong> Most LLMs correctly prefer candidates 
                with more qualifications (70-90% accuracy on strict pairs)</li>
            <li><strong>Low Discriminant Validity:</strong> LLMs rarely abstain when candidates 
                are equally qualified, making arbitrary choices instead</li>
            <li><strong>Position Bias:</strong> Some models consistently prefer the first 
                resume shown, regardless of content</li>
            <li><strong>Phrasing Sensitivity:</strong> Rewording without changing qualifications 
                can flip model decisions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h4 style="color: #1a1a2e;">💡 Why It Matters for Policy</h4>
        <ul style="color: #4a5568; line-height: 1.8;">
            <li><strong>Fairness:</strong> If AI makes arbitrary choices between equal candidates, 
                protected groups may be unfairly disadvantaged</li>
            <li><strong>Validity Standards:</strong> Employment tests must demonstrate 
                validity under EEOC guidelines — do AI screeners meet this bar?</li>
            <li><strong>Transparency:</strong> Job seekers deserve to know how AI 
                evaluates their applications</li>
            <li><strong>Regulation:</strong> Policymakers need tools to audit AI 
                hiring systems before widespread deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Privacy notice
st.markdown('<a name="privacy"></a>', unsafe_allow_html=True)
st.markdown("## 🔒 Privacy & Data Handling")

with st.container():
    st.markdown("""
    #### Your Data, Your Control
    
    We understand that resumes contain sensitive personal information. Here's how we handle your data:
    
    - **🚫 No Storage:** Resumes are processed only to generate results and are **not stored, saved, or retained** after your session ends.
    - **🤖 Automated Processing:** All analysis is fully automated — **no humans review** your resume or any LLM outputs.
    - **🔗 Third-Party APIs:** When you connect to external LLM providers (OpenAI, Anthropic, OpenRouter, etc.), your data is sent to those services. Their data handling is governed by **their privacy policies**, not ours.
    - **🏠 Local Option:** For maximum privacy, connect to a **locally-hosted LLM** (Ollama, vLLM) — your data never leaves your machine.
    - **✏️ Anonymization Recommended:** We suggest removing or anonymizing particularly sensitive information (exact addresses, phone numbers) before testing, especially when using third-party APIs.
    
    *This tool is provided for research and educational purposes. We make no guarantees about AI evaluation accuracy.*
    """)

st.markdown("<br>", unsafe_allow_html=True)

# FAQ section
st.markdown('<a name="faq"></a>', unsafe_allow_html=True)
st.markdown("## ❓ Frequently Asked Questions")

with st.expander("What is 'validity' in the context of hiring?"):
    st.markdown("""
    **Validity** refers to whether a hiring tool actually measures what it claims to measure — 
    in this case, whether AI resume screeners actually identify the most qualified candidates.
    
    There are several types of validity:
    - **Criterion Validity**: Does the tool's score predict job performance?
    - **Content Validity**: Does the tool measure job-relevant attributes?
    - **Construct Validity**: Does the tool measure the intended psychological construct?
    
    Our framework focuses on a specific aspect: given two resumes where one candidate is 
    objectively more qualified (has more required qualifications), does the AI prefer that candidate?
    """)

with st.expander("How is this different from bias testing?"):
    st.markdown("""
    While related, validity and bias are distinct concepts:
    
    - **Bias** asks: Does the tool treat different demographic groups differently?
    - **Validity** asks: Does the tool measure what it's supposed to measure?
    
    A tool can be unbiased but invalid (treats everyone equally poorly), or valid but biased 
    (correctly identifies qualifications but applies them inconsistently across groups).
    
    Our tool primarily tests validity — whether AI correctly identifies better-qualified candidates — 
    though the methods can be extended to bias testing by varying demographic attributes.
    """)

with st.expander("Can I use this to game AI resume screeners?"):
    st.markdown("""
    Our tool is designed for **understanding** and **auditing** AI systems, not gaming them.
    
    The insights you gain can help you:
    - Ensure your qualifications are clearly communicated
    - Understand which skills different AI systems value
    - Identify if your resume's phrasing might be misinterpreted
    
    We believe transparency about how AI hiring systems work is important for both 
    job seekers and policymakers.
    """)

with st.expander("How do I connect my own LLM?"):
    st.markdown("""
    Our tool supports any OpenAI-compatible API:
    
    1. **Local LLMs**: Ollama (`http://localhost:11434`), vLLM (`http://localhost:8000`), 
       LocalAI (`http://localhost:8080`)
    2. **Cloud APIs**: OpenRouter, Together AI, Anyscale, Fireworks
    3. **Custom endpoints**: Any server implementing the OpenAI chat completions API
    
    Just provide your API base URL and model name in the LLM Provider view.
    """)

with st.expander("What metrics do you measure?"):
    st.markdown("""
    Our validity framework measures four key metrics:
    
    | Metric | Description | Ideal Value |
    |--------|-------------|-------------|
    | **Criterion Validity** | % of correct decisions when one candidate is better | 100% |
    | **Unjustified Abstention** | % of times model abstains when one is clearly better | 0% |
    | **Discriminant Validity** | % of times model abstains when candidates are equal | 100% |
    | **Selection Rate (First)** | % of times model picks first candidate on equal pairs | 50% |
    """)

st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.9rem;">
    <p>
        Built with ❤️ for transparent AI hiring • 
        <a href="#privacy">Privacy Policy</a> • 
        <a href="#">Terms of Use</a> • 
        <a href="https://github.com/">GitHub</a>
    </p>
    <p style="font-size: 0.85rem;">
        This is a research tool. Results should be interpreted as exploratory findings, not definitive assessments.
    </p>
</div>
""", unsafe_allow_html=True)
