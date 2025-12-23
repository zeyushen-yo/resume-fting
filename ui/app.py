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
    page_title="Resume Validity Benchmark",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# Landing page content
st.markdown('<h1 class="hero-title">🎯 Resume Validity Benchmark</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">A comprehensive framework for evaluating AI hiring systems — '
    'test validity, detect bias, and benchmark LLM performance.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# Two cards side by side using HTML/CSS grid for equal heights
st.markdown("""
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: stretch;">
    <div class="card" style="height: 100%;">
        <div class="card-title">👤 Job Applicant View</div>
        <p style="color: #94a3b8; margin-bottom: 1rem;">
            Test your resume against multiple AI hiring systems. 
            Discover which qualifications matter most and how different 
            models evaluate your candidacy.
        </p>
        <ul style="color: #94a3b8; margin-left: 1rem;">
            <li>Upload or paste your resume</li>
            <li>Provide target job description</li>
            <li>Choose which AI models to test against</li>
            <li>Get detailed insights on what matters</li>
        </ul>
    </div>
    <div class="card" style="height: 100%;">
        <div class="card-title">🔬 LLM Provider View</div>
        <p style="color: #94a3b8; margin-bottom: 1rem;">
            Benchmark your LLM's hiring decision quality. 
            Test validity metrics across multiple resume pairs 
            and job descriptions.
        </p>
        <ul style="color: #94a3b8; margin-left: 1rem;">
            <li>Connect your custom LLM via OpenAI-compatible API</li>
            <li>Upload job descriptions and resumes</li>
            <li>Run comprehensive validity benchmarks</li>
            <li>Get detailed metrics and analysis</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Buttons below the cards
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Job_Applicant.py", label="🚀 Test My Resume", use_container_width=True)

with col2:
    st.page_link("pages/2_LLM_Provider.py", label="📊 Benchmark My LLM", use_container_width=True)

st.markdown("---")

# How it works section
st.markdown("## How It Works")

st.markdown("""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1rem;">
    <div class="card">
        <div class="card-title">1️⃣ Extract Qualifications</div>
        <p style="color: #94a3b8;">
            We analyze job descriptions to extract required (basic) 
            and preferred (bonus) qualifications.
        </p>
    </div>
    <div class="card">
        <div class="card-title">2️⃣ Generate Test Pairs</div>
        <p style="color: #94a3b8;">
            We create resume variants: underqualified (missing skills), 
            preferred (added skills), and reworded (same content, different phrasing).
        </p>
    </div>
    <div class="card">
        <div class="card-title">3️⃣ Evaluate & Measure</div>
        <p style="color: #94a3b8;">
            We test if models correctly identify the better candidate 
            and measure validity metrics across all test cases.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Validity metrics explanation
with st.expander("📐 Understanding Validity Metrics"):
    st.markdown("""
    ### What We Measure
    
    **1. Underqualified Detection Rate**
    - Can the model identify when a candidate is missing required qualifications?
    - We remove one qualification at a time and check if the model prefers the original
    
    **2. Preferred Skill Recognition**
    - Does the model value bonus/preferred qualifications?
    - We add one bonus qualification and check if the model prefers the enhanced resume
    
    **3. Phrasing Invariance**
    - Is the model robust to superficial changes?
    - We reword resumes without changing qualifications and check if the model abstains
    
    ### For LLM Providers
    
    Common integration methods for benchmarks:
    
    - **OpenAI-compatible API** (recommended): Most inference servers (vLLM, TGI, Ollama) 
      expose this format. Just provide your endpoint URL and model name.
    - **Direct model access**: For local testing, you can use HuggingFace model IDs
    - **Result submission**: Run evals locally and submit results (coming soon)
    """)
