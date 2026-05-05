#!/usr/bin/env python3
"""
Resume Screening Validity Benchmark — landing page.
"""
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.styles import inject_styles

st.set_page_config(
    page_title="Resume Screening Validity Benchmark",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

with st.sidebar:
    # st.markdown("### Resume Validity Benchmark")
    # st.markdown("---")
    st.markdown("""
**Links**
- [Paper](https://arxiv.org/abs/2602.18550)
- [GitHub](https://github.com/zeyushen-yo/resume-fting)
""")

st.markdown("""
<h1 class="hero-title">Resume Screening Validity Benchmark</h1>
<h4 class="hero-subtitle">
    An open-source benchmark for measuring whether LLM-based resume screeners make <strong>valid</strong>
    hiring decisions — do they prefer objectively more qualified candidates, and do they behave
    consistently across equivalent resumes?
</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# Feature cards rendered as a single HTML flex row — guarantees equal height
st.markdown("""
<div style="display:flex; gap:2rem; align-items:stretch;">
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
                padding:2rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); box-sizing:border-box;">
        <div style="color:#1a1a2e; font-size:1.4rem; font-weight:600; margin-bottom:10px;">
            Upload Job Description
        </div>
        <div style="color:#4a5568; font-size:1.05rem; line-height:1.7;">
            <p style="color:#4a5568; margin-bottom:0.6rem;">Paste or upload any job description. We extract required and preferred qualifications,
            let you mark priority qualifications and specify demographic dimensions of interest,
            then generate controlled resume pairs you can download or run through a suite of LLMs.</p>
            <ul style="color:#4a5568; margin:5px 0 0 1.1rem; line-height:1.9;">
                <li style="color:#4a5568;">Generates unequal and equal pairs per our framework</li>
                <li style="color:#4a5568;">Download as JSON in standard dataset format</li>
                <li style="color:#4a5568;">Run a small or large LLM suite to measure criterion and discriminant validity</li>
            </ul>
        </div>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:16px;
                padding:2rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); box-sizing:border-box;">
        <div style="color:#1a1a2e; font-size:1.4rem; font-weight:600; margin-bottom:10px;">
            Generate Dataset for Occupation
        </div>
        <div style="color:#4a5568; font-size:1.05rem; line-height:1.7;">
            <p style="color:#4a5568; margin-bottom:0.6rem;">Select from our pre-harvested occupations. Download the job description dataset —
            with extracted qualifications — as a <code style="background:#f1f5f9; padding:0.1rem 0.3rem; border-radius:3px;">.json</code> file or a
            <code style="background:#f1f5f9; padding:0.1rem 0.3rem; border-radius:3px;">.zip</code>
            archive with a README, ready for practitioners to generate pairs and test their own tools.</p>
            <ul style="color:#4a5568; margin:5px 0 0 1.1rem; line-height:1.9;">
                <li style="color:#4a5568;">10 occupations across tech, business, and operations</li>
                <li style="color:#4a5568;">100 real job postings per role with structured qualifications</li>
                <li style="color:#4a5568;">Benchmark results from our paper for each occupation</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation buttons sit below the cards in their respective columns
st.markdown("<div style='padding-top:12px;'></div>", unsafe_allow_html=True)
btn1, btn2 = st.columns(2, gap="large")
with btn1:
    if st.button("Upload a Job Description", key="nav_jd", use_container_width=True):
        st.switch_page("pages/1_Job_Description.py")
with btn2:
    if st.button("Browse Occupation Datasets", key="nav_occ", use_container_width=True):
        st.switch_page("pages/2_Occupation_Dataset.py")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("## How It Works")

# Three-column how-it-works cards — also a single flex row for equal height
st.markdown("""
<div style="display:flex; gap:1.5rem; align-items:stretch; margin-top:1rem;">
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); box-sizing:border-box;">
        <h4 style="color:#0d9488; margin-top:0;">1. Extract Qualifications</h4>
        <p style="color:#4a5568;">
            Each job description is parsed into <strong>required</strong> (basic) and
            <strong>preferred</strong> (bonus) qualifications using an LLM.
        </p>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); box-sizing:border-box;">
        <h4 style="color:#0d9488; margin-top:0;">2. Build Controlled Pairs</h4>
        <p style="color:#4a5568;">
            A base resume is generated from the required qualifications. Variants are created
            by removing qualifications (unequal), adding bonuses (unequal), or rephrasing (equal).
        </p>
    </div>
    <div style="flex:1; background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
                padding:1.5rem; box-shadow:0 2px 8px rgba(0,0,0,0.04); box-sizing:border-box;">
        <h4 style="color:#0d9488; margin-top:0;">3. Measure Validity</h4>
        <p style="color:#4a5568;">
            Models are asked to choose between the two resumes or abstain. We compute
            <strong>criterion validity</strong>, <strong>discriminant validity</strong>,
            and position bias from their decisions.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem;">
    <a href="https://arxiv.org/abs/2602.18550">Castleman et al., 2026</a> &nbsp;•&nbsp;
    <a href="https://github.com/zeyushen-yo/resume-fting">GitHub</a>
</div>
""", unsafe_allow_html=True)
