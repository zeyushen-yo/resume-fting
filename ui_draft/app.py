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

from ui_draft.styles import inject_styles

# Page config
st.set_page_config(
    page_title="Resume Screening Validity Benchmark",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# Additional inline styles to ensure they're applied
st.markdown("""
<style>
    /* Force font sizes */
    h1 {
        font-size: 32px !important;
        color: #000000 !important;
    }
    
    p {
        font-size: 24px !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
<h1 style="font-size: 32px; font-weight: 700; color: #000000; margin-bottom: 0.5rem;">
    Resume Screening Validity Benchmark
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 24px; color: #000000; line-height: 1.6; margin-bottom: 3rem;">
    This study evaluates whether AI hiring systems make valid decisions. We test whether 
    AI systems actually prefer more qualified candidates or are influenced by irrelevant factors.
</p>
""", unsafe_allow_html=True)

# Two main action buttons - larger with centered text and light green background
st.markdown("<br>", unsafe_allow_html=True)

# Add CSS for button alignment and styling
st.markdown("""
<style>
    /* Ensure columns align buttons vertically */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    
    [data-testid="column"] > div {
        display: flex !important;
        flex-direction: column !important;
        flex: 1 !important;
    }
    
    /* Ensure page link buttons are same height and aligned */
    [data-testid="stPageLink"],
    .stPageLink {
        display: flex !important;
        align-items: stretch !important;
        height: 100% !important;
        flex: 1 !important;
    }
    
    /* Force light green background on page link buttons */
    [data-testid="stPageLink"] > a,
    .stPageLink > a {
        background: #AFF589 !important;
        background-color: #AFF589 !important;
        color: #000000 !important;
        min-height: 60px !important;
    }
    
    [data-testid="stPageLink"] > a:hover,
    .stPageLink > a:hover {
        background: #9FE579 !important;
        background-color: #9FE579 !important;
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/1_Generate_Dataset.py", label="Generate Dataset", use_container_width=True)

with col2:
    st.page_link("pages/2_Test_Job_Description.py", label="Test Job Description", use_container_width=True)
