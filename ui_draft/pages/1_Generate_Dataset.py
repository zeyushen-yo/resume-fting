#!/usr/bin/env python3
"""
Generate Dataset Page - Select a job and download evaluation pairs.
"""
import streamlit as st
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ui_draft.styles import inject_styles

# Page config
st.set_page_config(
    page_title="Generate Dataset",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# Job options
JOBS = [
    "Software Engineer",
    "Nurse Practitioner",
    "Wind Turbine Technician"
]

# Directory for storing job JSON files
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "generated_datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_job_json_path(job_name: str) -> Path:
    """Get the path to the JSON file for a specific job."""
    # Convert job name to filename-friendly format
    filename = job_name.lower().replace(" ", "_") + ".json"
    return DATA_DIR / filename

def load_job_json(job_name: str) -> dict:
    """Load JSON data for a job, creating empty JSON if it doesn't exist."""
    json_path = get_job_json_path(job_name)
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            return {}
    else:
        # Create empty JSON file
        empty_data = {}
        with open(json_path, 'w') as f:
            json.dump(empty_data, f, indent=2)
        return empty_data

st.markdown("""
<h1 style="font-size: 3rem; font-weight: 700; color: #000000; margin-bottom: 0.5rem;">
    Generate Dataset
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size: 1.25rem; color: #000000; line-height: 1.6; margin-bottom: 2rem;">
    Select a job to view and download evaluation pairs.
</p>
""", unsafe_allow_html=True)

# Job selection dropdown
selected_job = st.selectbox(
    "Select Job",
    options=JOBS,
    index=0
)

if selected_job:
    # Load or create JSON for selected job
    job_data = load_job_json(selected_job)
    
    # Display results section
    st.markdown("### Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display graph/table using placeholder
        placeholder_path = Path(__file__).parent.parent / "resources" / "placeholder.png"
        if placeholder_path.exists():
            st.image(str(placeholder_path), caption=f"Results for {selected_job}")
        else:
            st.info("Results visualization will appear here")
    
    with col2:
        # Download button
        json_path = get_job_json_path(selected_job)
        if json_path.exists():
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{selected_job.lower().replace(' ', '_')}_evaluation_pairs.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("No data available for download")
