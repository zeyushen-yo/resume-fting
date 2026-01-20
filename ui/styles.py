"""Shared CSS styles for the Resume Validity UI - AI2 Playground inspired design."""

SHARED_CSS = """
<style>
    /* Import fonts - similar to AI2 Playground */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    /* Light theme global styles - matching AI2 Playground */
    .stApp {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f5f0e8 !important;
        color: #1a1a2e !important;
    }
    
    /* Override Streamlit's dark backgrounds */
    .main .block-container {
        background-color: #f5f0e8 !important;
        max-width: 1200px;
    }
    
    /* Hero header - refined styling */
    .hero-title {
        font-size: 2.75rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        color: #4a5568;
        font-weight: 400;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Cards - light theme with subtle shadows */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: box-shadow 0.2s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-description {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Feature cards for landing page */
    .feature-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 2rem;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        border-color: #0d9488;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.75rem;
    }
    
    .feature-text {
        color: #4a5568;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Insight cards - semantic colors */
    .insight-success {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-color: #22c55e;
    }
    
    .insight-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-color: #f59e0b;
    }
    
    .insight-info {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        border-color: #0ea5e9;
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
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    .result-incorrect {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .result-neutral {
        background: #f3f4f6;
        color: #4b5563;
        border: 1px solid #e5e7eb;
    }
    
    /* Stats boxes */
    .stat-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0d9488;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Teal accent color - matching AI2's green/teal */
    .accent-teal {
        color: #0d9488;
    }
    
    .accent-pink {
        color: #ec4899;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Textarea styling - light theme */
    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1) !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #1a1a2e !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1) !important;
    }
    
    /* Button styling - teal primary */
    .stButton > button {
        background: #0d9488 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: #0f766e !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #0d9488 !important;
        border: 2px solid #0d9488 !important;
    }
    
    /* Page link buttons */
    .stPageLink > a {
        background: #0d9488 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 500 !important;
        color: #1a1a2e !important;
        background-color: #ffffff !important;
    }
    
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    /* Progress bar - teal */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0d9488 0%, #14b8a6 100%) !important;
    }
    
    /* Metric cards for benchmark - light theme */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .metric-good { color: #059669; }
    .metric-warning { color: #d97706; }
    .metric-bad { color: #dc2626; }
    .metric-na { color: #6b7280; }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling - darker like AI2 */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: #0d9488 !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        border: 1px solid #e2e8f0;
        border-bottom: none;
        color: #4a5568;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-color: #0d9488;
        color: #0d9488;
    }
    
    /* Alert/info boxes */
    .stAlert {
        background-color: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        color: #0369a1 !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #0d9488 !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent;
    }
    
    /* Privacy notice styling */
    .privacy-notice {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 0.85rem;
        color: #64748b;
        margin: 1rem 0;
    }
    
    .privacy-notice strong {
        color: #1a1a2e;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0d9488;
    }
    
    /* Paper citation box */
    .paper-citation {
        background: linear-gradient(135deg, #f0fdf4 0%, #f0f9ff 100%);
        border: 1px solid #0d9488;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .paper-title {
        font-weight: 600;
        color: #1a1a2e;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .paper-authors {
        color: #4a5568;
        font-size: 0.95rem;
    }
    
    /* Quick benchmark info card */
    .quick-bench-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Link styling */
    a {
        color: #0d9488 !important;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Markdown text color fix */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1a1a2e !important;
    }
    
    /* Override any remaining dark text */
    .element-container {
        color: #1a1a2e !important;
    }
</style>
"""


def inject_styles():
    """Inject shared CSS styles into the Streamlit app."""
    import streamlit as st
    st.markdown(SHARED_CSS, unsafe_allow_html=True)
