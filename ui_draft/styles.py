"""Shared CSS styles for the Resume Validity UI - AI2 Playground inspired design."""

SHARED_CSS = """
<style>
    /* Import fonts - similar to AI2 Playground */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 24px !important;
    }
    
    /* Override Streamlit's dark backgrounds */
    .main .block-container {
        background-color: #ffffff !important;
        max-width: 1200px;
    }
    
    /* Sidebar styling - light background with black text */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div > div,
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    /* Force all sidebar text to be black */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #000000 !important;
        font-size: 24px !important;
    }
    
    [data-testid="stSidebar"] a:hover {
        color: #0d9488 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Caption text in sidebar */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: rgba(0, 0, 0, 0.7) !important;
        font-size: 24px !important;
    }
    
    /* Hero header - refined styling */
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #000000;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #000000;
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
        font-size: 1.25rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-description {
        color: #000000;
        font-size: 1.05rem;
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
        font-size: 1.4rem;
        font-weight: 600;
        color: #000000;
        margin-bottom: 0.75rem;
    }
    
    .feature-text {
        color: #000000;
        font-size: 1.05rem;
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
    
    /* Textarea styling - light theme with larger fonts */
    .stTextArea textarea {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.05rem !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1) !important;
    }
    
    /* Text input styling - larger fonts */
    .stTextInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
        font-size: 1.05rem !important;
    }
    
    .stTextInput input:focus {
        border-color: #0d9488 !important;
        box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.1) !important;
    }
    
    /* Button styling - larger buttons with centered text */
    .stButton > button {
        background: #0d9488 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1.25rem 3rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-size: 1.25rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        text-align: center !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
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
        font-size: 1.25rem !important;
        padding: 1.25rem 3rem !important;
    }
    
    /* Page link buttons - larger with centered text and light green background */
    [data-testid="stPageLink"] > a,
    .stPageLink > a,
    a[data-testid="stPageLink"] {
        background: #AFF589 !important;
        background-color: #AFF589 !important;
        color: #000000 !important;
        border-radius: 8px !important;
        padding: 1.25rem 3rem !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        text-align: center !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 60px !important;
        border: none !important;
        width: 100% !important;
    }
    
    [data-testid="stPageLink"] > a:hover,
    .stPageLink > a:hover,
    a[data-testid="stPageLink"]:hover {
        background: #9FE579 !important;
        background-color: #9FE579 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(175, 245, 137, 0.3) !important;
    }
    
    /* Ensure buttons are vertically aligned */
    [data-testid="stPageLink"],
    .stPageLink {
        display: flex !important;
        align-items: stretch !important;
        height: 100% !important;
        width: 100% !important;
    }
    
    /* Column containers for button alignment */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    
    [data-testid="column"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
    }
    
    [data-testid="column"] [data-testid="stPageLink"] {
        flex: 1 !important;
        display: flex !important;
    }
    
    /* Expander styling - larger fonts */
    .streamlit-expanderHeader {
        font-family: 'Source Sans Pro', sans-serif !important;
        font-weight: 500 !important;
        color: #000000 !important;
        background-color: #ffffff !important;
        font-size: 1.1rem !important;
    }
    
    [data-testid="stExpander"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stExpander"] .stMarkdown {
        color: #000000 !important;
        font-size: 1.05rem !important;
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
    
    /* Sidebar styling - light background */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #000000 !important;
        font-size: 24px !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: #0d9488 !important;
        color: white !important;
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
        color: #000000;
        font-weight: 500;
        font-size: 1.05rem !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        border-color: #0d9488;
        color: #0d9488;
    }
    
    /* Alert/info boxes - ensure readable text */
    .stAlert {
        background-color: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stAlert [data-baseweb="stMarkdownContainer"] {
        color: #000000 !important;
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
    
    /* Select boxes - larger fonts */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border-color: #d1d5db !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stSelectbox label {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Radio buttons - larger fonts */
    .stRadio > div {
        background-color: transparent;
    }
    
    .stRadio label {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Privacy notice styling - larger fonts */
    .privacy-notice {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-size: 1.05rem;
        color: #000000;
        margin: 1rem 0;
    }
    
    .privacy-notice strong {
        color: #000000;
    }
    
    /* Section headers - larger fonts */
    .section-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #000000;
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
        color: #000000;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .paper-authors {
        color: #000000;
        font-size: 1.05rem;
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
    
    /* Markdown text color fix - larger fonts, black text */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stMarkdown h1 {
        color: #000000 !important;
        font-size: 2.5rem !important;
    }
    
    .stMarkdown h2 {
        color: #000000 !important;
        font-size: 2rem !important;
    }
    
    .stMarkdown h3 {
        color: #000000 !important;
        font-size: 1.5rem !important;
    }
    
    .stMarkdown h4 {
        color: #000000 !important;
        font-size: 1.25rem !important;
    }
    
    /* Override any remaining dark text */
    .element-container {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Ensure all text elements have minimum font size */
    p, span, div, li, td, th {
        font-size: 24px !important;
        color: #000000 !important;
    }
    
    /* Labels and form elements */
    label {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Checkbox and other form controls */
    .stCheckbox label {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* File uploader text */
    [data-testid="stFileUploader"] label {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Success/Error/Info/Warning messages */
    .stSuccess {
        background-color: #d1fae5 !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stError {
        background-color: #fee2e2 !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stInfo {
        background-color: #dbeafe !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    .stWarning {
        background-color: #fef3c7 !important;
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Ensure all Streamlit text containers have black text */
    [data-testid="stText"] {
        color: #000000 !important;
        font-size: 1.05rem !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #000000 !important;
        font-size: 1rem !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
        font-size: 1rem !important;
    }
    
    pre {
        background-color: #f3f4f6 !important;
        color: #000000 !important;
        font-size: 1rem !important;
    }
</style>
"""


def inject_styles():
    """Inject shared CSS styles into the Streamlit app."""
    import streamlit as st
    st.markdown(SHARED_CSS, unsafe_allow_html=True)
