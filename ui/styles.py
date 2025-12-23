"""Shared CSS styles for the Resume Validity UI."""

SHARED_CSS = """
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
        color: #e2e8f0 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #6b7280 !important;
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
    
    /* Metric cards for benchmark */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 1rem;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-good { color: #34d399; }
    .metric-warning { color: #fbbf24; }
    .metric-bad { color: #f87171; }
    .metric-na { color: #9ca3af; }  /* Gray for N/A - visible on dark background */
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
</style>
"""


def inject_styles():
    """Inject shared CSS styles into the Streamlit app."""
    import streamlit as st
    st.markdown(SHARED_CSS, unsafe_allow_html=True)

