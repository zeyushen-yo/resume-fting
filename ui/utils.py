"""Shared utilities for the Resume Validity UI."""
import streamlit as st
from typing import List


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file (PDF, DOCX, TXT)."""
    if uploaded_file is None:
        return ""
    
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    try:
        if file_type == "application/pdf" or file_name.endswith(".pdf"):
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith(".docx"):
            import docx
            import io
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        
        elif file_type == "text/plain" or file_name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8").strip()
        
        else:
            # Try to read as text
            return uploaded_file.read().decode("utf-8").strip()
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


def extract_texts_from_files(uploaded_files) -> List[str]:
    """Extract text from multiple uploaded files."""
    texts = []
    for f in uploaded_files:
        # Reset file pointer for each file
        f.seek(0)
        text = extract_text_from_file(f)
        if text:
            texts.append(text)
    return texts

