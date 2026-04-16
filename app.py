import streamlit as st
import pandas as pd
import PyPDF2
from io import BytesIO
from docx import Document
import os
import spacy
from job_recommender_nlp import load_data_and_embeddings, recommend_for_uploaded_text, load_and_preprocess_data, generate_embeddings

# Ensure spaCy model is installed
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    st.info("Downloading spaCy model 'en_core_web_sm'...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Ensure working directory exists
if not os.path.exists("./working"):
    os.makedirs("./working")
    # Generate working files if missing
    combined_data, postings_sample_df, resume_sample_df = load_and_preprocess_data()
    job_embeddings, resume_embeddings, _ = generate_embeddings(postings_sample_df, resume_sample_df)

# Load data and embeddings
combined_data, job_embeddings, resume_embeddings = load_data_and_embeddings()

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 20px;
    }
    .upload-box {
        background-color: #F5F6FA;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .recommendation-card {
        background-color: #E8F0FE;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2E86C1;
    }
    .recommendation-title {
        font-size: 20px;
        font-weight: bold;
        color: #1A5276;
        margin-bottom: 10px;
    }
    .recommendation-text {
        font-size: 16px;
        color: #34495E;
    }
    .stRadio > label {
        font-size: 16px;
        color: #2E86C1;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<div class="main-title">Job Recommendation System</div>', unsafe_allow_html=True)

# Header image or icon
st.markdown("📄 **Upload your document to get personalized recommendations**", unsafe_allow_html=True)

# Upload section
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a resume or job description (.txt, .pdf, or .docx)", type=["txt", "pdf", "docx"])
    
    if uploaded_file is not None:
        try:
            # Extract text based on file type
            if uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    text += page_text or ""
                if not text.strip():
                    st.error("Failed to extract text from the PDF. It might be a scanned document or empty.")
                    text = None
            elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                doc = Document(BytesIO(uploaded_file.read()))
                text = "\n".join([para.text for para in doc.paragraphs])
                if not text.strip():
                    st.error("Failed to extract text from the Word document. It might be empty.")
                    text = None
            else:
                st.error("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
                text = None

            # Proceed if text was successfully extracted
            if text:
                input_type = st.radio("Select input type", ["Resume", "Job Description"], help="Choose whether you uploaded a resume or a job description.")
                
                # Load model for dynamic recommendations
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                
                recommendations = recommend_for_uploaded_text(text, input_type, job_embeddings, resume_embeddings, combined_data, model)
                
                # Display recommendations
                st.markdown('<div class="recommendation-title">Your Recommendations</div>', unsafe_allow_html=True)
                if input_type == "Resume":
                    for idx, score in recommendations:
                        job = combined_data[combined_data['type'] == 'job'].iloc[idx]
                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                                <div class="recommendation-text">
                                    <strong>Job {idx}</strong> (ID: {job['job_id']})<br>
                                    <strong>Title:</strong> {job['title']}<br>
                                    <strong>Similarity Score:</strong> {score:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    for idx, score in recommendations:
                        resume = combined_data[combined_data['type'] == 'resume'].iloc[idx]
                        resume_id = resume['ID']
                        resume_text_snippet = resume['Resume_str'][:100] + "..." if pd.notna(resume['Resume_str']) else "No description available"
                        st.markdown(
                            f"""
                            <div class="recommendation-card">
                                <div class="recommendation-text">
                                    <strong>Resume ID:</strong> {resume_id}<br>
                                    <strong>Snippet:</strong> {resume_text_snippet}<br>
                                    <strong>Similarity Score:</strong> {score:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing the uploaded file: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)