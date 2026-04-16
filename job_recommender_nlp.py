# Import Libraries

import pandas as pd
import numpy as np
import spacy
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Preprocess and Extract Features from the Job Description Dataset
# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Define skill and domain keywords
SKILLS = ['javascript', 'node.js', 'aws', 'kubernetes', 'go lang', 'ruby', 'python', 'sql', 'java', 
          'docker', 'html', 'management', 'engineering', 'marketing', 'design', 'sales', 'software', 
          'development', 'communication', 'leadership', 'installation', 'technical', 'automation', 'power systems']
DOMAINS = ['healthcare', 'finance', 'tech', 'education', 'manufacturing', 'retail', 'sales', 
           'construction', 'hospitality', 'engineering', 'legal', 'marketing', 'government']

def preprocess_text(text):
    """Preprocess text by lemmatizing and removing stop words."""
    if pd.isna(text):
        return ""
    text = text.lower()
    doc = nlp(text)
    original_terms = [token.text for token in doc]
    lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    return {'original': original_terms, 'lemmatized': lemmatized}

def extract_skills(text_data):
    """Extract skills from preprocessed text data."""
    if pd.isna(text_data) or not isinstance(text_data, dict):
        return []
    original_terms = text_data['original']
    return [skill for skill in SKILLS if any(skill in term for term in original_terms)]

def extract_domains(text_data, skills=None):
    """Extract domains from preprocessed text data and inferred skills."""
    if pd.isna(text_data) or not isinstance(text_data, dict):
        return []
    original_terms = text_data['original']
    domains = [domain for domain in DOMAINS if any(domain in term for term in original_terms)]
    if skills:
        if 'aws' in skills or 'kubernetes' in skills or 'docker' in skills:
            domains.append('tech')
        if 'management' in skills or 'leadership' in skills:
            domains.append('business')
    return list(set(domains))

def load_and_preprocess_data(postings_path='./datasets/postings.csv', resume_path='./datasets/Resume.csv'):
    """Load and preprocess job postings and resume data."""
    postings_df = pd.read_csv(postings_path)
    resume_df = pd.read_csv(resume_path)
    postings_sample_df = postings_df.sample(1000, random_state=42).copy()
    resume_sample_df = resume_df.sample(1000, random_state=42).copy()

    postings_sample_df['processed_desc'] = postings_sample_df['description'].apply(preprocess_text)
    postings_sample_df['job_skills'] = postings_sample_df['processed_desc'].apply(extract_skills)
    postings_sample_df['job_domain'] = postings_sample_df.apply(lambda x: extract_domains(x['processed_desc'], x['job_skills']), axis=1)

    resume_sample_df['processed_resume'] = resume_sample_df['Resume_str'].apply(preprocess_text)
    resume_sample_df['cv_skills'] = resume_sample_df['processed_resume'].apply(extract_skills)
    resume_sample_df['cv_domain'] = resume_sample_df.apply(lambda x: extract_domains(x['processed_resume'], x['cv_skills']), axis=1)

    job_data = postings_sample_df[['job_id', 'title', 'processed_desc', 'job_skills', 'job_domain']].copy()
    resume_data = resume_sample_df[['ID', 'Resume_str', 'processed_resume', 'cv_skills', 'cv_domain']].copy()
    combined_data = pd.concat([job_data.assign(type='job'), resume_data.assign(type='resume')], ignore_index=True)
    combined_data.to_csv('./working/cleaned_data.csv', index=False)
    return combined_data, postings_sample_df, resume_sample_df

def generate_embeddings(postings_sample_df, resume_sample_df):
    """Generate embeddings for job postings and resumes."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_texts = postings_sample_df['description'].fillna('').astype(str).tolist()
    resume_texts = resume_sample_df['Resume_str'].fillna('').astype(str).tolist()
    job_embeddings = model.encode(job_texts, show_progress_bar=True)
    resume_embeddings = model.encode(resume_texts, show_progress_bar=True)
    np.save('./working/job_embeddings.npy', job_embeddings)
    np.save('./working/resume_embeddings.npy', resume_embeddings)
    return job_embeddings, resume_embeddings, model

def load_data_and_embeddings():
    """Load preprocessed data and embeddings."""
    combined_data = pd.read_csv('./working/cleaned_data.csv')
    job_embeddings = np.load('./working/job_embeddings.npy')
    resume_embeddings = np.load('./working/resume_embeddings.npy')
    return combined_data, job_embeddings, resume_embeddings

def generate_recommendations(job_embeddings, resume_embeddings, combined_data):
    """Generate top 5 resume recommendations for each job."""
    similarity_matrix = cosine_similarity(job_embeddings, resume_embeddings)
    top_k = 5
    job_recommendations = []
    for i in range(len(job_embeddings)):
        top_indices = similarity_matrix[i].argsort()[-top_k-1:-1][::-1]
        job_recommendations.append([(combined_data[combined_data['type'] == 'resume'].index[j], similarity_matrix[i][j]) for j in top_indices])
    
    np.random.seed(42)
    random_job_indices = np.random.choice(len(job_embeddings), 3, replace=False)
    return job_recommendations, random_job_indices, similarity_matrix

def recommend_for_uploaded_text(text, input_type, job_embeddings, resume_embeddings, combined_data, model):
    """Generate recommendations for an uploaded resume or job description."""
    processed_text = preprocess_text(text)
    embedding = model.encode([processed_text["lemmatized"]])
    
    if input_type == "Resume":
        similarities = cosine_similarity(embedding, job_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        recommendations = [(idx, similarities[idx]) for idx in top_indices]
    else:
        similarities = cosine_similarity(embedding, resume_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        recommendations = [(idx, similarities[idx]) for idx in top_indices]
        #recommendations = [(combined_data[combined_data['type'] == 'resume'].index[idx], similarities[idx]) for idx in top_indices]
    
    return recommendations