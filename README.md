# 🔍 Job Recommender NLP: AI-Powered Job & Talent Matching with Sentence-BERT

## 📘 Overview

This project builds an **AI-powered job and resume recommendation engine** using Sentence-BERT embeddings to match resumes with job postings — and vice versa. It was initially developed as a demo on Kaggle to showcase how **semantic similarity** can improve recruitment and job searching over traditional keyword-based systems.

The system has now been enhanced with a **Streamlit web interface** for interactive use and deployment.

It processes a subset of the [LinkedIn Job Postings Dataset (2023-24)](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) and the [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset), recommending the **top 5 matches in both directions**:
- Resumes per job posting (for recruiters)
- Job postings per resume (for job seekers)

---

## 🔗 Links

- 📘 **Kaggle Notebook**: [Job Recommender NLP](https://www.kaggle.com/code/muhammadaliasghar01/job-recommender-nlp/)
- 💻 **GitHub Repository**: [Job-Recommender-NLP](https://github.com/MuhammadAliAsgher/Job-Recommender-NLP)
- 📄 **LinkedIn Job Postings Dataset**: [View Dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)
- 📄 **Resume Dataset**: [View Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

---

## 🔑 Key Features

### 🧠 Bi-Directional Semantic Recommendation Engine
- Matches **1,000 job postings** with **1,000 resumes** using **semantic similarity**
- Recommends:
  - 🔍 Top 5 jobs for an uploaded resume *(job-seeker perspective)*
  - 🧑‍💼 Top 5 resumes for an uploaded job description *(recruiter perspective)*

### ⚙️ Implementation Details
- Built with **Sentence-BERT (all-MiniLM-L6-v2)** for embeddings
- Features:
  - Text preprocessing using **spaCy** (lemmatization, skill/domain extraction)
  - Cosine similarity computation
  - Interactive frontend using **Streamlit**
  - Auto-generation of working files if missing

### 🌟 New Features
- 🎨 **Visually Appealing UI**: Custom-styled Streamlit interface
- 📁 **File Uploads**: Supports `.txt`, `.pdf`, `.docx` for resumes and job descriptions
- ⚙️ **Dynamic Setup**: Automatically creates the working directory if absent

---

## 📁 Repository Contents

- `app.py`: Streamlit app for interactive resume/job matching
- `job_recommender_nlp.py`: Backend logic (data processing, embeddings, recommendation)
- `job-recommender-nlp.ipynb`: Jupyter Notebook version
- `datasets/`: Includes `postings.csv` and `Resume.csv`

---

## 🚀 How to Run

### ✅ Prerequisites

- Python **3.10+**

Install dependencies:

```bash
pip install numpy pandas spacy sentence-transformers scikit-learn PyPDF2 python-docx streamlit

```

Install the spaCy model:

```bash
python -m spacy download en_core_web_sm
```
---
## 🛠️ Local Setup
### 1. Clone the Repository
```bash
git clone https://github.com/MuhammadAliAsgher/Job-Recommender-NLP
```
```bash
cd Job-Recommender-NLP
```
### 2. Prepare Datasets
- Datasets (`postings.csv`, `Resume.csv`) are inside the `datasets/` directory

- Managed using Git LFS — large files auto-downloaded when you clone the repo

### Run the App
```bash
streamlit run app.py
```
---


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙋‍♂️ Contact

Created by **Muhammad Ali Asghar**  
📧 Connect on [LinkedIn](https://www.linkedin.com/in/muhammad-ali-asghar-82b87121b/)  
🌐 [Github/MuhammadAliAsgher](https://github.com/MuhammadAliAsgher)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" />
