# 🧠 AI Resume Ranker

An intelligent resume screening and ranking system built with Python, Flask, NLP, and Machine Learning. It parses resumes (PDF and scanned images via OCR), extracts key skills, matches them to a given job description, ranks candidates using semantic similarity scores, and presents a dashboard for recruiters with filters and exports.

---

## The Problem

Recruiters spend an average of **23 hours** screening resumes for a single hire. Manual screening is slow, inconsistent, and prone to bias. Traditional ATS tools rely on simple keyword matching, missing context, synonyms, and transferable skills.

**Resume Ranker AI** solves this by combining keyword analysis with semantic NLP to understand the *meaning* behind resume content — not just exact word matches.

---

## Scoring Methodology

Each resume is scored on a **0–100% weighted composite**:

| Component             | Weight | Description                                                    |
|-----------------------|--------|----------------------------------------------------------------|
| **Keyword Match**     | 40%    | Ratio of required job keywords found in resume text            |
| **Semantic Similarity** | 40%  | Cosine similarity of Sentence-BERT embeddings (resume vs. JD)  |
| **Keyword Density**   | 10%    | Density of relevant keywords throughout the resume             |
| **Resume Quality**    | 10%    | Word count heuristic — optimal range 300–2,000 words           |

**Bonuses:** +10% if keyword match ≥ 70%.  
**Penalties:** −15% if keyword match < 20%.

### Tag Thresholds

| Tag              | Score     |
|------------------|-----------|
| 🟢 Top Candidate | ≥ 80%     |
| 🔵 Shortlisted   | 65% – 79% |
| 🟡 Review        | 45% – 64% |
| 🔴 Rejected      | < 45%     |

---

## Features

- 📄 Upload and parse resumes (PDF + scanned images via OCR)
- 🧠 Semantic scoring using Sentence-BERT (`all-MiniLM-L6-v2`)
- 🔍 Keyword extraction and matching with spaCy NLP
- 🏷️ Automatic candidate tagging based on score thresholds
- 📊 Recruitment analytics dashboard with top candidates
- 🧾 Export filtered results to Excel (`.xlsx`) or CSV
- 📧 Send tailored emails (acceptance, interview, rejection) via SMTP
- 🔐 User authentication with data isolation per account

---

## Tech Stack

| Layer       | Technology                                     |
|-------------|------------------------------------------------|
| Backend     | Python 3.10+, Flask                            |
| NLP         | spaCy (`en_core_web_sm`), Sentence-BERT        |
| OCR         | pytesseract + Pillow                           |
| Database    | MongoDB (pymongo)                              |
| Export      | pandas + XlsxWriter                           |
| Email       | Flask-Mail (SMTP)                              |
| Deployment  | Gunicorn, Render                               |

---

## Setup Instructions

### 1. Clone and Install

```bash
git clone https://github.com/Gajanan9960/ai_resume_ranker.git
cd ai_resume_ranker
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
SECRET_KEY=your-random-secret-key-here
MONGO_URI=mongodb://localhost:27017/
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

> **Note:** For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833) — not your account password.

### 3. Run

```bash
python app.py
```

The app will be available at `http://localhost:5001`.

---

## Project Structure

```
ai_resume_ranker/
├── app.py                # Main Flask application
├── templates/            # HTML templates (home, dashboard, upload, results, about)
├── static/               # CSS styles
├── uploads/              # Uploaded resume files (gitignored)
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (gitignored)
├── Procfile              # Gunicorn start command
├── render.yaml           # Render deployment config
└── README.md
```

---

## License

MIT License — Free to use, fork, and improve!
