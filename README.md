# 🧠 AI Resume Ranker

An intelligent resume screening and ranking system built with Python, Flask, NLP, and Machine Learning. It parses resumes (PDF/images), extracts key skills, matches them to a given job description, ranks candidates using semantic similarity scores, and presents a dashboard for recruiters with charts and filters.

---

## 🚀 Features

- 📄 Upload and parse resumes (PDF + Image)
- 🔍 Extract & match skills using NLP and fuzzy logic
- 🧠 Semantic scoring using Sentence-BERT
- 🏷️ Automatic skill tagging and threshold badges
- 📊 Resume analytics dashboard (charts, filters, tables)
- 🧾 Export filtered results to Excel/CSV
- 🔐 Admin login for secure access

---

## 🧱 Project Structure
ai_resume_ranker/
├── app.py                # Main Flask application
├── templates/            # HTML templates (dashboard, upload, results)
├── static/               # CSS and JS files
├── utils/                # Resume parsing, scoring, skill extraction
├── models/               # spaCy, Sentence-BERT
├── uploads/              # Uploaded resume files
├── output/               # Exported Excel or CSV files
├── requirements.txt      # Python dependencies
└── README.md             # You’re here!

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **NLP**: spaCy (large model), Sentence-BERT, FuzzyWuzzy
- **Vector Search**: FAISS (optional for scaling)
- **Database**: MongoDB + GridFS
- **Visualization**: Chart.js, DataTables.js
- **Export**: pandas, XlsxWriter

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

`bash
git clone https://github.com/Gajanan9960/ai_resume_ranker.git
cd ai_resume_ranker
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
python app.py

🔐 Admin Credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


  Screenshots
	•	Resume Upload Interface
	•	Matched Skills Table
	•	Shortlisted Candidates View
	•	Tag-based Charts (Pie + Line)
	•	Export to Excel/CSV

 📜 License

MIT License — Free to use, fork, and improve!
