# 🧠 AI Resume Ranker

**Stop manually screening resumes.** AI Resume Ranker is a web application that lets recruiters upload candidate resumes (PDF or image), paste a job description, and instantly receive an AI-ranked shortlist — scored and categorised automatically using NLP and semantic similarity.

---

## 🎯 Problem Statement

Every open role attracts dozens or hundreds of applications. Manual resume screening is:

- **Slow** — reading each resume takes minutes; large hiring drives take days
- **Inconsistent** — different reviewers judge the same resume differently
- **Biased** — unconscious patterns influence who gets shortlisted
- **Error-prone** — fatigue causes qualified candidates to be overlooked

**AI Resume Ranker** automates the first-pass screening step, giving recruiters an objective, ranked candidate list in seconds so human effort focuses where it matters most.

---

## 🚀 Features

- 📄 Upload and parse resumes — PDF and image (JPG/PNG) supported
- 🧠 Semantic scoring using Sentence-BERT (`all-MiniLM-L6-v2`)
- 🔍 Skill extraction and gap analysis (matched vs. missing skills)
- 🏷️ Automatic candidate tags: **Top Candidate**, **Shortlisted**, **Review**, **Rejected**
- 📊 Recruiter dashboard with stats and top-10 ranked candidates
- 📧 One-click status emails to candidates
- 📥 Export filtered results to Excel or CSV
- 🔐 Per-user data isolation — each recruiter only sees their own uploads

---

## ⚙️ How the Score is Calculated

| Component | Weight | Description |
|-----------|--------|-------------|
| Keyword Match | 40% | Fraction of required skills found in the resume |
| Semantic Similarity | 40% | Sentence-BERT cosine similarity between resume and job description |
| Keyword Density | 10% | How naturally and frequently matched keywords appear |
| Resume Quality | 10% | Length/completeness signal (optimal: 300–2000 words) |

Score thresholds: **Top Candidate ≥ 80%**, **Shortlisted ≥ 65%**, **Review ≥ 45%**, **Rejected < 45%**

---

## 🧱 Project Structure

```
ai_resume_ranker/
├── app.py              # Flask application — routes, scoring logic, helpers
├── templates/          # Jinja2 HTML templates
├── static/             # CSS styles
├── uploads/            # Uploaded resume files (auto-created)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3, Flask
- **NLP / AI**: Sentence-BERT, spaCy (`en_core_web_lg`)
- **Resume Parsing**: pdfminer.six (PDF), pytesseract + Pillow (images)
- **Database**: MongoDB Atlas
- **Export**: pandas, XlsxWriter
- **Email**: Flask-Mail (SMTP)

---

## ⚙️ Setup Instructions

### 1. Clone and install

```bash
git clone https://github.com/Gajanan9960/ai_resume_ranker.git
cd ai_resume_ranker
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://<user>:<pass>@cluster.mongodb.net/
SECRET_KEY=your-random-secret-key
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=your-app-password
```

### 3. Run

```bash
python app.py
```

Open `http://localhost:10000` in your browser, register an account, and start uploading resumes.

---

## 📜 License

MIT License — Free to use, fork, and improve.
