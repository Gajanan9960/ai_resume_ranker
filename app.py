import os
import re
import io
import cv2
import pytesseract
import spacy
import pandas as pd
from math import ceil
from datetime import datetime, timedelta
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient, ASCENDING, DESCENDING
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
from gridfs import GridFS
from markupsafe import Markup
from difflib import SequenceMatcher

# --- App Setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Models ---
nlp = spacy.load("en_core_web_lg")
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- DB Setup ---
client = MongoClient("mongodb://localhost:27017")
db = client['resume_db']
resumes_collection = db['resumes']
users_collection = db['users']
synonyms_collection = db['skill_synonyms']
fs = GridFS(db)

# --- Skills Setup ---
TECHNICAL_SKILLS = {
    "programming": ["python", "java", "c++", "c", "javascript", "typescript", "ruby", "go", "rust"],
    "web": ["html", "css", "react", "angular", "vue", "django", "flask", "node.js", "express"],
    "database": ["mysql", "postgresql", "mongodb", "oracle", "sql", "nosql", "sqlite"],
    "tools": ["git", "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "terraform"],
    "ml_ai": ["machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn", "nlp", "computer vision"],
    "data": ["pandas", "numpy", "matplotlib", "seaborn", "powerbi", "tableau", "data analysis"]
}

NOISY_TERMS = {
    "skill", "skills", "developer", "stack", "framework", "tools",
    "language", "technologies", "programming", "software", "web", "experience"
}

TECH_SKILL_SET = set(
    skill.lower()
    for group in TECHNICAL_SKILLS.values()
    for skill in group
    if skill.lower() not in NOISY_TERMS
)

ALL_SYNONYMS = {doc['keyword']: doc['synonyms'] for doc in synonyms_collection.find()}
for syns in ALL_SYNONYMS.values():
    for syn in syns:
        if syn.lower() not in NOISY_TERMS:
            TECH_SKILL_SET.add(syn.lower())

# --- Utilities ---
def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}", text)
    return match.group(0) if match else "—"

def extract_text_from_pdf(path):
    return extract_text(path)

def extract_text_from_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(cv2.medianBlur(thresh, 3))

def extract_section(text, section):
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    headers = [
        r'skills?', r'technical\s+skills?', r'experience', r'work\s+experience',
        r'certifications?', r'education', r'projects?', r'qualifications?',
        r'work\s+history', r'professional\s+summary', r'technical\s+proficiencies'
    ]
    pattern = rf'(?i)\b{section}\b[\s:\-]*\n([\s\S]+?)(?=\n(?:{"|".join(headers)})[\s:\-]*\n|\Z)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""

def get_synonyms(keyword):
    return ALL_SYNONYMS.get(keyword.lower(), [])

def normalize_keyword(term):
    term = term.strip().lower()
    for k, syns in ALL_SYNONYMS.items():
        if term == k or term in syns:
            return k
    return term

def extract_keywords_spacy(text):
    doc = nlp(text)
    keywords = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop and token.is_alpha:
            norm = normalize_keyword(token.lemma_.lower())
            if len(norm) >= 3 and norm not in NOISY_TERMS:
                keywords.add(norm)
    return list(keywords)

def is_partial_match(keyword, text, threshold=0.85):
    for word in text.split():
        if SequenceMatcher(None, keyword, word).ratio() >= threshold:
            return True
    return False

def extract_technical_skills(text, job_keywords):
    text_lower = text.lower()
    expanded_keywords = set()
    for kw in job_keywords:
        expanded_keywords.add(kw)
        expanded_keywords.update(get_synonyms(kw))
        for group in TECHNICAL_SKILLS.values():
            if kw in group:
                expanded_keywords.update(group)

    matched = []
    for kw in expanded_keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) or is_partial_match(kw, text_lower):
            matched.append(kw)

    if len(matched) < 4:
        for tech in TECH_SKILL_SET:
            if tech not in matched and tech in text_lower:
                matched.append(tech)
                if len(matched) >= 10:
                    break

    return matched[:10]

def semantic_similarity(text, job_desc):
    res_emb = model.encode(text, convert_to_tensor=True)
    job_emb = model.encode(job_desc, convert_to_tensor=True)
    return round(util.cos_sim(res_emb, job_emb).item() * 100, 2)

def keyword_density(text, keywords):
    text = text.lower()
    word_count = len(text.split()) or 1
    hits = sum(text.count(kw) for kw in keywords)
    return min(hits / word_count, 0.04)

def weighted_resume_score(text, job_keywords, job_desc):
    sections = {
        'skills': extract_section(text, "skills") or text,
        'experience': extract_section(text, "experience") or text,
        'certifications': extract_section(text, "certifications") or text,
        'education': extract_section(text, "education") or text,
        'projects': extract_section(text, "projects") or text
    }

    weights = {
        'skills': 0.3,
        'projects': 0.2,
        'experience': 0.2,
        'certifications': 0.15,
        'education': 0.1
    }

    total_keywords = len(job_keywords) or 1
    total_score = 0
    matched_all = []

    for section, weight in weights.items():
        matched = extract_technical_skills(sections[section], job_keywords)
        unique_matches = set(matched)
        matched_all.extend(unique_matches)
        section_score = len(unique_matches) / total_keywords
        total_score += section_score * weight

    # Global matches as fallback boost
    global_matches = extract_technical_skills(text, job_keywords)
    global_unique = set(global_matches) - set(matched_all)
    matched_all.extend(global_unique)

    # Semantic similarity gets slightly higher weight now
    sim = semantic_similarity(text, job_desc)
    total_score += 0.25 * (sim / 100)

    # Keyword density boost
    density_ratio = keyword_density(text, job_keywords)
    total_score += density_ratio * 0.5

    # Penalty if less than 30% of keywords matched globally
    match_ratio = len(set(matched_all)) / total_keywords
    if match_ratio < 0.3:
        total_score *= 0.85  # reduce score if very low match

    # Normalize and scale score to 100
    final_score = round(min(total_score * 100, 100), 2)

    tag = "Shortlisted" if final_score >= 85 else "Rejected" if final_score < 50 else "Review"
    top_skills = [s for s, _ in Counter(matched_all).most_common(10)]
    missing = list(set(job_keywords) - set(top_skills))

    return final_score, top_skills, tag, missing

def humanize_time(dt):
    if not dt:
        return "unknown"
    delta = datetime.utcnow() - dt if isinstance(dt, datetime) else datetime.utcnow()
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} sec ago"
    elif seconds < 3600:
        return f"{seconds // 60} min ago"
    elif seconds < 86400:
        return f"{seconds // 3600} hrs ago"
    else:
        return f"{seconds // 86400} days ago"
# Routes remain same. Dashboard and Results templates will now include DataTables for sorting/exporting.

# Routes and remaining logic are assumed to be previously integrated correctly.
# Let me know if you want them copied into this file too, or if you need a packaged project structure.ex
# --- Routes ---
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

# --- UPLOAD ROUTE ---
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "admin" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        job_title = request.form.get("job_role", "Unknown")
        job_desc = request.form.get("job_description", "")
        job_keywords_raw = extract_keywords_spacy(job_desc)

        # Normalize and expand job keywords
        job_keywords = set()
        for kw in job_keywords_raw:
            norm = normalize_keyword(kw)
            job_keywords.add(norm)
            job_keywords.update(get_synonyms(norm))

        for file in request.files.getlist("resume"):
            if not file.filename:
                continue

            original_filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            file.seek(0)
            file_id = fs.put(file.read(), filename=filename, content_type=file.content_type)

            # Extract resume text from PDF or image
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                resume_text = extract_text_from_image(filepath)
            else:
                resume_text = extract_text_from_pdf(filepath)
            resume_text = re.sub(r"\s+", " ", resume_text.strip())

            # Score resume
            score, matched_skills, tag, missing_skills = weighted_resume_score(resume_text, job_keywords, job_desc)

            # Store in MongoDB
            resumes_collection.insert_one({
                "name": original_filename.rsplit(".", 1)[0].split("_")[0],
                "filename": filename,
                "file_id": file_id,
                "job_title": job_title,
                "job_description": job_desc,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "score": score,
                "tag": tag,
                "email": extract_email(resume_text),
                "uploaded_at": datetime.utcnow(),
                "resume_text": resume_text
            })

        session["upload_success"] = True
        return redirect(url_for("results"))

    return render_template("upload.html")
# --- RESULTS ROUTE ---
@app.route("/results")
def results():
    if 'user' not in session and 'admin' not in session:
        return redirect(url_for('login'))

    # --- Request Parameters ---
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    search = request.args.get('search', '').strip()
    tag_filter = request.args.get('tag', '').strip()
    job_filter = request.args.get('job_filter', '').strip()
    sort_by = request.args.get('sort_by', 'uploaded_at')

    # --- Mongo Query Construction ---
    query = {}
    if search:
        query['$or'] = [
            {"name": {"$regex": search, "$options": "i"}},
            {"matched_skills": {"$regex": search, "$options": "i"}},
            {"resume_text": {"$regex": search, "$options": "i"}}
        ]
    if tag_filter:
        query['tag'] = tag_filter
    if job_filter:
        query['job_title'] = job_filter

    # --- Sorting Logic ---
    if sort_by == "score":
        sort_order = [("score", DESCENDING)]
    elif sort_by == "job_title":
        sort_order = [("job_title", ASCENDING)]
    else:
        sort_order = [("uploaded_at", DESCENDING)]

    # --- Pagination ---
    total = resumes_collection.count_documents(query)
    skips = per_page * (page - 1)
    resumes = list(
        resumes_collection.find(query)
        .sort(sort_order)
        .skip(skips)
        .limit(per_page)
    )
    total_pages = ceil(total / per_page)

    # --- Post-process Resumes ---
    for r in resumes:
        r['name'] = r.get('name', 'Unknown')
        r['filename'] = r.get('filename', 'N/A')
        r['job_title'] = r.get('job_title', 'Unknown')
        r['tag'] = r.get('tag', '—')
        r['matched_skills'] = list(map(str, r.get('matched_skills', [])))
        r['missing_skills'] = list(map(str, r.get('missing_skills', [])))
        r['score'] = round(float(r.get('score', 0.0)), 2)

        uploaded_at = r.get('uploaded_at')
        if isinstance(uploaded_at, str):
            try:
                uploaded_at = datetime.fromisoformat(uploaded_at)
            except ValueError:
                uploaded_at = None
        elif not isinstance(uploaded_at, datetime):
            uploaded_at = None

        r['uploaded_str'] = humanize_time(uploaded_at)

    # --- Dropdown Filter Options ---
    all_tags = sorted(filter(None, resumes_collection.distinct("tag")))
    all_titles = sorted(filter(None, resumes_collection.distinct("job_title")))

    # --- Flash Message for Upload Success ---
    success = session.pop("upload_success", False)

    # --- Render Page ---
    return render_template(
        "results.html",
        resumes=resumes,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        all_tags=all_tags,
        all_titles=all_titles,
        search=search,
        tag_filter=tag_filter,
        job_filter=job_filter,
        sort_by=sort_by,
        success=success,
        active_page='results'
    )
@app.route("/export_excel")
def export_excel():
    job_filter = request.args.get("job_filter", "")
    tag_filter = request.args.get("tag", "")
    sort_by = request.args.get("sort_by", "uploaded_at")

    query = {}
    if job_filter:
        query["job_title"] = job_filter
    if tag_filter:
        query["tag"] = tag_filter

    sort_order = [("uploaded_at", DESCENDING)]
    if sort_by == "score":
        sort_order = [("score", DESCENDING)]

    resumes = list(resumes_collection.find(query).sort(sort_order))

    data = []
    for r in resumes:
        uploaded = r.get("uploaded_at")
        data.append({
            "Name": r.get("name", "Unknown"),
            "Filename": r.get("filename", "N/A"),
            "Job Title": r.get("job_title", "Unknown"),
            "Score (%)": round(float(r.get("score", 0.0)), 2),
            "Tag": r.get("tag", "—"),
            "Matched Skills": ", ".join(r.get("matched_skills", [])),
            "Missing Skills": ", ".join(r.get("missing_skills", [])),
            "Uploaded At": uploaded.strftime('%Y-%m-%d %H:%M:%S') if isinstance(uploaded, datetime) else str(uploaded)
        })

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Filtered_Results')

    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="filtered_resume_results.xlsx",
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
@app.route('/clear_resumes', methods=['POST'])
def clear_resumes():
    if 'admin' not in session:
        return redirect(url_for('login'))

    resumes_collection.delete_many({})
    flash("🗑️ All resumes have been cleared successfully.", "success")
    return redirect(url_for('results'))



@app.route("/dashboard")
def dashboard():
    if "admin" not in session:
        return redirect(url_for("login"))

    tag_filter = request.args.get("tag")
    title_filter = request.args.get("job_title")

    query = {}
    if tag_filter:
        query["tag"] = tag_filter
    if title_filter:
        query["job_title"] = title_filter

    resumes = list(resumes_collection.find(query))
    total_resumes = len(resumes)

    all_titles = sorted(resumes_collection.distinct("job_title"))
    all_tags = sorted(resumes_collection.distinct("tag"))

    if total_resumes == 0:
        return render_template("dashboard.html", **{
            "total_resumes": 0,
            "average_score": 0,
            "tag_counts": {},
            "all_tags": all_tags,
            "all_titles": all_titles,
            "selected_tag": tag_filter,
            "selected_title": title_filter,
            "top_candidates": [],
            "shortlisted": [],
            "top_skills": [],
            "top_missing_skill": "N/A",
            "high_score_pct": 0,
            "recent": [],
            "stats": [],
            "dates": [],
            "counts": [],
            "now": datetime.now()
        })

    scores = [r.get("score", 0) for r in resumes]
    avg_score = round(sum(scores) / total_resumes, 2)
    high_score_pct = round(len([s for s in scores if s >= 80]) / total_resumes * 100, 1)

    today = datetime.utcnow()
    last_30_days = [(today - timedelta(days=i)).strftime('%Y-%m-%d') for i in reversed(range(30))]
    trend_counter = Counter(
        r["uploaded_at"].strftime('%Y-%m-%d')
        for r in resumes if isinstance(r.get("uploaded_at"), datetime)
    )
    dates = last_30_days
    counts = [trend_counter.get(date, 0) for date in last_30_days]

    top_candidates = sorted([
        {
            "name": r.get("name", "N/A"),
            "score": r.get("score", 0),
            "tag": r.get("tag", "Untagged"),
            "job_title": r.get("job_title", "N/A"),
            "uploaded": r.get("uploaded_at", datetime.now()).strftime('%Y-%m-%d'),
            "matched_skills": r.get("matched_skills", [])
        }
        for r in resumes
    ], key=lambda r: r["score"], reverse=True)[:10]

    shortlisted = [
        {
            "name": r.get("name", "N/A"),
            "score": r.get("score", 0),
            "job_title": r.get("job_title", "N/A"),
            "email": extract_email(r.get("resume_text", ""))
        }
        for r in resumes
        if r.get("tag") == "Shortlisted" and r.get("score", 0) >= 85
    ]

    all_matched_skills = [skill for r in resumes for skill in r.get("matched_skills", [])]
    skill_counter = Counter(all_matched_skills)
    most_common_skills = skill_counter.most_common(5)
    max_skill_count = most_common_skills[0][1] if most_common_skills else 1
    top_skills = [
        {
            "name": skill,
            "count": count,
            "percent": int((count / max_skill_count) * 100)
        } for skill, count in most_common_skills
    ]

    all_missing = [s for r in resumes for s in r.get("missing_skills", [])]
    top_missing_skill = Counter(all_missing).most_common(1)[0][0] if all_missing else "N/A"

    recent_uploads = sorted(
        [r for r in resumes if isinstance(r.get("uploaded_at"), datetime)],
        key=lambda r: r["uploaded_at"], reverse=True
    )[:4]
    recent = [
        {
            "message": f"New resume uploaded by {r.get('name', 'Unknown')}",
            "time": r["uploaded_at"].strftime('%b %d %Y %H:%M')
        } for r in recent_uploads
    ]

    stats = [
        {"label": "Total Resumes", "value": total_resumes},
        {"label": "Top Candidates (Score > 85)", "value": len([s for s in scores if s > 85])},
        {"label": "Average Resume Score", "value": avg_score},
        {"label": "Needing Review (Score < 50)", "value": len([s for s in scores if s < 50])},
        {"label": "Most Sought Job Title", "value": title_filter or "—"},
    ]

    return render_template("dashboard.html", 
    total_resumes=total_resumes,
    average_score=avg_score,
    tag_counts=Counter([...]),
    all_tags=all_tags,
    all_titles=all_titles,
    selected_tag=tag_filter,
    selected_title=title_filter,
    top_candidates=top_candidates,
    shortlisted_candidates=shortlisted,   # ✅ this must be passed
    top_skills=top_skills,
    top_missing_skill=top_missing_skill,
    high_score_pct=high_score_pct,
    recent=recent,
    stats=stats,
    dates=dates,
    counts=counts,
    now=datetime.now()
    )
# --- Email Extractor (Helper Function) ---

@app.route('/download/csv')
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Filename', 'Job Title', 'Score', 'Tag', 'Matched Skills', 'Missing Skills'])

    for r in resumes_collection.find():
        writer.writerow([
            r.get('name', 'N/A'),
            r.get('filename', 'N/A'),
            r.get('job_title', 'Unknown'),
            r.get('score', 0),
            r.get('tag', '—'),
            ', '.join(r.get('matched_skills', [])),
            ', '.join(r.get('missing_skills', []))
        ])
    output.seek(0)
    return send_file(
        io.BytesIO(output.read().encode()),
        download_name="resumes.csv",
        as_attachment=True,
        mimetype='text/csv'
    )
@app.route('/download/excel')
def download_excel():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Resumes"
    ws.append(['Name', 'Filename', 'Job Title', 'Score', 'Tag', 'Matched Skills', 'Missing Skills'])

    for r in resumes_collection.find():
        ws.append([
            r.get('name', 'N/A'),
            r.get('filename', 'N/A'),
            r.get('job_title', 'Unknown'),
            r.get('score', 0),
            r.get('tag', '—'),
            ', '.join(r.get('matched_skills', [])),
            ', '.join(r.get('missing_skills', []))
        ])

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return send_file(
        out,
        download_name="resumes.xlsx",
        as_attachment=True,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # Replace this check with actual credential validation if needed
        if username == "admin" and password == "admin":
            session["admin"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")
@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("login"))

@app.route('/resume/<filename>')
@app.route('/uploads/<filename>')
def serve_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)