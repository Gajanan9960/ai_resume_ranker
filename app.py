import os
import io
import csv
import math
from math import ceil
from datetime import datetime, timedelta
from collections import Counter
from flask import (
    Flask, request, render_template, redirect, url_for, flash,
    session, send_file, send_from_directory
)
from werkzeug.utils import secure_filename
from pymongo import MongoClient, DESCENDING, ASCENDING
from sentence_transformers import SentenceTransformer, util
from pdfminer.high_level import extract_text
import pandas as pd
import openpyxl
import spacy
from dotenv import load_dotenv

# ------------------ INITIAL SETUP ------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = "resume_secret_key"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# MongoDB Atlas connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["resume_db"]
resumes_collection = db["resumes"]

# NLP and Transformer models
nlp = spacy.load("en_core_web_lg")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ HELPER FUNCTIONS ------------------
def extract_text_from_pdf(file_path):
    """Extract text safely from a PDF file."""
    try:
        return extract_text(file_path)
    except Exception:
        return ""

def extract_email(text):
    """Extract an email address from resume text."""
    import re
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "N/A"

def humanize_time(dt):
    """Convert a datetime into a human-friendly relative time."""
    if not dt:
        return "Unknown"
    delta = datetime.utcnow() - dt
    if delta.days > 0:
        return f"{delta.days} days ago"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours} hours ago"
    minutes = delta.seconds // 60
    return f"{minutes} minutes ago"

def extract_technical_skills(text, keywords):
    found = []
    for word in keywords:
        if word.lower() in text.lower():
            found.append(word)
    return found

def semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)

def keyword_density(text, keywords):
    if not text:
        return 0
    total = len(keywords)
    matches = sum(word.lower() in text.lower() for word in keywords)
    return matches / total if total > 0 else 0

def weighted_resume_score(text, job_keywords, job_desc):
    sections = {
        'skills': text,
        'projects': text,
        'experience': text,
        'certifications': text,
        'education': text
    }

    weights = {
        'skills': 0.3,
        'projects': 0.2,
        'experience': 0.25,
        'certifications': 0.15,
        'education': 0.1
    }

    total_keywords = len(job_keywords) or 1
    total_score = 0
    matched_all = set()

    for section, weight in weights.items():
        matched = extract_technical_skills(sections[section], job_keywords)
        unique = set(matched)
        matched_all.update(unique)
        ratio = len(unique) / total_keywords
        density = keyword_density(sections[section], job_keywords)
        total_score += (0.6 * ratio + 0.4 * density) * weight

    sim = semantic_similarity(text, job_desc)
    sim_score = 1 / (1 + math.exp(-0.08 * (sim - 55)))
    total_score += sim_score * 0.2

    match_ratio = len(matched_all) / total_keywords
    if match_ratio < 0.3:
        total_score *= 0.7
    elif match_ratio < 0.5:
        total_score *= 0.85
    elif match_ratio > 0.8:
        total_score *= 1.05

    final_score = min(max(total_score * 100, 25), 95)
    final_score = round(final_score, 2)

    tag = (
        "Top Candidate" if final_score >= 85 else
        "Shortlisted" if final_score >= 70 else
        "Review" if final_score >= 50 else
        "Rejected"
    )

    all_matches = extract_technical_skills(text, job_keywords)
    top_skills = [s for s, _ in Counter(all_matches).most_common(10)]
    missing = list(set(job_keywords) - set(top_skills))

    return final_score, top_skills, tag, missing

# ------------------ AUTH ROUTES ------------------
@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    """Admin login page."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == "admin" and password == "admin":
            session["admin"] = True
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

@app.route("/logout")
def logout():
    """Logout admin session."""
    session.pop("admin", None)
    return redirect(url_for("login"))

# ------------------ UPLOAD ROUTE ------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    # Handle GET requests (render upload form)
    if request.method == "GET":
        return render_template("upload.html")

    # Handle POST requests (file uploads)
    try:
        uploaded_files = request.files.getlist("resume")
        job_desc = request.form.get("job_description", "").strip()
        job_title = request.form.get("job_role", "").strip() or "Unknown Role"

        # Optional keyword extraction (if user provides)
        keywords_raw = request.form.get("keywords", "")
        job_keywords = [kw.strip().lower() for kw in keywords_raw.split(",") if kw.strip()]

        # Validation
        if not uploaded_files or not job_desc:
            flash("Please upload at least one resume and provide a job description.", "danger")
            return redirect(url_for("upload"))  # ✅ fixed: stay on same page

        uploaded_count = 0

        for file in uploaded_files:
            if not file or file.filename == "":
                continue

            filename = secure_filename(file.filename)
            if not filename.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
                flash(f"❌ Skipped invalid file type: {filename}", "warning")
                continue

            # --- Save File ---
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # --- Extract Resume Text ---
            if filename.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(file_path)
            else:
                resume_text = extract_text_from_image(file_path)

            if not resume_text.strip():
                flash(f"⚠️ Could not extract text from {filename}. Skipped.", "warning")
                continue

            # --- Score Resume ---
            score, matched_skills, tag, missing_skills = weighted_resume_score(
                resume_text, job_keywords, job_desc
            )

            # --- Save to MongoDB ---
            resumes_collection.insert_one({
                "name": os.path.splitext(filename)[0].title(),
                "filename": filename,
                "job_title": job_title,
                "job_keywords": job_keywords,
                "score": float(score),
                "tag": tag,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "uploaded_at": datetime.utcnow(),
                "resume_text": resume_text
            })
            uploaded_count += 1

        # --- Feedback and Redirect ---
        if uploaded_count > 0:
            session["upload_success"] = True
            flash(f"✅ Successfully uploaded and analyzed {uploaded_count} resume(s).", "success")
        else:
            flash("⚠️ No valid resumes were processed.", "warning")

        return redirect(url_for("results"))  # ✅ only one redirect after success

    except Exception as e:
        print("Upload error:", e)
        flash(f"❌ Upload failed due to an unexpected error: {str(e)}", "danger")
        return redirect(url_for("upload"))  # ✅ fixed: redirect to upload, not results
# ------------------ RESULTS PAGE ------------------
@app.route("/results")
def results():
    if "admin" not in session:
        return redirect(url_for("login"))

    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    search = request.args.get("search", "").strip()
    tag_filter = request.args.get("tag", "").strip()
    job_filter = request.args.get("job_filter", "").strip()
    sort_by = request.args.get("sort_by", "uploaded_at")

    query = {}
    if search:
        query["$or"] = [
            {"filename": {"$regex": search, "$options": "i"}},
            {"matched_skills": {"$regex": search, "$options": "i"}},
            {"resume_text": {"$regex": search, "$options": "i"}}
        ]
    if tag_filter:
        query["tag"] = tag_filter
    if job_filter:
        query["job_title"] = job_filter

    sort_order = [("uploaded_at", DESCENDING)]
    if sort_by == "score":
        sort_order = [("score", DESCENDING)]

    total = resumes_collection.count_documents(query)
    skips = per_page * (page - 1)
    resumes = list(
        resumes_collection.find(query)
        .sort(sort_order)
        .skip(skips)
        .limit(per_page)
    )
    total_pages = ceil(total / per_page)

    for r in resumes:
        r["uploaded_str"] = humanize_time(r.get("uploaded_at"))
        r["score"] = round(float(r.get("score", 0)), 2)

    all_tags = sorted(filter(None, resumes_collection.distinct("tag")))
    all_titles = sorted(filter(None, resumes_collection.distinct("job_title")))

    success = session.pop("upload_success", False)

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
        active_page="results"
    )

# ------------------ EXPORT ROUTES ------------------
@app.route("/export_excel")
def export_excel():
    """Export filtered resumes as Excel file."""
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
            "Filename": r.get("filename", "N/A"),
            "Job Title": r.get("job_title", "Unknown"),
            "Score (%)": round(float(r.get("score", 0.0)), 2),
            "Tag": r.get("tag", "—"),
            "Matched Skills": ", ".join(r.get("matched_skills", [])),
            "Missing Skills": ", ".join(r.get("missing_skills", [])),
            "Uploaded At": uploaded.strftime("%Y-%m-%d %H:%M:%S") if isinstance(uploaded, datetime) else str(uploaded)
        })

    df = pd.DataFrame(data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Filtered_Results")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="filtered_resume_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/download/csv")
def download_csv():
    """Export all resumes as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Filename", "Job Title", "Score", "Tag", "Matched Skills", "Missing Skills"])
    for r in resumes_collection.find():
        writer.writerow([
            r.get("filename", "N/A"),
            r.get("job_title", "Unknown"),
            r.get("score", 0),
            r.get("tag", "—"),
            ", ".join(r.get("matched_skills", [])),
            ", ".join(r.get("missing_skills", []))
        ])
    output.seek(0)
    return send_file(
        io.BytesIO(output.read().encode()),
        download_name="resumes.csv",
        as_attachment=True,
        mimetype="text/csv"
    )

# ------------------ ADMIN DASHBOARD ------------------
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

    # Empty state
    if total_resumes == 0:
        return render_template(
            "dashboard.html",
            total_resumes=0,
            average_score=0,
            high_score_pct=0,
            tag_counts={},
            all_tags=[],
            all_titles=[],
            top_skills=[],
            now=datetime.now()
        )

    # --- Compute stats ---
    scores = [r.get("score", 0) for r in resumes]
    avg_score = round(sum(scores) / total_resumes, 2)
    high_score_pct = round(len([s for s in scores if s >= 80]) / total_resumes * 100, 1)

    # --- Tag & title stats ---
    tag_counts = Counter([r.get("tag", "Unknown") for r in resumes])
    all_tags = sorted(resumes_collection.distinct("tag"))
    all_titles = sorted(resumes_collection.distinct("job_title"))

    # --- Top 5 skills ---
    all_skills = [s for r in resumes for s in r.get("matched_skills", [])]
    skill_counter = Counter(all_skills)
    top_skills = skill_counter.most_common(5)

    # --- Recent uploads ---
    recent_uploads = sorted(
        [r for r in resumes if isinstance(r.get("uploaded_at"), datetime)],
        key=lambda r: r["uploaded_at"],
        reverse=True
    )[:5]

    recent = [
        {
            "message": f"New resume uploaded: {r.get('filename', 'N/A')}",
            "time": r["uploaded_at"].strftime("%b %d, %Y %H:%M")
        }
        for r in recent_uploads
    ]

    # --- Render Dashboard ---
    return render_template(
        "dashboard.html",
        total_resumes=total_resumes,
        average_score=avg_score,
        high_score_pct=high_score_pct,
        tag_counts=tag_counts,
        all_tags=all_tags,
        all_titles=all_titles,
        top_skills=top_skills,
        recent=recent,
        now=datetime.now()
    )
# ------------------ OTHER ROUTES ------------------
@app.route("/clear_resumes", methods=["POST"])
def clear_resumes():
    if "admin" not in session:
        return redirect(url_for("login"))
    resumes_collection.delete_many({})
    flash("All resumes cleared successfully.", "success")
    return redirect(url_for("results"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/uploads/<filename>")
def serve_resume(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)