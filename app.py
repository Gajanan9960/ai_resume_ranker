import os
import io
import csv
import math
from math import ceil
from datetime import datetime, timedelta
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify, send_from_directory
from flask_mail import Mail, Message
import pymongo
from werkzeug.security import generate_password_hash, check_password_hash
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
app.secret_key = "supersecretkey"  # Change this in production
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# --- Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')
mail = Mail(app)

# --- MongoDB Setup ---
# MongoDB Atlas connection
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["resume_db"]
resumes_collection = db["resumes"]
users_collection = db["users"]

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
    """Extract technical skills with fuzzy matching and case-insensitive search."""
    found = []
    text_lower = text.lower()
    
    for word in keywords:
        word_lower = word.lower().strip()
        # Check for exact match or word boundaries
        if word_lower in text_lower:
            found.append(word)
    return found

def extract_keywords_from_text(text):
    """Extract important keywords from resume text using spaCy."""
    doc = nlp(text[:5000])  # Limit to first 5000 chars for performance
    
    # Extract nouns, proper nouns, and technical terms
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            keywords.append(token.text)
    
    # Extract entities (organizations, technologies, etc.)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'SKILL']:
            keywords.append(ent.text)
    
    return list(set(keywords))

def semantic_similarity(text1, text2):
    """Calculate semantic similarity between two texts."""
    # Truncate texts to avoid memory issues
    text1 = text1[:2000]
    text2 = text2[:2000]
    
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score * 100, 2)

def keyword_density(text, keywords):
    """Calculate keyword density in text."""
    if not text or not keywords:
        return 0
    total = len(keywords)
    matches = sum(1 for word in keywords if word.lower() in text.lower())
    return matches / total if total > 0 else 0

def weighted_resume_score(text, job_keywords, job_desc):
    """
    Calculate weighted resume score based on multiple factors.
    Returns: (score, matched_skills, tag, missing_skills)
    """
    if not text or not job_desc:
        return 25.0, [], "Rejected", job_keywords
    
    # If no keywords provided, extract from job description
    if not job_keywords:
        job_keywords = extract_keywords_from_text(job_desc)[:20]  # Top 20 keywords
    
    # Extract matched skills
    matched_skills = extract_technical_skills(text, job_keywords)
    unique_matches = list(set(matched_skills))
    
    # Calculate keyword match score (40% weight)
    keyword_match_ratio = len(unique_matches) / len(job_keywords) if job_keywords else 0
    keyword_score = keyword_match_ratio * 40
    
    # Calculate semantic similarity (40% weight)
    similarity = semantic_similarity(text, job_desc)
    semantic_score = (similarity / 100) * 40
    
    # Calculate keyword density bonus (10% weight)
    density = keyword_density(text, job_keywords)
    density_score = density * 10
    
    # Resume length and quality bonus (10% weight)
    word_count = len(text.split())
    length_score = 0
    if 300 <= word_count <= 2000:
        length_score = 10
    elif 200 <= word_count < 300 or 2000 < word_count <= 3000:
        length_score = 7
    else:
        length_score = 5
    
    # Calculate total score
    total_score = keyword_score + semantic_score + density_score + length_score
    
    # Apply bonuses and penalties
    if keyword_match_ratio >= 0.7:
        total_score *= 1.1  # 10% bonus for high keyword match
    elif keyword_match_ratio < 0.2:
        total_score *= 0.85  # 15% penalty for low keyword match
    
    # Ensure score is within bounds
    final_score = min(max(total_score, 30), 98)
    final_score = round(final_score, 2)
    
    # Determine tag based on score
    if final_score >= 80:
        tag = "Top Candidate"
    elif final_score >= 65:
        tag = "Shortlisted"
    elif final_score >= 45:
        tag = "Review"
    else:
        tag = "Rejected"
    
    # Get top matched skills (limit to 10)
    skill_counts = Counter(matched_skills)
    top_skills = [skill for skill, _ in skill_counts.most_common(10)]
    
    # Calculate missing skills
    missing_skills = list(set(job_keywords) - set(top_skills))[:10]  # Limit to 10
    
    return final_score, top_skills, tag, missing_skills


# ------------------ AUTH ROUTES ------------------
# ------------------ AUTH ROUTES ------------------
@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if users_collection.find_one({"email": email}):
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.utcnow()
        })
        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = users_collection.find_one({"email": email})

        if user and check_password_hash(user["password"], password):
            session["user_id"] = str(user["_id"])
            session["username"] = user["username"]
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
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

            # --- Extract Email ---
            email = extract_email(resume_text)
            
            # --- Save to MongoDB ---
            resumes_collection.insert_one({
                "name": os.path.splitext(filename)[0].title(),
                "filename": filename,
                "email": email,
                "job_title": job_title,
                "job_keywords": job_keywords,
                "score": float(score),
                "tag": tag,
                "matched_skills": matched_skills,
                "missing_skills": missing_skills,
                "uploaded_at": datetime.utcnow(),
                "resume_text": resume_text,
                "user_id": session.get("user_id")
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
    if "user_id" not in session:
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
@app.route("/send_email/<resume_id>", methods=["POST"])
def send_email_to_candidate(resume_id):
    """Send tailored email to candidate."""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    from bson import ObjectId
    resume = resumes_collection.find_one({"_id": ObjectId(resume_id)})
    
    if not resume:
        flash("Resume not found.", "danger")
        return redirect(url_for("dashboard"))
    
    email = resume.get("email", "N/A")
    if email == "N/A":
        flash("No email found for this candidate.", "warning")
        return redirect(url_for("dashboard"))
    
    # Get tailored message based on tag
    tag = resume.get("tag", "Review")
    name = resume.get("name", "Candidate")
    job_title = resume.get("job_title", "the position")
    
    email_templates = {
        "Top Candidate": f"""Dear {name},

Congratulations! We are thrilled to inform you that you have been selected for the {job_title} position at our company.

Your impressive skills and experience stood out among the applicants, and we believe you will be a fantastic addition to our team.

We would like to schedule an onboarding meeting to discuss the next steps. Please let us know your availability for the coming week.

Welcome aboard!

Best regards,
Recruitment Team""",
        
        "Shortlisted": f"""Dear {name},

Congratulations! We are pleased to inform you that your application for the {job_title} position has been shortlisted.

We were impressed with your qualifications and would like to invite you for the next round of interviews. We will be sending you a calendar invite shortly.

Best regards,
Recruitment Team""",
        
        "Review": f"""Dear {name},

Thank you for your interest in the {job_title} position. Your application is currently under active review by our hiring team.

We appreciate your patience and will notify you as soon as a decision has been made.

Best regards,
Recruitment Team""",
        
        "Rejected": f"""Dear {name},

Thank you for giving us the opportunity to review your application for the {job_title} position.

Although your qualifications are impressive, we have decided to move forward with other candidates who more closely match our current specific needs for this role.

We will keep your resume on file for future openings that may be a better fit. We wish you the very best in your job search.

Best regards,
Recruitment Team"""
    }
    
    message = email_templates.get(tag, email_templates["Review"])
    subject = f"Application Update - {job_title}"

    try:
        if not app.config['MAIL_USERNAME'] or not app.config['MAIL_PASSWORD']:
            raise Exception("SMTP credentials not configured")

        msg = Message(subject, recipients=[email])
        msg.body = message
        mail.send(msg)
        flash(f"Email successfully sent to {name}!", "success")
    except Exception as e:
        print(f"Email error: {e}")
        # Fallback to mailto if backend sending fails
        import urllib.parse
        body = urllib.parse.quote(message)
        subject_enc = urllib.parse.quote(subject)
        mailto_link = f"mailto:{email}?subject={subject_enc}&body={body}"
        flash(f"Could not send directly (Error: {e}). Opening email client...", "warning")
        return redirect(mailto_link)

    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
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

    # --- Sort by Score (High to Low) ---
    resumes.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # --- Top 10 for Display ---
    display_resumes = resumes[:10]

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
    
    # Add humanized time to all resumes
    for r in resumes:
        r["uploaded_str"] = humanize_time(r.get("uploaded_at"))

    # --- Render Dashboard ---
    return render_template(
        "dashboard.html",
        resumes=display_resumes,  # Only pass top 10
        total_resumes=total_resumes,
        average_score=avg_score,
        high_score_pct=high_score_pct,
        tag_counts=tag_counts,
        tag_filter=tag_filter,
        title_filter=title_filter,
        all_tags=all_tags,
        all_titles=all_titles,
        top_skills=top_skills,
        recent=recent,
        now=datetime.now()
    )
# ------------------ OTHER ROUTES ------------------
@app.route("/clear_resumes", methods=["POST"])
def clear_resumes():
    if "user_id" not in session:
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