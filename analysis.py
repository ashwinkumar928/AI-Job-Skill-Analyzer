import pandas as pd


# Keep exact matching available if scikit-learn has not been installed.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    SKLEARN_AVAILABLE = False

# These labels affect display only; matching still uses lowercase skill names.
SKILL_DISPLAY_NAMES = {
    "sql": "SQL",
    "python": "Python",
    "javascript": "JavaScript",
    "node.js": "Node.js",
    "react": "React",
    "machine learning": "Machine Learning",
    "power bi": "Power BI",
    "rest apis": "REST APIs",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "scikit-learn": "Scikit-learn",
    "html": "HTML",
    "css": "CSS",
    "express.js": "Express.js",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "java": "Java",
    "git": "Git",
    "excel": "Excel",
    "aws": "AWS",
    "docker": "Docker",
    "typescript": "TypeScript",
    "ci/cd": "CI/CD",
    "spring boot": "Spring Boot",
    "fastapi": "FastAPI",
    "linux": "Linux",
    "kubernetes": "Kubernetes",
    "terraform": "Terraform",
    "tableau": "Tableau",
}


SKILL_ALIASES = {
    "nodejs": "node.js", "node js": "node.js",
    "powerbi": "power bi", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "rest api": "rest apis",
    "postgres": "postgresql", "reactjs": "react",
}


def normalize_text(value):
    return "" if pd.isna(value) else " ".join(str(value).split()).casefold()


def clean_jobs(raw):
    """Return clean display data and the number of duplicate postings removed."""
    columns = ["job_title", "company", "location", "job_description", "skills"]
    missing = set(columns) - set(raw.columns)
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(sorted(missing)))
    jobs = raw[columns].copy()
    for column in columns:
        jobs[column] = jobs[column].fillna("").astype(str).map(lambda text: " ".join(text.split()))
    jobs["skills"] = jobs["skills"].map(normalize_skills).map(display_skills)
    # Compare normalized copies; keep the first readable spelling for display.
    for column in ["job_title", "company", "location"]:
        labels = {}
        jobs[column] = jobs[column].map(
            lambda label: labels.setdefault(normalize_text(label), label)
        )
    keys = jobs.map(normalize_text)
    keys["skills"] = jobs["skills"].map(lambda text: ",".join(sorted(normalize_skills(text))))
    duplicates = keys.duplicated()
    jobs = jobs.loc[~duplicates].copy()
    # Keep unknown skills/descriptions empty rather than inventing requirements.
    jobs["job_title"] = jobs["job_title"].replace("", "Unknown role")
    return jobs.reset_index(drop=True), int(duplicates.sum())


def display_skill(skill):
    # Use title case as a simple fallback for skills outside the dictionary.
    return SKILL_DISPLAY_NAMES.get(skill, skill.title())


def display_skills(skills):
    return ", ".join(display_skill(skill) for skill in skills)


def normalize_skills(skill_text):
    # Use the same comparison rules for user input and job requirements.
    if pd.isna(skill_text):
        return []
    skills = []
    for skill in str(skill_text).split(","):
        skill = normalize_text(skill)
        skill = SKILL_ALIASES.get(skill, skill)
        if skill and skill not in skills:
            skills.append(skill)
    return skills


def count_values(column):
    # Exclude missing and blank labels from metrics and charts.
    return column.dropna().astype(str).str.strip().replace("", pd.NA).value_counts()


def calculate_similarity(jobs, user_skills):
    if jobs.empty:
        return []
    if not SKLEARN_AVAILABLE:
        # Preserve rule-based ranking until the optional dependency is installed.
        # The UI labels these unavailable scores rather than displaying zeros.
        return [0.0] * len(jobs)

    # Missing fields contribute empty text, not the literal word "nan".
    text_columns = ["job_title", "job_description", "skills"]
    job_text = jobs.reindex(columns=text_columns).fillna("").astype(str)
    combined_text = job_text.agg(" ".join, axis=1).map(normalize_text)
    query_text = " ".join(user_skills)

    # Fit jobs and query together so all vectors use the same vocabulary and IDF.
    # Query-only words are retained too, even when no job contains them.
    documents = combined_text.tolist() + [query_text]
    vectorizer = TfidfVectorizer()

    # The default tokenizer ignores punctuation and single-character words.
    # If nothing can be tokenized, there is no text similarity to calculate.
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(document) for document in documents):
        return [0.0] * len(jobs)

    tfidf_matrix = vectorizer.fit_transform(documents)
    job_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1:]
    similarities = cosine_similarity(query_vector, job_vectors).flatten()

    # Clamp tiny floating-point errors before converting 0-1 scores to 0-100.
    return (similarities.clip(0, 1) * 100).tolist()


def match_jobs(jobs, user_skills):
    user_skills = normalize_skills(",".join(user_skills))
    similarity_scores = calculate_similarity(jobs, user_skills)
    results = []
    # Position keeps each similarity attached to its job, even with custom indices.
    for position, (_, job) in enumerate(jobs.iterrows()):
        required = normalize_skills(job["skills"])
        # Matched requirements are present in the user's normalized list.
        matched = [skill for skill in required if skill in user_skills]
        # Missing requirements are the ones the user has not listed.
        missing = [skill for skill in required if skill not in user_skills]
        # An unknown requirement list receives zero, avoiding division by zero.
        percentage = len(matched) / len(required) * 100 if required else 0.0
        results.append({
            "title": job["job_title"],
            "company": job["company"],
            "location": job["location"],
            "matched": matched,
            "missing": missing,
            "percentage": percentage,
            "ai_similarity": similarity_scores[position],
            "has_requirements": bool(required),
        })
    # Sort by text similarity first, then exact skill overlap to break ties.
    # Keep full precision; completely tied jobs retain their dataset order.
    return sorted(
        results,
        key=lambda job: (job["ai_similarity"], job["percentage"]),
        reverse=True,
    )
