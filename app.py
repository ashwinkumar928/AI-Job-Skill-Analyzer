from pathlib import Path

import pandas as pd
import streamlit as st

# Keep the dashboard available before the new dependency is installed.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ModuleNotFoundError as error:
    if error.name != "sklearn":
        raise
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="AI Job Skill Analyzer", page_icon="📊", layout="wide")


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
}


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
        skill = " ".join(skill.lower().split())
        if skill and skill not in skills:
            skills.append(skill)
    return skills


def count_values(column):
    # Exclude missing and blank labels from metrics and charts.
    return column.dropna().astype(str).str.strip().replace("", pd.NA).value_counts()


def show_chart(title, counts):
    with st.container(border=True):
        st.subheader(title)
        if counts.empty:
            st.info("No data available.")
        else:
            st.bar_chart(counts.rename("Jobs"), horizontal=True, sort=False)


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
    combined_text = job_text.agg(" ".join, axis=1).str.lower()
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


# Resolve the CSV relative to this file, even when launched from another folder.
data_path = Path(__file__).parent / "data" / "jobs.csv"
try:
    df = pd.read_csv(data_path)
except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
    st.error(f"Could not load data/jobs.csv: {error}")
    st.stop()

required_columns = {"job_title", "company", "location", "skills"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(sorted(missing_columns)))
    st.stop()

# Keep df unchanged for the dataset overview. Calculate summaries separately.
skill_lists = df["skills"].apply(normalize_skills)
skill_counts = skill_lists.explode().dropna().value_counts()
# Rename a separate Series for presentation without changing comparison values.
display_skill_counts = skill_counts.rename(index=display_skill)
company_counts = count_values(df["company"])
role_counts = count_values(df["job_title"])
location_counts = count_values(df["location"])

with st.sidebar:
    st.title("Skill Analyzer")
    st.caption("Job market insights")
    page = st.radio("Navigation", ["Home", "Market Analysis", "Job Matcher", "Dataset Overview"])
    st.divider()
    st.caption("Source: data/jobs.csv")
    st.caption("Insights reflect the loaded dataset, not the entire job market.")

if page == "Home":
    st.title("AI-Powered Job Market Skill Analyzer")
    st.write("Analyze job market trends, discover in-demand skills, and identify skill gaps.")
    st.divider()

    columns = st.columns(4)
    metrics = [
        ("Total Jobs", len(df)),
        ("Total Companies", len(company_counts)),
        ("Total Locations", len(location_counts)),
        ("Most Demanded Skill", display_skill_counts.index[0] if not skill_counts.empty else "N/A"),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.divider()
    show_chart("Top 10 demanded skills", display_skill_counts.head(10))
    left, right = st.columns(2)
    with left:
        show_chart("Top job roles", role_counts.head(10))
    with right:
        show_chart("Jobs by location", location_counts)
    st.caption("Use Job Matcher in the sidebar to compare your skills with job requirements.")

elif page == "Market Analysis":
    st.title("Market Analysis")
    st.write("Explore the skills, roles, and locations represented in your dataset.")
    st.caption("Each skill is counted once per job. Tied counts share the same demand level.")
    st.divider()
    show_chart("Top demanded technical skills", display_skill_counts.head(10))
    left, right = st.columns(2)
    with left:
        show_chart("Most common job roles", role_counts.head(10))
    with right:
        show_chart("Jobs by location", location_counts)

    st.subheader("Skill demand percentage")
    st.caption("Percentage of all job postings that list each skill. Percentages need not sum to 100%.")
    if skill_counts.empty:
        st.info("No listed skills available.")
    else:
        demand = display_skill_counts.rename_axis("Skill").reset_index(name="Jobs")
        demand["Demand (%)"] = (demand["Jobs"] / len(df) * 100).round(1)
        st.dataframe(demand, hide_index=True, width="stretch")

elif page == "Job Matcher":
    st.title("Job Matcher")
    st.write("Enter comma-separated skills to find matching jobs and see what you could learn next.")
    if not SKLEARN_AVAILABLE:
        st.warning("AI similarity is unavailable. Results use exact skill matching until you install scikit-learn and restart the app.")
        st.code(r".\venv\Scripts\python.exe -m pip install scikit-learn", language="powershell")
    with st.form("skill_matcher"):
        skill_input = st.text_input("Enter your skills", placeholder="Python, SQL, React")
        analyze = st.form_submit_button("Analyze", type="primary")

    if analyze:
        user_skills = normalize_skills(skill_input)
        if not user_skills:
            st.warning("Please enter at least one skill, such as Python or SQL.")
        elif df.empty:
            st.info("There are no jobs to match in the dataset.")
        else:
            results = match_jobs(df, user_skills)
            best = results[0]
            st.caption("Your skills: " + display_skills(user_skills))
            st.divider()
            st.subheader("Best match summary")
            columns = st.columns(4)
            has_match = best["ai_similarity"] > 0 or best["percentage"] > 0
            columns[0].metric("Best Matching Role", best["title"] if has_match else "No match")
            columns[1].metric("Skill Match Score", f"{best['percentage']:.1f}%")
            columns[2].metric("AI Similarity Score", f"{best['ai_similarity']:.1f}%" if SKLEARN_AVAILABLE else "Unavailable")
            columns[3].metric("Missing Skill Count", len(best["missing"]) if has_match and best["has_requirements"] else "N/A")
            if has_match:
                ranking = "AI similarity, then skill match" if SKLEARN_AVAILABLE else "skill match only"
                st.caption(f"{best['company']} | {best['location']}. Ranked by {ranking}.")
            else:
                st.info("No matches were found using the available scoring methods. Review the requirements below.")

            st.divider()
            st.subheader("Top 5 matching jobs")
            st.caption("Skill Match = matched required skills / total required skills × 100. AI Similarity = TF-IDF cosine similarity × 100 across title, description, and skills. Neither score is a hiring probability.")
            for rank, job in enumerate(results[:5], start=1):
                with st.container(border=True):
                    details, score = st.columns([3, 1])
                    details.subheader(f"{rank}. {job['title']}")
                    details.write(f"{job['company']} | {job['location']}")
                    score.metric("Skill Match Score", f"{job['percentage']:.1f}%")
                    score.metric("AI Similarity Score", f"{job['ai_similarity']:.1f}%" if SKLEARN_AVAILABLE else "Unavailable")
                    if SKLEARN_AVAILABLE:
                        st.progress(job["ai_similarity"] / 100, text="AI Similarity")
                    else:
                        st.progress(job["percentage"] / 100, text="Skill Match")
                    left, right = st.columns(2)
                    left.write("**Matched skills**")
                    left.write(display_skills(job["matched"]) or "None")
                    right.write("**Missing skills**")
                    right.write(display_skills(job["missing"]) or ("None" if job["has_requirements"] else "Unknown"))
                    if not job["has_requirements"]:
                        st.caption("No requirements listed; Skill Match defaults to 0%. AI Similarity can still use the title and description.")

elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("Inspect the source data and check its completeness before interpreting results.")
    columns = st.columns(4)
    columns[0].metric("Rows", len(df))
    columns[1].metric("Columns", len(df.columns))
    columns[2].metric("Missing Values", int(df.isna().sum().sum()))
    columns[3].metric("Duplicate Rows", int(df.duplicated().sum()))
    st.caption("Duplicates are exact repeated rows after the first occurrence; they remain in the analysis.")
    st.divider()
    st.subheader("Full dataset")
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Column names")
    st.write(", ".join(df.columns))
    st.subheader("Missing values by column")
    st.dataframe(
        df.isna().sum().rename_axis("Column").reset_index(name="Missing Values"),
        hide_index=True,
        width="stretch",
    )
    st.caption("Missing values use Pandas isna(); whitespace-only cells are not counted as missing.")
    if df.duplicated().any():
        st.subheader("Duplicate rows (additional occurrences)")
        st.dataframe(df[df.duplicated()], hide_index=True, width="stretch")
