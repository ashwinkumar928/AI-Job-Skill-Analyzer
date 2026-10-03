from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import (
    SKLEARN_AVAILABLE, clean_jobs, count_values, display_skill,
    display_skills, match_jobs, normalize_skills,
)

st.set_page_config(page_title="AI Job Skill Analyzer", page_icon="📊", layout="wide")


def show_chart(title, counts):
    with st.container(border=True):
        st.subheader(title)
        if counts.empty:
            st.info("No data available.")
        else:
            st.bar_chart(counts.rename("Jobs"), horizontal=True, sort=False)


# Resolve the CSV relative to this file, even when launched from another folder.
data_path = Path(__file__).parent / "data" / "jobs.csv"
try:
    df = pd.read_csv(data_path)
except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
    st.error(f"Could not load data/jobs.csv: {error}")
    st.stop()

try:
    df, removed_duplicates = clean_jobs(df)
except ValueError as error:
    st.error(str(error))
    st.stop()

if removed_duplicates:
    st.info(f"Removed {removed_duplicates} duplicate job record(s) before analysis.")

# All charts, metrics, recommendations, and the overview use cleaned data.
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
    st.caption("Synthetic sample jobs and fictional companies for demonstration.")
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
    columns[2].metric("Missing Values", int(df.eq("").sum().sum()))
    columns[3].metric("Duplicates Removed", removed_duplicates)
    st.caption("Cleaned data: duplicate comparisons ignore case, spacing, and skill order. Blank companies and locations are excluded from category counts.")
    st.divider()
    st.subheader("Full dataset")
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Column names")
    st.write(", ".join(df.columns))
    st.subheader("Missing values by column")
    st.dataframe(
        df.eq("").sum().rename_axis("Column").reset_index(name="Missing Values"),
        hide_index=True,
        width="stretch",
    )
    st.caption("Missing values count empty cleaned cells. Missing titles display as Unknown role; unknown requirements receive a 0% exact match.")
