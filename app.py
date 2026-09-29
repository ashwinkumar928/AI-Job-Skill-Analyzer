from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Job Skill Analyzer", page_icon="📊", layout="wide")


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


def match_jobs(jobs, user_skills):
    results = []
    for _, job in jobs.iterrows():
        required = normalize_skills(job["skills"])
        matched = [skill for skill in required if skill in user_skills]
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
            "has_requirements": bool(required),
        })
    # Keep full precision while sorting; round only for display.
    return sorted(results, key=lambda job: job["percentage"], reverse=True)


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
        ("Most Demanded Skill", skill_counts.index[0] if not skill_counts.empty else "N/A"),
    ]
    for column, (label, value) in zip(columns, metrics):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.divider()
    show_chart("Top 10 demanded skills", skill_counts.head(10))
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
    show_chart("Top demanded technical skills", skill_counts.head(10))
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
        demand = skill_counts.rename_axis("Skill").reset_index(name="Jobs")
        demand["Demand (%)"] = (demand["Jobs"] / len(df) * 100).round(1)
        st.dataframe(demand, hide_index=True, width="stretch")

elif page == "Job Matcher":
    st.title("Job Matcher")
    st.write("Enter comma-separated skills to find matching jobs and see what you could learn next.")
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
            st.caption("Your normalized skills: " + ", ".join(user_skills))
            st.divider()
            st.subheader("Best match summary")
            columns = st.columns(4)
            has_match = best["percentage"] > 0
            columns[0].metric("Best Matching Role", best["title"] if has_match else "No match")
            columns[1].metric("Match Score", f"{best['percentage']:.1f}%")
            columns[2].metric("Matched Skill Count", len(best["matched"]) if has_match else 0)
            columns[3].metric("Missing Skill Count", len(best["missing"]) if has_match else "N/A")
            if has_match:
                st.caption(f"{best['company']} | {best['location']}. Tied scores retain dataset order.")
            else:
                st.info("None of the listed requirements match your skills. Review the gaps below.")

            st.divider()
            st.subheader("Top 5 matching jobs")
            st.caption("Score = matched required skills / total required skills × 100. This measures skill overlap, not hiring probability.")
            for rank, job in enumerate(results[:5], start=1):
                with st.container(border=True):
                    details, score = st.columns([3, 1])
                    details.subheader(f"{rank}. {job['title']}")
                    details.write(f"{job['company']} | {job['location']}")
                    score.metric("Match", f"{job['percentage']:.1f}%")
                    st.progress(job["percentage"] / 100)
                    left, right = st.columns(2)
                    left.write("**Matched skills**")
                    left.write(", ".join(job["matched"]) or "None")
                    right.write("**Missing skills**")
                    right.write(", ".join(job["missing"]) or ("None" if job["has_requirements"] else "Unknown"))
                    if not job["has_requirements"]:
                        st.caption("No requirements listed; score defaults to 0%.")

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
