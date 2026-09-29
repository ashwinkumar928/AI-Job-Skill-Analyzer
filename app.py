import streamlit as st
import pandas as pd


def normalize_skills(skill_text):
    # Missing skill values should become an empty list.
    if pd.isna(skill_text):
        return []

    normalized_skills = []
    for skill in skill_text.split(","):
        # Ignore capitalization and extra spaces when comparing skills.
        skill = " ".join(skill.lower().split())
        if skill and skill not in normalized_skills:
            normalized_skills.append(skill)

    return normalized_skills


st.title("AI-Powered Job Market Skill Analyzer")

st.write(
    "Analyze job market trends, discover in-demand skills, "
    "and find jobs matching your technical skills."
)

# Load dataset
df = pd.read_csv("data/jobs.csv")

st.subheader("Job Market Dataset")

st.dataframe(df)

st.write("Total Jobs:", len(df))

skills = st.text_input(
    "Enter your skills",
    placeholder="Python, SQL, React, JavaScript"
)

if st.button("Analyze"):
    user_skills = normalize_skills(skills)

    if not user_skills:
        st.warning("Please enter at least one skill, such as Python or SQL.")
    else:
        st.write("Your normalized skills:", user_skills)
        results = []

        # Compare the user's skills with each job's unique requirements.
        for _, job in df.iterrows():
            required_skills = normalize_skills(job["skills"])
            matched_skills = []
            missing_skills = []

            for skill in required_skills:
                if skill in user_skills:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)

            # Avoid dividing by zero when a job has no listed skills.
            if required_skills:
                match_percentage = len(matched_skills) / len(required_skills) * 100
            else:
                match_percentage = 0.0

            results.append({
                "Job Title": job["job_title"],
                "Company": job["company"],
                "Location": job["location"],
                "Matched Skills": ", ".join(matched_skills) or "None",
                "Missing Skills": ", ".join(missing_skills) or "None",
                "Match Percentage (%)": round(match_percentage, 2),
            })

        st.subheader("Skill Matching Results")
        st.dataframe(pd.DataFrame(results), hide_index=True)
        st.caption(
            "Match percentage = matched required skills / total required skills × 100. "
            "Jobs with no listed skills receive 0% because their requirements are unknown."
        )
