# AI-Powered Job Market Skill Analyzer

## Overview

A Python application that analyzes job-market data and recommends relevant job roles based on user skills. The interactive Streamlit dashboard combines descriptive analytics, exact skill matching, and TF-IDF cosine similarity to help users explore opportunities and identify skill gaps.

The included dataset contains **127 synthetic job postings**, **13 roles**, **18 fictional companies**, and **9 Indian locations**. It is a demonstration dataset, not a collection of live vacancies or evidence of actual market demand.

## Features

- Job market analysis: total jobs, companies, locations, and top roles
- Skill demand analysis: top 10 skills and percentage of postings requiring each skill
- Role recommendations: best matching role and top five job postings
- Skill gap analysis: matched and missing requirements for each recommendation
- Exact skill matching with case and whitespace normalization
- TF-IDF based text representation
- Cosine similarity based recommendations
- Interactive Streamlit dashboard with Home, Market Analysis, Job Matcher, and Dataset Overview pages
- Data validation, missing-value handling, and duplicate removal before analysis

## Tech Stack

- **Python** — application logic and data generation
- **Pandas** — loading, cleaning, and analyzing data
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **Streamlit** — interactive dashboard

## Project Workflow

```text
Job Dataset
→ Data Cleaning
→ Skill Analysis
→ User Skills
→ Exact Skill Matching
→ TF-IDF
→ Cosine Similarity
→ Job Recommendations
→ Skill Gap Analysis
```

## Dataset and Data Validation

`data/jobs.csv` uses five columns:

| Column | Contents |
| --- | --- |
| job_title | Role name |
| company | Fictional employer |
| location | Indian city |
| job_description | Responsibilities, project context, and experience expectations |
| skills | Comma-separated technical requirements |

Roles include AI Analyst, Data Analyst, Business Analyst, Machine Learning Engineer, Data Scientist, Software Engineer, Frontend Developer, Backend Developer, Full Stack Developer, Java Developer, Python Developer, Cloud Engineer, and DevOps Engineer.

Locations include Bangalore, Hyderabad, Pune, Mumbai, Delhi, Noida, Chennai, Gurugram, and Kolkata. Posting counts and optional skills vary between roles.

Before any analysis, the app validates the schema, trims whitespace, handles missing cells, canonicalizes common skill aliases (for example, `NodeJS` becomes `Node.js`), and removes duplicate postings using normalized comparison values. Skill order and repeated skills do not create distinct postings. Readable display labels are preserved. Duplicate removal is reported when applicable and counted in Dataset Overview.

Unknown descriptions and skills remain empty; missing titles display as `Unknown role`. Blank companies and locations are excluded from their category counts. Jobs with unknown skills remain in the total-job denominator, but have no counted requirements and receive a 0% exact skill match. All charts, metrics, recommendations, and the overview use the cleaned dataset. Each skill counts at most once per job, and demand percentages use all cleaned postings as their denominator.

To reproduce the synthetic dataset deterministically:

```bash
python scripts/generate_dataset.py
```

This overwrites `data/jobs.csv` with the demo dataset. Generation uses Python's standard library and requires no API keys.

## Matching Logic

Exact matching compares normalized user skills with the unique required skills in each posting:

```text
Skill Match (%) = matched skills / required skills * 100
```

For example, entering `Python, SQL` for a job requiring `Python, SQL, Pandas, Machine Learning` gives a **50%** match. Pandas and Machine Learning appear as missing skills. Extra user skills do not reduce the percentage. An empty requirement list receives 0%, with requirements marked as unknown.

## TF-IDF

TF-IDF converts text into numerical vectors. Words frequent within one document but less common across the collection receive more weight. Here, each job's title, description, and skills are combined into a document; user skills form the query document. Jobs and the query are fitted together so they share a vocabulary and weighting scheme, including query-only words.

This is a lexical baseline: it recognizes shared words, not semantic meaning or experience. The default tokenizer splits punctuation and does not preserve every multiword skill as a single unit. Exact matching separately compares complete normalized skill names.

## Cosine Similarity

Cosine similarity compares the direction of the user's TF-IDF vector with each job document's vector. A larger score indicates greater weighted textual overlap with the job's title, description, and skills. Scores are multiplied by 100 for display.

Recommendations rank by cosine similarity first, with exact skill match percentage breaking ties. Both scores are shown alongside matched and missing skills. Neither score is a hiring probability. If scikit-learn is unavailable, the app retains exact-match ranking and clearly labels similarity as unavailable.

## Installation

Use Python 3.12 (the tested version). In Windows Command Prompt:

```bat
git clone https://github.com/ashwinkumar928/AI-Job-Skill-Analyzer.git
cd AI-Job-Skill-Analyzer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

In PowerShell, activate with `.\venv\Scripts\Activate.ps1`. On macOS/Linux, use `source venv/bin/activate`. Open the local URL printed by Streamlit.

## Testing Recommendations

Open **Job Matcher**, enter comma-separated skills, and select **Analyze**. These examples should favor the following role families; individual postings can rank differently based on their requirements and descriptions:

| Input | Relevant role families |
| --- | --- |
| Python, SQL | Python Developer, AI Analyst, Data Analyst, Data Scientist |
| React, JavaScript | Frontend Developer, Full Stack Developer |
| Java, SQL | Java Developer, Software Engineer |
| Python, Pandas, Machine Learning | AI Analyst, Data Scientist, Machine Learning Engineer |
| AWS, Docker, Node.js | Backend Developer, DevOps Engineer, Cloud Engineer, Full Stack Developer |

Check that matched skills occur in your input, missing skills explain the gap, and the exact percentage follows the formula. Blank input should request a skill; an unknown term such as `unlistedskillxyz` should yield no positive matches.

Run automated checks for dataset coverage, cleaning, duplicate removal, normalization, scoring, the five recommendation examples, missing requirements, and fallback behavior:

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```text
app.py                       Streamlit pages and presentation
analysis.py                  Cleaning, normalization, and recommendation logic
data/jobs.csv                Synthetic job postings
scripts/generate_dataset.py  Reproducible dataset generator
tests/test_analysis.py       Data and matching regression tests
tests/test_dashboard.py      Dashboard navigation and form smoke test
requirements.txt            Python dependencies
README.md                    Documentation
```

## Live Demo

[Open the Streamlit dashboard](https://ai-job-skill-analyzer-wqzumg8xfcwiifstaejwtu.streamlit.app/)

## Future Improvements

- Real job APIs for current vacancies
- Larger real-world datasets with source and freshness tracking
- Semantic embeddings for matching beyond shared words
- Personalized recommendations using interests and experience
- Resume parsing to extract user skills automatically
