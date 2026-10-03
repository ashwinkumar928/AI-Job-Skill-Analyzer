"""Run with: python -m unittest discover -s tests -v"""
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from analysis import (
    SKLEARN_AVAILABLE, calculate_similarity, clean_jobs, count_values,
    match_jobs, normalize_skills,
)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "data" / "jobs.csv"
        cls.jobs, cls.removed = clean_jobs(pd.read_csv(path))

    def test_dataset_coverage(self):
        self.assertGreaterEqual(len(self.jobs), 100)
        self.assertEqual(self.removed, 0)
        self.assertEqual(self.jobs.job_title.nunique(), 13)
        self.assertEqual(self.jobs.location.nunique(), 9)
        self.assertFalse(self.jobs.eq("").any().any())

    def test_cleaning_and_duplicate_comparison(self):
        raw = pd.DataFrame([
            [" Data Analyst ", "Example Labs", "Pune", "Analyze data", "SQL, Python,SQL"],
            ["data analyst", "example labs", " PUNE ", " analyze   DATA ", "python,sql"],
            [None, None, "  ", None, None],
        ], columns=self.jobs.columns)
        cleaned, removed = clean_jobs(raw)
        self.assertEqual(removed, 1)
        self.assertEqual(cleaned.iloc[0].job_title, "Data Analyst")
        self.assertEqual(cleaned.iloc[0].skills, "SQL, Python")
        self.assertFalse(cleaned.isna().any().any())
        self.assertEqual(len(count_values(cleaned.company)), 1)
        self.assertEqual(len(count_values(cleaned.location)), 1)
        again, count = clean_jobs(cleaned)
        pd.testing.assert_frame_equal(cleaned, again)
        self.assertEqual(count, 0)

    def test_missing_schema(self):
        with self.assertRaisesRegex(ValueError, "job_description"):
            clean_jobs(self.jobs.drop(columns="job_description"))

    def test_skill_normalization(self):
        self.assertEqual(normalize_skills(" PYTHON, python, NodeJS, PowerBI, ,REST API"),
                         ["python", "node.js", "power bi", "rest apis"])

    def test_exact_match_and_gaps(self):
        jobs = self.jobs.iloc[:1].copy()
        jobs["skills"] = "Python, SQL, Pandas, Machine Learning"
        result = match_jobs(jobs, ["PYTHON", "SQL"])[0]
        self.assertEqual(result["percentage"], 50)
        self.assertEqual(result["matched"], ["python", "sql"])
        self.assertEqual(result["missing"], ["pandas", "machine learning"])

    def test_recommendation_examples(self):
        self.assertTrue(SKLEARN_AVAILABLE, "Install requirements.txt to test TF-IDF")
        examples = {
            "Python, SQL": {"Python Developer", "AI Analyst", "Data Analyst", "Data Scientist"},
            "React, JavaScript": {"Frontend Developer", "Full Stack Developer"},
            "Java, SQL": {"Java Developer", "Software Engineer"},
            "Python, Pandas, Machine Learning": {"AI Analyst", "Data Scientist", "Machine Learning Engineer"},
            "AWS, Docker, Node.js": {"Backend Developer", "DevOps Engineer", "Cloud Engineer", "Full Stack Developer"},
        }
        for query, expected in examples.items():
            with self.subTest(query=query):
                results = match_jobs(self.jobs, normalize_skills(query))
                exact_best = max(results, key=lambda job: job["percentage"])
                self.assertIn(results[0]["title"], expected)
                self.assertIn(exact_best["title"], expected)
                self.assertTrue(all(job["matched"] for job in results[:5]))
                self.assertTrue(all(0 <= job["ai_similarity"] <= 100 for job in results))
                print(f"{query}: TF-IDF -> {results[0]['title']}; exact -> {exact_best['title']}")

    def test_empty_unknown_and_missing_requirements(self):
        self.assertEqual(match_jobs(self.jobs.iloc[:0], ["python"]), [])
        for query in [[], ["unlistedskillxyz"]]:
            results = match_jobs(self.jobs, query)
            self.assertTrue(all(job["percentage"] == job["ai_similarity"] == 0 for job in results))
        blank = pd.DataFrame([["", "", "", "", ""]], columns=self.jobs.columns)
        self.assertEqual(calculate_similarity(blank, []), [0.0])
        result = match_jobs(blank, ["python"])[0]
        self.assertFalse(result["has_requirements"])
        self.assertEqual(result["percentage"], 0)

    def test_exact_fallback_without_sklearn(self):
        with patch("analysis.SKLEARN_AVAILABLE", False):
            results = match_jobs(self.jobs, ["java", "sql"])
        self.assertTrue(all(job["ai_similarity"] == 0 for job in results))
        self.assertEqual(results[0]["percentage"], max(job["percentage"] for job in results))


if __name__ == "__main__":
    unittest.main()
