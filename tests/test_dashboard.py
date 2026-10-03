"""Smoke-test the existing dashboard pages without opening a browser."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class DashboardTests(unittest.TestCase):
    def test_pages_and_recommendation_form(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py"),
            default_timeout=30,
        ).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.metric[0].value, "127")
        self.assertEqual(app.metric[1].value, "18")
        self.assertEqual(app.metric[2].value, "9")
        for page in ["Market Analysis", "Dataset Overview", "Job Matcher"]:
            app.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, page)
        app.text_input[0].set_value("React, JavaScript")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIn(app.metric[0].value, ["Frontend Developer", "Full Stack Developer"])
        app.text_input[0].set_value("")
        app.button[0].click().run()
        self.assertTrue(any("at least one skill" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
