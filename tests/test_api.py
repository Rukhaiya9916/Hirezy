import os
import unittest
from fastapi.testclient import TestClient

from main import app
from matching_engine import load_jobs, evaluate_job_match, find_top_matches
from resume_parser import extract_text_from_pdf
from database import init_db, record_match_analysis, get_analytics_summary

class TestSkillora(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        init_db()

    def test_root_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("HIREZY", response.text)
        self.assertIn("AI-Powered Career & Hiring Intelligence", response.text)

    def test_jobs_json_catalog(self):
        jobs = load_jobs()
        self.assertGreaterEqual(len(jobs), 15)
        first_job = jobs[0]
        for key in ["id", "title", "company", "location", "experience", "description", "required_skills", "preferred_skills"]:
            self.assertIn(key, first_job)

    def test_deterministic_matching(self):
        sample_resume = (
            "Experienced software developer skilled in Python, FastAPI, Docker, PostgreSQL, REST APIs, and Git. "
            "Built scalable microservices and database backends."
        )
        matches = find_top_matches(sample_resume, top_n=5)
        self.assertEqual(len(matches), 5)
        top_match = matches[0]
        self.assertEqual(top_match["title"], "Backend Developer")
        self.assertGreaterEqual(top_match["match_score"], 70)
        self.assertIn("Python", top_match["matching_skills"])
        self.assertIn("FastAPI", top_match["matching_skills"])
        self.assertIn("Docker", top_match["matching_skills"])

    def test_api_match_jobs_endpoint(self):
        payload = {
            "resume_text": "Proficient in JavaScript, React, HTML, CSS, REST APIs, Git."
        }
        response = self.client.post("/api/match-jobs", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["jobs"]), 5)
        self.assertIn(data["jobs"][0]["title"], ["Frontend Developer", "React Developer", "Full Stack Developer"])

    def test_api_ai_test_endpoint(self):
        response = self.client.get("/api/ai/test")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("success", data)

    def test_sqlite_analytics(self):
        record_match_analysis(
            job_title="Frontend Developer",
            match_score=85,
            matching_skills=["React", "JavaScript"],
            missing_skills=["TypeScript"]
        )
        summary = get_analytics_summary()
        self.assertGreaterEqual(summary["jobs_analyzed"], 1)
        self.assertGreater(summary["average_match_score"], 0)

    def test_api_analytics_endpoint(self):
        response = self.client.get("/api/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("jobs_analyzed", data)
        self.assertIn("average_match_score", data)

    def test_api_skill_proof_endpoint(self):
        response = self.client.post("/api/skill-proof", json={"skill": "Docker"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["skill"], "Docker")
        self.assertGreaterEqual(len(data["challenges"]), 3)

    def test_extract_resume_validation(self):
        response = self.client.post(
            "/api/extract-resume",
            files={"file": ("test.txt", b"plain text", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
