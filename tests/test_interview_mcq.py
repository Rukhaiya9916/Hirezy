import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import gemini_service
from main import app


def make_questions():
    return [
        {
            "category": "Technical",
            "topic": f"Skill {index}",
            "question": f"Question {index} grounded in the supplied role?",
            "options": ["First", "Second", "Third", "Fourth"],
            "correct_index": index % 4,
            "correct_answer": ["First", "Second", "Third", "Fourth"][index % 4],
            "explanation": "This choice best addresses the role requirement."
        }
        for index in range(10)
    ]


class TestInterviewMCQ(unittest.TestCase):
    def test_generator_uses_context_and_validates_answer_keys(self):
        with patch(
            "gemini_service.call_gemini_json",
            return_value={"questions": make_questions()}
        ) as generate:
            result = gemini_service.generate_interview_mcq(
                "Resume evidence: Python and FastAPI",
                "Backend Engineer",
                "Build Python APIs",
                ["Python"],
                ["Docker"]
            )

        prompt = generate.call_args.args[0]
        self.assertIn("Python and FastAPI", prompt)
        self.assertIn("Backend Engineer", prompt)
        self.assertEqual(len(result["questions"]), 10)
        self.assertEqual(result["questions"][0]["correct_index"], 0)
        self.assertEqual(result["questions"][0]["correct_answer"], "First")

    def test_generator_does_not_create_fallback_questions(self):
        with patch("gemini_service.call_gemini_json", return_value=None):
            with self.assertRaises(RuntimeError):
                gemini_service.generate_interview_mcq(
                    "Resume context",
                    "Backend Engineer"
                )

    def test_generator_rejects_ambiguous_answer_key(self):
        questions = make_questions()
        questions[0]["options"] = ["Same", "Same", "Other", "Fourth"]
        with patch(
            "gemini_service.call_gemini_json",
            return_value={"questions": questions}
        ):
            with self.assertRaises(ValueError):
                gemini_service.generate_interview_mcq(
                    "Resume context",
                    "Backend Engineer"
                )

    def test_mcq_api_returns_questions_and_surfaces_ai_unavailability(self):
        with TestClient(app) as client:
            with patch(
                "main.generate_interview_mcq",
                return_value={"questions": make_questions()}
            ):
                response = client.post(
                    "/api/interview-mcq",
                    json={
                        "resume_text": "Resume context",
                        "job_title": "Backend Engineer",
                        "job_description": "Build Python APIs",
                        "matching_skills": ["Python"]
                    }
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["questions"]), 10)
            self.assertEqual(response.json()["questions"][0]["correct_index"], 0)
            self.assertEqual(response.json()["questions"][0]["correct_answer"], "First")

            with patch(
                "main.generate_interview_mcq",
                side_effect=RuntimeError("AI generation unavailable.")
            ):
                unavailable = client.post(
                    "/api/interview-mcq",
                    json={
                        "resume_text": "Resume context",
                        "job_title": "Backend Engineer"
                    }
                )

            self.assertEqual(unavailable.status_code, 503)
            self.assertEqual(unavailable.json()["detail"], "AI generation unavailable.")


if __name__ == "__main__":
    unittest.main()
