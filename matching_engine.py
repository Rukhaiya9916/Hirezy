import json
import os
import re
from typing import List, Dict, Any, Tuple

# Predefined common aliases and variations to make matching robust and accurate
SKILL_ALIASES: Dict[str, List[str]] = {
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs", "node"],
    "python": ["python", "python3"],
    "sql": ["sql", "mysql", "sqlite", "tsql", "plsql"],
    "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "fastapi": ["fastapi", "fast-api"],
    "docker": ["docker", "containerization", "containers"],
    "git": ["git", "github", "gitlab"],
    "rest apis": ["rest api", "rest apis", "restful", "restful api", "restful apis", "rest"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks", "dl"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "ci/cd": ["ci/cd", "ci cd", "cicd", "continuous integration", "github actions"],
    "aws": ["aws", "amazon web services"],
    "cloud computing": ["cloud computing", "cloud infrastructure", "gcp", "azure", "aws"],
    "generative ai": ["generative ai", "genai", "gen ai", "foundation models"],
    "llms": ["llm", "llms", "large language models", "transformer models"],
    "data structures": ["data structures", "dsa"],
    "algorithms": ["algorithms", "algorithmic problem solving", "algorithm design"],
    "ui design": ["ui design", "user interface design", "ui"],
    "ux research": ["ux research", "user experience research", "user research", "ux"],
    "data visualization": ["data visualization", "data viz", "dashboarding", "charts"],
    "data pipelines": ["data pipeline", "data pipelines", "data ingestion"],
    "business analysis": ["business analysis", "business requirements", "ba"],
    "design systems": ["design system", "design systems", "component library"],
}

def load_jobs(filepath: str = None) -> List[Dict[str, Any]]:
    if filepath is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(base_dir, "data", "jobs.json")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def skill_in_text(skill_name: str, text_lower: str) -> bool:
    """
    Checks if a skill or its known aliases exist in the lowercased text
    using word-boundary regex patterns to avoid partial word collisions.
    """
    skill_clean = skill_name.lower().strip()
    patterns_to_check = [skill_clean]
    
    if skill_clean in SKILL_ALIASES:
        patterns_to_check.extend([a.lower() for a in SKILL_ALIASES[skill_clean]])
        
    for pat in patterns_to_check:
        # Escape special regex characters except when handling slashes/dots
        escaped = re.escape(pat)
        # Handle word boundary: if pattern starts/ends with alphanumeric, enforce boundary
        regex = r"(?<![a-zA-Z0-9])" + escaped + r"(?![a-zA-Z0-9])"
        if re.search(regex, text_lower):
            return True
            
    return False

def evaluate_job_match(resume_text: str, job: Dict[str, Any]) -> Dict[str, Any]:
    text_lower = resume_text.lower()
    
    required_skills = job.get("required_skills", [])
    preferred_skills = job.get("preferred_skills", [])
    
    matching_required = [
        skill for skill in required_skills if skill_in_text(skill, text_lower)
    ]
    missing_required = [
        skill for skill in required_skills if skill not in matching_required
    ]
    
    matching_preferred = [
        skill for skill in preferred_skills if skill_in_text(skill, text_lower)
    ]
    
    total_req = len(required_skills)
    matched_req_count = len(matching_required)
    
    if total_req > 0:
        base_score = (matched_req_count / total_req) * 100.0
    else:
        base_score = 0.0
        
    # Small bonus for preferred skills without overwhelming required qualifications
    if preferred_skills and len(preferred_skills) > 0:
        pref_ratio = len(matching_preferred) / len(preferred_skills)
        bonus = pref_ratio * 10.0
        final_score = round(min(100.0, (base_score * 0.9) + bonus))
    else:
        final_score = round(base_score)
        
    return {
        "id": job.get("id"),
        "title": job.get("title"),
        "company": job.get("company"),
        "location": job.get("location"),
        "experience": job.get("experience", "1-3 years"),
        "match_score": int(final_score),
        "matching_skills": matching_required,
        "missing_skills": missing_required,
        "preferred_matches": matching_preferred,
        "total_required": total_req,
        "matched_required_count": matched_req_count,
        "description": job.get("description", "")
    }

def find_top_matches(resume_text: str, top_n: int = 5) -> List[Dict[str, Any]]:
    jobs = load_jobs()
    evaluated = [evaluate_job_match(resume_text, job) for job in jobs]
    
    # Sort primarily by match_score descending, then by matched_required_count
    evaluated.sort(key=lambda x: (x["match_score"], x["matched_required_count"]), reverse=True)
    return evaluated[:top_n]
