import sqlite3
import os
import json
from collections import Counter
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillora.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database table if it doesn't already exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                job_title TEXT NOT NULL,
                match_score INTEGER NOT NULL,
                matching_skills TEXT NOT NULL,
                missing_skills TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()

def record_match_analysis(job_title: str, match_score: int, matching_skills: List[str], missing_skills: List[str]):
    """Records an analysis event into SQLite for screening analytics."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO analysis_records (job_title, match_score, matching_skills, missing_skills)
            VALUES (?, ?, ?, ?)
            """,
            (
                job_title,
                match_score,
                json.dumps(matching_skills),
                json.dumps(missing_skills)
            )
        )
        conn.commit()
    finally:
        conn.close()

def get_analytics_summary() -> Dict[str, Any]:
    """Computes summary analytics based on stored analysis records."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_records ORDER BY id DESC")
        rows = cursor.fetchall()
    finally:
        conn.close()
        
    if not rows:
        return {
            "jobs_analyzed": 0,
            "average_match_score": 0.0,
            "most_common_skill_gap": "None recorded yet",
            "top_matching_skill": "None recorded yet",
            "recent_records": []
        }
        
    total_count = len(rows)
    total_score = sum(r["match_score"] for r in rows)
    avg_score = round(total_score / total_count, 1)
    
    all_matching = []
    all_missing = []
    recent_records = []
    
    for r in rows:
        try:
            m_skills = json.loads(r["matching_skills"])
            all_matching.extend(m_skills)
        except Exception:
            pass
            
        try:
            gap_skills = json.loads(r["missing_skills"])
            all_missing.extend(gap_skills)
        except Exception:
            pass
            
        if len(recent_records) < 5:
            recent_records.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "job_title": r["job_title"],
                "match_score": r["match_score"],
                "matching_skills": json.loads(r["matching_skills"]) if isinstance(r["matching_skills"], str) else [],
                "missing_skills": json.loads(r["missing_skills"]) if isinstance(r["missing_skills"], str) else []
            })
            
    top_match = Counter(all_matching).most_common(1)
    top_match_str = f"{top_match[0][0]} ({top_match[0][1]}x)" if top_match else "None"
    
    top_gap = Counter(all_missing).most_common(1)
    top_gap_str = f"{top_gap[0][0]} ({top_gap[0][1]}x)" if top_gap else "None"
    
    return {
        "jobs_analyzed": total_count,
        "average_match_score": avg_score,
        "most_common_skill_gap": top_gap_str,
        "top_matching_skill": top_match_str,
        "recent_records": recent_records
    }
