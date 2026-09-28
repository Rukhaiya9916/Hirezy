import os
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, record_match_analysis, get_analytics_summary
from resume_parser import extract_text_from_pdf
from matching_engine import find_top_matches, load_jobs
from schemas import (
    ResumeExtractResponse,
    MatchRequest,
    MatchResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    RoadmapRequest,
    RoadmapResponse,
    ImproveResumeRequest,
    ImproveResumeResponse,
    RejectionAnalyzerRequest,
    RejectionAnalyzerResponse,
    InterviewQuestionsRequest,
    InterviewQuestionsResponse,
    SkillProofRequest,
    SkillProofResponse,
    AnalyticsResponse
)
from gemini_service import (
    test_gemini_connection,
    analyze_resume_for_job,
    generate_roadmap,
    suggest_resume_improvements,
    analyze_rejection_risks,
    generate_interview_questions,
    generate_skill_proof
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure SQLite database and tables exist
    init_db()
    yield

app = FastAPI(
    title="Skillora - AI Resume Screening & Job Matching",
    description="Turn Your Resume Into Your Career Roadmap with AI-Powered Matching, Skill Gap Analysis & Proof Engine",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static directory for CSS/assets if needed
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Skillora Backend is running. Please place static/index.html in the static directory."}

@app.get("/api/ai/test")
def test_ai():
    """Tests whether Gemini API connectivity is active."""
    return test_gemini_connection()

@app.get("/api/jobs")
def get_all_jobs():
    """Returns the full catalog of benchmark job listings."""
    try:
        jobs = load_jobs()
        return {"success": True, "jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-resume", response_model=ResumeExtractResponse)
async def extract_resume(file: UploadFile = File(...)):
    """Accepts a PDF resume, extracts text in-memory, and returns the parsed text."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF resume files (.pdf) are supported.")

    file_bytes = await file.read()
    success, result = extract_text_from_pdf(file_bytes)

    if not success:
        return ResumeExtractResponse(
            success=False,
            text="",
            message=result,
            character_count=0
        )

    return ResumeExtractResponse(
        success=True,
        text=result,
        message="Resume extracted successfully.",
        character_count=len(result)
    )

@app.post("/api/match-jobs", response_model=MatchResponse)
def match_jobs(payload: MatchRequest):
    """
    Evaluates candidate resume text against all jobs in data/jobs.json.
    Returns the top 5 matching jobs with match percentages, matching skills, and missing skills.
    """
    resume_text = payload.resume_text.strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text is empty.")

    try:
        top_jobs = find_top_matches(resume_text, top_n=5)
        return MatchResponse(
            success=True,
            jobs=top_jobs,
            total_jobs_evaluated=len(load_jobs()),
            message=f"Found {len(top_jobs)} relevant positions."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching calculation error: {str(e)}")

@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_job(payload: AnalyzeRequest):
    """
    Deep Gemini analysis of resume against a specific target job.
    Records analysis to SQLite for bias-aware tracking analytics.
    """
    resume_text = payload.resume_text.strip()
    job = payload.job

    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")
    if not job or "title" not in job:
        raise HTTPException(status_code=400, detail="Valid job details must be provided.")

    analysis = analyze_resume_for_job(resume_text, job)
    
    # Store record in SQLite database
    try:
        record_match_analysis(
            job_title=job.get("title", "Unknown Role"),
            match_score=analysis.get("match_score", 0),
            matching_skills=analysis.get("matching_skills", []),
            missing_skills=analysis.get("missing_skills", [])
        )
    except Exception as e:
        print(f"[DB save error] {e}")

    return AnalyzeResponse(
        success=True,
        match_score=analysis.get("match_score", 0),
        matching_skills=analysis.get("matching_skills", []),
        missing_skills=analysis.get("missing_skills", []),
        experience_match=analysis.get("experience_match", ""),
        gaps=analysis.get("gaps", []),
        recommendations=analysis.get("recommendations", [])
    )

@app.post("/api/roadmap", response_model=RoadmapResponse)
def get_roadmap(payload: RoadmapRequest):
    """Generates an actionable multi-phase learning roadmap for missing skills."""
    res = generate_roadmap(payload.resume_text, payload.missing_skills, payload.target_role or "")
    return RoadmapResponse(
        success=True,
        target_role=res.get("target_role", payload.target_role or "Target Role"),
        summary=res.get("summary", ""),
        phases=res.get("phases", [])
    )

@app.post("/api/improve-resume", response_model=ImproveResumeResponse)
def improve_resume(payload: ImproveResumeRequest):
    """Generates concrete resume improvement suggestions without inventing fake achievements."""
    res = suggest_resume_improvements(payload.resume_text, payload.target_role or "")
    return ImproveResumeResponse(
        success=True,
        overall_critique=res.get("overall_critique", ""),
        formatting_score=res.get("formatting_score", 75),
        suggestions=res.get("suggestions", []),
        key_keywords_to_add=res.get("key_keywords_to_add", [])
    )

@app.post("/api/rejection-analyzer", response_model=RejectionAnalyzerResponse)
def rejection_analyzer(payload: RejectionAnalyzerRequest):
    """Audits potential screening risks and ATS blockers using ethical, objective phrasing."""
    res = analyze_rejection_risks(
        payload.resume_text,
        payload.job_title,
        payload.job_description or "",
        payload.match_analysis
    )
    return RejectionAnalyzerResponse(
        success=True,
        disclaimer=res.get("disclaimer", "Possible screening concerns include the following points."),
        risk_level=res.get("risk_level", "Moderate"),
        screening_concerns=res.get("screening_concerns", []),
        ats_formatting_risks=res.get("ats_formatting_risks", []),
        recommended_fixes=res.get("recommended_fixes", [])
    )

@app.post("/api/interview-questions", response_model=InterviewQuestionsResponse)
def interview_questions(payload: InterviewQuestionsRequest):
    """Generates 3 technical, 2 project, 2 behavioral, and 2 job-specific questions."""
    res = generate_interview_questions(payload.resume_text, payload.job_title, payload.job_description or "")
    return InterviewQuestionsResponse(
        success=True,
        job_title=payload.job_title,
        technical_questions=res.get("technical_questions", []),
        project_questions=res.get("project_questions", []),
        behavioral_questions=res.get("behavioral_questions", []),
        job_specific_questions=res.get("job_specific_questions", [])
    )

@app.post("/api/skill-proof", response_model=SkillProofResponse)
def skill_proof(payload: SkillProofRequest):
    """Skill Proof Engine: Generates 3-5 assessment questions/tasks for a given skill."""
    skill = payload.skill.strip()
    if not skill:
        raise HTTPException(status_code=400, detail="Skill name cannot be empty.")
    res = generate_skill_proof(skill)
    return SkillProofResponse(
        success=True,
        skill=skill,
        challenges=res.get("challenges", [])
    )

@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics():
    """Returns stored screening records and statistical analytics."""
    data = get_analytics_summary()
    return AnalyticsResponse(
        success=True,
        jobs_analyzed=data["jobs_analyzed"],
        average_match_score=data["average_match_score"],
        most_common_skill_gap=data["most_common_skill_gap"],
        top_matching_skill=data["top_matching_skill"],
        recent_records=data["recent_records"]
    )
