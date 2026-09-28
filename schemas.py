from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ResumeExtractResponse(BaseModel):
    success: bool
    text: str
    message: Optional[str] = None
    character_count: Optional[int] = 0

class JobModel(BaseModel):
    id: int
    title: str
    company: str
    location: str
    experience: str
    description: str
    required_skills: List[str]
    preferred_skills: List[str] = []

class JobMatchResult(BaseModel):
    id: int
    title: str
    company: str
    location: str
    experience: str
    match_score: int
    matching_skills: List[str]
    missing_skills: List[str]
    preferred_matches: Optional[List[str]] = []
    total_required: int
    matched_required_count: int

class MatchRequest(BaseModel):
    resume_text: str

class MatchResponse(BaseModel):
    success: bool
    jobs: List[JobMatchResult]
    total_jobs_evaluated: int
    message: Optional[str] = None

class AnalyzeRequest(BaseModel):
    resume_text: str
    job: Dict[str, Any]

class AnalyzeResponse(BaseModel):
    success: bool
    match_score: int
    matching_skills: List[str]
    missing_skills: List[str]
    experience_match: str
    gaps: List[str]
    recommendations: List[str]
    bias_audit_note: Optional[str] = (
        "Evaluation strictly verified based on objective skills, experience, and project qualifications."
    )

class RoadmapRequest(BaseModel):
    resume_text: str
    missing_skills: List[str]
    target_role: Optional[str] = ""

class RoadmapPhase(BaseModel):
    phase: str
    timeframe: str
    focus_skills: List[str]
    learning_objectives: List[str]
    recommended_resources: List[str]
    hands_on_project: str

class RoadmapResponse(BaseModel):
    success: bool
    target_role: str
    summary: str
    phases: List[RoadmapPhase]

class ImproveResumeRequest(BaseModel):
    resume_text: str
    target_role: Optional[str] = ""

class ImprovementSuggestion(BaseModel):
    category: str  # e.g., 'Action Verbs & Impact', 'Quantifiable Metrics', 'Structure & ATS'
    current_issue: str
    actionable_fix: str
    example: str

class ImproveResumeResponse(BaseModel):
    success: bool
    overall_critique: str
    formatting_score: int
    suggestions: List[ImprovementSuggestion]
    key_keywords_to_add: List[str]

class RejectionAnalyzerRequest(BaseModel):
    resume_text: str
    job_title: str
    job_description: Optional[str] = ""
    match_analysis: Optional[Dict[str, Any]] = None

class ScreeningConcern(BaseModel):
    concern_area: str
    possible_concern: str
    mitigation_strategy: str

class RejectionAnalyzerResponse(BaseModel):
    success: bool
    disclaimer: str = (
        "Possible screening concerns include the following points. "
        "These are algorithmic risk indicators and do not represent an official rejection notice."
    )
    risk_level: str  # Low, Moderate, High
    screening_concerns: List[ScreeningConcern]
    ats_formatting_risks: List[str]
    recommended_fixes: List[str]

class InterviewQuestionsRequest(BaseModel):
    resume_text: str
    job_title: str
    job_description: Optional[str] = ""

class InterviewQuestion(BaseModel):
    category: str  # Technical, Project, Behavioral, Job-Specific
    question: str
    context_or_intent: str
    suggested_talking_points: List[str]

class InterviewQuestionsResponse(BaseModel):
    success: bool
    job_title: str
    technical_questions: List[InterviewQuestion]
    project_questions: List[InterviewQuestion]
    behavioral_questions: List[InterviewQuestion]
    job_specific_questions: List[InterviewQuestion]

class SkillProofRequest(BaseModel):
    skill: str

class SkillProofItem(BaseModel):
    type: str  # Coding Task, Conceptual Challenge, Scenario Problem
    question: str
    difficulty: str
    evaluation_criteria: str
    sample_answer_outline: str

class SkillProofResponse(BaseModel):
    success: bool
    skill: str
    challenges: List[SkillProofItem]

class AnalyticsResponse(BaseModel):
    success: bool
    jobs_analyzed: int
    average_match_score: float
    most_common_skill_gap: str
    top_matching_skill: str
    recent_records: List[Dict[str, Any]]
