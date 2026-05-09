from pydantic import BaseModel, Field
from typing import List, Optional

class CandidateProfile(BaseModel):
    skills: List[str] = Field(default_factory=list, description="List of technical and soft skills extracted from the CV")
    experience_years: Optional[int] = Field(None, description="Total years of experience")
    role_titles: List[str] = Field(default_factory=list, description="Previous or current job titles")
    location_preference: Optional[str] = Field(None, description="Preferred work location if mentioned")
    summary: Optional[str] = Field(None, description="A brief summary of the candidate's profile")

class SearchQueries(BaseModel):
    queries: List[str] = Field(description="List of 3 to 5 optimized job search queries based on the candidate profile")

class JobResult(BaseModel):
    title: str = Field(description="Job title")
    company: Optional[str] = Field("Unknown", description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    description: str = Field(description="Job description snippet or full text")
    url: Optional[str] = Field(None, description="Link to the job posting")

class EvaluatedJob(JobResult):
    score: int = Field(description="Match score from 1 to 100")
    reasoning: str = Field(description="A short reasoning paragraph explaining why this job is a strong or weak match")

class SearchResponse(BaseModel):
    candidate_profile: CandidateProfile
    matched_jobs: List[EvaluatedJob]

class BatchEvaluatedJobs(BaseModel):
    evaluations: List[EvaluatedJob] = Field(description="List of evaluated job matches")

class EmailRequest(BaseModel):
    email: str
    candidate_profile: CandidateProfile
    matched_jobs: List[EvaluatedJob]

class ApplyRequest(BaseModel):
    candidate_profile: CandidateProfile
    job: EvaluatedJob

class ApplyResponse(BaseModel):
    cover_letter: str
    application_tips: List[str]
