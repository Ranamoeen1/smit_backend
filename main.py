import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

from models import SearchResponse, EmailRequest, ApplyRequest, ApplyResponse
from services.document_parser import parse_cv
from services.vector_store import store_cv_in_vector_db
from services.agent import run_agent_workflow, generate_cover_letter
from services.email_service import send_job_summary_email

load_dotenv()

app = FastAPI(title="JobScout AI", description="Agentic AI for Job Searching based on CVs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/jobs/search", response_model=SearchResponse)
async def search_jobs(file: UploadFile = File(...)):
    """
    Endpoint to upload a CV (PDF or DOCX) and get an autonomous job search result.
    """
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
        
    try:
        # Read file contents
        contents = await file.read()
        
        # Parse text from CV
        text = parse_cv(contents, file.filename)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the provided file.")
            
        # Initialize Vector Database (RAG)
        vectorstore = store_cv_in_vector_db(text)
        
        # Run Agent Workflow (Plan, Search, Evaluate)
        response = run_agent_workflow(vectorstore)
        
        return response
        
    except ResourceExhausted:
        raise HTTPException(
            status_code=429, 
            detail="Gemini quota exceeded. Our agent is resting to recover limits. Please try again in 1 minute."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/api/v1/jobs/email")
async def email_jobs(request: EmailRequest):
    """
    Endpoint to send the job match summary via email.
    """
    try:
        send_job_summary_email(request)
        return {"message": "Email sent successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.post("/api/v1/jobs/apply", response_model=ApplyResponse)
async def apply_to_job(request: ApplyRequest):
    """
    Endpoint to generate a tailored application (cover letter) for a specific job.
    """
    try:
        response = generate_cover_letter(request.candidate_profile, request.job)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate application: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "JobScout AI is running"}
