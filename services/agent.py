import os
import time
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from models import CandidateProfile, SearchQueries, JobResult, EvaluatedJob, SearchResponse, BatchEvaluatedJobs, ApplyResponse
from google.api_core.exceptions import ResourceExhausted

def get_llm():
    # 1. Check for Groq (Recommended for stability/speed)
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_api_key,
            temperature=0.2
        )

    # 2. Check for Google Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key.startswith("AIza"):
        # Switch to stable gemini-2.0-flash for better free tier limits
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
    # 3. Fallback to OpenAI-compatible (AI ML API)
    return ChatOpenAI(
        model="gemini-1.5-pro",
        openai_api_key=api_key,
        openai_api_base="https://api.aimlapi.com",
        temperature=0.2
    )

def invoke_with_retry(runnable, input_data, retries=3):
    """Helper to handle ResourceExhausted with exponential backoff."""
    for i in range(retries):
        try:
            return runnable.invoke(input_data)
        except ResourceExhausted:
            if i == retries - 1:
                raise
            wait = (2 ** i) * 10
            print(f"Rate limited (Gemini). Waiting {wait}s before retry {i+1}/{retries}")
            time.sleep(wait)
        except Exception as e:
            # Re-raise other exceptions immediately
            raise e

def run_agent_workflow(vectorstore) -> SearchResponse:
    """
    Executes the autonomous agentic loop: Ingest -> Plan -> Search -> Evaluate.
    """
    # Initialize the LLM
    llm = get_llm()
    
    # 1. Ingest (RAG)
    # Search for more specific terms to get a better profile
    docs = vectorstore.similarity_search("technical skills, professional experience, work history, education, certifications, projects", k=8)
    context = "\n".join([doc.page_content for doc in docs])
    
    print("Extracting candidate profile...")
    profile_llm = llm.with_structured_output(CandidateProfile)
    profile = invoke_with_retry(profile_llm, 
        f"Based on the following excerpts from a candidate's CV, extract their professional profile:\n\n"
        f"{context}\n\n"
        "Ensure you provide a comprehensive summary and extract all relevant technical skills."
    )
    
    # Ensure profile has a summary if LLM missed it
    if not profile.summary:
        profile.summary = f"Professional with experience in {', '.join(profile.skills[:5])}."

    # 2. Plan
    print("Generating search plan...")
    query_llm = llm.with_structured_output(SearchQueries)
    search_plan = invoke_with_retry(query_llm, 
        f"Candidate Profile: {profile.model_dump_json()}\n\n"
        "Generate 3 to 5 optimized job search queries to find real-time job listings that perfectly match this candidate's skills and experience. "
        "Focus on specific job boards or direct company career pages if possible."
    )
    
    # 3. Search
    from tavily import TavilyClient
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    raw_jobs = []
    seen_urls = set()
    
    print(f"Executing search queries: {search_plan.queries}")
    for query in search_plan.queries:
        try:
            # Increase max_results slightly for more breadth
            search_result = tavily.search(query=query, search_depth="basic", max_results=5)
            results = search_result.get("results", [])
            print(f"Query '{query}' found {len(results)} results.")
            
            for res in results:
                url = res.get("url", "")
                # Filter out obvious non-job URLs if any
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_jobs.append({
                        "description": res.get("content", ""),
                        "url": url,
                        "title": res.get("title", "Unknown Position")
                    })
        except Exception as e:
            print(f"Error during Tavily search for query '{query}': {e}")
            
    print(f"Total unique raw jobs found: {len(raw_jobs)}")
    
    # 4. Evaluate & Rank (BATCHED & CHUNKED)
    if not raw_jobs:
        print("No jobs found by Tavily. Returning empty list.")
        return SearchResponse(candidate_profile=profile, matched_jobs=[])

    # Sort out potentially irrelevant jobs (short snippets) if we have many
    if len(raw_jobs) > 15:
        raw_jobs = raw_jobs[:15]

    print(f"Evaluating {len(raw_jobs)} jobs in chunks...")
    
    all_evaluated_jobs = []
    # Smaller chunks are more reliable for structured output
    chunk_size = 4 
    
    for i in range(0, len(raw_jobs), chunk_size):
        chunk = raw_jobs[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}...")
        
        jobs_text = ""
        for j, job in enumerate(chunk):
            jobs_text += f"--- JOB #{j+1} ---\nTitle: {job['title']}\nURL: {job['url']}\nSnippet: {job['description'][:2000]}\n\n"

        prompt = (
            f"Candidate Profile:\n{profile.model_dump_json()}\n\n"
            f"Here are {len(chunk)} job listings found on the web:\n\n"
            f"{jobs_text}"
            "Evaluate these job matches against the candidate profile. For EACH job:\n"
            "1. Score it from 1 to 100 based on fit (technical skills, experience level, role).\n"
            "2. Extract the actual job title and company name from the snippet.\n"
            "3. Provide a concise reasoning paragraph.\n"
            "4. IMPORTANT: Include the original URL for each job."
        )
        
        try:
            # We use a fresh LLM instance with a clear schema for each chunk
            batch_eval_llm = llm.with_structured_output(BatchEvaluatedJobs)
            batch_result = invoke_with_retry(batch_eval_llm, prompt)
            
            if batch_result and batch_result.evaluations:
                # Ensure URLs are correctly mapped if missing
                for idx, eval_job in enumerate(batch_result.evaluations):
                    if idx < len(chunk) and (not eval_job.url or eval_job.url == "string"):
                        eval_job.url = chunk[idx]['url']
                
                all_evaluated_jobs.extend(batch_result.evaluations)
                
        except Exception as e:
            print(f"Error during chunk evaluation: {e}")
            # Continue to next chunk instead of failing entirely
            continue

    # Filter out very low quality matches (score < 30)
    final_jobs = [j for j in all_evaluated_jobs if j.score >= 30]
            
    # Sort jobs by score descending
    final_jobs.sort(key=lambda x: x.score, reverse=True)
    
    print(f"Workflow complete. Found {len(final_jobs)} qualified matches.")
    
    return SearchResponse(
        candidate_profile=profile,
        matched_jobs=final_jobs
    )

def generate_cover_letter(profile: CandidateProfile, job: EvaluatedJob) -> ApplyResponse:
    """
    Generates a tailored cover letter and application tips for a specific job.
    """
    llm = get_llm()
    
    prompt = (
        f"Candidate Profile: {profile.model_dump_json()}\n\n"
        f"Job Details:\nTitle: {job.title}\nCompany: {job.company}\nDescription: {job.description}\n\n"
        "Generate a professional and persuasive cover letter (max 300 words) tailored to this specific job. "
        "Also provide 3 specific tips for the candidate to stand out during the application process for this role. "
        "Return the output as a structured object with 'cover_letter' and 'application_tips' fields."
    )
    
    # We use structured output for consistency
    apply_llm = llm.with_structured_output(ApplyResponse)
    response = invoke_with_retry(apply_llm, prompt)
    
    return response

